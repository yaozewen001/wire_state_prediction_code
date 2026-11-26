"""复用整墙配对、功率、标签及选层，组装 WSP-Mamba 的三段历史。"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if __package__:
    from . import config  # 初始化 IDE 直跑所需的公共代码路径。
else:
    import config

from cnn_power_dataset import fit_normalization, load_sequences, power_history, power_offsets_us, window_temperature
from data_utils import OrderedLayerBatchSampler, ordered_rows, read_csv
from training_acceleration import WindowBatchCollator
from wall_dataset import WindowDataset as BaseWindowDataset
from wall_dataset import build_windows


class WindowDataset(BaseWindowDataset):
    """沿用图像及窗口接口，把短温度、长温度和功率按固定次序打包。"""

    packed_images = True  # 以 uint8 传输，进入 Atto 前再归一化。

    def __init__(self, sequences, windows, settings, normalization):
        super().__init__(sequences, windows, settings, normalization)
        self.power_offsets = power_offsets_us(settings)

    def __getitem__(self, index):
        images, _, label, index = super().__getitem__(index)
        window = self.windows[index]
        sequence = self.sequences[window["sequence_id"]]
        start, cutoff, selections = window_temperature(sequence, window, self.settings)
        values = sequence["values"][start:window["end_seq_idx"] + 1].reshape(-1)
        histories = []
        for indices, key in zip(selections, ("short_temperature_steps", "long_temperature_steps")):
            branch = np.zeros(self.settings[key], dtype=np.float32)
            if indices.size:
                # 因采集时间晚于图像截止而缺失的点不入窗，左侧以标准化后的零补齐。
                branch[-indices.size:] = (values[indices] - self.normalization["mean"]) / self.normalization["std"]
            histories.append(branch)
        power = power_history(sequence["power_schedule"], cutoff, sequence["image_times_us"][start],
                              self.power_offsets, f"{window['sequence_id']} 第 {window['layer_id']} 层")
        payload = np.concatenate((*histories, power.astype(np.float32)))[None, :]
        return images, torch.from_numpy(payload), label, index


class CachedFeatureDataset(Dataset):
    """保存同一冻结编码器一次前向得到的逐窗向量，保留原始窗口索引。"""

    def __init__(self, source, features):
        self.settings = source.settings
        self.windows = source.windows
        self.sequences = source.sequences
        self.features = features
        self.labels = torch.tensor([window["label"] for window in self.windows], dtype=torch.long)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return self.features[index], self.labels[index], index


def make_loader(dataset, settings, device, *, training=False):
    """复用同墙、同层、同连续段采样器及图像去重打包。"""
    cached = isinstance(dataset, CachedFeatureDataset)
    batch_key = "head_batch_size" if cached else ("batch_size" if training else "evaluation_batch_size")
    sampler = OrderedLayerBatchSampler(
        dataset.windows, settings[batch_key],
        layer_shuffle_seed=settings["seed"] if training else None,
    )
    workers = 0 if cached else settings["num_workers"]
    options = {}
    if workers:
        options.update(persistent_workers=settings["persistent_workers"], prefetch_factor=settings["prefetch_factor"])
    collate = None if cached else WindowBatchCollator(dataset.windows, settings["reuse_image_features"])
    return DataLoader(
        dataset, batch_sampler=sampler, num_workers=workers, collate_fn=collate,
        pin_memory=device.type == "cuda" and settings["pin_memory"],
        generator=torch.Generator().manual_seed(settings["seed"]), **options,
    )


def select_reference_windows(sequences, train_windows, settings):
    """从目标熔断帧找到同一预测跨度的训练窗，兼容旧末帧区间。"""
    selections = settings["reference_windows"]
    if not selections:
        raise ValueError("BPM 阶段需先填 config.py 的 REFERENCE_WINDOWS。")
    candidates = defaultdict(list)
    for index, window in enumerate(train_windows):
        if window["current_state"] == "stable" and window["target_state"] == "breakoff":
            candidates[window["sequence_id"], window["layer_id"]].append(index)
    source_frames = {}
    selected = {}
    for selection in selections:
        if not isinstance(selection, (list, tuple)) or len(selection) not in (3, 4):
            raise ValueError("REFERENCE_WINDOWS 每条填写（墙名, 层号, 目标熔断帧原始编号）。")
        wall, layer, first = selection[:3]
        target_anchor = len(selection) == 3
        # 三项按目标帧定位；旧四项仍按历史末帧的闭区间定位。
        last = first if target_anchor else selection[3]
        if not isinstance(wall, str) or type(first) is not int or type(last) is not int or first < 0 or last < first:
            raise ValueError(f"参考帧号格式错误：{selection}")
        if wall not in sequences or sequences[wall]["record"]["split"] != "train":
            raise ValueError(f"BPM 参考墙必须属于当前训练集：{wall}")
        if wall not in source_frames:
            manifest = Path(settings["data_root"]) / sequences[wall]["record"]["relative_folder"] / "frame_manifest.csv"
            rows = ordered_rows(read_csv(manifest), str(manifest))
            source_frames[wall] = [int(row["source_frame_number"]) for row in rows]
        frames = source_frames[wall]
        # 窗口清单已按原采集顺序和 FUTURE_FRAMES 配对，原始帧号只用于定位。
        anchor_key = "target_seq_idx" if target_anchor else "end_seq_idx"
        picks = []
        for index in candidates[wall, str(layer)]:
            window = train_windows[index]
            if first <= frames[window[anchor_key]] <= last:
                picks.append(index)
                selected[index] = dict(
                    sequence_id=wall, layer_id=window["layer_id"], training_window_index=index,
                    end_seq_idx=window["end_seq_idx"], end_source_frame_number=frames[window["end_seq_idx"]],
                    target_seq_idx=window["target_seq_idx"], target_source_frame_number=frames[window["target_seq_idx"]],
                    future_frames=settings["future_frames"],
                )
        if not picks:
            anchor = f"目标熔断帧 {first}" if target_anchor else f"历史末帧区间 {first}–{last}"
            raise ValueError(
                f"{wall} 第 {layer} 层的{anchor}没有满足提前 {settings['future_frames']} 帧的"
                "稳定→熔断训练窗口。请核对有效历史、窗口步长和标注。"
            )
        if target_anchor and len(picks) > 1:
            raise ValueError(f"{wall} 第 {layer} 层有多条训练窗口指向原始目标帧 {first}，请核对帧号。")
    return [selected[index] for index in sorted(selected)]

