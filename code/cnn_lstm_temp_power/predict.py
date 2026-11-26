"""在 IDE 回放一面墙，也可导入实时 FutureStatePredictor。"""

from collections import deque  # 缓存当前有效段的有限历史。
from pathlib import Path  # 定位待回放墙和输出文件。

import numpy as np  # 筛选截至当前的温度和功率。
import torch  # 使用冻结的训练权重前向计算。
from PIL import Image  # 接收采集图像或图片文件。

if __package__:
    from .config import get_settings
    from .model import CNNLSTMTempPower
    from .evaluate import POWER_SNAPSHOT, restore_run
    from cnn_power_dataset import (PowerSchedule, attach_signal_history, check_watts, pack_histories, parse_power_table,
                          power_history, power_offsets_us, temperature_indices, time_us)
else:
    from config import get_settings
    from model import CNNLSTMTempPower
    from evaluate import POWER_SNAPSHOT, restore_run
    from cnn_power_dataset import (PowerSchedule, attach_signal_history, check_watts, pack_histories, parse_power_table,
                          power_history, power_offsets_us, temperature_indices, time_us)

from data_utils import read_json, read_labels, read_paired_frames, sequence_boundaries, write_csv
from wall_dataset import image_positions, preprocess_image, read_image
from wall_runtime import load_weights, setup_environment


class WindowPredictor:
    def __init__(self, model_class, current_settings):
        self.run_dir, self.settings, checkpoint, _, self.report_dir = restore_run(current_settings)
        self.normalization = read_json(self.run_dir / "normalization.json")  # 使用训练时的均值和标准差。
        configuration = read_json(self.run_dir / "model_config.json")
        self.device = setup_environment(self.settings)
        self.model = model_class(**configuration["model_settings"]).to(self.device)
        load_weights(self.model, self.run_dir / checkpoint, self.device)
        self.model.eval()
        self.positions = image_positions(self.settings)
        self.offsets = power_offsets_us(self.settings)
        self.images = deque(maxlen=self.settings["window_size"])
        self.temperatures = deque()  # 只保留可能仍落在长窗内的温度组。
        self.reset()

    def _clear_segment(self):
        self.images.clear()
        self.temperatures.clear()
        self.segment_start_index = None
        self.segment_start_us = None

    def reset(self, sequence_id=None):
        self._clear_segment()
        self.sequence_id = None if sequence_id is None else str(sequence_id)
        self.last_index, self.last_time, self.last_key = None, None, None
        self.power_schedule = PowerSchedule()  # 新墙重新登记功率。
        self.status = "startup"

    def record_power(self, power_setpoint_w, effective_at):
        moment = time_us(effective_at)  # 与图片、温度使用同一采集时钟。
        self.power_schedule.append(moment, check_watts(power_setpoint_w))

    @torch.no_grad()
    def update(
        self, image, temperature, seq_idx, *, timestamp, temperature_timestamps,
        layer_id, processing_ready, segment_id="0", sequence_id=None,
        input_available=True, known_segment_end_index=None,
    ):
        if sequence_id is not None and str(sequence_id) != self.sequence_id:
            self.reset(sequence_id)  # 换墙时重新积累图温及功率。
        if type(seq_idx) is not int or seq_idx < 0 or (self.last_index is not None and seq_idx <= self.last_index):
            raise ValueError("同一面墙的 seq_idx 必须是递增的非负整数。")
        end_us = time_us(timestamp)
        if self.last_time is not None and end_us < self.last_time:
            raise ValueError("同一面墙的图像采集时间不能倒退。")
        if type(processing_ready) is not bool:
            raise ValueError("processing_ready 使用布尔值表示当前加工是否可预测。")
        key = (str(layer_id), str(segment_id))
        new_segment = (not self.images or key != self.last_key or seq_idx != self.last_index + 1)
        allowed = processing_ready and input_available
        if self.settings.get("exclude_first_layer", False) and key[0] == "1":
            allowed = False
        if not allowed:
            self._clear_segment()  # 启动期或忽略帧阻断连续输入。
            self.last_index, self.last_time, self.last_key = seq_idx, end_us, key
            self.status = "unavailable"
            return None

        ratio = self.settings["temperature_samples_per_image"]
        values = np.asarray(temperature, dtype=np.float64)
        times = np.asarray([time_us(value) for value in temperature_timestamps], dtype=np.int64)
        if values.shape != (ratio,) or times.shape != (ratio,) or not np.isfinite(values).all():
            raise ValueError(f"每张图需要 {ratio} 个有限温度值和对应采集时间。")
        size = self.settings["image_size"]
        if isinstance(image, (str, Path)):
            array = read_image(str(image), size)
        else:
            if not isinstance(image, Image.Image):
                pixels = np.asarray(image)
                if pixels.dtype != np.uint8:
                    raise ValueError("在线图像使用 uint8 灰度或 RGB 像素。")
                image = Image.fromarray(pixels)
            array = preprocess_image(image, size)
        if min(array.shape) < 16 or (not new_segment and array.shape != self.images[-1].shape):
            raise ValueError("同一有效段内的图像应尺寸一致，且短边至少 16 像素。")
        context = f"{self.sequence_id or '当前墙'} 第 {key[0]} 层"
        if new_segment:
            self.power_schedule.sample(np.asarray([end_us], dtype=np.int64), context)  # 新段首帧就检查设定功率。
            self._clear_segment()
            self.segment_start_index, self.segment_start_us = seq_idx, end_us
        self.last_index, self.last_time, self.last_key = seq_idx, end_us, key
        self.images.append(array)
        self.temperatures.append((times, values))
        length = self.settings["window_size"]
        lower = end_us - self.settings["long_history_us"]
        while len(self.temperatures) > length and self.temperatures[0][0].max() <= lower:
            self.temperatures.popleft()  # 同时保留最近短窗和所有还可能用于长窗的温度。
        if len(self.images) < length:
            self.status = "warming_up"
            return None
        target = seq_idx + self.settings["future_frames"]
        if known_segment_end_index is not None and target > known_segment_end_index:
            self.status = "segment_end"
            return None
        first_end = self.segment_start_index + length - 1
        if (seq_idx - first_end) % self.settings["window_stride"]:
            self.status = "waiting_stride"
            return None

        times = np.stack([group[0] for group in self.temperatures])
        values = np.concatenate([group[1] for group in self.temperatures])
        indices = temperature_indices(times, end_us, self.settings)
        power_kw = power_history(self.power_schedule, end_us, self.segment_start_us, self.offsets, context)
        histories = pack_histories(values, indices, power_kw, self.settings, self.normalization)
        images = np.stack([self.images[position] for position in self.positions])[None, :, None]
        logits = self.model(torch.from_numpy(images).to(self.device), torch.from_numpy(histories[None]).to(self.device))
        scores = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
        names = self.settings["class_names"]
        self.status = "ready"
        result = {
            "sequence_id": self.sequence_id, "layer_id": key[0], "segment_id": key[1],
            "end_seq_idx": seq_idx, "target_seq_idx": target, "end_time_s": end_us / 1_000_000,
            "target_time_s": end_us / 1_000_000 + self.settings["future_frames"] / self.settings["fps"],
            "power_setpoint_w": float(power_kw[-1] * 1000),
            "long_temperature_points": len(indices[1]), "predicted_class": names[int(scores.argmax())],
            "status": self.status,
        }
        result.update({"score_" + name: float(score) for name, score in zip(names, scores)})
        return result


def predict_main(predictor, current):
    chosen = current["predict_sequence_dir"].strip()
    if chosen:
        folder = Path(chosen).expanduser().resolve()
    else:
        split = read_json(predictor.run_dir / "split_snapshot.json")
        record = next(item for item in split["records"] if item["split"] == "test")
        folder = Path(predictor.settings["data_root"]) / record["relative_folder"]
    sequence, _ = read_paired_frames(
        folder, use_annotation_input_mask=True, exclude_unknown_inputs=True, temperature_input_policy="raw",
    )  # 回放使用当前帧质量标记和已有配对。
    names = predictor.settings["class_names"]
    labels, _, _ = read_labels(folder, len(sequence["values"]), names, annotation_only=True)
    table = parse_power_table((predictor.run_dir / POWER_SNAPSHOT).read_bytes())
    if folder.name not in table:  # 新墙读取当前登记表，原墙沿用训练时的功率。
        table = parse_power_table(Path(current["power_settings_file"]).read_bytes())
    attach_signal_history(sequence, folder, table)
    predictor.reset(folder.name)
    predictor.power_schedule = sequence["power_schedule"]  # 所有查询仍只取预测截止时已生效的值。
    layers, segments = sequence["layer_ids"], sequence["segments"]
    allowed = sequence["input_available"] & (labels >= 0)
    if predictor.settings.get("exclude_first_layer", False):
        allowed &= np.asarray(layers) != "1"
    boundaries = sequence_boundaries(layers, segments, allowed)  # 与训练共用有效段边界。
    rows = []
    for start, stop in zip(boundaries, boundaries[1:]):
        if not allowed[start]:
            continue  # 保留原帧号，下一次有效输入会从新段开始。
        for index in range(start, stop):
            result = predictor.update(
                sequence["paths"][index], sequence["values"][index], index,
                timestamp=sequence["image_times_us"][index] / 1_000_000,
                temperature_timestamps=sequence["temperature_times_us"][index] / 1_000_000,
                layer_id=layers[index], segment_id=segments[index], processing_ready=True,
                known_segment_end_index=stop - 1,
            )
            if result is not None:
                result["end_time_s"] = float(sequence["timestamps_s"][index])  # 回放表使用相对本墙首图的秒数。
                target = result["target_seq_idx"]
                result["target_time_s"] = float(sequence["timestamps_s"][target])  # 推理完成后附上目标采集时间供核对。
                rows.append(result)
    if not rows:
        raise ValueError(f"{folder.name} 没有可回放的有效预测窗口。")
    output = predictor.report_dir / ("online_predictions_" + folder.name + ".csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, rows, list(rows[0]))
    print(f"已回放 {folder.name} 的 {len(rows)} 个有效窗口。预测表 {output}", flush=True)
    return output


class FutureStatePredictor(WindowPredictor):
    def __init__(self, settings=None):
        super().__init__(CNNLSTMTempPower, get_settings() if settings is None else settings)


def main():
    settings = get_settings()
    return predict_main(FutureStatePredictor(settings), settings)  # 读取配置指定的墙和权重。

if __name__ == "__main__":  # 允许 IDE 直接运行，导入时只定义函数。
    main()
