"""三个模型共用数据接口：按unknown标注排除图温，旧模型按保存策略恢复。"""

import hashlib  # 保存输入文件内容标识。
import json  # 保存训练和划分快照。
from collections import Counter  # 汇总墙、分组和窗口数量。
from functools import lru_cache  # 缓存少量已经读取的图像。
from pathlib import Path  # 处理文件路径。

import numpy as np  # 处理图像、温度和标签数组。
import torch  # 把数组转换为模型张量。
from PIL import Image  # 读取灰度图像。
from torch.utils.data import Dataset  # 定义按历史窗口读取的数据集。

from data_utils import (  # 复用公共数据接口，划分与三个模型始终使用相同的读取规则。
    read_csv,  # 公共的read_csv接口，供本模块或训练评估入口复用。
    write_csv,  # 公共的write_csv接口，供本模块或训练评估入口复用。
    read_json,  # 公共的read_json接口，供本模块或训练评估入口复用。
    write_json,  # 公共的write_json接口，供本模块或训练评估入口复用。
    ordered_rows,  # 公共的ordered_rows接口，供本模块或训练评估入口复用。
    natural_key,  # 公共的natural_key接口，供本模块或训练评估入口复用。
    read_paired_frames,  # 公共的read_paired_frames接口，供本模块或训练评估入口复用。
    label_number,  # 公共的label_number接口，供本模块或训练评估入口复用。
    read_labels,  # 公共的read_labels接口，供本模块或训练评估入口复用。
    records_digest,  # 公共的records_digest接口，供本模块或训练评估入口复用。
    find_wall_folders,  # 公共的find_wall_folders接口，供本模块或训练评估入口复用。
    list_sequences,  # 公共的list_sequences接口，供本模块或训练评估入口复用。
    discover_sequences,  # 公共的discover_sequences接口，供本模块或训练评估入口复用。
    validate_membership,  # 公共的validate_membership接口，供本模块或训练评估入口复用。
    load_split,  # 公共的load_split接口，供本模块或训练评估入口复用。
    build_layer_windows,  # 三个模型共用按层建立有序窗口的逻辑。
    uses_startup_context,  # 恢复训练保存的启动策略；旧结果不采用新默认值。
    uses_annotation_startup,  # 新版按人工unknown标注完全排除图温输入。
    filter_training_layer_windows,  # 训练层来自固定划分表；不限制验证和测试。
)


def image_positions(settings):  # 公共接口只依赖已保存配置，不依赖某个模型的 config 模块。
    size, mode = settings["window_size"], settings["image_sampling"]  # 窗口长度和选图规则。
    if mode == "all":  # 全部图像按时间顺序输入。
        return list(range(size))  # 局部窗口允许与前一窗口重叠。
    if mode == "custom":  # 人工配置从 1 开始编号。
        return [position - 1 for position in settings["custom_frame_positions"]]  # 转成内部从 0 开始的位置。
    count = settings["image_steps"]  # 均匀采样只改变局部图像分支。
    return [size - 1] if count == 1 else [index * (size - 1) // (count - 1) for index in range(count)]  # 与三个模型原配置的规则一致。


def load_sequences(records, settings):  # 从完整单道或整墙清单读取输入和逐帧状态。
    sequences, digest = {}, hashlib.sha256()  # 收集配对数据与标签。
    for record in sorted(records, key=lambda row: row["sequence_id"]):  # 固定读取顺序。
        sequence_id = record["sequence_id"]  # 当前完整序列编号。
        folder = Path(settings["data_root"]) / record["relative_folder"]  # 直接使用原始目录。
        sequence, paired_digest = read_paired_frames(folder, use_annotation_input_mask=uses_startup_context(settings) or uses_annotation_startup(settings), exclude_unknown_inputs=uses_annotation_startup(settings), temperature_input_policy=settings.get("temperature_input_policy", "legacy_filtered"))  # 新版训练和回放共用unknown排除掩码。
        if sequence["temperature_samples_per_image"] != settings.get("temperature_samples_per_image", 1):  # 评估恢复保存的样本比例。
            raise ValueError(sequence_id + " 的每图温度数与本次训练配置不一致。")  # 不改变旧模型输入含义。
        labels, label_digest, source = read_labels(folder, len(sequence["values"]), settings["class_names"], record["constant_state"], annotation_only=settings.get("temperature_input_policy") == "raw")  # 展开对应的真实标签。
        if label_digest != record["label_digest"] or source != record["label_source"]:  # 标签变化后不能静默改变评估。
            raise ValueError(sequence_id + " 的标签已变化，请重新划分并训练。")  # 明确来源变更。
        sequence.update(record=record, labels=labels)  # 两种实验共用同一种内存结构。
        sequences[sequence_id] = sequence  # 按序列编号访问。
        digest.update((sequence_id + paired_digest + label_digest).encode("utf-8"))  # 固定本次输入版本。
    return sequences, digest.hexdigest()  # 交给同一训练或评估入口。


def build_windows(sequences, settings):  # 局部窗只使用当前层的有效历史。
    windows, skipped = build_layer_windows(sequences, settings, image_positions(settings))  # 先按统一规则生成局部窗口。
    return filter_training_layer_windows(windows, skipped, settings.get("training_layer_selection"))  # 层选择在划分时已确定，这里只按清单筛选监督。


def require_classes(windows, class_count):  # 训练需要有配置中的类别，验证和测试允许缺少个别类别。
    if any(not samples for samples in windows.values()):  # 三个集合都必须有窗口。
        raise ValueError("某个集合没有有效窗口，请查看 window_report.json，补标签、缩短窗口或补充实验数据。")  # 不临时混用其他集合。
    if {sample["label"] for sample in windows["train"]} != set(range(class_count)):  # 避免学习未出现的输出类别。
        raise ValueError("训练窗口缺少 CLASS_NAMES 中某类；仅研究稳定/熔断时设置两类，或补充对应训练序列。")  # 给出明确配置方向。


def fit_normalization(sequences, training_windows):  # 仅统计训练历史的原始温度点，允许配对重用时也不重复计入同一点。
    masks = {}  # 先去除重叠图像窗口重复访问的位置。
    for window in training_windows:  # 不读取验证、测试温度统计量。
        sequence_id = window["sequence_id"]  # 每面墙分别维护索引。
        if sequence_id not in masks:  # 第一次遇到该墙。
            masks[sequence_id] = np.zeros(len(sequences[sequence_id]["values"]), dtype=bool)  # 按图片组号标记。
        masks[sequence_id][window["start_seq_idx"]:window["end_seq_idx"] + 1] = True  # 局部温度分支读取窗口内全部组。
    collected = []  # 保持每面墙的独立温度索引作用域。
    for sequence_id, mask in masks.items():  # 遍历训练输入涉及的墙。
        sequence = sequences[sequence_id]  # 当前墙的配对记录。
        values = sequence["values"][mask].reshape(-1)  # 展开每图两个温度。
        source_indices = sequence.get("temperature_source_indices")  # 老数据可能没有原温度序号。
        if source_indices is not None:  # 同一原温度被配给相邻图时去重统计。
            selected_indices = source_indices[mask].reshape(-1)  # 仅对应训练历史中的点。
            _, first_positions = np.unique(selected_indices, return_index=True)  # 每个原采样点只取一次。
            values = values[np.sort(first_positions)]  # 保持原读取次序。
        collected.append(values)  # 不跨墙去重相同数字序号。
    if not collected:  # 空训练集不能生成有意义的统计量。
        raise ValueError("没有可用于归一化的训练输入，请检查启动排除与窗口设置。")  # 不借用验证测试温度。
    values = np.concatenate(collected)  # 汇总唯一训练温度。
    if not np.isfinite(values).all():  # 不可用温度不应出现在有效训练窗口中。
        raise ValueError("训练窗口仍含不可用温度，请核对逐帧标签和有效性标记。")  # 不插值或填零。
    return {"mean": float(values.mean()), "std": max(float(values.std()), 1e-6), "unique_training_points": int(values.size)}  # 保留原标准化接口。


def fit_resnet_normalization(sequences, training_windows, exclude_value, exclude_atol):  # 统计普通 ResNet 训练历史中的唯一原始温度点。
    masks = {}  # 按墙记录训练输入实际使用的图温组。
    for window in training_windows:  # 不读取验证集或测试集窗口。
        sequence_id = window["sequence_id"]  # 同一墙的重叠窗口共享一个掩码。
        if sequence_id not in masks:  # 首次遇到当前墙。
            masks[sequence_id] = np.zeros(len(sequences[sequence_id]["values"]), dtype=bool)  # 每张图及其原配对温度只统计一次。
        masks[sequence_id][window["start_seq_idx"]:window["end_seq_idx"] + 1] = True  # 温度分支读取窗口内全部图温组。
    collected = []  # 汇总不同墙的唯一原温度点。
    for sequence_id, mask in masks.items():  # 每面墙的原温度索引具有独立作用域。
        sequence = sequences[sequence_id]  # 读取内存中的配对数据，不修改磁盘文件。
        values = np.asarray(sequence["values"], dtype=np.float64)[mask].reshape(-1)  # 将每图两个温度点按原顺序展开。
        source_indices = sequence.get("temperature_source_indices")  # 同一个原温度点可能被相邻图片复用。
        if source_indices is not None:  # 有原索引时继续去重，避免配对重复影响统计。
            selected_indices = np.asarray(source_indices)[mask].reshape(-1)  # 只选当前训练历史涉及的原温度点。
            _, first_positions = np.unique(selected_indices, return_index=True)  # 每个原温度点只保留第一次出现的位置。
            values = values[np.sort(first_positions)]  # 去重不会重排原始温度次序。
        collected.append(values)  # 不跨墙去重相同数字索引。
    if not collected:  # 没有训练输入时不能借用其他集合计算统计量。
        raise ValueError("没有可拟合温度标准化的训练窗口，请检查划分、标注和窗口大小。")  # 明确数据准备问题。
    all_values = np.concatenate(collected)  # 此数组是副本，不会更改模型实际输入。
    if not np.isfinite(all_values).all():  # 799.9是有限占位值，NaN或无穷值需要另外修复。
        raise ValueError("训练历史含NaN或无穷温度，请先核对配对文件；程序不会自动填值。")  # 不静默制造温度。
    excluded = np.isclose(all_values, exclude_value, rtol=0.0, atol=exclude_atol)  # 仅匹配799.9及其浮点表示误差。
    fitted_values = all_values[~excluded]  # 只在统计副本中排除799.9。
    if fitted_values.size == 0:  # 所有训练温度均为799.9时没有可计算的有效均值。
        raise ValueError("训练历史中的温度全部为799.9，无法按要求计算均值和标准差。")  # 不从验证、测试集补统计量。
    measured_std = float(fitted_values.std(ddof=0))  # 采用总体标准差，与已有公共接口一致。
    result = {"mean": float(fitted_values.mean()), "std": max(measured_std, 1e-6),  # 保存训练集统一统计量，避免零除。
              "unique_training_points": int(all_values.size), "statistics_points_used": int(fitted_values.size),  # 区分总训练温度点和参与统计的点。
              "excluded_799_9_points": int(excluded.sum()), "excluded_value": float(exclude_value),  # 便于检查排除数量。
              "excluded_value_atol": float(exclude_atol), "std_ddof": 0,  # 完整记录计算口径。
              "std_floor_applied": measured_std < 1e-6, "fit_scope": "unique_raw_temperature_points_in_training_input_windows",  # 不统计未来目标温度。
              "input_799_9_policy": "keep_position_and_standardize_with_training_mean_std",  # 799.9输入仍使用同一均值、标准差变换。
              "validation_test_reuse_training_statistics": True}  # 验证、测试和在线推理都不重新拟合。
    print("温度统计：训练历史唯一原采样点{}个；排除799.9共{}个；参与统计{}个；均值{:.4f}，标准差{:.4f}。".format(result["unique_training_points"], result["excluded_799_9_points"], result["statistics_points_used"], result["mean"], result["std"]), flush=True)  # 启动时直接显示核对信息。
    return result  # 公共WindowDataset保留全部温度位置，并读取这里的mean和std。


def save_window_report(output, windows, skipped, training_order="wall_layer_time"):  # 保存有序窗口与实际训练排序方式。
    output = Path(output)  # 定位本次训练目录。
    report = {split: {"windows": len(samples), "sequences": len({sample["sequence_id"] for sample in samples}),  # 保留原墙数和窗口数。
              "layers": len({(sample["sequence_id"], sample["layer_id"]) for sample in samples}),  # 当前集合实际产生有效窗口的墙层数。
              "target_classes": dict(Counter(sample["target_state"] for sample in samples)),  # 直接显示未来目标的英文状态和数量。
              "state_changes": sum(sample["label"] != sample["current_label"] for sample in samples)} for split, samples in windows.items()}  # 检查当前到未来确有变化的样本数。
    report["training_order"] = training_order  # 训练时墙与墙之间可每轮重排。
    report["window_table_order"] = "wall_layer_time"  # 窗口表维持固定顺序，便于追溯。
    policy = skipped[0].get("startup_policy", "exclude") if skipped else "exclude"  # 三集合来自同一个训练配置。
    context_only = policy == "context_only"  # 只对旧v7结果报告启动上下文。
    report["history_scope"] = "current_layer_with_startup_context" if context_only else "current_layer_only_after_startup"
    report["startup_exclusion"] = {"rule": "exclude_unknown_inputs_by_annotation" if policy == "annotation_exclude" else "retain_inputs_mask_supervision_and_display" if context_only else "skip_first_frames_once_per_layer", "startup_policy": policy, "frames_per_layer": sorted({int(rule["skip_frames"]) for row in skipped for rule in row.get("layer_start_rules", {}).values()} or {int(row.get("layer_skip_frames", 0)) for row in skipped}), "per_wall_layer": {row["sequence_id"]: row.get("layer_start_rules", {}) for row in skipped}, "excluded_image_temperature_groups": sum(row.get("reasons", {}).get("startup_frames_excluded", 0) for row in skipped), "unknown_image_temperature_groups_excluded": sum(row.get("reasons", {}).get("unknown_image_temperature_groups_excluded", 0) for row in skipped), "masked_startup_labels": sum(row.get("reasons", {}).get("startup_labels_masked", 0) for row in skipped), "annotated_startup_frames": sum(row.get("reasons", {}).get("startup_annotation_prefix_frames", 0) for row in skipped), "available_startup_context_groups": sum(row.get("reasons", {}).get("startup_context_available", 0) for row in skipped)}  # 按标注直接排除输入，数量只用于核对。
    report["supervision_policy"] = {"unknown_is_output_class": False, "unknown_history_allowed": context_only, "current_and_future_must_be_known": context_only or policy == "annotation_exclude", "truth_used_as_model_input": False}  # 新版未知图片和温度都排除。
    report["filtered_windows"] = skipped  # 保存未知、忽略、温度无效和层段过短的原因。
    if any("first_layer_frame_groups_excluded" in row["reasons"] for row in skipped):  # 只有启用整层排除的运行写此统计。
        report["first_layer_exclusion"] = {split: sum(row["reasons"].get("first_layer_frame_groups_excluded", 0)  # 统计每个集合排除的图温组。
                                                     for row in skipped if row["split"] == split) for split in windows}  # 不混合训练、验证和测试。
    report["training_layer_selection"] = {row["sequence_id"]: row["training_layer_selection"] for row in skipped if "training_layer_selection" in row}  # 记录当前训练实际采用的层，不另建CSV。
    write_json(output / "window_report.json", report)  # 模型训练之前就能查看此报告。
    fields = ["sequence_id", "group_id", "split", "layer_id", "segment_id", "segment_start_seq_idx", "start_seq_idx", "end_seq_idx", "target_seq_idx", "end_time_s", "target_time_s", "current_label", "label", "current_state", "target_state", "selected_image_indices"]  # 窗口表明确包含层号。
    extra_fields = ["effective_start_seq_idx", "layer_frame_number_1based", "target_layer_frame_number_1based", "layer_skip_frames", "context_unknown_frames", "context_startup_frames", "has_recorded_timestamps"]
    fields.extend(field for field in extra_fields if field not in fields)  # 保留原窗口表常用列。
    rows = []  # 列表转成紧凑可查看的文本。
    for samples in windows.values():  # 保留固定集合和墙层顺序。
        for sample in samples:  # 每个样本对应一次监督输出。
            row = {key: value for key, value in sample.items() if key != "image_indices"}  # 不导出未来图像数据。
            row["selected_image_indices"] = ";".join(map(str, sample["image_indices"]))  # 局部选图索引。
            rows.append(row)  # 当前窗口加入输出。
    write_csv(output / "windows.csv", rows, fields)  # 查看每个窗口到底用了哪些图片及哪个未来目标。
    return report  # 供训练入口显示数据摘要。


def preprocess_image(image, size):  # 离线数据与在线缓存共用同一幅图像的处理方式。
    image = image.convert("L")  # 使用原始灰度特征，不往图像上绘制标签。
    if size is not None:  # None 表示不缩放。
        resampling = getattr(Image, "Resampling", Image)  # 兼容新旧 Pillow。
        image = image.resize((size, size), resample=resampling.BILINEAR)  # 仅在配置要求时改变大小。
    return np.array(image, dtype=np.uint8)  # 保留紧凑数组。


@lru_cache(maxsize=64)  # 原图模式避免建立过大的图像缓存。
def read_image(path, size):  # 按路径读取单张历史图像。
    with Image.open(path) as image:  # 读完即关闭原文件。
        return preprocess_image(image, size)  # 复用统一预处理。


class WindowDataset(Dataset):  # 一个样本包含历史输入与未来帧状态。
    deterministic_image_inputs = True  # 目前没有随机增强；加入增强时须关闭同批图像特征共享。
    packed_images = False  # 仅无记忆批处理的加载器副本使用uint8，其余调用维持原FP32接口。

    def __init__(self, sequences, windows, settings, normalization):  # 接口供训练和评估共用。
        self.sequences, self.windows = sequences, windows  # 保存序列数据和窗口索引。
        self.settings, self.normalization = settings, normalization  # 保存输入规则和训练统计量。

    def __len__(self):  # 返回集合中的窗口数。
        return len(self.windows)  # 供 DataLoader 迭代。

    def __getitem__(self, index):  # 动态读取一个窗口。
        window = self.windows[index]  # 当前样本。
        sequence = self.sequences[window["sequence_id"]]  # 所属完整序列。
        images = [read_image(str(sequence["paths"][frame]), self.settings["image_size"]) for frame in window["image_indices"]]  # 只读当前及过去。
        if len({image.shape for image in images}) != 1 or min(images[0].shape) < 16:  # 输入图像需要可堆叠且足够大。
            raise ValueError(window["sequence_id"] + " 的图像尺寸不一致或过小，请统一原图尺寸或配置 IMAGE_SIZE。")  # 提示配置位置。
        image_array = np.stack(images)[:, None, :, :]  # 图像时刻、单通道、高、宽；先保持原像素字节。
        if not self.packed_images:  # 逐窗模型、旧调用和在线接口仍使用原来的0到1图像。
            image_array = image_array.astype(np.float32) / 255.0  # 无记忆快速批处理由ResNet统一转换，减少内存与传输。
        values = sequence["values"][window["start_seq_idx"]:window["end_seq_idx"] + 1].reshape(-1)  # 局部窗口每张图的整组温度按时间展开，例如 20 图读取 40 点。
        temperatures = ((values - self.normalization["mean"]) / self.normalization["std"]).astype(np.float32)[None, :]  # 复用训练归一化。
        return torch.from_numpy(image_array), torch.from_numpy(temperatures), torch.tensor(window["label"], dtype=torch.long), index  # 未来标签只用于损失。
