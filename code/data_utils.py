# -*- coding: utf-8 -*-
"""公共数据读取与划分校验；只被其他文件导入，不需要直接运行。"""

import csv  # 按表头读取配对数据及标签清单。
import hashlib  # 核对标签、配对与固定划分的内容标识。
import json  # 读写划分快照和配置。
import os  # 在网络存储上按目录批量读取文件名。
import random  # 用独立随机源重排每轮训练墙。
import re  # 兼容旧数据文件名的自然排序。
from collections import Counter, defaultdict  # 检查同一完整实验组是否跨集合。
from datetime import datetime  # 从已保存的真实图像时间计算层间停顿，不重新校准设备。
from pathlib import Path  # 统一管理跨平台路径。

import numpy as np  # 读取温度数组与逐帧标签；公共划分不需要导入 PyTorch。

def read_csv(path):  # 读取 UTF-8 或带 BOM 的 CSV。
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:  # 支持 Excel 保存的 UTF-8 表格。
        return list(csv.DictReader(handle))  # 按列名读取每一行。


def write_csv(path, rows, fields):  # 保存顺序固定的表格。
    path = Path(path)  # 统一路径类型。
    path.parent.mkdir(parents=True, exist_ok=True)  # 创建输出目录。
    with path.open("w", encoding="utf-8-sig", newline="") as handle:  # 使用 Excel 可读取的编码。
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")  # 明确输出列，允许窗口内部增加元数据。
        writer.writeheader()  # 写入列名。
        writer.writerows(rows)  # 写入全部记录。


def read_json(path):  # 读取配置与快照。
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))  # 返回普通结构，同时兼容带 BOM 的文件。


def write_json(path, value):  # 保存可审阅的中文 JSON。
    path = Path(path)  # 统一路径类型。
    path.parent.mkdir(parents=True, exist_ok=True)  # 创建父目录。
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")  # 拒绝无效数值。


def ordered_rows(rows, description):  # 有 seq_idx 时按序号读取，否则按行顺序读取。
    if not rows:  # 空表格无法组成样本。
        raise ValueError(description + " 没有数据行。")  # 指出问题来源。
    if "seq_idx" not in rows[0]:  # 配对温度和图像清单允许省略序号列。
        return rows  # 此时行顺序就是采集顺序。
    rows = sorted(rows, key=lambda row: int(row["seq_idx"]))  # 按整数序号排序。
    if [int(row["seq_idx"]) for row in rows] != list(range(len(rows))):  # 防止缺号和重复序号。
        raise ValueError(description + " 的 seq_idx 必须从 0 连续编号，不能重复。")  # 保持连续配对关系。
    return rows  # 返回排好顺序的记录。


def natural_key(path):  # 只根据文件名中的数字排序，不分析时间戳。
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]  # img_2 在 img_10 前面。


def read_input_quality_mask(folder, count, exclude_unknown=False, prefer_annotations=False):  # raw策略直接使用人工区间，旧版保留原逐帧掩码。
    mask = np.ones(count, dtype=bool)  # 没有显式质量排除时允许真实可用观测。
    frame_file, interval_file = Path(folder) / "frame_labels.csv", Path(folder) / "label_intervals.csv"  # 复用已有表，不新增CSV。
    if prefer_annotations and (Path(folder) / "annotations" / "label_intervals.csv").is_file():  # 人工区间是v9唯一状态依据。
        interval_file = Path(folder) / "annotations" / "label_intervals.csv"  # 不读取旧温度规则生成的ignore作为真实状态。
    ignored = {"ignore", "忽略", "-2"}  # 旧上下文模式只排除显式质量忽略标志。
    if exclude_unknown:  # 只有新版按标注排除模式才完全去掉unknown图温输入。
        ignored |= {"unknown", "待确认", "-1", ""}  # 不按stable、stubbing、breakoff的类别值筛选。
    if frame_file.is_file() and not (prefer_annotations and interval_file.is_file()):  # 旧版或没有区间表的单道才使用逐帧表。
        for row in read_csv(frame_file):  # 只比较保留/忽略，不解析三类真实状态。
            index = int(row["seq_idx"])  # 使用原图温组编号。
            if not 0 <= index < count:  # 质量标志必须能对应到现有图片。
                raise ValueError(str(frame_file) + " 的质量标志seq_idx越界。")  # 不截断或猜测。
            if str(row["state"]).strip().lower() in ignored:  # 显式ignore同时禁止局部和记忆输入。
                mask[index] = False  # 与实时采集接口input_available=False含义一致。
    elif interval_file.is_file():  # 兼容模型直接读取的旧区间格式。
        for row in read_csv(interval_file):  # 不读取未来状态决定当前输入。
            if str(row["state"]).strip().lower() in ignored:  # 仅把明确的数据质量忽略区间映射为掩码。
                start, end = int(row["start_seq_idx"]), int(row["end_seq_idx"])  # 区间端点包含在内。
                if not 0 <= start <= end < count:  # 防止质量标志越界。
                    raise ValueError(str(interval_file) + " 的ignore区间越界。")  # 保留原文件等待修正。
                mask[start:end + 1] = False  # 不改变原标签和图片顺序。
    return mask  # 该布尔掩码不包含stable、stubbing或breakoff真值。


def validate_image_names(paths, folder):  # 批量核对清单中的图片文件名。
    if not paths:
        raise ValueError(str(folder) + " 图片清单为空。")
    expected = defaultdict(set)  # 将图片按所在目录组织，一次读取该目录的名字。
    for path in paths:
        expected[path.parent].add(os.path.normcase(path.name))  # 兼容 Windows 文件名大小写。
    for directory, names in expected.items():
        try:
            available = {os.path.normcase(name) for name in os.listdir(directory)}
        except (FileNotFoundError, NotADirectoryError) as error:
            raise ValueError(f"{folder} 的图片目录缺失：{directory}") from error
        missing = names - available
        if missing:
            raise ValueError(f"{folder} 的图片清单有 {len(missing)} 张未找到，例如 {min(missing)}。")


def read_paired_frames(folder, use_annotation_input_mask=False, exclude_unknown_inputs=False, temperature_input_policy="raw"):  # 新数据保留799.9原值；旧权重显式传legacy_filtered。
    folder = Path(folder)  # 每个目录只对应一条单道或一面墙。
    manifest, temperature_file = folder / "frame_manifest.csv", folder / "temperature.csv"  # 两种输入清单。
    digest = hashlib.sha256()  # 固定配对版本。
    if manifest.is_file():  # 图片清单明确排序和连续段。
        frames = ordered_rows(read_csv(manifest), str(manifest))  # 图像 seq_idx 从 0 连续编号。
        paths = [folder / "images" / row["image_name"] for row in frames]  # 保留原图名。
        layer_ids = [str(row.get("layer_id", "")).strip() for row in frames]  # 层号用于边界检查，不作为网络输入。
        segments = [str(row.get("segment_id", "0")).strip() or "0" for row in frames]  # 同一连续段才能取窗。
        available = [str(row.get("temperature_available", "1")).strip().lower() not in ("0", "false", "no") for row in frames]  # 一个组内任一温度不可用即跳过整组。
        input_allowed = [str(row.get("use_for_training", "1")).strip().lower() not in ("0", "false", "no") for row in frames]  # 只读取已有输入有效标记，不把真实类别作为模型特征。
        digest.update(manifest.read_bytes())  # 记录清单内容。
    else:  # 兼容旧的每图一个温度数据。
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}  # 图像格式保持不变。
        paths = sorted((path for path in (folder / "images").iterdir() if path.suffix.lower() in extensions), key=natural_key)  # 原文件名自然排序。
        layer_ids = [""] * len(paths)  # 老单道没有层号，作为单层处理。
        segments, available = ["0"] * len(paths), [True] * len(paths)  # 旧版本没有额外有效性标志。
        frames, input_allowed = [], [True] * len(paths)  # 老数据没有额外清单时不猜测状态。
    validate_image_names(paths, folder)  # 一次核对目录成员，实际解码由模型读取图像时完成。
    rows = ordered_rows(read_csv(temperature_file), str(temperature_file))  # 这里的 seq_idx 是温度行序号。
    column = "temperature_c" if "temperature_c" in rows[0] else "value"  # 保留旧温度列兼容性。
    if column not in rows[0]:  # 不猜测未知字段。
        raise ValueError(str(temperature_file) + " 需要 temperature_c 或 value 列。")  # 指明字段。
    if "image_seq_idx" in rows[0]:  # 新版明确标明每行温度属于哪张图。
        if "sample_in_image" not in rows[0] or len(rows) % len(paths):  # 每图样本数必须固定。
            raise ValueError(str(temperature_file) + " 温度分组不完整，需要 image_seq_idx、sample_in_image。")  # 防止缺少一行后整体错位。
        samples_per_image = len(rows) // len(paths)  # 两温度数据的结果为 2。
        if samples_per_image not in (1, 2):  # 当前基础接口仅支持已明确实现的两种比例。
            raise ValueError("当前数据接口支持每图一个或两个温度点。")  # 其他采样比例应另行明确设计。
        for index, row in enumerate(rows):  # 验证分组顺序，而不是只检查总行数。
            if int(row["image_seq_idx"]) != index // samples_per_image or int(row["sample_in_image"]) != index % samples_per_image:  # 必须为图 0 的 0/1、图 1 的 0/1……。
                raise ValueError(str(temperature_file) + " 温度与图像分组序号不连续或有重复。")  # 不按行硬配。
    else:  # 老版一图一温度没有分组字段。
        samples_per_image = 1  # 保留旧模型和旧数据的解释。
        if len(rows) != len(paths):  # 老版不允许仅把温度行数加倍。
            raise ValueError(str(temperature_file) + " 行数与图片数不一致；两温度格式必须有 image_seq_idx 和 sample_in_image。")  # 明确转换要求。
    values = np.asarray([float(row[column]) for row in rows], dtype=np.float64).reshape(len(paths), samples_per_image)  # 内存按图片分组，窗口内再展开为时间序列。
    finite_groups = np.isfinite(values).all(axis=1)  # 每图两个点都要检查；799.9是有限数值，允许原值输入。
    if temperature_input_policy not in ("raw", "legacy_filtered"):  # 不允许拼错策略后静默改变训练规则。
        raise ValueError("temperature_input_policy只能为raw或legacy_filtered。")  # 明确兼容边界。
    if temperature_input_policy == "raw":  # v9只按人工unknown/ignore筛选，旧可用性字段不参与删样本。
        if not finite_groups.all():  # 空值和无穷无法进行数值计算，报错而不是偷偷改标签或删窗口。
            raise ValueError(str(folder) + " 温度含NaN或无穷，请修正数值；不会自动覆盖人工状态。")  # 799.9不触发此错误。
        available, input_allowed = [True] * len(paths), [True] * len(paths)  # 覆盖旧版温度过滤和来源训练开关，仅在内存中生效。
        digest.update(b"temperature_input_policy=raw;annotation_only")  # 与旧规则的训练输入指纹明确区分。
    if any(not finite and is_available for finite, is_available in zip(finite_groups, available)):  # 不可用组可保留原值，但有效组不允许 NaN 或无穷。
        raise ValueError(str(folder) + " 的有效温度组含空值或无效数值，请核对 temperature_available 标记。")  # 不修改原始温度。
    source_indices = np.asarray([row["temperature_index"] for row in rows]).reshape(len(paths), samples_per_image) if "temperature_index" in rows[0] else None  # 原温度索引可重复，供训练统计去重。
    digest.update(json.dumps([path.name for path in paths]).encode("utf-8"))  # 固定原图读取顺序。
    digest.update(temperature_file.read_bytes())  # 固定所有温度点。
    timestamp_texts = [row.get("image_datetime") or row.get("source_datetime") or "" for row in frames]  # 优先使用图像清单中已确认的采集时间。
    if not timestamp_texts or not all(timestamp_texts):  # 旧清单没有图像时间时，只从原配对表读取已存在的记录。
        pairing_file = folder / "pairing.csv"  # 不重新配对温度，也不改任何匹配阈值。
        if pairing_file.is_file():  # 该表中的同一图像有两行温度，图像时间只取一次。
            pairing_times = {}  # 按图像组号定位时间，绝不按温度行号解释图像编号。
            for row in read_csv(pairing_file):  # 只读取时间与组号，忽略 source_state 等真实类别。
                if row.get("image_datetime") and row.get("image_seq_idx") is not None:  # 兼容没有图像时间的旧表。
                    pairing_times.setdefault(int(row["image_seq_idx"]), row["image_datetime"])  # 两温度共享同一图像时间。
            timestamp_texts = [pairing_times.get(index, "") for index in range(len(paths))]  # 必须覆盖每张图像。
            digest.update(pairing_file.read_bytes())  # 时间历史变化也属于模型输入版本变化。
    has_timestamps = len(timestamp_texts) == len(paths) and all(timestamp_texts)  # 只有完整真实时间才可解释层间冷却。
    timestamps_s = np.full(len(paths), np.nan, dtype=np.float64)  # 缺时间时保留缺失标志，禁止伪造跨层冷却间隔。
    if has_timestamps:  # 有真实记录时按同一面墙的首图计算相对秒数。
        try:  # 明确提示损坏时间，不静默退回整墙序号除以帧率。
            recorded_times = [datetime.fromisoformat(value.strip()) for value in timestamp_texts]  # 已配对数据使用 ISO 日期时间。
        except (TypeError, ValueError) as error:  # 对齐问题应定位到输入文件。
            raise ValueError(str(folder) + " 的图像采集时间格式无效，请核对现有清单。") from error  # 不修改原始记录。
        timestamps_s = np.asarray([(value - recorded_times[0]).total_seconds() for value in recorded_times], dtype=np.float64)  # 保留真实层间停顿。
        if np.any(np.diff(timestamps_s) < 0):  # 同一面墙必须按照真实采集顺序前进。
            raise ValueError(str(folder) + " 图像采集时间倒退，不能建立连续过程记忆。")  # 不通过重新排序改变配对关系。
    layer_counts, layer_positions = Counter(), []  # 每层单独计数，连续段变化不重新跳过 50 张。
    for layer in layer_ids:  # 顺序完全沿用已有图像清单。
        layer_positions.append(layer_counts[layer])  # 第 1 张对应位置 0，第 51 张对应位置 50。
        layer_counts[layer] += 1  # 同层后续段接着计数。
    input_available = np.asarray(available, dtype=bool) & np.asarray(input_allowed, dtype=bool) & finite_groups  # 输入有效性只来自源标志和数值，不来自类别标签。
    if use_annotation_input_mask:  # v7训练和回放必须使用同样的坏帧过滤；旧模型仍保留原输入语义。
        quality_mask = read_input_quality_mask(folder, len(paths), exclude_unknown=exclude_unknown_inputs, prefer_annotations=temperature_input_policy == "raw")  # v9直接按人工区间过滤unknown/ignore。
        input_available &= quality_mask  # 不通过三类真值或未来标签决定是否读取观测。
        digest.update(quality_mask.tobytes())  # 新训练的输入指纹包含实际质量掩码，旧训练指纹保持不变。
    return {"paths": paths, "values": values, "layer_ids": layer_ids, "segments": segments, "temperature_available": available,  # 图温按原组保持一一对应。
            "input_available": input_available, "source_input_allowed": input_allowed, "timestamps_s": timestamps_s,  # 供局部输入和整墙记忆共用。
            "has_recorded_timestamps": bool(has_timestamps), "layer_frame_positions": np.asarray(layer_positions, dtype=np.int64),  # 层首规则不依赖真实类别。
            "temperature_samples_per_image": samples_per_image, "temperature_source_indices": source_indices}, digest.hexdigest()  # 两类数据共用读取接口。


def label_number(value, names):  # 把人工状态名称转换成模型的类别编号。
    text = str(value).strip().lower()  # 允许首尾空格。
    aliases = {"稳定": "stable", "稳态": "stable", "顶丝": "stubbing", "熔断": "breakoff"}  # 支持中文状态名称。
    text = aliases.get(text, text)  # 统一成英文类别。
    if text in ("ignore", "-2", "忽略"):  # 显式忽略与仅仅未知状态分开处理。
        return -2  # 新模式不把ignore当历史输入，也不增加输出类别。
    if text in ("", "unknown", "-1", "待确认"):  # 未知状态仍可以有真实可用的图温观测。
        return -1  # unknown不增加为第四类；未知目标不参与分类损失。
    if text not in names:  # 避免类别拼写错误或不在本次任务范围内。
        raise ValueError("标签 {} 不在 CLASS_NAMES 中；不确定时填写 unknown。".format(value))  # 明确修正方法。
    return names.index(text)  # 类别编号由配置中的顺序决定。


def read_labels(folder, count, names, constant_state=None, annotation_only=False):  # v9直接读取人工区间；旧结果继续按保存来源恢复。
    folder = Path(folder)  # 定位当前单道或墙的原始目录。
    frame_file, interval_file = folder / "frame_labels.csv", folder / "label_intervals.csv"  # 两种标签文件。
    sync_path = folder / "annotations" / "label_sync.json"  # 新整理数据的区间表与逐帧表同步记录。
    if sync_path.is_file():  # 老数据没有同步记录时保留原标签接口。
        sync = read_json(sync_path)  # 只有整理或同步脚本负责更新该记录。
        checks = [(folder / "annotations" / "label_intervals.csv", "interval_sha256"), (frame_file, "frame_labels_sha256"), (folder / "frame_manifest.csv", "manifest_sha256")]  # 防止漏同步或直接改派生标签。
        if any(not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sync.get(key) for path, key in checks):  # 文件缺失或内容改变时不继续训练。
            raise ValueError(str(folder) + " 的区间标签、逐帧标签或清单已变化，请先运行 code/prepare_dataset.py 的 sync_labels 模式；不要手工修改清单。")  # 避免读到旧标签。
    authoritative = folder / "annotations" / "label_intervals.csv"  # 当前墙的唯一人工标签来源。
    if annotation_only and authoritative.is_file():  # 温度数值及派生训练开关不参与类别决定。
        labels, occupied = np.full(count, -1, dtype=np.int64), np.zeros(count, dtype=bool)  # 明确检查遗漏和重叠。
        for row in read_csv(authoritative):  # 只解释前三列，辅助列不覆盖人工状态。
            start, end = int(row["start_seq_idx"]), int(row["end_seq_idx"])  # 闭区间保留原图顺序。
            if not 0 <= start <= end < count or occupied[start:end + 1].any():  # 不猜测错误区间。
                raise ValueError(str(authoritative) + " 存在越界、倒置或重叠区间。")  # 保留人工文件等待明确修正。
            labels[start:end + 1] = label_number(row["state"], names)  # 只按人工状态产生监督。
            occupied[start:end + 1] = True  # 每张保留图片只能有一种状态。
        if not occupied.all():  # 未确定状态应明确标unknown，不能静默漏标。
            raise ValueError(str(authoritative) + " 存在未覆盖的图片，请补齐人工区间。")  # 不按温度填ignore。
        return labels, hashlib.sha256(authoritative.read_bytes()).hexdigest(), "annotations/label_intervals.csv"  # 保存真实标签来源和版本。
    if not frame_file.is_file() and not interval_file.is_file() and constant_state is not None:  # 现有单道没有细分标签时使用整段类别。
        number = label_number(constant_state, names)  # 类别来自现有单道清单，含义与文件夹标签一致。
        labels = np.full(count, number, dtype=np.int64)  # 只在内存中把已有整段标签展开到每帧。
        digest = hashlib.sha256(json.dumps([constant_state, count]).encode("utf-8")).hexdigest()  # 标识整段标签内容。
        return labels, digest, "sequence_state"  # 不生成新图片或标签文件。
    if frame_file.is_file() == interval_file.is_file():  # 真实墙必须有一种标签，任何序列都不能有两份冲突标签。
        raise ValueError(str(folder) + " 请提供 frame_labels.csv 或 label_intervals.csv 中的一种；可运行 prepare_labels.py 生成模板。")  # 不猜标签来源。
    labels = np.full(count, -1, dtype=np.int64)  # 未标注位置默认为待确认。
    if frame_file.is_file():  # 每一帧直接记录真实状态。
        seen = set()  # 检查重复标注。
        for row in read_csv(frame_file):  # 逐行读取帧号与状态。
            index = int(row["seq_idx"])  # 使用与图像清单相同的从 0 开始的帧号。
            if not 0 <= index < count or index in seen:  # 不允许越界或重复帧号。
                raise ValueError(str(frame_file) + " 存在越界或重复的 seq_idx。")  # 定位标签问题。
            labels[index] = label_number(row["state"], names)  # 标注的是该帧实际状态。
            seen.add(index)  # 记录已标注位置。
        chosen = frame_file  # 保存标签内容标识时使用该文件。
    else:  # 连续状态只需填写开始帧、结束帧与类别。
        occupied = np.zeros(count, dtype=bool)  # 区间之间不可重叠，即使类别相同也应明确边界。
        for row in read_csv(interval_file):  # 每行代表包含两端帧号的闭区间。
            start, end = int(row["start_seq_idx"]), int(row["end_seq_idx"])  # 区间起止帧号。
            if not 0 <= start <= end < count or occupied[start:end + 1].any():  # 检查倒置、越界和重叠。
                raise ValueError(str(interval_file) + " 存在倒置、越界或重叠区间。")  # 不覆盖已有标签。
            labels[start:end + 1] = label_number(row["state"], names)  # 自动展开整段真实状态。
            occupied[start:end + 1] = True  # 记录已占用区间。
        chosen = interval_file  # 保存实际使用的标签文件。
    return labels, hashlib.sha256(chosen.read_bytes()).hexdigest(), chosen.name  # 保存标签版本与实际使用的来源。


def records_digest(records):  # 标识完整序列、分组和标签版本。
    keys = ("sequence_id", "sequence_type", "group_id", "relative_folder", "constant_state", "labels_present", "label_digest", "label_source")  # 划分依赖的字段。
    rows = [{key: row[key] for key in keys} for row in sorted(records, key=lambda row: row["sequence_id"])]  # 固定内容顺序。
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()  # 返回稳定内容标识。


def find_wall_folders(root):  # 直接扫描数据根目录下的墙，同时兼容旧的 walls/ 中间层。
    root = Path(root).resolve()  # 明确数据根目录。
    if not root.is_dir():  # 尚未整理数据时不给出虚构记录。
        return root, []  # 后续入口会提示缺少墙数据。
    direct = sorted(path for path in root.iterdir() if path.is_dir() and path.name != "walls" and (path / "images").is_dir())  # 只识别含 images 的墙，排除 metadata 等辅助目录。
    legacy = root / "walls"  # 旧数据格式仍可以读取。
    if legacy.is_dir() and direct:  # 同一数据目录不能混用两套墙位置。
        raise ValueError("同一数据目录同时有直接墙目录和旧 walls/ 目录，请统一为 DATA_ROOT/墙编号/。")  # 避免重复纳入相同实验。
    base = legacy if legacy.is_dir() else root  # 新数据直接以根目录作为墙的父目录。
    folders = sorted(path for path in base.iterdir() if path.is_dir() and (path / "images").is_dir())  # 标注工具和训练入口复用相同发现规则。
    return base, folders  # 返回实际父目录与真实墙列表。


def list_sequences(settings):  # 自动识别已有单道格式或整面墙格式，不要求转换数据。
    root = Path(settings["data_root"]).resolve()  # 唯一的数据根目录。
    metadata_path = root / "dataset_metadata.json"  # 新整理数据明确记录图像与温度的两种采样率。
    if metadata_path.is_file():  # 老数据没有该文件时仍使用训练保存的配置。
        metadata = read_json(metadata_path)  # 分别读取图像帧率与每图温度样本数。
        image_fps = float(metadata["image_fps"])  # 标签和预测提前量仍按图像帧数解释。
        samples_per_image = int(metadata.get("temperature_samples_per_image", 1))  # 兼容旧的一图一温度版本。
        if samples_per_image != settings.get("temperature_samples_per_image", 1):  # 避免用旧温度长度读取新数据。
            raise ValueError("此数据每图有 {} 个温度，请在 code/data_config.py 设置 TEMPERATURE_SAMPLES_PER_IMAGE={}。".format(samples_per_image, samples_per_image))  # 明确修改位置。
        if abs(image_fps - float(settings.get("fps", 50))) > 1e-9:  # 不把 25 fps 数据默认为旧 50 fps。
            raise ValueError("数据版本的图像帧率为 {}，请在 code/data_config.py 设置 FPS={}。".format(image_fps, image_fps))  # 防止预测提前量换算错误。
    wall_base, wall_folders = find_wall_folders(root)  # 新格式直接扫描 DATA_ROOT/墙编号/。
    wall_layout = bool(wall_folders) or (root / "walls").is_dir() or (root / "metadata" / "walls.csv").is_file() or metadata_path.is_file()  # 兼容旧墙位置及尚未填完的墙清单。
    track_layout = (root / "metadata" / "runs.csv").is_file()  # 单道仍沿用原有入口。
    if wall_layout == track_layout:  # 不混用单道与墙的数据入口。
        raise ValueError("DATA_ROOT 应指向直接包含各墙文件夹的 wall_dataset，或含 metadata/runs.csv 的单道根目录；不要混用两种入口。")  # 无需再添加 walls 层。
    if wall_layout:  # 真实墙采用此前说明的目录。
        metadata = root / "metadata" / "walls.csv"  # 可选的完整墙分组清单。
        rows = read_csv(metadata) if metadata.is_file() else [{"wall_id": path.name} for path in wall_folders]  # 各墙直接位于数据根目录，辅助目录不作为墙。
    else:  # 当前单道沿用已有清单与图片目录。
        rows = read_csv(root / "metadata" / "runs.csv")  # 读取 run_id、state 和可选 group_id。
    records, seen = [], set()  # 收集独立记录，不合并不同实验。
    for row in rows:  # 每行是一条单道或一面墙的采集记录。
        sequence_id = row["wall_id" if wall_layout else "run_id"].strip()  # 统一内部编号。
        if not sequence_id or sequence_id in (".", "..") or Path(sequence_id).name != sequence_id or sequence_id in seen:  # 拒绝重复或非法编号。
            raise ValueError("序列编号必须是唯一文件夹名：" + sequence_id)  # 不通过改名合并数据。
        group_id = (row.get("group_id") or sequence_id).strip()  # 同一真实实验的分段可使用相同组号。
        state = None if wall_layout else row["state"].strip()  # 仅单道清单提供整段默认标签。
        if not group_id or (state is not None and (not state or state in (".", "..") or Path(state).name != state)):  # 路径类别和分组需要有效。
            raise ValueError(sequence_id + " 的 group_id 或 state 无效。")  # 明确清单问题。
        relative = (wall_base / sequence_id).relative_to(root) if wall_layout else Path("runs") / state / sequence_id  # 新格式为墙编号本身，旧格式保留原路径。
        records.append({"sequence_id": sequence_id, "sequence_type": "wall" if wall_layout else "single_track",  # 保存真实实验类型。
                        "group_id": group_id, "relative_folder": relative.as_posix(), "constant_state": state})  # 墙不会回退到整段类别。
        seen.add(sequence_id)  # 同一记录只读取一次。
    if not records:  # 根目录中没有可使用的序列。
        raise ValueError("未发现采集记录，请检查 DATA_ROOT 和对应的清单。")  # 不生成替代数据。
    return sorted(records, key=lambda row: row["sequence_id"])  # 返回确定顺序。


def discover_sequences(settings, report_progress=False):  # 读取两种目录后统一形成完整序列记录。
    root, names = Path(settings["data_root"]).resolve(), settings["class_names"]  # 数据地址与类别列表。
    records = []  # 保存已关联标签的序列。
    sequences = list_sequences(settings)
    for index, record in enumerate(sequences, start=1):  # 单道整段、薄壁墙整墙逐一读取。
        if report_progress:
            print(f"正在核对 {index}/{len(sequences)}：{record['sequence_id']}", flush=True)  # 显示当前正在读取的墙。
        folder = root / record["relative_folder"]  # 原始目录保持不变。
        paired, _ = read_paired_frames(folder)  # 读取已有图像温度配对，不检查帧率。
        if paired["temperature_samples_per_image"] != settings.get("temperature_samples_per_image", 1):  # 即使旧数据没有版本元信息，也核对实际分组。
            raise ValueError(str(folder) + " 的每图温度数与 TEMPERATURE_SAMPLES_PER_IMAGE 不一致。")  # 不让温度形状错误拖到网络内部。
        labels, digest, source = read_labels(folder, len(paired["values"]), names, record["constant_state"], annotation_only=settings.get("temperature_input_policy") == "raw")  # 新划分的状态直接取人工区间。
        present = sorted(set(int(value) for value in labels if value >= 0))  # 记录本序列已确认的类别。
        if not present:  # 全部未知时无法监督学习。
            raise ValueError(record["sequence_id"] + " 没有已确认标签，请先填写真实状态。")  # 不自动猜测状态。
        records.append({**record, "labels_present": present, "label_digest": digest, "label_source": source})  # 统一后进入相同划分流程。
    return records  # 两种输入都不生成中间数据副本。


def validate_membership(records):  # 同一独立实验的序列不能跨越多个集合。
    groups, seen = defaultdict(set), set()  # 跟踪组归属和重复记录。
    for row in records:  # 每条记录是一条单道或一面墙。
        if row["split"] not in ("train", "val", "test") or row["sequence_id"] in seen:  # 拒绝重复或未知集合。
            raise ValueError("划分清单有重复序列或未知集合，请重新运行 code/split_data.py。")  # 明确修复方向。
        groups[row["group_id"]].add(row["split"])  # 同组记录一起分配。
        seen.add(row["sequence_id"])  # 标记已读取的序列。
    if any(len(parts) > 1 for parts in groups.values()):  # 检查组交集。
        raise ValueError("同一实验组跨越多个集合，请重新划分。")  # 不允许随机拆分同墙的相邻窗口。


def parse_training_layer_ids(value):  # 训练层单元格支持3;4或JSON列表，显式[]表示不监督该墙。
    if isinstance(value, str):  # CSV中所有字段首先都是文字。
        text = value.strip()  # 容忍Excel单元格两端空格。
        if not text:  # 空白通常意味着漏填，不能静默删除整面训练墙。
            raise ValueError("train_layer_ids不能为空；填写层号如3;4，明确排除整墙请填[]。")  # 区分遗漏和主动排除。
        try:  # JSON列表用于兼容程序写入或用户习惯。
            values = json.loads(text) if text.startswith("[") else text.replace("；", ";").split(";")  # 普通表格使用分号分隔。
        except (ValueError, TypeError) as error:  # 不自动修复不完整的列表。
            raise ValueError("train_layer_ids格式错误，请填写3;4或[\"3\",\"4\"]。") from error  # 指明唯一需要编辑的列。
    else:  # 程序内部可以直接传递层号列表。
        values = value  # 保留给后面的统一检查。
    if not isinstance(values, (list, tuple)) or any(type(item) not in (str, int) for item in values):  # 不接受字典、布尔值或浮点层号。
        raise ValueError("训练层必须是整数或文字层号组成的列表。")  # 层号与原frame_manifest.csv一致。
    layers = [str(item).strip() for item in values]  # 不把09自动改成9。
    if any(not item for item in layers) or len(set(layers)) != len(layers):  # 拒绝空层号和重复层号。
        raise ValueError("train_layer_ids不能包含空项或重复层号。")  # 避免同层重复监督。
    return layers  # 保留用户层号含义，实际读取顺序仍服从原墙顺序。


def read_training_layer_selection(snapshot, csv_path, settings=None):  # 训练只读取划分时缓存的层信息和这张表，不重新搜索熔断层。
    cached = snapshot.get("training_layer_selection")  # 首次升级需由split_data.py生成该缓存。
    if not isinstance(cached, dict) or cached.get("format_version") != 1 or not isinstance(cached.get("layers_by_sequence"), dict):  # 旧划分缺少层信息时不能退回训练时推断。
        raise ValueError("划分尚未包含训练层清单，请先在IDE运行code/split_data.py一次。")  # 保留原墙归属后补列即可。
    if settings is not None and any(cached.get(name) != settings.get(name, default) for name, default in (("train_layer_policy", "from_pre_breakoff_layer_to_wall_end"), ("no_breakoff_training_policy", "all_valid_layers"))):  # 规则变更需显式刷新表格。
        raise ValueError("训练层选择规则已改变，请先在IDE运行code/split_data.py刷新训练表。")  # 训练不自行重选层。
    path = Path(csv_path)  # 仅train_sequences.csv具有监督层选择含义。
    if not path.is_file():  # 不根据JSON默默补造用户要审阅的表。
        raise FileNotFoundError("缺少train_sequences.csv，请先运行code/split_data.py。")  # 明确准备步骤。
    rows = read_csv(path)  # 不读取验证和测试表中的层号来限制评估。
    expected = {row["sequence_id"]: row for row in snapshot["records"] if row["split"] == "train"}  # 墙归属仍由固定JSON决定。
    if len(rows) != len(expected) or {row.get("sequence_id") for row in rows} != set(expected) or len({row.get("sequence_id") for row in rows}) != len(rows):  # 防止在CSV里偷偷换墙或删除重复墙。
        raise ValueError("train_sequences.csv的墙名单与固定划分不一致；只编辑train_layer_ids列，换墙请重新划分。")  # 不破坏整墙隔离。
    selected = {}  # 每面训练墙的层号与审阅来源。
    for row in rows:  # 按表格逐墙检查，不读取真实帧标签。
        sequence_id = row["sequence_id"]  # 当前完整实验序列。
        record = expected[sequence_id]  # 与正式墙划分核对。
        if any(str(row.get(name, "")) != str(record[name]) for name in ("group_id", "relative_folder", "split")):  # CSV不能更改组、目录或集合。
            raise ValueError(sequence_id + "的分组/目录/集合被修改，请恢复或重新运行split_data.py。")  # 只开放监督层列。
        if sequence_id not in cached["layers_by_sequence"] or "train_layer_ids" not in row:  # 不允许缺少某面训练墙的缓存。
            raise ValueError(sequence_id + "缺少训练层信息，请重新运行split_data.py。")  # 不在训练期间补算。
        metadata = cached["layers_by_sequence"][sequence_id]  # 其中的可用层与首次熔断信息在划分时已确定。
        if metadata.get("label_digest") != record["label_digest"] or metadata.get("label_source") != record["label_source"]:  # 缓存必须对应固定划分中的标签版本。
            raise ValueError(sequence_id + "的训练层缓存与标签版本不同，请重新运行split_data.py。")  # 不在训练时重新搜索层。
        if settings is not None and metadata.get("frame_manifest_sha256") is not None:  # 仅核对层清单字节版本，不重新判断熔断。
            frame_manifest = Path(settings["data_root"]) / record["relative_folder"] / "frame_manifest.csv"  # 保留数据根目录可迁移性。
            if not frame_manifest.is_file() or hashlib.sha256(frame_manifest.read_bytes()).hexdigest() != metadata["frame_manifest_sha256"]:  # 层号或配对索引变化后旧缓存不可复用。
                raise ValueError(sequence_id + "的frame_manifest.csv已变化，请先运行split_data.py刷新训练层。")  # 不静默套用旧层号。
        layers = parse_training_layer_ids(row["train_layer_ids"])  # 唯一可人工修改的监督层字段。
        available = metadata["available_layer_ids"]  # 全unknown/ignore的层不能变成监督层。
        if set(layers) - set(available):  # 校验只使用缓存层号，不搜索任何熔断标注。
            raise ValueError(sequence_id + "选择了不存在或全为unknown/ignore的层：" + str(sorted(set(layers) - set(available))))  # 保留数据供用户修改表格。
        layers = [layer for layer in available if layer in set(layers)]  # 无论填入顺序如何，都按原加工层顺序读取。
        selected[sequence_id] = {**metadata, "train_layer_ids": layers, "selection_reason": "manual_csv" if layers != metadata["train_layer_ids"] else metadata["selection_reason"]}  # 审阅结果进入训练快照。
    return {**cached, "layers_by_sequence": selected, "csv_file": path.name, "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}  # 保存这次实际读取的表内容标识。


def filter_training_layer_windows(windows, skipped, selection):  # 只限制训练监督；验证和测试窗口原样保留。
    if selection is None:  # 旧的已保存窗口和直接接口检查保持兼容。
        return windows, skipped  # 新训练由load_split保证已提供层清单。
    metadata = selection["layers_by_sequence"]  # 层号已由划分表决定。
    kept, removed = [], Counter()  # 保存需要反向学习的窗口以及排除数量。
    for window in windows["train"]:  # 监督层只约束训练集合。
        sequence_id = window["sequence_id"]
        if sequence_id not in metadata:  # 缺失清单属于准备错误。
            raise ValueError(sequence_id + "没有训练层清单，请运行split_data.py。")
        layer = str(window["layer_id"]).strip() or "0"  # 旧单道无层号时使用0。
        if layer in metadata[sequence_id]["train_layer_ids"]:
            kept.append(window)  # 保留同层的完整历史和未来目标。
        else:
            removed[sequence_id] += 1  # 窗口报告会登记未选中层的数量。
    windows = {**windows, "train": kept}  # val和test列表的顺序与内容保持完整。
    reports = []  # 更新每面训练墙的窗口排除原因。
    for item in skipped:  # 验证与测试的过滤记录不套用训练层条件。
        report = dict(item)  # 不修改上游对象。
        if item["split"] == "train":  # 只给训练墙登记层选择与数量。
            report["training_layer_selection"] = metadata[item["sequence_id"]]  # 原始自动选择和人工修改都可追溯。
            report["reasons"] = {**item.get("reasons", {}), "unselected_training_layer_windows": removed[item["sequence_id"]]}  # 与原unknown/ignore过滤分开。
        reports.append(report)  # 保持原墙序。
    return windows, reports  # 训练器不再搜索哪一层发生熔断。


def load_split(settings):  # 训练读取固定的完整序列划分，不重新随机分配窗口。
    path = Path(settings["split_dir"]) / "split_manifest.json"  # 正式划分文件。
    if not path.is_file():  # 用户尚未运行划分入口。
        raise FileNotFoundError("请先运行 code/split_data.py 划分完整单道或整面墙。")  # 明确下一步。
    snapshot = read_json(path)  # 读取保存的划分。
    selection = read_training_layer_selection(snapshot, path.parent / "train_sequences.csv", settings)  # 先读取划分阶段写好的监督层，不在训练时识别熔断层。
    if snapshot.get("format_version") != 4 or snapshot.get("class_names") != settings["class_names"]:  # 拒绝旧格式或已改变的类别。
        raise ValueError("请重新运行 code/split_data.py 生成当前版本的完整序列划分。")  # 不混用旧清单。
    saved_swap = snapshot.get("holdout_swap", {})  # 旧快照没有独立的对调种子。
    if saved_swap.get("seed") != settings.get("holdout_swap_seed"):  # 确保使用配置指定的固定划分。
        raise ValueError("划分清单的对调种子与配置不同，请选择正确的划分目录。")
    if snapshot.get("seed") != settings["seed"]:  # 保证清单与训练使用同一个公共种子。
        raise ValueError("划分清单的种子与 code/data_config.py 不一致，请运行 split_data.py。")
    records = discover_sequences(settings)  # 镜像数据地址可不同，序列、分组和标签必须一致。
    if snapshot["records_digest"] != records_digest(records):
        raise ValueError("序列列表、分组或标签已变化，请重新运行 code/split_data.py。")
    if records_digest(snapshot["records"]) != snapshot["records_digest"]:  # 快照内容被编辑时重新建立。
        raise ValueError("划分快照被修改，请重新运行 code/split_data.py。")  # 保持来源一致。
    validate_membership(snapshot["records"])  # 检查完整实验组隔离。
    snapshot["training_layer_selection"] = selection  # 实际CSV选择随本次训练一起冻结，后续评估不重新套用它。
    return snapshot  # 返回固定划分。


def resolve_layer_start(settings, sequence_id, layer_id, labels=None, offsets=None, indices=None):  # 新版按每层首个已知标注，旧版才查固定张数。
    if uses_annotation_startup(settings):  # 不把旧50张或特殊层表叠加到人工unknown标注上。
        if labels is None or offsets is None or indices is None:  # 在线没有标签时必须由外部提供阶段信号。
            raise ValueError("按标注启动需要当前层标注；实时update请传processing_ready启动信号。")  # 不自动回退固定50张。
        first = next((int(index) for index in indices if int(labels[index]) >= 0), None)  # 首个stable、stubbing或breakoff均是已知状态。
        return int(offsets[first]) if first is not None else max((int(offsets[index]) for index in indices), default=-1) + 1, ""  # 全层未知时不生成监督窗。
    default = int(settings.get("layer_skip_frames", 0))  # 旧快照缺少此项时保持旧输入规则。
    rule = settings.get("layer_start_overrides", {}).get(str(sequence_id), {}).get(str(layer_id), {})  # 只有墙名和层号同时匹配才覆盖默认值。
    return int(rule.get("skip_frames", default)), str(rule.get("first_image", ""))  # 返回排除数量及可选的首张有效图片锚点。


def uses_startup_context(settings):  # 仅用于恢复旧版保留启动上下文的模型。
    return settings.get("startup_policy", "exclude") == "context_only"  # 只有已保存的旧v7上下文模式保留未知历史。


def uses_annotation_startup(settings):  # 将新版人工unknown启动规则与旧的固定计数明确分开。
    return settings.get("startup_policy", "exclude") == "annotation_exclude"  # unknown图温都不入模。


def sequence_boundaries(layers, segments, allowed=None):  # 统一划分不能跨越的局部时间段。
    count = len(layers)  # 保留全墙原始索引，过滤时不重编号。
    return [0] + [index for index in range(1, count) if (str(layers[index]), str(segments[index])) != (str(layers[index - 1]), str(segments[index - 1])) or (allowed is not None and bool(allowed[index]) != bool(allowed[index - 1]))] + [count]  # unknown/坏帧前后不拼接成虚假连续历史。


def first_window_end(part_start, first_post_start, length, context_only):  # 离线和在线使用同一因果输出网格。
    return max(part_start + length - 1, first_post_start) if context_only else first_post_start + length - 1  # 新模式不在启动后重新凑窗。


def supervision_rejection(labels, allowed, start, end, target, context_only):  # 真值仅决定离线是否计算损失，不传入网络。
    if not allowed[start:end + 1].all():  # unknown不等于缺图、缺温度或显式ignore。
        return "input_unavailable"  # 不补造输入，不越过坏帧组成窗口。
    if context_only and int(labels[end]) < 0:  # 未知当前状态只积累上下文。
        return "current_unknown_or_startup"  # 全unknown窗口不列为可评估的分类结果。
    if not context_only and (labels[start:target + 1] < 0).any():  # 旧结果继续使用整段标签要求。
        return "unknown_or_ignored_supervision"  # 保持旧v5/v6口径。
    if int(labels[target]) < 0 or not allowed[target]:  # 未来目标必须已确认并且可用。
        return "target_unknown_ignored_or_unavailable"  # 未来图温不进入网络。
    return ""  # 空原因表示该窗口可计算分类损失。


def build_layer_windows(sequences, settings, positions):  # 在同一层和连续有效段内构建局部窗口。
    length, future = settings["window_size"], settings["future_frames"]  # 历史图数与未来目标距离仍按图像帧解释。
    default_startup = int(settings.get("layer_skip_frames", 0))  # 仅为旧结果保留固定张数的回退值。
    context_only = uses_startup_context(settings)  # 只有旧v7上下文模式允许启动输入。
    annotation_start = uses_annotation_startup(settings)  # 新规则只服从人工标注，不再要求逐层数量表。
    exclude_first_layer = settings.get("exclude_first_layer", False)  # 只排除原始编号1；旧快照及其他模型默认保留。
    windows, skipped = {"train": [], "val": [], "test": []}, []  # 固定整墙集合，不重新随机划分。
    for sequence_id in sorted(sequences):  # 整面墙按时间完成后再处理另一面墙。
        sequence = sequences[sequence_id]  # 当前墙的原始配对数据。
        record, labels = sequence["record"], np.asarray(sequence["labels"], dtype=np.int64).copy()  # 仅复制内存，不改原标签。
        count, rejected = len(labels), Counter()  # 统计本墙的图像组数和过滤原因。
        layers, segments = sequence.get("layer_ids", [""] * count), sequence["segments"]  # 局部窗口禁止跨层或跨连续段。
        offsets = sequence.get("layer_frame_positions")  # 位置从每层第一张图的 0 开始计数。
        if offsets is None:  # 兼容小型合成检查和旧内存结构。
            layer_counts, offsets = Counter(), []  # 同层换段不重复跳过启动期。
            for layer in layers:  # 仅按已经确认的清单顺序计数。
                offsets.append(layer_counts[str(layer)])  # 记录当前图的层内位置。
                layer_counts[str(layer)] += 1  # 下一张同层图接着计数。
        offsets = np.asarray(offsets, dtype=np.int64)  # 便于统一比较启动阶段。
        available = np.asarray(sequence.get("input_available", sequence.get("temperature_available", [True] * count)), dtype=bool)  # 输入筛选不使用真实类别。
        if any(len(field) != count for field in (layers, segments, offsets, available)):  # 各逐帧字段必须与图像组一一对应。
            raise ValueError(sequence_id + " 的层、连续段、层内位置或输入有效标记长度不一致。")  # 不截断数据补配对。
        layer_rules = {}  # 保存本墙每一层实际采用的数量和核对后的输入起点。
        for layer in dict.fromkeys(str(value) for value in layers):  # 按原层顺序处理，不改变墙内顺序。
            indices = [index for index, value in enumerate(layers) if str(value) == layer]  # 原始墙索引保持不变。
            startup, anchor = resolve_layer_start(settings, sequence_id, layer, labels, offsets, indices)  # 标注模式逐层读取首个已知状态。
            first = next((index for index in indices if offsets[index] >= startup), None)  # 该图本身是第一张允许输入的图片。
            first_name = Path(sequence["paths"][first]).name if first is not None else ""  # 不读取或改写图像内容。
            if anchor and first_name != anchor:  # 数据清单改变后不能静默偏移人工确认的起点。
                raise ValueError("{} 第{}层：跳过{}张后的首图是{}，与配置{}不一致，请核对启动规则。".format(sequence_id, layer, startup, first_name or "不存在", anchor))  # 明确指出计数差异。
            layer_rules[layer] = {"boundary_source": "first_known_annotation" if annotation_start else "configured_frame_count", "skip_frames": startup, "first_image": first_name, "first_input_seq_idx": int(indices[0]) if context_only else int(first) if first is not None else None, "first_supervised_seq_idx": int(first) if first is not None else None, "remaining_images": max(0, len(indices) - startup)}  # skip_frames是报告计算结果，不再是用户参数。
        thresholds = np.asarray([layer_rules[str(layer)]["skip_frames"] for layer in layers], dtype=np.int64)  # 每张图对应其所属层的排除数量。
        allowed = available & (labels >= 0) if annotation_start else available & (labels != -2) if context_only else available & (offsets >= thresholds)  # 按所选规则确定历史可用性。
        if exclude_first_layer:  # 可按配置排除整面墙的第1层。
            first_layer = np.asarray(layers, dtype=str) == "1"  # 以原始层号判断，缺少第1层的墙保留现有层。
            allowed &= ~first_layer  # 保留原配对索引，只屏蔽模型输入。
            rejected["first_layer_frame_groups_excluded"] = int(first_layer.sum())  # 分开记录整层排除的图温组数。
        if context_only and not annotation_start:  # 只有旧计数模式需要在内存中屏蔽标签，新版完全服从人工unknown。
            labels[offsets < thresholds] = -1  # 启动图不能充当熔断或当前状态的训练答案。
        rejected["startup_frames_excluded"] = 0 if context_only else int(np.count_nonzero(offsets < thresholds))  # 输入排除数。
        rejected["unknown_image_temperature_groups_excluded"] = int(np.count_nonzero(labels == -1)) if annotation_start else 0  # 记录整层内的unknown，不仅是开头。
        rejected["startup_labels_masked"] = int(np.count_nonzero(offsets < thresholds)) if context_only and not annotation_start else 0  # 新版不会再自动覆盖人工标签。
        rejected["startup_annotation_prefix_frames"] = int(np.count_nonzero(offsets < thresholds)) if annotation_start else 0  # 按标注得到的逐层启动上下文数量。
        rejected["startup_context_available"] = int(np.count_nonzero(allowed & (offsets < thresholds))) if context_only else 0  # 真实可用的启动上下文数。
        timestamps = np.asarray(sequence.get("timestamps_s", np.full(count, np.nan)), dtype=np.float64)  # 真实时间用于窗口起止和报告。
        real_time = bool(sequence.get("has_recorded_timestamps", False) and len(timestamps) == count and np.isfinite(timestamps).all())  # 旧数据明确标记缺少时间。
        boundaries = sequence_boundaries(layers, segments, allowed if annotation_start else None)  # 新版局部窗口不能跨过unknown或无效输入。
        for part_start, part_stop in zip(boundaries[:-1], boundaries[1:]):  # 原清单顺序决定真实过程顺序。
            if exclude_first_layer and str(layers[part_start]) == "1":  # 第1层不产生训练或评估窗口。
                continue  # 局部历史和未来目标都从同一层的保留段构建。
            if annotation_start and not allowed[part_start]:  # unknown/ignore/坏图温组成的区间不能产生输入或监督。
                rejected["excluded_annotation_or_input_blocks"] += 1  # 保留过滤原因，不压缩原配对索引。
                continue  # 下一段从自己的首张有效图片重新积累历史。
            startup = layer_rules[str(layers[part_start])]["skip_frames"]  # 本连续段使用所属层的规则，不重新计数。
            eligible = [index for index in range(part_start, part_stop) if offsets[index] >= startup]  # 启动期不会因为段号改变而重新计数。
            if not eligible:  # 全段位于启动期时跳过。
                rejected["startup_only_segments"] += 1  # 报告过滤原因。
                continue  # 继续本层后面的段。
            effective_start = part_start if context_only else eligible[0]  # 新模式的历史允许包含启动图。
            first_end = first_window_end(part_start, eligible[0], length, context_only)  # 新版从首张已知有效图片开始积累完整窗口。
            rejected["startup_candidate_windows_filtered"] += sum(end < first_end for end in range(part_start + length - 1, part_stop - future, settings["window_stride"]))  # 启动期不显示或监督的候选。
            if first_end >= part_stop - future:  # 当前完整历史和同段未来目标都必须存在。
                rejected["layer_or_segment_too_short_after_startup"] += 1  # 不借用上一层或下一段补足。
                continue  # 当前段无法监督。
            for end in range(first_end, part_stop - future, settings["window_stride"]):  # 与无标签在线推理的帧数网格一致。
                start, target = end - length + 1, end + future  # 输入只到 end；target 仅提供未来真值。
                reason = supervision_rejection(labels, allowed, start, end, target, context_only)  # 新版不允许未知历史，仅旧上下文模式放宽该要求。
                if reason:  # 当前或未来未知、输入坏帧都明确记录。
                    rejected[reason] += 1  # 三个集合使用同一监督规则。
                    continue  # 继续检查后一个候选窗口。
                current = int(labels[end])  # 当前标签只供状态转变报告，未知时保留 -1。
                windows[record["split"]].append({"sequence_id": sequence_id, "group_id": record["group_id"], "split": record["split"],  # 完整墙所属集合保持不变。
                    "layer_id": str(layers[end]), "segment_id": str(segments[end]), "segment_start_seq_idx": part_start, "effective_start_seq_idx": effective_start,  # 便于核对实际起点。
                    "start_seq_idx": start, "end_seq_idx": end, "target_seq_idx": target,  # 保留原墙图像组索引。
                    "layer_frame_number_1based": int(offsets[end]) + 1, "target_layer_frame_number_1based": int(offsets[target]) + 1, "layer_skip_frames": startup,  # 人工查看使用从 1 起的层内图号。
                    "end_time_s": float(timestamps[end]) if real_time else end / float(settings.get("fps", 50)), "target_time_s": float(timestamps[target]) if real_time else target / float(settings.get("fps", 50)),  # 有记录时使用真实时间。
                    "has_recorded_timestamps": real_time, "label": int(labels[target]), "current_label": current,  # 标签只供监督和统计。
                    "context_unknown_frames": int(np.count_nonzero(labels[start:end + 1] < 0)), "context_startup_frames": int(np.count_nonzero(offsets[start:end + 1] < thresholds[start:end + 1])),  # 仅作审计，不传入模型。
                    "target_state": settings["class_names"][int(labels[target])], "current_state": settings["class_names"][current] if current >= 0 else "unknown",  # 方便人工核对分类目标。
                    "image_indices": [start + position for position in positions]})  # 局部窗可重叠。
        skipped.append({"sequence_id": sequence_id, "split": record["split"], "frames": count, "startup_policy": settings.get("startup_policy", "exclude"), "layer_skip_frames": default_startup, "layer_start_rules": layer_rules, "reasons": dict(rejected)})  # 保存实际策略和标注边界。
    return windows, skipped  # 墙、层、时间顺序保持固定，重新训练不改变墙的划分。


class OrderedLayerBatchSampler:  # 批次限制在同墙同层同段；训练可重排墙或层。
    def __init__(self, windows, batch_size, wall_shuffle_seed=None, layer_shuffle_seed=None):  # 评估不提供打乱种子。
        if type(batch_size) is not int or batch_size < 1:  # 明确批量大小是每批窗口数。
            raise ValueError("BATCH_SIZE 必须是正整数。")  # 大于 1 也可以保持层内时间顺序。
        if wall_shuffle_seed is not None and layer_shuffle_seed is not None:  # 一次只使用一种组块顺序。
            raise ValueError("墙打乱和层打乱不能同时开启。")
        self.batch_size, self.wall_ranges = batch_size, {}  # 每面墙保存原顺序的层段范围。
        self.wall_shuffle_seed, self.epoch = wall_shuffle_seed, 0  # 固定种子与轮次产生可重复的墙顺序。
        self.layer_shuffle_seed = layer_shuffle_seed  # 独立窗口模型可按全训练集的层重排。
        start, previous_key, previous_end, seen = 0, None, -1, set()  # 记录组边界并检查有序性。
        for index, window in enumerate(windows):  # 顺着原窗口列表检查，不悄悄重排或混组。
            key = (window["sequence_id"], str(window.get("layer_id", "")), str(window.get("segment_id", "")), window.get("segment_start_seq_idx"))  # 批次不跨越被unknown隔开的有效输入段。
            if key != previous_key:  # 开始另一层、另一连续段或另一面墙。
                if key in seen:  # 同一个组出现、离开、再出现说明列表被交错。
                    raise ValueError("窗口列表存在墙或层交错，请按当前程序重新生成窗口。")  # 不允许 A、B、A 的窗口顺序。
                if index:  # 保存上一个完整组的闭开索引范围。
                    self.wall_ranges[previous_key[0]].append((start, index))  # 组尾不足一个完整批次时也保留。
                if previous_key is None or key[0] != previous_key[0]:  # 按整面墙建立顺序块。
                    if key[0] in self.wall_ranges:  # 原清单中的同一面墙必须连续。
                        raise ValueError("窗口列表交错了不同的墙，请重新生成有序窗口。")  # 拒绝拆开同墙历史。
                    self.wall_ranges[key[0]] = []  # 层段依次附到所属墙。
                start, previous_key, previous_end = index, key, -1  # 新组重新检查时间顺序。
                seen.add(key)  # 当前墙层段只允许连续出现一次。
            if int(window["end_seq_idx"]) <= previous_end:  # 同层窗口末端必须严格向后移动。
                raise ValueError("同一层内的窗口没有按时间递增，请重新生成窗口。")  # 避免倒序输入。
            previous_end = int(window["end_seq_idx"])  # 记录已经检查过的当前时刻。
        if windows:  # 最后一个组同样需要保留。
            self.wall_ranges[previous_key[0]].append((start, len(windows)))  # 保留最后一个层段。
        self.layer_ranges = {}  # 层块内部仍按原先的连续段顺序遍历。
        if layer_shuffle_seed is not None:  # 只有层打乱时才建立第二份索引。
            for wall, ranges in self.wall_ranges.items():  # 原清单已验证为墙内有序。
                for start, stop in ranges:  # 一个范围就是同层、同段的连续窗口。
                    layer = str(windows[start].get("layer_id", ""))  # 同墙同层合为一个块。
                    self.layer_ranges.setdefault((wall, layer), []).append((start, stop))

    def set_epoch(self, epoch):  # 训练入口在每轮创建加载迭代器前设置轮次。
        self.epoch = epoch  # 评估不使用此值。

    def wall_order(self):  # 给加载器和训练日志返回同一轮的墙顺序。
        walls = list(self.wall_ranges)  # 评估默认使用原窗口清单的墙顺序。
        if self.wall_shuffle_seed is not None:  # 只有训练阶段启用墙间打乱。
            random.Random(self.wall_shuffle_seed + self.epoch).shuffle(walls)  # 独立于模型初始化和随机失活。
        return walls  # 返回新列表，不修改原窗口或索引。

    def layer_order(self):  # 返回本轮实际使用的完整层块顺序。
        layers = list(self.layer_ranges)  # 拷贝键的顺序，不改索引本身。
        if self.layer_shuffle_seed is not None:  # 只在训练层打乱时重排。
            random.Random(self.layer_shuffle_seed + self.epoch).shuffle(layers)
        return layers  # 日志与加载器用同一规则获得同一列表。

    def __iter__(self):  # 重排单位决定遍历次序，每个批次仍来自原连续段。
        if self.layer_shuffle_seed is not None:  # 全训练集的层块共同打乱。
            groups = (self.layer_ranges[key] for key in self.layer_order())
        else:  # 评估及旧训练模式按墙完整遍历。
            groups = (self.wall_ranges[wall] for wall in self.wall_order())
        for ranges in groups:  # 保留每个墙块或层块内部的连续段次序。
            for start, stop in ranges:
                for first in range(start, stop, self.batch_size):  # 同段内仍用相邻窗口组批。
                    yield list(range(first, min(first + self.batch_size, stop)))  # 组尾保留短批次。

    def __len__(self):  # 总批次数与训练墙排列无关。
        return sum((stop - start + self.batch_size - 1) // self.batch_size  # 每个层段分别向上取整。
                   for ranges in self.wall_ranges.values() for start, stop in ranges)  # 不跨组填补短批次。
