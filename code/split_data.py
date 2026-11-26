# -*- coding: utf-8 -*-
"""在 IDE 中运行：按完整序列分组划分训练、验证、测试，之后才在各序列内生成窗口。"""

import json  # 将 CSV 中的列表写成明确的文本格式。
import hashlib  # 保存层清单的数据版本，防止修改层号后沿用旧缓存。
import os  # 按目录核对图片文件名。
import random  # 使用固定随机种子分配完整序列。
from time import perf_counter  # 显示各阶段实际耗时。
from collections import defaultdict  # 汇总每个完整实验组实际包含的类别。
from datetime import datetime  # 为旧划分生成归档名称。
from pathlib import Path  # 管理划分文件路径。

# 直接运行时先确认实际执行的是这份脚本，即使数据依赖还未加载。
if __name__ == "__main__":
    print("启动整墙划分程序：{}".format(Path(__file__).resolve()), flush=True)

from data_config import get_data_settings, validate_data_settings  # 读取公共数据地址与划分参数。
from data_utils import (list_sequences, read_json, records_digest, validate_membership, write_csv, write_json,  # 与所有模型共用数据读取和校验。
                        ordered_rows, read_csv, read_paired_frames, read_labels, read_training_layer_selection)  # 熔断层只在划分阶段确认，训练读取缓存。


def read_split_layers(folder, settings, verify_inputs=False):
    manifest = folder / "frame_manifest.csv"
    if not manifest.is_file():
        paired, _ = read_paired_frames(folder, temperature_input_policy=settings.get("temperature_input_policy", "raw"))
        if paired["temperature_samples_per_image"] != settings.get("temperature_samples_per_image", 1):
            raise ValueError("{} 的每图温度数与公共数据配置不一致。".format(folder))
        return paired["layer_ids"]  # 兼容早期无逐帧清单的单道。

    frames = ordered_rows(read_csv(manifest), str(manifest))
    if verify_inputs:
        # 划分只用帧顺序和标注。按目录读取名称，避免逐张查询文件属性。
        names_by_directory = defaultdict(set)
        for frame in frames:
            path = folder / "images" / frame["image_name"]
            names_by_directory[path.parent].add(os.path.normcase(path.name))
        for directory, expected in names_by_directory.items():
            names = {os.path.normcase(name) for name in os.listdir(directory)}
            missing = expected - names
            if missing:
                raise ValueError("{} 有 {} 张清单图片未找到，例如 {}。".format(folder, len(missing), min(missing)))
        if not (folder / "temperature.csv").is_file():
            raise FileNotFoundError("缺少已配对温度文件：{}".format(folder / "temperature.csv"))
    return [frame.get("layer_id", "") for frame in frames]


def discover_split_sequences(settings):
    root = Path(settings["data_root"]).resolve()
    sequences = list_sequences(settings)  # 复用原有完整墙与单道识别规则。
    records = []
    for index, record in enumerate(sequences, start=1):
        print("正在核对 {}/{}：{}".format(index, len(sequences), record["sequence_id"]), flush=True)
        folder = root / record["relative_folder"]
        layers = read_split_layers(folder, settings, verify_inputs=True)
        labels, digest, source = read_labels(
            folder, len(layers), settings["class_names"], record["constant_state"],
            annotation_only=settings.get("temperature_input_policy") == "raw",
        )
        present = sorted({int(value) for value in labels if value >= 0})
        if not present:
            raise ValueError(record["sequence_id"] + " 没有已确认标签，请先填写真实状态。")
        records.append({**record, "labels_present": present, "label_digest": digest, "label_source": source})
    return records


def assign_groups(records, settings):  # 一段序列中即使有多种状态，也始终整体划分。
    group_labels = defaultdict(set)  # 保存同组序列包含的类别集合。
    for record in records:  # 同一真实墙拆成多个文件夹时仍使用同一 group_id。
        group_labels[record["group_id"]].update(record["labels_present"])  # 合并各段的有效状态。
    groups, labels = sorted(group_labels), list(range(len(settings["class_names"])))  # 固定遍历顺序。
    if len(groups) < 3:  # 三个独立集合至少需要三组完整序列。
        raise ValueError("至少需要 3 个独立实验组，才能划分训练、验证、测试；不能把同一段序列的相邻窗口混分。")  # 不把同墙伪装成独立实验。
    totals = [sum(label in values for values in group_labels.values()) for label in labels]  # 每类出现的独立组数。
    if min(totals) == 0:  # 不为完全没有样本的状态训练输出节点。
        raise ValueError("数据缺少 CLASS_NAMES 中的类别；仅有稳定/熔断时请设置这两类，或补充对应数据。")  # 指明修改位置。
    test_count = max(1, round(len(groups) * settings["test_fraction"]))  # 测试组目标数量。
    val_count = max(1, round(len(groups) * settings["validation_fraction"]))  # 验证组目标数量。
    if test_count + val_count >= len(groups):  # 确保训练还有完整序列。
        raise ValueError("留出比例过大，请降低 VALIDATION_FRACTION 或 TEST_FRACTION。")  # 提示具体参数。
    generator, best, best_score = random.Random(settings["seed"]), None, None  # 初始化随机划分搜索。
    for _ in range(3000):  # 尝试保持各集合类别覆盖和组数比例。
        candidate = groups.copy()  # 不修改原始组列表。
        generator.shuffle(candidate)  # 只打乱完整组。
        parts = {"test": candidate[:test_count], "val": candidate[test_count:test_count + val_count],  # 留出完整序列。
                 "train": candidate[test_count + val_count:]}  # 其余墙用于训练。
        counts = {name: [sum(label in group_labels[group] for group in part) for label in labels]  # 按组计数。
                  for name, part in parts.items()}  # 避免长墙的窗口数量主导划分。
        if min(counts["train"]) == 0:  # 训练集必须包含所有要学习的类别。
            continue  # 继续寻找可用划分。
        missing = sum(value == 0 for name in ("val", "test") for value in counts[name])  # 优先减少评估集合的缺类。
        imbalance = sum(abs(counts[name][label] / totals[label] - len(part) / len(groups))  # 各类组数比例偏差。
                        for name, part in parts.items() for label in labels)  # 合并各集合偏差。
        score = (missing, imbalance)  # 先比较缺类数，再比较比例。
        if best_score is None or score < best_score:  # 保存更合适的完整组方案。
            best, best_score = parts, score  # 记录当前最佳方案。
    if best is None:  # 当前组数与比例没有找到训练类别齐全的方案。
        raise ValueError("未找到训练类别齐全的实验划分，请增加独立实验或降低验证、测试比例。")  # 不暗中复用测试墙。
    membership = {group: name for name, part in best.items() for group in part}  # 每组只属于一个集合。
    return [{**record, "split": membership[record["group_id"]]} for record in records]  # 给各段序列附上集合名。


def has_same_group_basis(records, settings, previous):  # 核对基准墙身份和分组规则，允许标签更新。
    identity = lambda row: (row["sequence_id"], row["group_id"], row["relative_folder"], row.get("sequence_type"), row.get("constant_state"))  # 排除可变标签摘要。
    same_settings = previous is not None and previous.get("format_version") == 4 and all(previous.get(key) == settings[key] for key in ("class_names", "seed", "validation_fraction", "test_fraction"))  # 种子、比例或类别改变时才按新配置划分。
    return same_settings and {identity(row) for row in records} == {identity(row) for row in previous["records"]}  # 不把新增或移除的墙静默塞入旧集合。


def assign_or_keep_groups(records, settings, previous=None):  # 只改标注时刷新摘要，不重新随机分配整墙。
    if has_same_group_basis(records, settings, previous):  # 修改unknown区间不会改变验证/测试墙归属。
        validate_membership(previous["records"])  # 拒绝旧名单已经混组的情况。
        membership = {row["sequence_id"]: row["split"] for row in previous["records"]}  # 只恢复归属，不恢复旧标签。
        return [{**record, "split": membership[record["sequence_id"]]} for record in records], True  # 写入新的标签版本。
    return assign_groups(records, settings), False  # 真正改变墙名单或划分配置时重新分组。


def assign_holdout_swap(records, settings, baseline):  # 只在固定基准的验证和测试组之间交换一对。
    if not has_same_group_basis(records, settings, baseline):  # 训练组必须精确锁定到基准名单。
        raise ValueError("当前墙名单或划分参数与基准不一致，无法保留原训练墙做对调。")  # 由用户决定是否重新建基准。
    validate_membership(baseline["records"])  # 同一个实验组只能属于一个集合。
    membership = {row["group_id"]: row["split"] for row in baseline["records"]}  # 基准名单决定所有原始归属。
    val_groups = sorted(group for group, split in membership.items() if split == "val")  # 固定抽样池顺序。
    test_groups = sorted(group for group, split in membership.items() if split == "test")  # 固定抽样池顺序。
    if not val_groups or not test_groups:  # 需要两个集合都有完整实验组。
        raise ValueError("基准验证集和测试集各需至少一个完整实验组。")  # 不改变训练组来补齐。
    seed = settings["holdout_swap_seed"]  # 此种子只负责对调。
    rng = random.Random(seed)  # 不影响模型初始化或训练顺序。
    val_to_test = rng.choice(val_groups)  # 从基准验证组抽一组。
    test_to_val = rng.choice(test_groups)  # 从基准测试组抽一组。
    membership[val_to_test], membership[test_to_val] = "test", "val"  # 按完整组互换归属。
    assigned = [{**row, "split": membership[row["group_id"]]} for row in records]  # 标签摘要取当前数据。
    return assigned, {"seed": seed, "val_to_test": val_to_test, "test_to_val": test_to_val}  # 写入可审计的对调结果。


def build_training_layer_selection(records, settings, previous=None, previous_csv=None):  # 在划分阶段固定监督层。
    policy = settings.get("train_layer_policy", "from_pre_breakoff_layer_to_wall_end")  # 从首次熔断前一有效层开始，保留后续全部有效层。
    if policy == "first_breakoff_plus_previous":  # 原来只监督两层的规则仅允许旧结果评估使用。
        raise ValueError("请在data_config.py将TRAIN_LAYER_POLICY改为from_pre_breakoff_layer_to_wall_end，再生成新划分。")  # 不再生成漏掉后续熔断的清单。
    no_breakoff = settings.get("no_breakoff_training_policy", "all_valid_layers")  # 无熔断墙保留作对照或显式不监督。
    previous_cache = previous.get("training_layer_selection", {}) if previous else {}  # 兼容第一次升级的旧划分。
    previous_selection = None  # 只有已有正式层缓存时，才解释旧CSV中的人工选择。
    old_csv = previous_csv if previous_csv is not None else Path(settings["split_dir"]) / "train_sequences.csv"  # 新对调版继承基准手选层。
    same_policy = previous_cache.get("train_layer_policy") == policy and previous_cache.get("no_breakoff_training_policy") == no_breakoff  # 改变全局策略时重建自动方案。
    if previous_cache.get("format_version") == 1 and same_policy:  # 层规则未变时保留用户明确的手工选择。
        previous_selection = read_training_layer_selection(previous, old_csv) if old_csv.is_file() else previous_cache  # 缺CSV时只能从已保存缓存恢复。
    result = {}  # 只缓存训练墙的层选择，不根据验证测试标签筛选评估层。
    for record in records:  # 整墙划分已经确定，同一组仍不会跨集合。
        if record["split"] != "train":  # 验证集和测试集完整保留。
            continue  # 不为评估集合生成训练层条件。
        folder = Path(settings["data_root"]) / record["relative_folder"]  # 原始配对数据只读。
        frame_manifest = folder / "frame_manifest.csv"  # 新墙的原始层号直接记录在逐帧清单中。
        layer_ids = read_split_layers(folder, settings)  # 刷新训练层只需逐帧清单和标注。
        labels, digest, source = read_labels(folder, len(layer_ids), settings["class_names"], record["constant_state"], annotation_only=settings.get("temperature_input_policy") == "raw")  # 沿用已验证的标注来源。
        if digest != record["label_digest"] or source != record["label_source"]:  # 防止划分过程中人工文件被并发修改。
            raise ValueError(record["sequence_id"] + "的标注在划分期间发生变化，请保存标注后重新运行。")  # 不写入混合版本清单。
        layers = [str(value).strip() or "0" for value in layer_ids]  # 没有层号的旧单道作为第0层处理。
        all_layers = list(dict.fromkeys(layers))  # 按原加工顺序，而不是对层号按字母排序。
        available = [layer for layer in all_layers if any(int(labels[index]) >= 0 for index, value in enumerate(layers) if value == layer)]  # 全unknown或ignore的层不参与监督。
        number = settings["class_names"].index("breakoff") if "breakoff" in settings["class_names"] else None  # 兼容仅稳定与顶丝等类别配置。
        breakoff_layers = [layer for layer in available if number is not None and any(int(labels[index]) == number for index, value in enumerate(layers) if value == layer)]  # unknown启动段不会被误认为正式熔断。
        first = breakoff_layers[0] if breakoff_layers else None  # 首次有效熔断以原层顺序确定。
        if policy == "all_valid_layers":  # 显式选择所有可监督的加工层。
            chosen, reason = list(available), "all_valid_layers"  # 作为全层训练对照。
        elif first is not None:  # 已经标注有效熔断的训练墙。
            position = available.index(first)  # 找到其前一有效加工层，而不是简单把层号减1。
            chosen, reason = available[max(0, position - 1):], "from_pre_breakoff_layer_to_wall_end"  # 包含后续恢复稳定、再次熔断及墙尾；首层熔断时从首层开始。
        else:  # 没有熔断的墙不能假装知道哪一层会熔断。
            chosen = list(available) if no_breakoff == "all_valid_layers" else []  # 保留完整稳定/顶丝对照或显式排除。
            reason = "no_breakoff_control_all_valid_layers" if chosen else "no_breakoff_excluded"  # 选择原因写在同一张表里。
        sequence_id = record["sequence_id"]  # 当前墙标识。
        if previous_selection is not None and sequence_id in previous_selection["layers_by_sequence"]:  # 保留同一训练墙以前的人工修改。
            old = previous_selection["layers_by_sequence"][sequence_id]  # 上次CSV实际读取值。
            baseline = previous_cache["layers_by_sequence"][sequence_id]  # 上次自动方案或已确认人工方案。
            manual = old["train_layer_ids"] != baseline["train_layer_ids"] or str(baseline["selection_reason"]).startswith("manual")  # 自动选择随新标签刷新，人工选择不默默覆盖。
            if manual:  # 用户确实手工指定过监督层。
                if set(old["train_layer_ids"]) - set(available):  # 删除或全改unknown的层不能继续监督。
                    raise ValueError(sequence_id + "之前手选的训练层已不存在或无有效标签，请先修改train_sequences.csv。")  # 不静默改成其他层。
                chosen, reason = [layer for layer in available if layer in old["train_layer_ids"]], "manual_preserved"  # 仍按原加工顺序排列。
        result[sequence_id] = {"all_layer_ids": all_layers, "available_layer_ids": available, "breakoff_layer_ids": breakoff_layers,  # 已在划分阶段缓存的事实。
                               "first_breakoff_layer_id": first, "train_layer_ids": chosen, "selection_reason": reason,  # 训练只消费最终层清单。
                               "label_digest": digest, "label_source": source, "frame_manifest_sha256": hashlib.sha256(frame_manifest.read_bytes()).hexdigest() if frame_manifest.is_file() else None}  # 缓存标签与层信息的版本。
    return {"format_version": 1, "train_layer_policy": policy, "no_breakoff_training_policy": no_breakoff, "layers_by_sequence": result}  # 与正式划分放在同一个JSON中。


def main():  # 保存一次可重复使用的完整实验序列划分和训练监督层。
    settings = get_data_settings()
    validate_data_settings(settings)  # 公共划分只读取数据配置。
    print("整墙划分种子：{} | 输出目录：{}".format(settings["seed"], settings["split_dir"]), flush=True)
    scan_started = perf_counter()
    print("正在核对数据目录、逐帧清单及标注：{}".format(settings["data_root"]), flush=True)
    records = discover_split_sequences(settings)  # 入口兼容服务器原有的数据工具。
    print("数据核对完成：{} 面墙/段序列，用时 {:.1f} 秒。".format(len(records), perf_counter() - scan_started), flush=True)
    output = Path(settings["split_dir"])  # 先读取当前固定名单，再决定是否需要重新分配墙。
    previous = output / "split_manifest.json"  # 训练入口读取的划分文件。
    old_snapshot = read_json(previous) if previous.is_file() else None  # 同时用于保留整墙归属与人工层选择。
    swap = None  # 基准划分沿用原有刷新方式。
    layer_source, layer_csv = old_snapshot, None  # 默认读取本版已有的手选层。
    if settings.get("holdout_swap_seed") is None:  # 明确选择基准划分。
        assigned, kept = assign_or_keep_groups(records, settings, old_snapshot)  # 仅改标注时保持原墙划分。
        if kept:  # 说明已沿用上一次完整墙分配。
            print("墙名单与划分参数未变：保留原训练/验证/测试归属，仅刷新标签摘要。", flush=True)
    else:  # 从基准名单重现对调，重复运行不会把两面墙交换回去。
        base_dir = Path(settings["base_split_dir"])  # 只读取这份旧清单的墙归属。
        base_manifest = base_dir / "split_manifest.json"  # 基准需预先存在。
        if not base_manifest.is_file():  # 缺基准时不能保证训练墙不变。
            raise FileNotFoundError("缺少用于锁定训练墙的基准划分：" + str(base_manifest))
        baseline = read_json(base_manifest)  # 基准标签摘要可以早于当前标签版本。
        assigned, swap = assign_holdout_swap(records, settings, baseline)  # 重新核对并生成新名单。
        swap["base_split"] = base_dir.name  # 使用可迁移的目录名称标识来源。
        if old_snapshot is None:  # 第一次生成时沿用基准已有的人工选层。
            layer_source, layer_csv = baseline, base_dir / "train_sequences.csv"
        print("保留基准训练组。对调种子 {}，验证→测试 {}，测试→验证 {}。".format(
            swap["seed"], swap["val_to_test"], swap["test_to_val"]), flush=True)  # 启动时可直接核对。
    validate_membership(assigned)  # 同一实验组不得跨集合。
    layer_started = perf_counter()
    print("正在读取训练墙逐帧清单和标注，刷新监督层。", flush=True)
    selection = build_training_layer_selection(assigned, settings, layer_source, layer_csv)
    print("训练层刷新完成：{} 面墙/段序列，用时 {:.1f} 秒。".format(len(selection["layers_by_sequence"]), perf_counter() - layer_started), flush=True)
    output.mkdir(parents=True, exist_ok=True)  # 首次运行时创建目录。
    manifest = {"format_version": 4, "data_root": str(Path(settings["data_root"]).resolve()),  # 保存统一墙格式版本。
                "class_names": settings["class_names"], "records_digest": records_digest(records),  # 固定标签与墙列表。
                "seed": settings["seed"], "validation_fraction": settings["validation_fraction"],  # 保存分组设置。
                "test_fraction": settings["test_fraction"], "records": assigned, "training_layer_selection": selection}  # 层缓存与整墙名单统一保存。
    if swap is not None:  # 对调版可随完全同步的配对数据上传到服务器。
        manifest.update(holdout_swap=swap, portable_data_root=True)  # 训练入口继续核对当前数据摘要。
    if previous.exists() and read_json(previous) != manifest:  # 重新划分时保留旧记录。
        archive = output / "history" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")  # 唯一归档路径。
        write_json(archive, read_json(previous))  # 归档上一份 JSON，不移动用户数据。
    for split in ("train", "val", "test"):  # 输出三个可以用 Excel 打开的序列清单。
        rows = [record for record in assigned if record["split"] == split]  # 本集合中的完整序列。
        csv_rows = [{key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value  # 列表使用 JSON 文本。
                     for key, value in row.items()} for row in rows]  # 保留每段序列的状态集合和来源。
        fields = list(assigned[0])  # 原有整墙列保持可读。
        if split == "train":  # 训练表直接包含可修改的监督层，不另建CSV。
            fields.extend(["available_layer_ids", "breakoff_layer_ids", "first_breakoff_layer_id", "train_layer_ids", "layer_selection_reason"])  # 只需人工修改train_layer_ids列。
            for row in csv_rows:  # 一面墙仍占一行。
                item = selection["layers_by_sequence"][row["sequence_id"]]  # 读取刚刚缓存的层方案。
                row.update(available_layer_ids=";".join(item["available_layer_ids"]), breakoff_layer_ids=";".join(item["breakoff_layer_ids"]),  # 分号文本方便在Excel查看。
                           first_breakoff_layer_id=item["first_breakoff_layer_id"] or "", train_layer_ids=";".join(item["train_layer_ids"]) if item["train_layer_ids"] else "[]",  # 显式[]表示整墙不监督。
                           layer_selection_reason=item["selection_reason"])  # 用原因区分首熔断方案和无熔断对照墙。
        write_csv(output / (split + "_sequences.csv"), csv_rows, fields)  # 训练层来自此CSV，整墙归属仍以JSON为准。
        counts = {name: sum(label in row["labels_present"] for row in rows)  # 每类有多少段完整序列包含它。
                  for label, name in enumerate(settings["class_names"])}  # 一段序列可同时计入多个类别。
        print("{}：{} 面墙/段序列，{} 个完整组，包含有效状态的墙/序列数 {}".format(split, len(rows), len({row["group_id"] for row in rows}), counts), flush=True)  # 显示摘要。
        if any(count == 0 for count in counts.values()):  # 数据较少时评估集合可能缺类。
            print("该集合有缺失类别，对应类别的评估不能据此得出结论。", flush=True)  # 清楚说明报告边界。
    write_json(previous, manifest)  # 保存训练使用的正式快照。
    print("以上统计各状态的墙数，墙可重复计入。基准划分种子 {}，训练批次顺序由模型配置决定。".format(settings["seed"]), flush=True)  # 区分划分和读取顺序。
    print("训练层已写入train_sequences.csv的train_layer_ids列；可改为3;4等层号，验证和测试仍使用完整墙。", flush=True)  # 明确模型读取哪个表。
    print("划分完成：{}\n下一步：在 IDE 中运行所选模型目录下的train.py。".format(output), flush=True)  # 同时支持新增ResNet34模型。


if __name__ == "__main__":  # 被其他模块导入时不自动划分。
    main()  # 从 IDE 或服务器终端直接运行本文件。
