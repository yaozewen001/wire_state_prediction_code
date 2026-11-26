# -*- coding: utf-8 -*-
"""更新已有墙：保留每图原采样编号，修复缺图、同步最新温度值与区间标签。"""

import bisect  # 按原seq_idx将人工区间映射回原图片，不读取旧辅助锚点作为标注指令。
import json  # 保存一次批量修复报告。
import math  # 检查温度值比较是否可靠。
import os  # 原子替换单个派生清单文件。
import shutil  # 写入前复制完整原清单备份，出错时回滚。
import tempfile  # 在同一墙目录下暂存已完成的新清单。
from collections import Counter, defaultdict  # 按图片定位原配对及统计标签。
from datetime import datetime, timezone  # 为备份和审计记录生成独立时间。
from pathlib import Path  # 限定读写目录。

from prepare_dataset import MANIFEST_FIELDS, PAIR_FIELDS, INTERVAL_FIELDS, read_csv, write_csv, write_json, sha256, flag, integer, normalize_state, make_intervals, expand_intervals, save_sync_record  # 复用现有文件格式，不重新按时间匹配温度。

SOURCE_FILES = ("frame_manifest.csv", "temperature.csv", "pairing.csv", "annotations/label_intervals.csv", "frame_labels.csv", "annotations/label_sync.json", "preparation_report.json")  # 修复计划与备份使用相同的源文件范围。


def _source_hashes(folder):  # 同时检查内容变化、文件新增和文件消失。
    return {relative: sha256(Path(folder) / relative) for relative in SOURCE_FILES if (Path(folder) / relative).is_file()}  # 不包含原图或原始采集文件。


class RepairError(ValueError):  # 保留可供IDE定位的具体修复问题。
    pass  # 不将模糊状态或配对冲突降级成自动猜测。


def _index_ranges(values):  # 将多个原编号压缩成连续范围，错误报告不逐帧刷屏。
    ranges = []  # 每项为包含两端的原seq范围。
    for value in sorted(set(values)):  # 只合并相邻编号，不改变原始数据。
        if ranges and value == ranges[-1][1] + 1:  # 同一连续范围继续延长。
            ranges[-1][1] = value  # 保留真实端点。
        else:  # 新建一个独立范围。
            ranges.append([value, value])  # 单帧范围两端相同。
    return ranges  # 方便用户一次修正一个区间。


def _unique_rows(rows, field, description):  # 建立唯一键，禁止重复后取第一条。
    result = {}  # 原编号或图片名映射。
    for row in rows:  # 不改变原表内容。
        value = row[field]  # 使用已有明确键名。
        if value in result:  # 同一图片不能指向多组原记录。
            raise RepairError(description + " 存在重复键：" + str(value))  # 需要先解决来源冲突。
        result[value] = row  # 一一对应后才能重建索引。
    return result  # 无重复的原始记录。


def _samples_by(rows, field):  # 原温度和追溯表各自按图片分组。
    groups = defaultdict(dict)  # 每组保存原sample_in_image=0和1。
    for row in rows:  # 不根据当前图片数量重新顺排温度。
        key, sample = str(row[field]), integer(row["sample_in_image"], "sample_in_image")  # 保留原组标识。
        if sample not in (0, 1) or sample in groups[key]:  # 每张图恰好两个不同采样位置。
            raise RepairError("温度分组重复或sample_in_image不为0/1：" + key)  # 不平均或复制温度补齐。
        groups[key][sample] = row  # 保存原行的温度索引、值和时间。
    return groups  # 缺少采样点的保留图片在使用时明确报错。


def _same_temperature(left, right):  # 两张现存追溯表的配对必须一致。
    if str(left["temperature_index"]) != str(right["temperature_index"]) or str(left["temperature_datetime"]) != str(right["temperature_datetime"]):  # 原采样序号和时刻不得错位。
        return False  # 不擅自选择其中一张表。
    a, b = float(left["temperature_c"]), float(right["temperature_c"])  # 仅比较，不改写原温度字符串。
    return a == b or (math.isnan(a) and math.isnan(b))  # 允许同一不可用原值继续保留并由有效性规则排除。


def _label_original_indices(intervals, frames, retained):  # 用户确认区间仍使用删图前的原seq_idx。
    keys = sorted(frames)  # 原清单可以有缺号，但每个现存键必须唯一。
    labels, owners, errors, conflicts = {}, {}, [], defaultdict(list)  # 不静默覆盖用户新标注。
    blank_rows, duplicate_assignments, aliases = [], 0, []  # 记录可确定的格式整理。
    for line, row in enumerate(intervals, start=2):  # 输出真实CSV行号，便于在Excel定位。
        start_text, end_text, state_text = (str(row.get(name, "") or "").strip() for name in ("start_seq_idx", "end_seq_idx", "state"))  # 辅助图名和原帧号不是新的标签指令。
        if not start_text and not end_text and not state_text:  # 清空了前三列但旧辅助列仍在的行，不再视作有效区间。
            blank_rows.append(line)  # 原文件会完整备份，报告说明被忽略的空行。
            continue  # 不能用旧辅助列复活已被人工清空的标注。
        try:  # 不猜测只有一端的区间或不明确的状态。
            start, end = integer(start_text, "start_seq_idx"), integer(end_text, "end_seq_idx")  # 原seq_idx闭区间。
            state = normalize_state(state_text)  # unknow明确规范为unknown，其他未知拼写仍报错。
            if start < 0 or start > end or start < keys[0] or end > keys[-1]:  # 拒绝倒置与超出原清单的范围。
                raise ValueError("区间必须位于原清单 {}～{} 内且起点不大于终点".format(keys[0], keys[-1]))  # 不截断人工区间。
        except (TypeError, ValueError) as error:  # 将本行问题汇总，而不是只看到第一个错误。
            errors.append("第{}行：{}".format(line, error))  # 提供具体修正位置。
            continue  # 本行不会改变任何已标注状态。
        if state_text.lower() == "unknow":  # 这是已有明确用途的unknown拼写，不推断物理类别。
            aliases.append({"line": line, "from": state_text, "to": "unknown"})  # 规范化过程写入修复报告。
        selected = keys[bisect.bisect_left(keys, start):bisect.bisect_right(keys, end)]  # 按原编号查原图片。
        for index in selected:  # 已删除图片可以没有标签，保留图片必须有唯一状态。
            if index not in retained:  # 删除位置不再参与模型，无需猜测其标签。
                continue  # 缺口仍在后面的段号中记录。
            if index in labels and labels[index] != state:  # 两种不同状态争用同一保留帧。
                conflicts[(owners[index], line, labels[index], state)].append(index)  # 必须人工明确，不逐帧刷屏。
            elif index in labels:  # 相同类别的重复覆盖可合并，不改变标注语义。
                duplicate_assignments += 1  # 报告冗余，不重复生成数据。
            else:  # 第一次获得该帧的人工状态。
                labels[index], owners[index] = state, line  # 不使用旧frame_labels覆盖人工修改。
    for (first_line, second_line, first_state, second_state), indices in conflicts.items():  # 将同一对冲突标注合并报告。
        errors.append("第{}行和第{}行状态冲突 {} / {}，原seq_idx范围：{}".format(first_line, second_line, first_state, second_state, _index_ranges(indices)))  # 不自动选择类别。
    uncovered = sorted(set(retained) - set(labels))  # 仍有图片但人工区间没有覆盖。
    if uncovered:  # 这些帧不能当作已经删除，也不能自动标稳定。
        errors.insert(0, "保留图片漏标{}张，原seq_idx范围：{}".format(len(uncovered), _index_ranges(uncovered)[:20]))  # 明确显示遗漏，不能把它们当已删图。
    if errors:  # 在任何写入之前停止该墙。
        raise RepairError("标注需要先修正：\n" + "\n".join(errors[:30]) + ("\n其余{}条冲突未展开。".format(len(errors) - 30) if len(errors) > 30 else ""))  # 不自动选择冲突类别。
    return labels, {"ignored_blank_interval_lines": blank_rows, "same_state_overlap_assignments_merged": duplicate_assignments, "normalized_spellings": aliases}  # 原始人工文件仍会备份。


def plan_repair(folder):  # 只读生成修复计划，调用此函数不会改真实墙文件。
    folder = Path(folder).resolve()  # 仅处理用户明确选择的一面已配对墙。
    source_hashes = _source_hashes(folder)  # 必须在读取CSV前固定版本，不能读取后才记录新版本哈希。
    manifest_path, temperature_path, pairing_path = (folder / name for name in ("frame_manifest.csv", "temperature.csv", "pairing.csv"))  # 现存配对是唯一依据。
    interval_path = folder / "annotations" / "label_intervals.csv"  # 人工新改标注。
    original_manifest = read_csv(manifest_path, MANIFEST_FIELDS)  # 不要求旧seq_idx仍连续，但必须唯一。
    if not original_manifest:  # 没有原记录就不能安全恢复图温关系。
        raise RepairError("原frame_manifest为空，不能凭图片行号猜温度。")  # 保留文件等待原备份。
    frames = {}  # 用原seq_idx查原帧号、文件名和温度关系。
    for item in original_manifest:  # 不改变传入行。
        index = integer(item["seq_idx"], "原seq_idx")  # 原编号可以存在删除后的缺口。
        if index < 0 or index in frames:  # 重复旧编号将使区间映射不唯一。
            raise RepairError("原清单seq_idx重复或为负：" + str(index))  # 不随意去重。
        frames[index] = item  # 原文件名作为更强的温度追溯键。
    _unique_rows(original_manifest, "image_name", "原图片清单")  # 文件名必须唯一。
    existing_names = {path.name for path in (folder / "images").iterdir() if path.is_file()}  # 不打开、删除或改名图片。
    expected_names = {row["image_name"] for row in original_manifest}  # 原清单中的名字。
    extra = sorted(name for name in existing_names - expected_names if Path(name).suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))  # 额外图片不能猜对应温度。
    if extra:  # 不能把新加图片按目录位置插入旧配对。
        raise RepairError("图片目录存在原清单没有记录的图片：" + str(extra[:10]))  # 需先提供其来源。
    retained = sorted(index for index, row in frames.items() if row["image_name"] in existing_names)  # 按用户确认的原编号保持时间顺序。
    missing = sorted(set(frames) - set(retained))  # 已删除图片不会从原始采集目录补回来。
    if not retained:  # 空墙不发布成可训练数据。
        raise RepairError("本墙已无保留图片，不能重建训练清单。")  # 不修改任何文件。
    labels, notes = _label_original_indices(read_csv(interval_path, ["start_seq_idx", "end_seq_idx", "state"]), frames, retained)  # 辅助start_source_frame_number不覆盖新的原seq区间。
    structural_repair = bool(missing) or sorted(frames) != list(range(len(frames)))  # 没有删图或缺号时保留原段号，重复更新不叠加后缀。
    original_temperatures = read_csv(temperature_path, ["seq_idx", "image_seq_idx", "sample_in_image", "temperature_c", "temperature_index", "temperature_datetime"])  # 温度值与原采样索引保持原样。
    original_pairing = read_csv(pairing_path, PAIR_FIELDS)  # 含image_name的追溯表不依赖人工重新编号。
    temperature_groups, pairing_groups = _samples_by(original_temperatures, "image_seq_idx"), _samples_by(original_pairing, "image_name")  # 双重核对避免温度顺移。
    manifest, temperatures, pairings, state_rows, index_map = [], [], [], [], []  # 全部先在内存完成。
    previous, segment_count = None, 0  # 删除缺口必须切分连续段。
    for new_index, old_index in enumerate(retained):  # 新seq_idx连续，但原始采集帧号不变。
        original = frames[old_index]  # 原图片对应关系。
        raw_frame = integer(original["source_frame_number"], "source_frame_number")  # 原始帧号仅用于核验与断点追溯。
        match = re_frame(original["image_name"])  # 图片名含明确原帧号时额外核验。
        if match is not None and match != raw_frame:  # 原清单和原图名冲突不能自动修正。
            raise RepairError("{}的source_frame_number与文件名不一致。".format(original["image_name"]))  # 不猜测正确来源。
        group = pairing_groups.get(original["image_name"], {})  # 按图片文件名恢复它自己的两个温度。
        if set(group) != {0, 1}:  # 保留图片必须有两条原配对记录。
            raise RepairError("{}在pairing.csv缺少完整的两条温度。".format(original["image_name"]))  # 不挪用后一张图的温度。
        for sample in (0, 1):  # 核对原温度CSV与原追溯表。
            pair = group[sample]  # 该图该采样点的原记录。
            if str(pair["source_frame_number"]) != str(original["source_frame_number"]) or str(pair["layer_id"]) != str(original["layer_id"]):  # 原图身份必须完全对应。
                raise RepairError("{}的原帧号或层号与pairing.csv不一致。".format(original["image_name"]))  # 拒绝错配。
            old_samples = temperature_groups.get(str(old_index), {})  # 原image_seq_idx仍按用户确认的旧编号解释。
            if old_samples and (set(old_samples) != {0, 1} or not _same_temperature(pair, old_samples[sample])):  # 两张表存在不同原值时不能选一张覆盖。
                raise RepairError("{}的temperature.csv与pairing.csv原温度不一致。".format(original["image_name"]))  # 明确定位到图片。
        gap = previous is not None and (old_index != previous[0] + 1 or raw_frame != previous[1] + 1 or original["segment_id"] != previous[2] or original["layer_id"] != previous[3])  # 包含真实删除、层与原连续段变化。
        if previous is None or gap:  # 每个保留连续段使用独立模型段号。
            segment_count += 1  # 不因类别改变切段。
        segment = str(original["segment_id"]) + "__R{:04d}".format(segment_count) if structural_repair else str(original["segment_id"])  # 只有结构修复才增加缺口段号。
        current = dict(original, seq_idx=new_index, segment_id=segment, temperature_start_seq_idx=2 * new_index, temperature_end_seq_idx=2 * new_index + 1)  # 只更新派生索引。
        manifest.append(current)  # 图片名、真实时刻和原帧号均不改写。
        for sample in (0, 1):  # 两个温度点一起随原图片重建模型序号。
            pair = dict(group[sample], seq_idx=2 * new_index + sample, image_seq_idx=new_index, sample_in_image=sample, segment_id=segment)  # 原温度索引、值和时间保持字符串原样。
            pairings.append(pair)  # source_state仍为原始来源状态，人工新标签另存区间表。
            temperatures.append({name: pair[name] for name in ("seq_idx", "image_seq_idx", "sample_in_image", "temperature_c", "temperature_index", "temperature_datetime")})  # 模型需要的长表。
        state_rows.append(dict(current, source_state=labels[old_index]))  # 用新的人工状态生成规范区间。
        index_map.append({"old_seq_idx": old_index, "new_seq_idx": new_index, "image_name": original["image_name"], "source_frame_number": raw_frame, "old_segment_id": original["segment_id"], "new_segment_id": segment, "temperature_index_0": group[0]["temperature_index"], "temperature_index_1": group[1]["temperature_index"]})  # 对照表放在备份中。
        previous = (old_index, raw_frame, original["segment_id"], original["layer_id"])  # 不使用新编号判断原始缺口。
    intervals = make_intervals(state_rows)  # 重建辅助列，原start_source_frame_number旧值不再残留。
    frame_labels = expand_intervals(intervals, manifest)  # 温度无效和源训练开关继续按原规则处理。
    if _source_hashes(folder) != source_hashes:  # 用户在生成计划期间保存CSV时，必须重新读取完整版本。
        raise RepairError("生成修复计划期间CSV已变化，请关闭编辑后重新运行。")  # 不安装可能过时的标注。
    report = {"wall_id": folder.name, "old_frames": len(frames), "retained_frames": len(retained), "removed_missing_images": len(missing), "removed_old_seq_idx": missing, "paired_temperature_points": len(temperatures), "continuous_segments": segment_count, "label_counts": dict(Counter(row["state"] for row in frame_labels)), "temperature_pairing_preserved": True, "interval_reference": "original_seq_idx", **notes}  # 所有结论来自已完成的计划。
    return {"folder": folder, "manifest": manifest, "temperatures": temperatures, "pairings": pairings, "intervals": intervals, "labels": frame_labels, "annotation_states": [labels[index] for index in retained], "index_map": index_map, "report": report, "requires_structural_repair": structural_repair, "source_hashes": source_hashes, "existing_image_names": existing_names}  # 保留人工真实状态，避免旧温度无效标记掩盖新标签。


def re_frame(name):  # 只解析约定的文件名前缀，不重新推测时间或图像帧率。
    import re  # 本函数只处理纯文本文件名。
    match = re.match(r"frame_(\d+)_", str(name))  # 当前文件名包含原采集帧号。
    return int(match.group(1)) if match else None  # 不符合该格式时以原清单为准。


def _check_temperature_source(plan):  # 同时兼容不读取源温度的旧修复模式。
    if "temperature_source" in plan and sha256(plan["temperature_source"]) != plan["temperature_source_hash"]:  # 不能覆盖用户预检查后刚保存的新温度版本。
        raise RepairError("源温度已改变，请关闭编辑后重新运行：" + str(plan["temperature_source"]))  # 此检查不会修改源CSV。


def _temperature_source_path(folder, temperature_root, mapping):  # 按明确映射及原报告确定来源，不能按文件排列顺序猜墙。
    import re  # 只用文件名中的采集时间核对来源身份。
    folder, temperature_root = Path(folder), Path(temperature_root).resolve()  # 允许整个项目迁移到其他磁盘。
    report_path = folder / "preparation_report.json"  # 原报告保存了首次使用的温度文件名。
    report = json.loads(report_path.read_text(encoding="utf-8-sig")) if report_path.is_file() else {}  # 原绝对地址只取文件名。
    mapped = str(mapping.get(folder.name, "") or "")  # 优先使用配置里明确的图温映射。
    recorded = str(report.get("source_temperature_file", "") or "")  # 映射没有此墙时可使用原报告。
    names = [Path(text.replace("\\", "/")).name for text in (mapped, recorded) if text]  # 同时兼容Windows和服务器路径分隔符。
    if not names:  # 不能仅凭图像墙时间与温度时间接近就自动配对。
        raise RepairError("缺少本墙温度来源，请在TEMPERATURE_FILES中填写：" + folder.name)  # 用户只需配置明确来源。
    keys = [re.search(r"(\d{8}_\d{6})", name) for name in names]  # 历史修订后缀不属于采集身份。
    if any(key is None for key in keys) or len({key.group(1) for key in keys}) != 1:  # 配置与原报告不一致时拒绝猜测。
        raise RepairError("TEMPERATURE_FILES与原报告的温度采集时间不一致：" + str(names))  # 保持原图温对应关系。
    canonical = "serial_50hz_" + keys[0].group(1) + ".csv"  # 用户已统一清理为这个标准文件名。
    candidates = list(dict.fromkeys([canonical] + names))  # 优先标准名，兼容尚未统一命名的旧目录。
    for name in candidates:  # 只尝试明确属于同一次采集的文件。
        path = temperature_root / name  # 总是使用当前配置根目录，忽略旧电脑的绝对地址。
        if path.is_file():  # 不从备份目录或旧修订里自动挑选温度。
            if path.resolve().parent != temperature_root:  # 不接受指向温度根目录之外的符号链接。
                raise RepairError("温度文件不在配置根目录内：" + str(path))  # 防止读错来源。
            return path  # 返回本次唯一确定的来源。
    raise RepairError("当前温度目录找不到本墙来源：" + str(temperature_root / canonical))  # 不会新建空温度文件。


def plan_refresh(folder, temperature_path, invalid_values, minimum_valid):  # 只读计划：缺图修复与新温度、标签在同一版本内生成。
    temperature_path = Path(temperature_path).resolve()  # 固定当前读取的来源文件。
    source_hash = sha256(temperature_path)  # 在读取前记录版本。
    lookup, previous_index = {}, None  # 原采样编号是唯一更新键，不按剩余图片行号重新配温度。
    for row in read_csv(temperature_path, ["index", "time", "value"]):  # time只保留在源CSV，不拿舍入后的时间重新匹配。
        index = integer(row["index"], "源温度index")  # 同时兼容CSV索引列中的首尾空白。
        if index < 0 or index in lookup or (previous_index is not None and index <= previous_index):  # 不允许重复编号或改变采集顺序。
            raise RepairError("源温度index重复、为负或倒序：{}，{}".format(temperature_path.name, index))  # 不按行数猜采样身份。
        value_text = str(row["value"]).strip()  # 保留用户最终保存的温度数值文本。
        try:  # 无效占位值可以保留，但不能把非数字文字静默替换成温度。
            numeric = float(value_text)  # 只用于数值有效性判断。
        except (TypeError, ValueError) as error:  # 定位具体原采样编号。
            raise RepairError("源温度value不是数值：index={}".format(index)) from error  # 不会修改原文件。
        lookup[index] = (value_text, numeric)  # 更新值及有效性都来自同一条采样记录。
        previous_index = index  # 后续原编号必须继续向前。
    plan = plan_repair(folder)  # 先核验旧图温对应、缺图、人工区间与编号，失败时不写本墙。
    updated_rows, changed_samples, available_images = [], 0, 0  # 分别统计配对行、变化数和可用图像数。
    groups = _samples_by(plan["pairings"], "image_seq_idx")  # 使用修复后的图片编号，仍是每图原来的0/1两个采样。
    for image in plan["manifest"]:  # 保持原图顺序与层内缺口边界。
        samples = groups[str(image["seq_idx"])]  # 本图的两个原采样编号已由plan_repair核对。
        group_available = True  # 只检查数值能否计算；799.9及旧时间标记不覆盖人工状态。
        for slot in (0, 1):  # 不平均、不插值、不挪用下一张图的温度。
            pair = samples[slot]  # 在计划中更新，不碰磁盘。
            index = integer(pair["temperature_index"], "配对temperature_index")  # 通过原采样编号查当前源温度值。
            if index not in lookup:  # 源温度删除了行或改了编号时不能继续原对应。
                raise RepairError("{}的原温度采样index={}在当前源CSV中不存在。".format(image["image_name"], index))  # 保持整墙未写入。
            text, numeric = lookup[index]  # 同一原编号的新数值。
            old_value = float(pair["temperature_c"])  # 仅统计真实数值变化，不受小数位文本格式影响。
            changed_samples += int(not (old_value == numeric or (math.isnan(old_value) and math.isnan(numeric))))  # 允许源无效值按规则留存。
            usable = math.isfinite(numeric)  # 799.9是有限数值，保留原值。
            pair.update(temperature_c=text, sample_available=int(usable))  # 只改温度值及其有效性，原采样编号与原配对时刻不变。
            group_available = group_available and usable  # 时间对齐记录只追溯，不作为状态或窗口过滤条件。
        image["temperature_available"] = int(group_available)  # 重新计算，避免修复了799.9却仍沿用旧的不可用标记。
        available_images += int(group_available)  # 统计本次实际可用图片组数。
        for slot in (0, 1):  # 同步追溯表中的整组有效性。
            pair = samples[slot]  # 使用刚刚更新的同一条计划记录。
            pair.update(temperature_available=int(group_available), use_for_training=image["use_for_training"])  # 原人工训练开关保持不变。
            updated_rows.append(pair)  # 顺序仍为图0的0/1、图1的0/1。
    state_rows = [dict(image, source_state=state) for image, state in zip(plan["manifest"], plan["annotation_states"])]  # 真实状态只来自本轮人工区间，不由温度值决定。
    plan["pairings"] = updated_rows  # 两份温度接口必须取同一版本的样本。
    plan["temperatures"] = [{name: row[name] for name in ("seq_idx", "image_seq_idx", "sample_in_image", "temperature_c", "temperature_index", "temperature_datetime")} for row in updated_rows]  # 训练长表仍为每图两行。
    plan["intervals"] = make_intervals(state_rows)  # 更新区间端点、图片名、层与有效性辅助列。
    plan["labels"] = expand_intervals(plan["intervals"], plan["manifest"])  # 严格展开人工状态，温度不生成ignore。
    plan.update(temperature_source=temperature_path, temperature_source_hash=source_hash)  # 安装前需要再次核验来源版本。
    plan["report"].update(operation="refresh_existing", source_temperature_file=str(temperature_path), source_temperature_sha256=source_hash, changed_paired_temperature_rows=changed_samples, available_temperature_frames=available_images, unavailable_temperature_frames=len(plan["manifest"]) - available_images, temperature_input_policy="raw", original_temperature_indices_preserved=True, original_pairing_datetimes_preserved=True, label_counts=dict(Counter(row["state"] for row in plan["labels"])))  # 明确这是按采样编号更新值，而非重新按时间配对。
    _check_temperature_source(plan)  # 读取期间源CSV有变化就停止。
    if _source_hashes(plan["folder"]) != plan["source_hashes"]:  # 温度更新计划期间同样不能覆盖刚保存的新标注。
        raise RepairError("生成统一更新计划期间清单或标注已改变，请关闭编辑后重新运行。")  # 所有文件仍保持原样。
    return plan  # 调用本函数不会写真实墙数据。


def refresh_batch(data_root, wall_ids, temperature_root, mapping, backup_root, invalid_values, minimum_valid):  # prepare_dataset.py唯一统一更新入口。
    run_folder = Path(backup_root) / ("refresh_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))  # 每次独立保存备份、旧新编号映射和批量报告。
    plans, outcomes = [], []  # 先检查所有选中墙，有任何输入错误则整批不写。
    print("统一更新 {} 面墙：先检查全部图片清单、区间标签与源温度，检查完成前不写墙数据。".format(len(wall_ids)), flush=True)  # IDE控制台清楚显示当前阶段。
    for wall_id in wall_ids:  # 墙名单由入口配置确定。
        try:  # 收集所有墙的问题，避免用户一次只能看到一个错误。
            folder = Path(data_root) / wall_id  # 只处理现有墙目录。
            source = _temperature_source_path(folder, temperature_root, mapping)  # 从明确映射或原报告确定当前温度文件。
            plan = plan_refresh(folder, source, invalid_values, minimum_valid)  # 预检查完全只读。
            plans.append(plan)  # 仅保存检查通过的计划。
            print("检查通过 {}：保留{}张，缺图{}张，温度变化{}个配对行。".format(wall_id, plan["report"]["retained_frames"], plan["report"]["removed_missing_images"], plan["report"]["changed_paired_temperature_rows"]), flush=True)  # 变化行数可能包含被多张图使用的同一原温度点。
        except (ValueError, OSError, KeyError) as error:  # 漏标、冲突、缺原编号或源文件缺失都不猜测。
            outcomes.append({"wall_id": wall_id, "status": "precheck_failed", "error": str(error)})  # 报告具体墙名与原因。
            print("检查未通过 {}：{}".format(wall_id, error), flush=True)  # 保留原seq_idx范围方便回到现有区间表修改。
    run_folder.mkdir(parents=True, exist_ok=False)  # 即使检查失败也保存本次错误报告。
    report_path = run_folder / "refresh_report.json"  # 仅一个批量报告，原文件备份按墙保留。
    if outcomes:  # 任何输入问题都阻止整批发布，避免用户误以为全部已更新。
        outcomes.extend({"wall_id": plan["folder"].name, "status": "not_written_due_to_batch_precheck"} for plan in plans)  # 通过的墙也明确记录未写入。
        write_json(report_path, {"status": "precheck_failed_no_wall_modified", "walls": outcomes})  # 只写错误报告，不改wall_dataset。
        raise RepairError("统一更新已停止，所有墙保持不变。请修正控制台中的标注或来源问题后重跑。报告：" + str(report_path))  # 此时不要运行split或训练。
    for position, plan in enumerate(plans):  # 全部预检查通过后逐墙备份并发布。
        try:  # 每面墙在发布前重新核验CSV、图片集合和源温度版本。
            apply_repair(plan, run_folder / plan["folder"].name)  # 写失败时由原函数恢复本墙全部已替换CSV。
            outcomes.append({**plan["report"], "status": "refreshed"})  # 只将真实完成的墙算作成功。
            print("已更新 {}：{}张图片、{}个温度行，标签与清单已同步。".format(plan["folder"].name, len(plan["manifest"]), len(plan["temperatures"])), flush=True)  # 不启动模型。
        except (ValueError, OSError, KeyError) as error:  # 文件占用或并发改动需用户先处理，不继续覆盖后面的墙。
            outcomes.append({"wall_id": plan["folder"].name, "status": "write_failed_this_wall_rolled_back_or_not_started", "error": str(error)})  # 已完成的前序墙在报告中保持真实成功状态。
            outcomes.extend({"wall_id": other["folder"].name, "status": "not_started"} for other in plans[position + 1:])  # 明确哪些墙没有执行。
            write_json(report_path, {"status": "incomplete_do_not_split_or_train", "walls": outcomes})  # 不能将部分成功当成完成。
            raise RepairError("统一更新未全部完成，请先处理错误并重新运行；不要划分或训练。报告：" + str(report_path)) from error  # 原错误仍可在IDE查看。
    write_json(report_path, {"status": "completed", "temperature_pairing_policy": "preserve_existing_sample_indices_and_datetimes_update_values_by_index", "interval_reference": "pre_run_seq_idx", "walls": outcomes})  # 成功报告可用于确认全部墙已更新。
    print("统一更新完成：{}面墙。备份与报告：{}。下一步在IDE运行code/split_data.py，然后运行所选模型的train.py。".format(len(plans), run_folder), flush=True)  # 仅完成后才提示进入训练流程。
    return outcomes  # 返回真实更新结果，不自动划分或训练。


def apply_repair(plan, backup_folder):  # 只在全部配对、标签与索引校验通过后写派生文件。
    folder, backup_folder = plan["folder"], Path(backup_folder)  # 不写原始采集目录。
    _check_temperature_source(plan)  # 统一更新时检查源温度在预检查后没有被再次修改。
    if _source_hashes(folder) != plan["source_hashes"]:  # 包含计划后新增的同步记录等文件，不能覆盖其他操作。
        raise RepairError("预检查后CSV文件集合或内容已变化，请重新运行。")  # 此时尚未写备份或目标。
    for relative, expected in plan["source_hashes"].items():  # 防止用户在预检查后又改CSV。
        if sha256(folder / relative) != expected:  # 正在编辑的表不能被旧计划覆盖。
            raise RepairError("预检查后文件已变化，请重新运行：" + relative)  # 此时尚未写任何墙文件。
    if {path.name for path in (folder / "images").iterdir() if path.is_file()} != plan["existing_image_names"]:  # 删除或新增图后需要重做计划。
        raise RepairError("预检查后图片集合已经改变，请重新运行。")  # 不使用过时的索引映射。
    backup_folder.mkdir(parents=True, exist_ok=False)  # 不覆盖以前的修复备份。
    for relative in plan["source_hashes"]:  # 写入前把用户当前所有清单原样复制出去。
        destination = backup_folder / "original" / relative  # 备份保留原目录结构。
        destination.parent.mkdir(parents=True, exist_ok=True)  # 每面墙各自保存。
        shutil.copy2(folder / relative, destination)  # 不删除或移动原始采集文件。
        if sha256(destination) != plan["source_hashes"][relative]:  # 备份不完整时不得继续写入。
            raise RepairError("备份校验失败：" + relative)  # 原墙此时仍未改写。
    write_csv(backup_folder / "index_map.csv", plan["index_map"], ["old_seq_idx", "new_seq_idx", "image_name", "source_frame_number", "old_segment_id", "new_segment_id", "temperature_index_0", "temperature_index_1"])  # 后续可按原帧追溯所有新编号。
    write_json(backup_folder / "repair_report.json", plan["report"])  # 保留已删图片的原编号清单。
    replacements = ["frame_manifest.csv", "temperature.csv", "pairing.csv", "frame_labels.csv", "annotations/label_intervals.csv", "annotations/label_sync.json", "preparation_report.json"]  # 不改图片、原温度来源或原始标注快照。
    installed = []  # 出错时仅回滚本次已经写入的派生文件。
    with tempfile.TemporaryDirectory(prefix=".repairing_", dir=folder) as temporary:  # 暂存区只含新CSV，不复制整面墙图片。
        staged = Path(temporary)  # 与目标在同一磁盘，单文件替换保持原子性。
        write_csv(staged / "frame_manifest.csv", plan["manifest"], MANIFEST_FIELDS)  # 新seq_idx从0连续。
        write_csv(staged / "temperature.csv", plan["temperatures"], ["seq_idx", "image_seq_idx", "sample_in_image", "temperature_c", "temperature_index", "temperature_datetime"])  # 每图原来那两个温度。
        write_csv(staged / "pairing.csv", plan["pairings"], PAIR_FIELDS)  # 原温度追溯保持不变。
        write_csv(staged / "frame_labels.csv", plan["labels"], ["seq_idx", "state"])  # 新标签按新图片索引展开。
        write_csv(staged / "annotations" / "label_intervals.csv", plan["intervals"], INTERVAL_FIELDS)  # 重建全部辅助列。
        save_sync_record(staged)  # 为新清单生成一致的同步摘要。
        old_report = folder / "preparation_report.json"  # 保留原始配对说明与层锚点。
        report = json.loads(old_report.read_text(encoding="utf-8-sig")) if old_report.is_file() else {}  # 不重算人眼确认的配对。
        operation = "existing_data_refresh" if "temperature_source" in plan else "deletion_repair"  # 区分统一更新与旧的只修复删图模式。
        report.update(copied_frames=len(plan["manifest"]), paired_frames=len(plan["manifest"]), paired_temperature_points=len(plan["temperatures"]), images_excluded=max(int(report.get("source_frames", plan["report"]["old_frames"])) - len(plan["manifest"]), 0), continuous_segments=plan["report"]["continuous_segments"], training_label_counts=plan["report"]["label_counts"], unavailable_temperature_frames=sum(not flag(row["temperature_available"]) for row in plan["manifest"]))  # 同步更新数量和温度有效性摘要。
        report[operation] = {**plan["report"], "backup_folder": str(backup_folder), "updated_at_utc": datetime.now(timezone.utc).isoformat()}  # 保留原层首锚点和本次操作说明。
        if "temperature_source" in plan:  # 记录本次真正读取的源温度版本。
            report.update(source_temperature_file=str(plan["temperature_source"]), source_temperature_sha256=plan["temperature_source_hash"])  # 下次更新仍可核对来源。
        uses = Counter(row["temperature_index"] for row in plan["pairings"])  # 删图后追溯统计也要对应实际保留样本。
        report.update(unique_source_temperature_points=len(uses), reused_temperature_assignments=sum(count - 1 for count in uses.values()), retained_frames_by_layer=dict(Counter(str(row["layer_id"]) for row in plan["manifest"])), time_aligned_frames=sum(flag(row["time_alignment_ok"]) for row in plan["pairings"] if integer(row["sample_in_image"], "sample_in_image") == 0))  # 原层首锚点仍保留，当前数量单独更新。
        write_json(staged / "preparation_report.json", report)  # 原报告也在备份中。
        if _source_hashes(folder) != plan["source_hashes"] or {path.name for path in (folder / "images").iterdir() if path.is_file()} != plan["existing_image_names"]:  # 暂存期间仍不允许人工改清单或图片集合。
            raise RepairError("暂存期间源清单或图片集合变化，本墙未写入，请重新运行。")  # 保留备份，避免覆盖较新的标注。
        _check_temperature_source(plan)  # 安装前再次确认源温度未改变。
        try:  # 不能把部分写入当作整面墙修复成功。
            for relative in replacements:  # 按固定清单更新。
                (folder / relative).parent.mkdir(parents=True, exist_ok=True)  # 只涉及原墙下的annotations。
                os.replace(staged / relative, folder / relative)  # Excel锁定时抛错并回滚。
                installed.append(relative)  # 记录确实安装的文件。
        except BaseException:  # 中断或权限错误也要尝试恢复原清单。
            for relative in reversed(installed):  # 从完整原备份恢复已改文件。
                original = backup_folder / "original" / relative  # 不从不完整暂存结果恢复。
                if original.is_file():  # 修复前已存在的文件原样还原。
                    shutil.copy2(original, folder / relative)  # 不触碰图片。
                elif (folder / relative).is_file():  # 只清理这次失败事务新建的派生文件，不涉及任何原文件。
                    (folder / relative).unlink()  # 恢复到修复前不存在该派生文件的状态。
            raise  # 不隐藏失败原因，备份仍完整保留。
    return plan["report"]  # 只有全部替换完成才报告成功。


def repair_batch(data_root, wall_ids, backup_root):  # 只处理配置明确列出的墙，错误墙不猜标签。
    run_folder = Path(backup_root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 每次修复建立独立备份。
    plans, outcomes = [], []  # 先检查全部选中墙，之后逐墙发布。
    for wall_id in wall_ids:  # 不扫描或修复未列出的后续墙。
        try:  # 标注冲突不能阻碍其他墙生成明确检查结果。
            plan = plan_repair(Path(data_root) / wall_id)  # 这一步只读。
            if not plan["requires_structural_repair"]:  # 重跑时不反复添加段号或备份同一套清单。
                outcomes.append({"wall_id": wall_id, "status": "no_repair_needed"})  # 需要改状态时另用sync_labels。
                print("无需结构修复 {}：图片完整且seq_idx连续；如只改状态请用sync_labels。".format(wall_id), flush=True)  # 不覆盖本次新改的标注。
                continue  # 不写这面墙。
            plans.append(plan)  # 只把可唯一恢复的墙列入写入计划。
            print("预检查通过 {}：原{}张，保留{}张，已删{}张。".format(wall_id, plan["report"]["old_frames"], plan["report"]["retained_frames"], plan["report"]["removed_missing_images"]), flush=True)  # 用户知道处理范围。
        except (ValueError, OSError, KeyError) as error:  # 非法区间、文件缺失与配对冲突都不会写入该墙。
            outcomes.append({"wall_id": wall_id, "status": "not_modified", "error": str(error)})  # 不伪装为修复成功。
            print("未修改 {}：{}".format(wall_id, error), flush=True)  # 明确需要人工处理的行或原编号。
    run_folder.mkdir(parents=True, exist_ok=False)  # 检查报告和原清单备份统一放在records。
    for plan in plans:  # 每面墙独立备份、独立提交。
        try:  # 若文件在预检查后又变更，停止该墙写入。
            report = apply_repair(plan, run_folder / plan["folder"].name)  # 不删图片，不重新匹配温度。
            outcomes.append({**report, "status": "repaired"})  # 只报告真实成功的墙。
            print("已修复 {}，新seq_idx为0～{}；原配对未变。".format(plan["folder"].name, report["retained_frames"] - 1), flush=True)  # 明确新索引范围。
        except (ValueError, OSError, KeyError) as error:  # 锁定文件等运行问题保留备份并报告。
            outcomes.append({"wall_id": plan["folder"].name, "status": "failed", "error": str(error)})  # 不能静默跳过。
            print("修复失败 {}：{}".format(plan["folder"].name, error), flush=True)  # 提供原错误。
    write_json(run_folder / "repair_report.json", {"walls": outcomes, "temperature_pairing_policy": "preserve_original_two_samples_per_retained_image", "interval_reference": "original_seq_idx"})  # 一份批量结果包含所有成功和未修改墙。
    success = sum(row["status"] == "repaired" for row in outcomes)  # 不将未修改墙算作修复成功。
    unchanged = sum(row["status"] == "no_repair_needed" for row in outcomes)  # 已修复墙重跑时保持不动。
    print("修复结束：成功{}面，无需修复{}面，未完成{}面。备份和报告：{}。下一步在IDE运行split_data.py刷新索引和标签版本，再重新训练。".format(success, unchanged, len(outcomes) - success - unchanged, run_folder), flush=True)  # 不自动启动训练。
    return outcomes  # 便于内部定点检查和后续界面调用。
