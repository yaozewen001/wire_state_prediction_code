# -*- coding: utf-8 -*-
"""使用已完成的预测，统计一帧稳定转熔断与每层首发。"""

from pathlib import Path  # 组织每轮与最终评估的报告目录。

from data_utils import write_csv, write_json  # 复用项目中的CSV和JSON写入方式。


def _first_transitions(dataset, split_name, stable, breakoff):  # 从逐帧真值确定首发，再找可评估窗口。
    selected = dataset.settings.get("training_layer_selection")  # 训练只评价已选监督层。
    first_by_layer, walls = {}, {}  # 墙的计数也包含没有事件的对照墙。
    for wall, sequence in dataset.sequences.items():  # 使用原始时间顺序与真实层号。
        if sequence["record"]["split"] != split_name:  # 三个集合各自统计。
            continue
        walls[wall] = {"sequence_id": wall, "transition_windows": 0, "transition_hits": 0,
                       "first_events": 0, "evaluable_first_events": 0, "first_hits": 0,
                       "negative_windows": 0, "false_positive_windows": 0}  # 记录所有墙的分母。
        allowed_layers = None  # 验证和测试使用评估范围内的完整墙。
        if split_name == "train" and selected is not None:  # 已保存的训练层清单定义训练报告范围。
            allowed_layers = set(selected["layers_by_sequence"][wall]["train_layer_ids"])
        labels = sequence["labels"]  # 只用于离线判定稳定→熔断的真实转变。
        layers = sequence["layer_ids"]  # 从原清单取加工层，防止跨层构造事件。
        for index in range(1, len(labels)):  # 先找真实相邻记录，跳过序列首帧。
            layer = str(layers[index])  # 同一墙中每层最多定义一个首发。
            if str(layers[index - 1]) != layer or labels[index - 1] != stable or labels[index] != breakoff:
                continue  # 同层末帧稳定且下一记录熔断才定义转变。
            if (allowed_layers is not None and layer not in allowed_layers) or (layer == "1" and dataset.settings.get("exclude_first_layer", False)):
                continue  # 老模型保存的明确排除范围仍适用。
            first_by_layer.setdefault((wall, layer), index)  # 保留第一次真实转变，即使前面历史不足。
    return first_by_layer, walls  # 窗口资格在预测表中另行核对。


def save_layer_event_report(destination, dataset, result, class_names, split_name, timeline_title=None):  # 不再执行额外模型推理。
    destination = Path(destination)  # 与整体三分类报告分开保存。
    destination.mkdir(parents=True, exist_ok=True)  # 允许首次创建子目录。
    summary = {"split": split_name, "model_epoch": result.get("model_epoch"), "transition_f1": None,
               "checkpoint": result.get("checkpoint", "best_model.pt"),
               "future_frames": dataset.settings["future_frames"], "decision_rule": "argmax",
               "definition": "当前窗口末帧真值为稳定，且目标帧真值为熔断。每层从原标注取首次真实转变。",
               "time_note": "预测间隔是输入末帧到目标帧的采集时间差。错报窗口逐窗计数。"}  # 记录判定口径。
    if summary["future_frames"] < 1 or not {"stable", "breakoff"}.issubset(class_names):  # 其他预测目标单独注明。
        summary["status"] = "requires_next_frame_stable_breakoff_labels"
        if timeline_title is not None:  # 一帧首发统计不适用于其他目标。
            (destination.parent / "paper_breakoff_timeline.pdf").unlink(missing_ok=True)
        write_json(destination / "summary.json", summary)
        return summary
    stable, breakoff = class_names.index("stable"), class_names.index("breakoff")  # 遵从配置中的类别顺序。
    first_by_layer, walls = _first_transitions(dataset, split_name, stable, breakoff)  # 真实首发独立于模型预测。
    indices = result["indices"]  # 评估可按任意批量执行，输出通过原窗口索引对应。
    if len(indices) != len(dataset.windows) or len(set(indices)) != len(indices) or any(
        len(result[key]) != len(indices) for key in ("labels", "predictions", "scores")
    ):  # 此报告以本集合全部窗口为分母。
        raise ValueError("首发报告需要集合内全部窗口各预测一次。")
    transitions, false_positives, by_target = [], [], {}  # 分开保存真正的转变和一帧错报。
    for position, index in enumerate(indices):  # 逐条关联已保存的分类分数。
        window = dataset.windows[int(index)]  # 窗口来自本次训练保存的同一清单。
        if int(window["current_label"]) != stable:  # 已熔断或顶丝等末帧不属于稳定转熔断前提。
            continue
        actual = int(result["labels"][position])  # 下一帧真值。
        predicted = int(result["predictions"][position])  # 三分类最大分数对应的类别。
        wall, layer = window["sequence_id"], str(window["layer_id"])  # 原墙和加工层。
        counts = walls[wall]  # 逐墙统计误报和命中。
        if actual == breakoff:  # 一条真实转变，无论是否命中都计入分母。
            counts["transition_windows"] += 1
            counts["transition_hits"] += int(predicted == breakoff)
            status = "hit" if predicted == breakoff else "miss"  # 同一个转变只能计一次。
        else:  # 稳定末帧且下一帧仍未熔断的有效窗口。
            counts["negative_windows"] += 1
            if predicted != breakoff:  # 正确未报熔断只更新分母。
                continue
            counts["false_positive_windows"] += 1  # 一帧错报，独立于报警段合并规则。
            status = "false_positive"
        target = int(window["target_seq_idx"])  # 用目标索引与原始首发对齐。
        end_time, target_time = window.get("end_time_s"), window.get("target_time_s")  # 保存采集时刻。
        interval = None  # 无实测时间时保留空值。
        if window.get("has_recorded_timestamps") and end_time is not None and target_time is not None:
            interval = round((target_time - end_time) * 1000.0, 3)  # 只计算当前至目标的一步间隔。
        row = {"sequence_id": wall, "layer_id": layer, "end_seq_idx": int(window["end_seq_idx"]),
               "target_seq_idx": target, "end_time_s": end_time, "target_time_s": target_time,
               "one_step_interval_ms": interval, "true_class": class_names[actual],
               "predicted_class": class_names[predicted], "score_breakoff": float(result["scores"][position][breakoff]),
               "is_first_in_layer": target == first_by_layer.get((wall, layer)),
               "status": status}  # 当前样本的可审计结果。
        if actual == breakoff:
            transitions.append(row)  # 同一层重复转变继续保留。
            by_target[(wall, layer, target)] = row  # 首发只能匹配同墙同层同一目标帧。
        else:
            false_positives.append(row)  # 保留错报对应的采集时间和分数。
    first_rows = []  # 首发表包含没有有效前窗的事件。
    for (wall, layer), target in sorted(first_by_layer.items()):  # 报告便于逐墙逐层核查。
        counts = walls[wall]  # 真实首次转变总数包括无法评价的事件。
        counts["first_events"] += 1
        row = by_target.get((wall, layer, target))  # 后续重复转变不能替代首发。
        if row is None:  # 历史不足、段边界、步长等均可能导致该窗不可用。
            sequence = dataset.sequences[wall]  # 无预测也尽量保留真实事件的采集时刻。
            target_time = float(sequence["timestamps_s"][target]) if sequence.get("has_recorded_timestamps") else None
            first_rows.append({"sequence_id": wall, "layer_id": layer, "target_seq_idx": target,
                               "target_time_s": target_time,
                               "true_class": "breakoff", "is_first_in_layer": True, "status": "no_valid_window"})
            continue
        first_rows.append(row)  # 命中与漏判均有逐窗证据。
        counts["evaluable_first_events"] += 1
        counts["first_hits"] += int(row["status"] == "hit")
    wall_rows = [walls[wall] for wall in sorted(walls)]  # 含没有熔断的对照墙。
    count_fields = ("transition_windows", "transition_hits", "first_events", "evaluable_first_events",
                    "first_hits", "negative_windows", "false_positive_windows")  # 统一定义所有分母。
    summary.update({key: sum(row[key] for row in wall_rows) for key in count_fields})  # 三个集合分别汇总。
    summary["status"] = "ok"  # 零事件的报告仍完整保留误报分母。
    summary["first_misses"] = summary["evaluable_first_events"] - summary["first_hits"]
    summary["unevaluable_first_events"] = summary["first_events"] - summary["evaluable_first_events"]
    summary["event_walls"] = sum(row["first_events"] > 0 for row in wall_rows)  # 区分层事件数量与独立墙数。
    for rate, numerator, denominator in (("first_recall", "first_hits", "evaluable_first_events"),
                                          ("transition_recall", "transition_hits", "transition_windows"),
                                          ("false_positive_rate", "false_positive_windows", "negative_windows")):
        summary[rate] = summary[numerator] / summary[denominator] if summary[denominator] else None  # 空分母留空。
    if summary["transition_windows"]:  # 验证集有真实转变时，才用该F1选权重。
        hits = summary["transition_hits"]  # 真实稳定转熔断且预测熔断的窗口数。
        denominator = summary["transition_windows"] + hits + summary["false_positive_windows"]  # 真实转变数加报熔断数。
        summary["transition_f1"] = 2 * hits / denominator  # 只使用输入末帧真值为稳定的窗口。
    fields = ("sequence_id", "layer_id", "end_seq_idx", "target_seq_idx", "end_time_s", "target_time_s",
              "one_step_interval_ms", "true_class", "predicted_class", "score_breakoff", "is_first_in_layer", "status")
    write_csv(destination / "first_by_layer.csv", first_rows, fields)  # 首发真实事件、命中和覆盖。
    write_csv(destination / "transition_windows.csv", transitions, fields)  # 包含同层后续转变。
    write_csv(destination / "false_positive_windows.csv", false_positives, fields)  # 每次一帧错报。
    write_csv(destination / "by_wall.csv", wall_rows, ("sequence_id",) + count_fields)  # 逐墙计数。
    write_json(destination / "summary.json", summary)  # 写完明细后再发布汇总。
    if timeline_title is not None:  # 每轮验证仍只写原来的事件数据。
        from resnet_result_reports import save_breakoff_timeline  # 仅在最终报告调用绘图。

        save_breakoff_timeline(destination.parent / "paper_breakoff_timeline.pdf", dataset, result, breakoff,
                               first_rows, false_positives, timeline_title)
    return summary
