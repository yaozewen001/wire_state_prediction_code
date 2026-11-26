# -*- coding: utf-8 -*-
"""将两个 ResNet34 模型的离线评估整理为 Excel 总表与首发案例图。"""

from json import JSONDecodeError  # 识别尚未完整写出的报告。
from pathlib import Path  # 统一运行目录、表格和图像路径。
from tempfile import NamedTemporaryFile  # 先写完整工作簿，再替换旧总表。

from data_utils import read_json  # 沿用项目的 UTF-8 报告读取接口。


RESNET_RESULT_MODELS = ("resnet34_ms_tcn", "resnet34_convlstm",
                        "resnet34_ms_tcn_temp_power", "resnet34_lstm_temp_power", "resnet34_convlstm_temp_power",
                        "cnn_lstm_temp_power", "cnn_tcn_temp_power")  # ResNet34 和含功率 CNN 对照使用同一报表。
SPLITS = (("val", "validation", "验证集"), ("test", "evaluation", "测试集"))  # 工作簿固定按验证、测试排列。
CLASS_LABELS = {"stable": "稳定", "stubbing": "顶丝", "breakoff": "熔断", "non_breakoff": "非熔断"}  # 表内使用中文状态。
PERCENT = "0.00%"  # 单元格仍保存可计算的数值。


def _workbook_class():  # 报表依赖只在这两个模型用到。
    try:
        from openpyxl import Workbook  # 服务器使用 Python 直接生成 Excel。
    except ImportError as exc:
        raise RuntimeError("生成评估总表需要 openpyxl，请在当前解释器安装模型目录 requirements.txt 中的依赖。") from exc
    return Workbook


def require_result_workbook(model_name):  # 在耗时的训练或推理之前检查依赖。
    if model_name in RESNET_RESULT_MODELS:
        _workbook_class()


def _matching_report(path, identity):  # 仅合并当前运行、同一权重及预测目标的评估。
    try:
        report = read_json(path)  # 从已经写入的报告读取数值。
    except (FileNotFoundError, JSONDecodeError):
        return {}  # 另一集合可以还没评估，或尚未完整写出。
    return report if all(key in report and report[key] == value for key, value in identity.items()) else {}


def _split_reports(root, checkpoint, model_epoch, future_frames):  # 按工作簿列顺序收集三个统计范围。
    overall, breakoff, transitions = [], [], []  # 每个列表均为验证、测试两项。
    for split, directory, _ in SPLITS:
        folder = root / directory  # 各集合保留自己的原报告。
        identity = {"split": split, "checkpoint": checkpoint, "model_epoch": model_epoch}
        manifest = _matching_report(folder / "scope_manifest.json", identity)  # 完成标记及统计口径。
        metrics_identity = {**identity, "future_frames": future_frames}
        whole = _matching_report(folder / "metrics.json", {**metrics_identity, "scope": "all_windows"}) if manifest else {}
        overall.append(whole)  # 没有对应报告时保持空白。
        binary = _matching_report(folder / "breakoff_layers" / "metrics.json",
                                  {**metrics_identity, "scope": "breakoff_layers", "positive_class": "breakoff"}) if whole else {}
        breakoff.append(binary)  # 只读取采用熔断正类口径的子集。
        event = {}  # 整体报告缺失时，事件列也尚未评估。
        if whole:
            event = {"status": "disabled"}  # 区分关闭事件统计与没有真实事件。
            if manifest.get("layer_event_report") == "layer_events/summary.json":
                event = _matching_report(folder / "layer_events" / "summary.json", metrics_identity)
        transitions.append(event)
    return overall, breakoff, transitions


def _metric_rows(reports, fields):  # 两个集合使用相同的行名和单元格格式。
    return [(label, fmt, *(report.get(key) for report in reports)) for label, key, fmt in fields]


def _status(report):  # 无分母时的空值通过状态说明来区分。
    return {
        "ok": "已评估",
        "empty": "无有效窗口",
        "disabled": "未启用",
        "requires_next_frame_stable_breakoff_labels": "仅适用于下一帧任务",
    }.get(report.get("status"), "尚未评估")


def _write_sheet(sheet, title, context, note, rows):  # 保持三个页签的阅读顺序与样式一致。
    from openpyxl.styles import Alignment, Font, PatternFill  # Excel 显示格式。

    sheet.title = title  # 第一个页签直接打开整体准确率。
    sheet.sheet_view.showGridLines = False
    for column, width in (("A", 36), ("B", 23), ("C", 23), ("D", 21)):
        sheet.column_dimensions[column].width = width  # 预留中文和百分数空间。
    for row, value, height in ((1, title, 30), (2, context, 24), (3, note, 30)):
        sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=4)
        sheet.cell(row, 1, value)
        sheet.row_dimensions[row].height = height
    sheet["A1"].font = Font(name="Microsoft YaHei", size=17, bold=True, color="FFFFFF")
    sheet["A1"].fill = PatternFill("solid", fgColor="17334B")
    for cell in (sheet["A2"], sheet["A3"]):
        cell.font = Font(name="Microsoft YaHei", size=10, color="445B6E")
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    sheet.append(["指标", *(label for _, _, label in SPLITS)])  # 两列对应同一权重。
    for cell in sheet[4][:3]:
        cell.font = Font(name="Microsoft YaHei", bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="29536B")
    for label, fmt, val, test in rows:
        sheet.append([label, val if val is not None else "—", test if test is not None else "—"])
        for cell in sheet[sheet.max_row][:3]:
            cell.font = Font(name="Microsoft YaHei", size=11, color="17334B", bold=sheet.max_row == 5)
            cell.alignment = Alignment(vertical="center", horizontal="left" if cell.column == 1 else "center")
            cell.fill = PatternFill("solid", fgColor="D5EFEB" if sheet.max_row == 5 else
                                    "F0F5F9" if sheet.max_row % 2 == 0 else "FFFFFF")
            if cell.column > 1 and fmt:
                cell.number_format = fmt  # 百分比、样本数继续可以计算。
        sheet.row_dimensions[sheet.max_row].height = 26 if sheet.max_row == 5 else 19
    sheet.freeze_panes = "B5"  # 保留表头和指标名称。


def _append_confusion_matrices(sheet, reports, names):  # 每个集合保留一张原始计数矩阵。
    from openpyxl.styles import Alignment, Font, PatternFill

    for (_, _, title), report in zip(SPLITS, reports):
        classes = report.get("class_order", names)  # 使用实际保存的类别顺序。
        labels = [CLASS_LABELS.get(name, name) for name in classes]
        sheet.append([])  # 以留白分隔上下两个集合。
        sheet.append([title + "混淆矩阵（数量）"])
        sheet.cell(sheet.max_row, 1).font = Font(name="Microsoft YaHei", bold=True, color="17334B")
        sheet.append(["真实类别 / 预测类别", *labels])
        for cell in sheet[sheet.max_row][:len(labels) + 1]:
            cell.fill = PatternFill("solid", fgColor="29536B")
            cell.font = Font(name="Microsoft YaHei", bold=True, color="FFFFFF")
        matrix = report.get("confusion_matrix", [["—"] * len(labels) for _ in labels])
        for label, counts in zip(labels, matrix):
            sheet.append([label, *counts])
            for cell in sheet[sheet.max_row][:len(labels) + 1]:
                cell.font = Font(name="Microsoft YaHei", size=11, color="17334B")
                cell.alignment = Alignment(horizontal="left" if cell.column == 1 else "center")


def save_result_workbook(root, model_name, checkpoint, model_epoch, future_frames, class_names):  # 从已有推理结果生成总表。
    root = Path(root)  # 每个选模结果有独立的总表。
    overall, binary, events = _split_reports(root, checkpoint, model_epoch, future_frames)
    Workbook = _workbook_class()  # 取得服务器上的 Excel 写入类。
    workbook = Workbook()  # 新建三个固定页签。
    context = f"{model_name}  |  权重 {checkpoint}  |  第 {model_epoch if model_epoch is not None else '—'} 轮  |  提前 {future_frames} 帧"
    for metrics in overall:  # 兼容尚未增加正确窗口数字段的已有报告。
        if metrics:
            metrics["correct_windows"] = sum(row[i] for i, row in enumerate(metrics["confusion_matrix"]))
    rows = _metric_rows(overall, (
        ("准确率（整体三分类）", "accuracy", PERCENT),
        ("有效窗口数", "window_count", "#,##0"),
        ("预测正确的窗口数", "correct_windows", "#,##0"),
        ("Macro-F1（三类平均）", "macro_f1", PERCENT),
        ("分类损失", "loss", "0.0000"),
    ))
    rows.append(("评估状态", None, *(_status(report) for report in overall)))
    for name in class_names:  # 每类显示精确率、召回率、F1 和真实支持数。
        class_reports = [report.get("per_class", {}).get(name, {}) for report in overall]
        label = CLASS_LABELS.get(name, name)
        rows.extend(_metric_rows(class_reports, (
            (f"{label} · 精确率", "precision", PERCENT),
            (f"{label} · 召回率", "recall", PERCENT),
            (f"{label} · F1", "f1", PERCENT),
            (f"{label} · 真实窗口数", "support", "#,##0"),
        )))
    _write_sheet(workbook.active, "整体三分类", context, "统计全部有效预测窗口。— 表示尚未评估或指标无分母。", rows)
    _append_confusion_matrices(workbook.active, overall, class_names)

    rows = _metric_rows([report.get("per_class", {}).get("breakoff", {}) for report in binary], (
        ("熔断 F1", "f1", PERCENT),
        ("熔断精确率", "precision", PERCENT),
        ("熔断召回率", "recall", PERCENT),
        ("真实熔断窗口数", "support", "#,##0"),
    ))
    rows.extend(_metric_rows(binary, (
        ("熔断层有效窗口数", "window_count", "#,##0"),
        ("熔断命中窗口数", "true_positive", "#,##0"),
        ("熔断漏报窗口数", "false_negative", "#,##0"),
        ("熔断误报窗口数", "false_positive", "#,##0"),
    )))
    rows.append(("评估状态", None, *(_status(report) for report in binary)))
    sheet = workbook.create_sheet()  # 第二页专门看熔断层正类。
    _write_sheet(sheet, "熔断层", context, "统计发生过熔断的层内全部有效窗口。稳定、顶丝归入非熔断。", rows)
    _append_confusion_matrices(sheet, binary, ("non_breakoff", "breakoff"))

    rows = _metric_rows(events, (
        ("稳定转熔断 F1", "transition_f1", PERCENT),
        ("真实转变窗口命中率", "transition_recall", PERCENT),
        ("命中窗口数", "transition_hits", "#,##0"),
        ("真实转变窗口数", "transition_windows", "#,##0"),
        ("误报窗口率", "false_positive_rate", PERCENT),
        ("误报窗口数", "false_positive_windows", "#,##0"),
        ("真实非熔断窗口数", "negative_windows", "#,##0"),
        ("每层首次转变命中率", "first_recall", PERCENT),
        ("命中首发的层数", "first_hits", "#,##0"),
        ("可评估首发的层数", "evaluable_first_events", "#,##0"),
        ("真实首发的层数", "first_events", "#,##0"),
        ("无有效前窗的首发层数", "unevaluable_first_events", "#,##0"),
    ))
    rows.append(("评估状态", None, *(_status(report) for report in events)))
    _write_sheet(workbook.create_sheet(), "稳定转熔断", context,
                 "窗口末帧真值为稳定，目标帧熔断为正例。误报按窗口计数，首发按层计数。", rows)

    output = root / "评估总表.xlsx"  # 打开结果目录即可找到准确率。
    with NamedTemporaryFile(dir=root, prefix=".评估总表-", suffix=".xlsx", delete=False) as handle:
        temporary = Path(handle.name)  # Windows 上先关闭临时句柄，再由 openpyxl 写入。
    try:
        workbook.save(temporary)
        try:
            temporary.replace(output)  # 读者只会看到写完的工作簿。
        except PermissionError:
            print(f"评估总表暂未刷新，请关闭 Excel 中打开的 {output} 后重新运行评估。", flush=True)
            return
    finally:
        temporary.unlink(missing_ok=True)  # 写入失败时也清理本次临时文件。
    print(f"评估总表  {output}", flush=True)


def save_breakoff_timeline(path, dataset, result, breakoff, first_rows, false_positives, title):  # 仅绘制最终评估。
    from matplotlib import pyplot as plt  # 复用离线绘图后端。
    from PIL import Image  # 读取案例的末帧和目标帧原图。

    cases = []  # 每种情况至多展示一例。
    for status, rows, label, color, marker in (
        ("hit", first_rows, "First transition hit", "#0072B2", "o"),
        ("miss", first_rows, "First transition miss", "#A44A00", "X"),
        ("false_positive", false_positives, "False alarm", "#A32453", "^"),
    ):
        case = next((row for row in rows if row["status"] == status), None)
        if case is not None:
            cases.append((case, label, color, marker))
    if not cases:
        path.unlink(missing_ok=True)  # 重评无案例时清理旧的同名生成图。
        return

    scored = [(dataset.windows[int(index)], float(score[breakoff]))
              for index, score in zip(result["indices"], result["scores"])]  # 使用已推理的分数。
    with plt.rc_context({"pdf.fonttype": 42, "font.size": 9}):  # 保留矢量文字与线条。
        figure = plt.figure(figsize=(8.8, 2.45 * len(cases) + .55), layout="constrained")
        grid = figure.add_gridspec(len(cases), 3, width_ratios=[4.4, 1, 1])
        figure.suptitle(title + " | Next-frame breakoff scores", fontsize=11)
        for position, (case, label, color, marker) in enumerate(cases):
            wall, layer, target = case["sequence_id"], case["layer_id"], case["target_seq_idx"]
            same_layer = [(window, score) for window, score in scored
                          if window["sequence_id"] == wall and str(window["layer_id"]) == layer]
            reference = next(window for window, _ in same_layer if window["target_seq_idx"] == target)
            # 在案例同一连续段取附近的分数，横轴用输入末帧的位置。
            nearby = [(window, score) for window, score in same_layer
                      if window.get("segment_start_seq_idx", window.get("segment_id")) ==
                      reference.get("segment_start_seq_idx", reference.get("segment_id"))
                      and target - 25 <= window["end_seq_idx"] <= target + 10]
            nearby.sort(key=lambda item: item[0]["end_seq_idx"])
            use_time = reference.get("target_time_s") is not None and all(
                window.get("has_recorded_timestamps") and window.get("end_time_s") is not None for window, _ in nearby)
            if use_time:
                x = [(window["end_time_s"] - reference["target_time_s"]) * 1000 for window, _ in nearby]
                decision_x = (reference["end_time_s"] - reference["target_time_s"]) * 1000
                xlabel = "Input end relative to reference (ms)"
            else:
                x = [window["end_seq_idx"] - target for window, _ in nearby]
                decision_x = reference["end_seq_idx"] - target
                xlabel = "Input end relative to reference (frames)"
            axis = figure.add_subplot(grid[position, 0])
            axis.plot(x, [score for _, score in nearby], color="#33566E", lw=1.3, marker=".", ms=3)
            reference_label = "True transition" if case["is_first_in_layer"] else "Forecast target"
            axis.axvline(0, ls="--", lw=1, color="#555555", label=reference_label)
            axis.scatter(decision_x, case["score_breakoff"], color=color, marker=marker,
                         s=42, zorder=3, label="Case input end")  # 标记该次实际分类所用的输入时刻。
            axis.set(title=f"{label}  |  {wall} / layer {layer}", xlabel=xlabel,
                     ylabel="Breakoff score", ylim=(-.03, 1.03), yticks=[0, .5, 1])
            axis.grid(axis="y", alpha=.2)
            axis.legend(loc="upper left", fontsize=8, frameon=False)
            paths = dataset.sequences[wall]["paths"]  # 用原始图索引回查案例。
            for column, index, caption in (
                (1, reference["end_seq_idx"], "Input end"),
                (2, target, f"Target\n{case['true_class']}"),
            ):
                image_axis = figure.add_subplot(grid[position, column])
                try:  # 目标帧只用于出图，原图读不到时仍保留完整的评估指标。
                    with Image.open(paths[index]) as picture:
                        image_axis.imshow(picture.convert("L"), cmap="gray", vmin=0, vmax=255)
                except OSError:
                    image_axis.text(.5, .5, "Image unavailable", ha="center", va="center", fontsize=8)
                image_axis.set_title(caption, fontsize=8)
                image_axis.set_axis_off()
        figure.savefig(path, dpi=220)  # 曲线、刻度及文字保留为 PDF 矢量元素。
        plt.close(figure)
