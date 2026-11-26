"""训练、验证、测试统一报告：全部有效窗口和所有熔断层窗口；仅分析已有预测。"""

from collections import Counter  # 汇总逐类数量和窗口被排除的原因。
from pathlib import Path  # 管理各统计范围的输出目录。
import matplotlib  # 在服务器离线绘图。
matplotlib.use("Agg")  # 不弹出图形窗口。
import matplotlib.pyplot as plt  # 生成混淆矩阵及分类曲线。
import numpy as np  # 处理标签、分数和筛选掩码。
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score  # 使用标准分类定义。
from data_utils import write_csv, write_json, resolve_layer_start, uses_startup_context, uses_annotation_startup, sequence_boundaries, first_window_end, supervision_rejection  # 评估与取窗共用规则。
from layer_event_reports import save_layer_event_report  # 用同一次预测补充层首发与错报。
from resnet_result_reports import RESNET_RESULT_MODELS, save_result_workbook  # 两个 ResNet 模型共用总表。

REPORT_FORMAT_VERSION = 3  # 第3版的熔断层报告采用二分类口径。
BREAKOFF_CLASSES = ("non_breakoff", "breakoff")  # 混淆矩阵按阴性、阳性排序。

SCOPES = {  # 所有模型使用同样的统计名称。
    "all_windows": ("All windows", "全部有效预测窗口"),
    "breakoff_layers": ("All breakoff layers\nBreakoff vs non-breakoff", "所有出现过有效熔断的层，保留整层有效窗口并统计熔断对非熔断"),
}  # 条件子集不替代总体结果，也不代表首次事件预警率。


def _metrics(labels, predictions, names):  # 空范围与缺类范围都保持固定类别顺序。
    labels, predictions = np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)  # 只读取已保存的预测类别。
    size = len(labels)  # 当前范围内的窗口数。
    class_ids = list(range(len(names)))
    if size:
        matrix = confusion_matrix(labels, predictions, labels=class_ids)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, labels=class_ids, zero_division=0
        )
    else:
        # 保留零窗口层的记录，准确率和 F1 由下方的 return 留空。
        matrix = np.zeros((len(names), len(names)), dtype=int)
        support = np.zeros(len(names), dtype=int)
        precision = recall = f1 = np.zeros(len(names), dtype=float)
    return {"window_count": size, "status": "ok" if size else "empty", "class_order": list(names),  # 空范围明确标记。
            "accuracy": float(np.mean(labels == predictions)) if size else None,  # 空范围不能报告0%准确率。
            "macro_f1": float(f1.mean()) if size else None,  # 非空范围固定对配置类别平均。
            "balanced_accuracy": float(recall[support > 0].mean()) if size else None,  # 只平均实际出现的真值类别。
            "confusion_matrix": matrix.tolist(),  # 保存原始数量，便于人工核对。
            "per_class": {name: {"precision": float(precision[i]) if size else None, "recall": float(recall[i]) if size else None, "f1": float(f1[i]) if size else None, "support": int(support[i]), "predicted_count": int(matrix[:, i].sum())} for i, name in enumerate(names)}}  # 保存每类支持数和预测数。


def _breakoff_metrics(rows):  # 只报告熔断正类，保留所有非熔断窗口作为负例。
    labels = [row["true_breakoff"] for row in rows]  # 稳定和顶丝的未来真值都归为非熔断。
    predictions = [row["predicted_breakoff"] for row in rows]  # 归并原三分类决策，保持既有推理规则。
    metrics = _metrics(labels, predictions, BREAKOFF_CLASSES)  # 复用固定类别顺序、空范围和数量矩阵。
    (tn, fp), (fn, tp) = metrics["confusion_matrix"]  # 行是真值，列是预测。
    metrics.update(true_positive=tp, false_positive=fp, false_negative=fn, true_negative=tn)  # 明确列出命中与错误数。
    metrics.pop("macro_f1")  # 本子集只使用熔断的F1。
    metrics.pop("balanced_accuracy")  # 保留普通二分类准确率作为辅助指标。
    positive = metrics["per_class"]["breakoff"]  # 只保留熔断正类的精确率、召回率、F1和样本数。
    if tp + fp == 0:  # 没有任何熔断预测时，精确率没有分母。
        positive["precision"] = None  # JSON中使用null，CSV中留空。
    if tp + fn == 0:  # 范围可能有窗口，但未来目标中没有熔断。
        positive["recall"] = None  # 无法评估熔断召回。
    if tp + fp + fn == 0:  # 没有真实或预测熔断时，F1同样没有分母。
        positive["f1"] = None  # 区分缺少样本和实际分类错误。
    metrics["per_class"] = {"breakoff": positive}  # 顶丝、稳定和非熔断都不单独输出逐类评分。
    return metrics  # 同时用于熔断层总体、逐墙和逐层报告。


def _catalog(dataset, split_name, breakoff_index):  # 根据原逐帧清单识别所有层，不能从有效窗口反推层号。
    entries = []  # 保留没有窗口的层。
    settings = dataset.settings  # 使用训练保存的窗口和启动设置。
    exclude_first_layer = settings.get("exclude_first_layer", False)  # 使用训练保存的整层范围。
    for sequence_id, sequence in dataset.sequences.items():  # 原数据字典可能同时包含三个集合。
        record = sequence["record"]  # 判断当前墙的固定划分。
        if record["split"] != split_name:  # 不能把训练墙混入验证或测试统计。
            continue  # 只统计本集合原始清单。
        labels = np.asarray(sequence["labels"], dtype=int).copy()  # 内存副本只用于离线筛选和诊断。
        layers = [str(value) for value in sequence.get("layer_ids", [""] * len(labels))]  # 保留原始层顺序。
        segments = [str(value) for value in sequence.get("segments", ["0"] * len(labels))]  # 连续段不能拼接补足局部窗口。
        available = np.asarray(sequence.get("input_available", sequence.get("temperature_available", [True] * len(labels))), dtype=bool)  # 与原生成器使用同一源有效性字段。
        offsets = sequence.get("layer_frame_positions")  # 读取每层原始图像位置。
        if offsets is None:  # 小型检查和旧内存结构没有该字段时按原层顺序累计。
            counts, positions = Counter(), []  # 同层切段不重新计算启动期。
            for layer in layers:  # 不按字符串排序层号。
                positions.append(counts[layer])  # 当前图从0起的层内位置。
                counts[layer] += 1  # 同层下一张图继续计数。
            offsets = positions  # 得到与原生成器相同的回退位置。
        offsets = np.asarray(offsets, dtype=int)  # 向量化启动判断。
        context_only = uses_startup_context(settings)  # 旧输出保持原排除方式，新输出允许未知历史。
        annotation_start = uses_annotation_startup(settings)  # 新版完全排除unknown图温。
        if annotation_start:  # 排除输入不等于把unknown重标成稳定。
            available = available & (labels >= 0)  # 与训练及回放采用同样的输入范围。
        if context_only:  # 显式ignore是坏帧；unknown仅仅没有状态监督。
            available = available & (labels != -2)  # 与实际构窗、记忆输入过滤一致。
            for layer in dict.fromkeys(layers):  # 每层独立屏蔽主动启动过程。
                indices = np.flatnonzero(np.asarray(layers) == layer)  # 每层单独识别首个已知状态。
                skip, _ = resolve_layer_start(settings, sequence_id, layer, labels, offsets, indices)  # 新版用标注，旧版用保存的数量规则。
                labels[(np.asarray(layers) == layer) & (offsets < skip)] = -1  # 启动熔断不计入条件子集。
        if exclude_first_layer:  # 报告保留原始层顺序，但第1层没有可评估图温。
            available = available & (np.asarray(layers) != "1")  # 只改当前报告的掩码，不改原配对数据。
        current_entries = []  # 当前墙全部原始层。
        for order, layer in enumerate(dict.fromkeys(layers), start=1):  # 首次出现的先后决定实际层顺序。
            indices = np.flatnonzero(np.asarray(layers) == layer)  # 当前原始层包含的图像组。
            skip, _ = resolve_layer_start(settings, sequence_id, layer, labels, offsets, indices)  # 与构窗使用相同的逐层起点。
            retained = indices[offsets[indices] >= skip]  # 排除启动后的原始图，尚未因窗口长度或标签过滤。
            first_layer_excluded = exclude_first_layer and layer == "1"  # 按原始层号标明主动排除。
            rejected = Counter({"first_layer_excluded": 1}) if first_layer_excluded else Counter()  # 区分整层排除和有效帧不足。
            boundaries = [] if first_layer_excluded else sequence_boundaries(layers, segments, available if annotation_start else None)  # 排除层不核算候选窗口。
            for first, stop in zip(boundaries[:-1], boundaries[1:]):  # 独立核查每个原始层中的有效输入段。
                if layers[first] != layer:  # 当前层之外的段在后面的层条目中登记。
                    continue  # 不重复统计其他层。
                if annotation_start and not available[first]:  # unknown或无效图温不能形成模型窗口。
                    rejected["excluded_annotation_or_input_blocks"] += 1  # 空层原因明确记录。
                    continue  # 不能把两侧有效段拼接起来。
                eligible = [i for i in range(first, stop) if offsets[i] >= skip]  # 本段第一张非启动图片。
                if not eligible:  # 本段完全被启动规则排除。
                    rejected["startup_only_segment"] += 1  # 不把它误报为模型漏读。
                    continue  # 继续后续原始段。
                first_end = first_window_end(first, eligible[0], settings["window_size"], context_only)  # 因果帧数决定网格。
                if first_end >= stop - settings["future_frames"]:  # 短段不能补用其他层。
                    rejected["insufficient_contiguous_history_and_future"] += 1  # 明确短窗口原因。
                    continue  # 该段不产生可监督候选。
                for end in range(first_end, stop - settings["future_frames"], settings["window_stride"]):  # 与实际构窗一致。
                    start, target = end - settings["window_size"] + 1, end + settings["future_frames"]  # 当前和未来严格分离。
                    reason = supervision_rejection(labels, available, start, end, target, context_only)  # 未知历史允许，未知目标不计指标。
                    if reason:  # 明确记录没有可评估窗口的原因。
                        rejected[reason] += 1  # 三模型使用同一监督规则。
            current_entries.append({"sequence_id": sequence_id, "group_id": record["group_id"], "split": split_name, "layer_id": layer, "layer_order": order, "raw_frames": int(len(indices)), "skip_frames": int(skip), "post_start_frames": int(len(retained)), "available_post_start_frames": int(available[retained].sum()), "post_start_breakoff_frames": int(((labels[retained] == breakoff_index) & available[retained]).sum()) if breakoff_index is not None else 0, "filter_reasons": dict(rejected)})  # 全层零窗口也登记；坏帧不用于认定熔断层。
        training_selection = settings.get("training_layer_selection", {}).get("layers_by_sequence", {}).get(sequence_id)  # 训练集只评价保存的监督层。
        for entry in current_entries:  # 给所有层添加统计范围标志。
            selected_for_training = training_selection is None or (entry["layer_id"].strip() or "0") in training_selection["train_layer_ids"]  # 训练仍服从原监督层清单。
            entry["included_in_split_evaluation"] = "first_layer_excluded" not in entry["filter_reasons"] and (split_name != "train" or selected_for_training)  # 三集合都排除第1层。
            entry["has_post_start_breakoff"] = entry["post_start_breakoff_frames"] > 0  # 真值只定义离线条件子集。
        entries.extend(current_entries)  # 保留原始墙层清单。
    return entries  # 该函数从不调用模型或重建训练样本。


def _flat_metrics(base, metrics, rows):  # 把逐墙逐层结果展开成易读的一行。
    output = dict(base)  # 保留墙层及空范围原因。
    columns = ("window_count", "status", "accuracy", "macro_f1", "balanced_accuracy",
               "true_positive", "false_positive", "false_negative", "true_negative")  # 二分类另外记录熔断计数。
    output.update({key: metrics[key] for key in columns if key in metrics})  # 各范围输出适用的指标。
    for name, class_metrics in metrics["per_class"].items():  # 熔断层只展开熔断正类。
        for key, value in class_metrics.items():  # 支持数、预测数和三项指标。
            output[name + "_" + key] = value  # 每个指标独立一列。
        output[name + "_current_count"] = sum(row["current_true_class"] == name for row in rows)  # 当前类别只用于离线核对，不作为预测输入。
    return output  # 空指标以None保存为CSV空单元格。


def _write_table(path, rows, default_fields):  # 空表也保留可读表头。
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else default_fields  # 保持首次出现的字段顺序。
    write_csv(path, rows, fields)  # 复用现有UTF-8表格接口。


def _draw_matrix(axis, metrics, normalized):  # 单张混淆矩阵和论文双图共用绘制口径。
    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)  # 不改变数量矩阵。
    support = matrix.sum(axis=1, keepdims=True)  # 逐真实类别计算分母。
    values = np.divide(matrix, support, out=np.zeros_like(matrix), where=support != 0) if normalized else matrix  # 无真实样本的行不除零。
    artist = axis.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized else max(1, float(matrix.max())))  # 归一化色标固定0到1。
    for i in range(len(matrix)):  # 遍历真实类别。
        for j in range(len(matrix)):  # 遍历预测类别。
            label = "N/A" if normalized and support[i, 0] == 0 else "{:.1%}".format(values[i, j]) if normalized else str(int(matrix[i, j]))  # 空行不伪报0%召回。
            axis.text(j, i, label, ha="center", va="center", fontsize=13, color="white" if values[i, j] > (0.5 if normalized else matrix.max() / 2) else "black")  # 保证深浅底色上的可读性。
    names = metrics["class_order"]  # 使用已保存的类别顺序。
    axis.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names,
             xlabel="Predicted future state", ylabel="True future state")  # 行是真值，列是预测。
    return artist  # 单图和双图按各自布局加色标。


def _plot_matrix(path, metrics, title, normalized=False):  # 空子集也画固定类别矩阵并注明无可评估样本。
    fig, axis = plt.subplots(figsize=(7.1, 6.0))  # 原有单图的输出大小保持一致。
    artist = _draw_matrix(axis, metrics, normalized)
    suffix = "Row-normalized" if normalized else "Counts"  # 图题说明统计含义。
    axis.set_title(f"{title}\n{suffix} | windows={metrics['window_count']}")  # 标注集合、范围与窗口数。
    if metrics["window_count"] == 0:  # 空范围必须明确说明。
        fig.text(.5, .015, "No evaluable windows; see metrics.json and per_layer_metrics.csv", ha="center", fontsize=9)  # 指向原因表。
    fig.colorbar(artist, ax=axis, shrink=.8)  # 提供数值刻度。
    fig.tight_layout(rect=(0, .025, 1, 1))  # 给底部说明留出位置。
    fig.savefig(str(path), dpi=180)  # 保存静态科研图。
    plt.close(fig)  # 释放图形内存。


def _save_confusion_matrices(destination, metrics, title):  # 数量和按真值归一化的两张图使用固定文件名。
    _plot_matrix(destination / "confusion_matrix.png", metrics, title)  # 显示实际窗口数。
    _plot_matrix(destination / "confusion_matrix_normalized.png", metrics, title, normalized=True)  # 显示各真值类别的预测比例。


def _save_paper_matrices(path, scopes, title):  # 一张论文图并排比较整体与熔断层。
    from matplotlib.ticker import PercentFormatter  # 色标与矩阵单元格都显示百分比。

    with plt.rc_context({"pdf.fonttype": 42, "font.size": 9}):
        figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.9), layout="constrained")
        for axis, (panel, scope) in zip(axes, (("(a) All windows", "all_windows"), ("(b) Breakoff layers", "breakoff_layers"))):
            metrics = scopes[scope]  # 使用与 JSON 和 Excel 一致的原始数量。
            image = _draw_matrix(axis, metrics, normalized=True)  # 依真实类别按行归一化。
            labels = [name.replace("_", "-").capitalize() for name in metrics["class_order"]]
            axis.set_xticks(range(len(labels)), labels)
            axis.set_yticks(range(len(labels)), labels)
            axis.set_title(f"{panel}  |  {metrics['window_count']:,} windows", fontsize=10)
        figure.suptitle(title + " | Row-normalized confusion matrices", fontsize=11)
        figure.colorbar(image, ax=axes, shrink=.78, format=PercentFormatter(xmax=1))
        figure.savefig(path, dpi=220)  # 保存可用于论文排版的 PDF。
        plt.close(figure)


def _plot_class_metrics(destination, metrics, names, split_title):  # 兼容原三个模型的逐类指标柱图。
    fig, axis = plt.subplots(figsize=(8, 4.8))  # 逐类指标并列展示。
    for offset, key, label in ((-.25, "precision", "Precision"), (0, "recall", "Recall"), (.25, "f1", "F1")):  # 三种互补分类指标。
        values = [metrics["per_class"][name][key] or 0 for name in names]  # 空范围由总体状态解释。
        axis.bar(np.arange(len(names)) + offset, values, width=.23, label=label)  # 固定并列位置。
    axis.set(xticks=range(len(names)), xticklabels=names, ylim=(0, 1.05), title=split_title + "\nPer-class metrics", ylabel="Score")  # 分行标明集合与范围。
    axis.legend()  # 标明各指标颜色。
    fig.tight_layout()  # 避免标签裁切。
    fig.savefig(str(destination / "class_metrics.png"), dpi=180)  # 保存总体每类表现。
    plt.close(fig)  # 释放内存。


def _plot_ranking_curves(destination, result, names, split_title, positive_class=None):  # 可只绘制熔断正类曲线。
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))  # ROC和PR分别展示。
    ranking = {}  # 保存可复核的面积和阳性比例。
    curve_rows = []  # 数量、混淆矩阵外，曲线的每个坐标点也保留为CSV。
    labels, scores = np.asarray(result["labels"]), np.asarray(result["scores"])  # 不改变原始分数或决策阈值。
    for i, name in enumerate(names):  # 全部窗口按每类对其余类别计算。
        if positive_class is not None and name != positive_class:  # 熔断层只读取原有熔断分数。
            continue  # 跳过其他类别的ROC、PR及AP。
        positive = labels == i  # 当前类别的二值真值。
        if not positive.any() or positive.all():  # 没有正例或负例时ROC面积没有定义。
            ranking[name] = {"roc_auc": None, "average_precision": None, "reason": "requires_positive_and_negative_windows"}  # 不伪造曲线面积。
            continue  # 保留其他合法类别。
        fpr, tpr, roc_thresholds = roc_curve(positive, scores[:, i])  # 按原始类别分数排序。
        precision, recall, pr_thresholds = precision_recall_curve(positive, scores[:, i])  # 获取精确率、召回率及阈值。
        curve_rows.extend({"curve": "ROC", "class": name, "x": float(x), "y": float(y), "threshold": float(threshold) if np.isfinite(threshold) else None} for x, y, threshold in zip(fpr, tpr, roc_thresholds))  # ROC横轴为假阳率，纵轴为真阳率。
        curve_rows.extend({"curve": "PR", "class": name, "x": float(x), "y": float(y), "threshold": float(pr_thresholds[j]) if j < len(pr_thresholds) and np.isfinite(pr_thresholds[j]) else None} for j, (x, y) in enumerate(zip(recall, precision)))  # PR横轴为召回率，纵轴为精确率；末点无阈值。
        auc, ap = float(roc_auc_score(positive, scores[:, i])), float(average_precision_score(positive, scores[:, i]))  # AP不是梯形积分面积。
        axes[0].plot(fpr, tpr, label="{} AUC={:.3f}".format(name, auc))  # 显示该类别ROC。
        line, = axes[1].plot(recall, precision, label="{} AP={:.3f}".format(name, ap))  # 显示该类别PR。
        axes[1].axhline(float(positive.mean()), ls=":", color=line.get_color(), alpha=.45)  # 阳性比例作为PR参考。
        ranking[name] = {"roc_auc": auc, "average_precision": ap, "positive_fraction": float(positive.mean())}  # 明确指标定义。
    axes[0].plot([0, 1], [0, 1], "--", color="gray", alpha=.5)  # ROC随机参考。
    for axis, title, xlabel, ylabel in ((axes[0], "ROC (one-vs-rest)", "False positive rate", "True positive rate"), (axes[1], "Precision-recall", "Recall", "Precision")):  # 设置各曲线坐标。
        axis.set(title=split_title + "\n" + title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.02))  # 分行显示，避免两幅曲线图标题互相覆盖。
        handles, _ = axis.get_legend_handles_labels()  # 无合法曲线时不生成空图例警告。
        if handles:  # 至少一类可计算。
            axis.legend(fontsize=8)  # 标注面积。
        else:  # 空子集或只有单一真值类别时，不伪造可计算的ROC/PR面积。
            axis.text(.5, .5, "No class has both positive and negative windows", ha="center", va="center", wrap=True, transform=axis.transAxes)  # 数值原因同时写入ranking_metrics.json。
        axis.grid(alpha=.2)  # 提供读数参照。
    fig.tight_layout()  # 避免图题重叠。
    fig.savefig(str(destination / "roc_pr_curves.png"), dpi=180)  # 保存总体概率排序曲线。
    plt.close(fig)  # 释放图形。
    write_json(destination / "ranking_metrics.json", ranking)  # 保存曲线对应数值。
    _write_table(destination / "curve_points.csv", curve_rows, ["curve", "class", "x", "y", "threshold"])  # 支持复画曲线，不必重新运行模型。


def save_evaluation_reports(destination, dataset, result, class_names, split_name):  # 训练集保存混淆矩阵，验证和测试保存完整报告。
    if split_name not in ("train", "val", "test"):  # 防止图题和数据集合混淆。
        raise ValueError("报告集合只能为train、val或test。")  # 不把训练集误标为测试集。
    destination = Path(destination)  # 训练集使用所选权重的报告根目录。
    destination.mkdir(parents=True, exist_ok=True)  # 不写入原始数据目录。
    count = len(result["indices"])  # 预测与数据集窗口索引必须一一对应。
    if any(len(result[key]) != count for key in ("labels", "predictions", "scores")) or len(set(int(i) for i in result["indices"])) != count:  # 不截断错位输出。
        raise ValueError("预测、分数、真值与窗口索引长度不一致，或窗口索引重复。")  # 避免重复计数。
    checkpoint = result.get("checkpoint", "best_model.pt")  # 指定候选轮次时记录实际文件。
    model_epoch = result.get("model_epoch")  # 当前评估所用的权重轮次。
    split_title = {"train": "Training", "val": "Validation", "test": "Test"}[split_name]  # 图题明确区分集合。
    split_title += " | Best checkpoint" if checkpoint == "best_model.pt" else f" | Epoch {model_epoch} checkpoint"
    if dataset.settings.get("exclude_first_layer", False):  # 图题标明训练时的层范围。
        split_title += "\nLayer 1 excluded"  # 沿用训练保存的第一层排除配置。
    if split_name == "train":  # 固定权重重评的训练集只保存根目录两张图。
        metrics = _metrics(result["labels"], result["predictions"], class_names)  # 使用当前权重对实际训练窗的三分类预测。
        _save_confusion_matrices(destination, metrics, split_title)  # 训练图片与所用权重的其他结果放在同一根目录。
        return  # 训练集的最终出图到此完成。
    resnet_report = dataset.settings.get("model_name") in RESNET_RESULT_MODELS  # 其他模型继续使用既有报告格式。
    paper_title = f"{split_title.split(' | ')[0]} | Epoch {model_epoch if model_epoch is not None else 'N/A'} | {dataset.settings['future_frames']}-frame lead"
    scope_metrics = {}  # 保存已经计算过的两张矩阵。
    breakoff_index = class_names.index("breakoff") if "breakoff" in class_names else None  # 兼容没有熔断类别的任务。
    catalog = _catalog(dataset, split_name, breakoff_index)  # 层定义来自原始标注，包含零窗口层。
    by_layer = {(row["sequence_id"], row["layer_id"]): row for row in catalog}  # 以墙与层共同定位。
    base_fields = ["sequence_id", "group_id", "split", "layer_id", "segment_id", "start_seq_idx", "end_seq_idx", "target_seq_idx", "end_time_s", "target_time_s"]  # 追溯输入和未来目标。
    rows = []  # 保存本集合全部已完成的预测。
    for position, index in enumerate(result["indices"]):  # 不重新推理，也不按未来真值改变输入。
        window = dataset.windows[int(index)]  # 恢复本次预测的监督窗口。
        if window["split"] != split_name or int(result["labels"][position]) != int(window["label"]):  # 固定集合与未来目标。
            raise ValueError("评估集合或未来标签与保存窗口不一致。")  # 不悄悄更改提前帧数。
        entry = by_layer[(window["sequence_id"], str(window["layer_id"]))]  # 查询整层离线标记。
        if not entry["included_in_split_evaluation"]:  # 训练报告必须与保存的训练监督窗口一致。
            raise ValueError("训练报告出现了未选作监督的层，请检查保存窗口和训练层快照。")  # 不把早层记忆上下文当作训练监督样本。
        current = int(window["current_label"])  # 当前真值仅供离线查看。
        row = {field: window.get(field, "") for field in base_fields}  # 保留原始索引。
        row.update(current_true_class=class_names[current] if current >= 0 else "unknown", true_class=class_names[int(result["labels"][position])], predicted_class=class_names[int(result["predictions"][position])], layer_order=entry["layer_order"], layer_has_post_start_breakoff=entry["has_post_start_breakoff"])  # 整层标记不输入模型。
        row.update({"score_" + name: float(result["scores"][position, i]) for i, name in enumerate(class_names)})  # 保留原始三类概率。
        rows.append(row)  # 每个有效预测窗口只保存一次。
    best_epoch = model_epoch if checkpoint == "best_model.pt" else None  # best_epoch仍专指默认最佳验证损失。
    manifest = {"report_format_version": REPORT_FORMAT_VERSION, "split": split_name, "checkpoint": checkpoint, "best_epoch": best_epoch, "model_epoch": model_epoch,
                "layer_order_source": "original_frame_manifest_order", "effective_layer_definition": "available_original_frames_after_startup_exclusion",  # 不靠预测是否正确决定层归属。
                "evaluation_window_set": "saved_training_supervision_windows" if split_name == "train" else "all_saved_evaluable_windows",  # 训练只评价实际监督层，验证测试保留完整墙。
                "breakoff_layer_definition": "at_least_one_available_breakoff_label_after_startup_exclusion",  # 排除unknown、ignore和不可用观测。
                "selection_note": "真实标签只用于预测后的离线统计。熔断层保留整层有效窗口，稳定与顶丝统一计为非熔断。",  # 不筛选真阳性窗口冒充整层结果。
                "evaluation_mode": "fixed_weights_eval_no_grad", "layers": catalog, "scopes": {}}  # 保留零窗口层及筛选依据。
    if dataset.settings.get("exclude_first_layer", False):  # 在报告和图题写明整体层范围。
        manifest["excluded_layer_ids"] = ["1"]  # 供离线程序核对评估口径。
    for scope, (english, chinese) in SCOPES.items():  # 只生成全部窗口和熔断层两种范围。
        breakoff_only = scope == "breakoff_layers"  # 子集使用熔断对非熔断的二分类口径。
        folder = destination / scope if breakoff_only else destination  # 总体与熔断层仍分别保存。
        folder.mkdir(parents=True, exist_ok=True)  # 两套结果分开保存。
        def selected_layer(entry):  # 只在推理结束后筛选离线报告。
            return entry["included_in_split_evaluation"] and (scope == "all_windows" or entry["has_post_start_breakoff"])  # 稳定层只排除在熔断层子集之外。
        positions = [i for i, row in enumerate(rows) if selected_layer(by_layer[(row["sequence_id"], str(row["layer_id"]))])]  # 不按窗口的真实未来状态过滤。
        selected_rows = [rows[i] for i in positions]  # 选中层中的全部有效窗口保留。
        if breakoff_only:  # 为熔断层单独保存二分类真值与决策，原始三类预测仍可追溯。
            selected_rows = [dict(row, true_breakoff=int(row["true_class"] == "breakoff"),
                                  predicted_breakoff=int(row["predicted_class"] == "breakoff")) for row in selected_rows]

        def summarize(items):  # 同一范围的总体、逐墙、逐层使用同一统计口径。
            if breakoff_only:  # 在原预测上统计熔断，不重新决策。
                return _breakoff_metrics(items)
            return _metrics([class_names.index(row["true_class"]) for row in items],
                            [class_names.index(row["predicted_class"]) for row in items], class_names)  # 总体使用原三类定义。
        metrics = summarize(selected_rows)  # 计算当前范围的分类指标。
        metrics.update(split=split_name, scope=scope, scope_description=chinese, conditional_subset=breakoff_only,
                       future_frames=dataset.settings["future_frames"], checkpoint=checkpoint, best_epoch=best_epoch, model_epoch=model_epoch,
                       empty_reason="no_qualifying_evaluable_windows" if not selected_rows else "")  # 记录实际评估轮次。
        if breakoff_only:  # 记录阳性定义与原分类决策的归并方式。
            metrics.update(positive_class="breakoff", decision_rule="argmax_is_breakoff")
        else:  # 仅总体有对应完整评估损失与耗时。
            metrics.update({key: result["metrics"][key] for key in ("loss", "seconds") if key in result["metrics"]})  # 不编造子集损失。
        if resnet_report:  # 直接打开 JSON 时先看到整体准确率及分子、分母。
            accuracy = metrics["accuracy"]
            correct = sum(row[i] for i, row in enumerate(metrics["confusion_matrix"]))
            metrics = {"accuracy_percent": f"{accuracy:.2%}" if accuracy is not None else None,
                       "accuracy": accuracy, "correct_windows": correct, "window_count": metrics["window_count"],
                       **metrics}  # 原字段仍是 0 到 1 的准确率，便于脚本继续读取。
            scope_metrics[scope] = metrics  # 图和表复用同一组指标。
        write_json(folder / "metrics.json", metrics)  # 数值结果可由后续脚本直接读取。
        fields = base_fields + ["current_true_class", "true_class", "predicted_class", "layer_order", "layer_has_post_start_breakoff"] + ["score_" + name for name in class_names]  # 空子集也保留完整表头。
        if breakoff_only:  # 空子集也保留二分类字段表头。
            fields += ["true_breakoff", "predicted_breakoff"]  # 1代表熔断，0代表非熔断。
        _write_table(folder / "predictions.csv", selected_rows, fields)  # 每条结果对应原墙、层、窗口和目标帧。
        title = split_title + "\n" + english  # 图题分行写明集合、范围和最佳权重。
        _save_confusion_matrices(folder, metrics, title)  # 验证与测试各范围的两张矩阵图。
        layer_rows = []  # 原层即使没有可评估窗口也登记。
        for entry in catalog:  # 不按有无预测删除层。
            items = [row for row in selected_rows if row["sequence_id"] == entry["sequence_id"] and str(row["layer_id"]) == entry["layer_id"]]  # 当前范围中的本层预测。
            included = selected_layer(entry)  # 判断整层是否属于当前统计范围。
            reason = ""  # 有窗口且入选时无需空原因。
            if "first_layer_excluded" in entry["filter_reasons"]:  # 三集合都明确保留第1层的排除原因。
                reason = "first_layer_excluded"  # 保留零窗口记录便于查表。
            elif not entry["included_in_split_evaluation"]:  # 训练早层只用于记忆上下文，不属于训练监督报告。
                reason = "not_selected_for_training_supervision"  # 与实际无有效窗口区分。
            elif not included:  # 稳定层不属于熔断层子集。
                reason = "outside_scope"  # 保留目录中的原层记录。
            elif not items:  # 入选层没有足够历史或未来标签等。
                reason = ";".join(entry["filter_reasons"]) or "no_evaluable_windows"  # 明确空层原因。
            base = {key: value for key, value in entry.items() if key != "filter_reasons"}  # 展开原始层信息。
            base.update(scope=scope, selected_by_scope=included, empty_reason=reason, filter_reasons=str(entry["filter_reasons"]))  # 便于核查。
            layer_rows.append(_flat_metrics(base, summarize(items), items))  # 保存当前范围的数量和指标。
        wall_rows = []  # 每面墙汇总一行，包含零窗口墙。
        for wall in dict.fromkeys(entry["sequence_id"] for entry in catalog):  # 保持原墙顺序。
            items = [row for row in selected_rows if row["sequence_id"] == wall]  # 本范围中的全部窗口。
            layers = [entry for entry in catalog if entry["sequence_id"] == wall]  # 原墙层清单。
            base = {"sequence_id": wall, "group_id": layers[0]["group_id"], "split": split_name, "scope": scope, "original_layer_count": len(layers), "selected_layer_count": sum(selected_layer(entry) for entry in layers), "empty_reason": "no_qualifying_evaluable_windows" if not items else ""}  # 不省略没有熔断的对照墙。
            wall_rows.append(_flat_metrics(base, summarize(items), items))  # 与当前范围的逐层指标口径一致。
        _write_table(folder / "per_layer_metrics.csv", layer_rows, ["sequence_id", "layer_id", "window_count", "empty_reason"])  # 不按层再拆出多个文件。
        _write_table(folder / "per_wall_metrics.csv", wall_rows, ["sequence_id", "window_count", "empty_reason"])  # 所有墙保存在一张表。
        if not resnet_report:  # 两个 ResNet34 模型的逐类指标列入 Excel。
            _plot_class_metrics(folder, metrics, list(metrics["per_class"]), title)  # 熔断层的柱图只显示熔断。
        scope_result = {key: np.asarray(result[key])[np.asarray(positions, dtype=int)] for key in ("labels", "scores")}  # 复用同次预测的真值和分数。
        _plot_ranking_curves(folder, scope_result, class_names, title,
                             positive_class="breakoff" if breakoff_only else None)  # 熔断层只输出熔断的ROC、PR和AP。
        manifest["scopes"][scope] = {"description": chinese, "windows": len(selected_rows),
                                    "selected_layers": sum(selected_layer(entry) for entry in catalog),
                                    "class_order": metrics["class_order"], "directory": scope if breakoff_only else "."}  # 明确各范围的类别口径。
    if dataset.settings.get("layer_event_reports", False):  # 启用时保存本集合的逐层首发、全部转变和错报。
        save_layer_event_report(destination / "layer_events", dataset, result, class_names, split_name,
                                timeline_title=paper_title if resnet_report else None)
        manifest["layer_event_report"] = "layer_events/summary.json"  # 指向同一权重的事件评估。
    elif resnet_report:  # 本次关闭事件统计时，避免旧案例图混入当前报告。
        (destination / "paper_breakoff_timeline.pdf").unlink(missing_ok=True)
    if resnet_report:
        _save_paper_matrices(destination / "paper_confusion_matrices.pdf", scope_metrics, paper_title)
    write_json(destination / "scope_manifest.json", manifest)  # 记录真实标签筛选依据和实际评估轮次。
    if resnet_report:
        save_result_workbook(destination.parent, dataset.settings["model_name"], checkpoint, model_epoch,
                             dataset.settings["future_frames"], class_names)  # 只并列同一权重的验证与测试。
    return manifest  # 供入口和检查程序使用。
