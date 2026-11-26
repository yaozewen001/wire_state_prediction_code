"""加载固定权重，按所选集合逐批推理并保存报告。"""

from pathlib import Path  # 定位本次训练的快照和权重。

import numpy as np  # 保存概率与分类指标。
import torch  # 在无梯度模式下执行推理。

if __package__:
    from .config import FORMAT_VERSION, get_settings, validate_settings
    from .model import ResNet34LSTMTempPower
    from .dataset import WindowDataset, load_sequences
    from .reports import _metrics, save_evaluation_reports
else:
    from config import FORMAT_VERSION, get_settings, validate_settings
    from model import ResNet34LSTMTempPower
    from dataset import WindowDataset, load_sequences
    from reports import _metrics, save_evaluation_reports

from data_utils import read_json, validate_membership
from resnet_result_reports import require_result_workbook
from training_acceleration import PERFORMANCE_KEYS, PrecisionPolicy
from training_console import PhaseProgress
from wall_runtime import (_batch_forward, _batch_loss_sum, _phase_metrics, _phase_start, load_weights,
                          make_criterion, make_loader, setup_environment, training_run_sort_key)


BEST_LOSS_FILE = "best_model.pt"  # 最低验证损失权重。
BEST_TRANSITION_FILE = "best_transition_f1.pt"  # 最高稳定转熔断 F1 权重。
POWER_SNAPSHOT = "power_settings.csv"  # 本次训练实际读取的功率表副本。
PHASE_TITLES = {"train": "训练集重评", "val": "验证", "test": "测试"}
PHASE_DIRS = {"train": ".", "val": "validation", "test": "evaluation"}


def main():  # 在 IDE 运行本文件评估已保存的权重。
    current = get_settings()
    output, settings, checkpoint, model_epoch, report_root = restore_run(current)
    require_result_workbook(settings["model_name"])  # 提前检查 Excel 依赖。
    split = read_json(output / "split_snapshot.json")
    validate_membership(split["records"])  # 训练、验证、测试以当时冻结的墙清单为准。
    snapshot = read_json(output / "dataset_snapshot.json")
    sequences, fingerprint = load_sequences(split["records"], settings, (output / POWER_SNAPSHOT).read_bytes())
    if fingerprint != snapshot["fingerprint"]:
        raise ValueError("当前图温配对、标注或结果目录里的功率表快照与训练时不一致。")
    normalization = read_json(output / "normalization.json")  # 评估复用训练统计。
    configuration = read_json(output / "model_config.json")
    device = setup_environment(settings)
    precision = PrecisionPolicy(settings, device)
    model = ResNet34LSTMTempPower(**configuration["model_settings"]).to(device)
    load_weights(model, output / checkpoint, device)
    criterion = make_criterion(configuration["class_weights"], device)
    selector = current["evaluation_split"]
    splits = {"all": ("train", "val", "test"), "both": ("val", "test")}.get(selector, (selector,))
    for phase in splits:
        dataset = WindowDataset(sequences, snapshot["windows"][phase], settings, normalization)
        loader = make_loader(dataset, settings, device=device, evaluation=True)
        result = evaluate_loader(model, loader, criterion, device, precision, PHASE_TITLES[phase])
        result.update(model_epoch=model_epoch, checkpoint=checkpoint)
        destination = report_root / PHASE_DIRS[phase]
        save_evaluation_reports(destination, dataset, result, settings["class_names"], phase)
        metrics = result["metrics"]
        print(f"{PHASE_TITLES[phase]}完成，准确率 {metrics['accuracy']:.2%}，Macro-F1 {metrics['macro_f1']:.4f}。报告 {destination}", flush=True)
    return report_root


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device, precision, phase):  # 一个集合只遍历一次。
    settings = loader.dataset.settings
    model.eval()  # 关闭随机失活并固定归一化统计。
    started = _phase_start(device)
    progress = PhaseProgress(len(loader), settings, phase)
    total_loss, nominal, encoded = 0.0, 0, 0  # 累计损失和实际读图指标。
    labels, scores, indices = [], [], []  # 保存完整结果供报告使用。
    try:
        for number, batch in enumerate(loader, start=1):
            with precision.context():  # 训练和评估共用批次前向计算。
                losses, logits, truth, batch_indices, batch_nominal, batch_encoded = _batch_forward(
                    model, batch, loader.dataset, criterion, device,
                )
            loss_sum = _batch_loss_sum(losses)
            total_loss += loss_sum
            nominal += batch_nominal
            encoded += batch_encoded
            labels.extend(truth)
            scores.extend(torch.softmax(logits.detach().float(), dim=1).cpu().numpy())
            indices.extend(batch_indices)
            last = loader.dataset.windows[indices[-1]]
            progress.update(number, total_loss / len(indices), last["sequence_id"], last["layer_id"])
    finally:
        progress.close()  # 正常结束或中断时都清理进度显示。
    if not indices:
        raise ValueError(f"{phase}没有可用的预测窗口。")
    return summarize_predictions(loader, device, started, total_loss, nominal, encoded, labels, scores, indices)


def restore_run(current):
    specified = current["evaluation_run_dir"].strip()
    candidates = ([Path(specified).expanduser().resolve()] if specified else
                  [item.parent for item in Path(current["output_root"]).glob("*/training_summary.json")])
    accepted = []
    for path in candidates:
        summary_path = path / "training_summary.json"
        if summary_path.is_file():
            summary = read_json(summary_path)
            if (summary.get("format_version") == FORMAT_VERSION and summary.get("model_name") == current["model_name"]
                    and summary.get("completed") and (path / BEST_LOSS_FILE).is_file()):
                accepted.append((path, summary))
    if not accepted:
        raise ValueError(f"找不到 {current['model_name']} 的完整 v10 训练结果，请先运行本模型的 train.py 或检查 EVALUATION_RUN_DIR。")
    output, summary = max(accepted, key=lambda item: training_run_sort_key(item[0]))
    settings = read_json(output / "config.json")  # 保留训练保存的全部时间窗口和数据规则。
    names = ("device", "evaluation_batch_size", "num_workers", "cpu_threads", "progress_mode",
             "progress_refresh_seconds", "plain_log_interval_seconds", "data_root", "evaluation_epoch", "evaluation_split") + PERFORMANCE_KEYS
    for key in names:
        settings[key] = current[key]  # 可更换运行设备及数据目录，评估仍校验实际数据指纹。
    validate_settings(settings)
    selector = current["evaluation_epoch"]
    if selector is None:
        checkpoint, model_epoch, report_root = BEST_LOSS_FILE, summary["best_epoch"], output
    elif selector == "transition_f1":
        checkpoint, model_epoch, report_root = BEST_TRANSITION_FILE, summary["best_transition_f1_epoch"], output / "transition_f1_evaluation"
    else:
        checkpoint, model_epoch = f"checkpoints/epoch_{selector:03d}.pt", selector
        report_root = output / "epoch_evaluations" / f"epoch_{selector:03d}"
    if model_epoch is None or not (output / checkpoint).is_file():
        raise FileNotFoundError(f"该训练结果未保存所选权重 {checkpoint}。")
    return output, settings, checkpoint, model_epoch, report_root


def summarize_predictions(loader, device, started, total_loss, nominal, encoded, labels, scores, indices):
    labels, scores = np.asarray(labels), np.asarray(scores)  # 保留逐窗标签和概率。
    predicted = scores.argmax(axis=1)  # 最大分数对应预测类别。
    metrics = _metrics(labels, predicted, loader.dataset.settings["class_names"])  # 沿用总体分类统计。
    _phase_metrics(metrics, total_loss, len(indices), device, started, nominal, encoded)
    return {"metrics": metrics, "labels": labels, "scores": scores, "predictions": predicted, "indices": indices}

if __name__ == "__main__":  # 允许 IDE 直接运行，导入时只定义函数。
    main()
