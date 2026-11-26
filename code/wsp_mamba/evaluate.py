"""加载一次 WSP-Mamba 权重，评估固定划分里的有效窗口。"""

from pathlib import Path

import numpy as np
import torch

if __package__:
    from .blocks import select_scan_backend
    from .config import FORMAT_VERSION, MODEL_NAME, get_settings, model_options, validate_settings
    from .dataset import WindowDataset, load_sequences, make_loader
    from .model import WSPMamba
else:
    from blocks import select_scan_backend
    from config import FORMAT_VERSION, MODEL_NAME, get_settings, model_options, validate_settings
    from dataset import WindowDataset, load_sequences, make_loader
    from model import WSPMamba

from data_utils import read_json, validate_membership
from evaluation_reports import _metrics, save_evaluation_reports
from resnet_result_reports import save_result_workbook
from training_acceleration import PERFORMANCE_KEYS, PrecisionPolicy, transfer_tensor
from training_console import PhaseProgress
from wall_runtime import (_batch_forward, _batch_loss_sum, _phase_metrics, _phase_start, load_weights,
                          make_criterion, setup_environment, training_run_sort_key)

BEST_LOSS_FILE = "best_model.pt"
BEST_TRANSITION_FILE = "best_transition_f1.pt"
POWER_SNAPSHOT = "power_settings.csv"
PHASE_TITLES = {"train": "训练集重评", "val": "验证", "test": "测试"}
PHASE_DIRS = {"train": ".", "val": "validation", "test": "evaluation"}


def completed_runs(output_root, specified):
    """按新到旧读取此模型完整运行的选模信息。"""
    if specified:
        candidates = [Path(specified)]
    else:
        candidates = [path.parent for path in Path(output_root).glob("*/training_summary.json")]
        candidates.sort(key=training_run_sort_key, reverse=True)
    for run in candidates:
        manifest = run / "training_summary.json"
        if not manifest.is_file():
            continue
        summary = read_json(manifest)
        if (summary.get("completed") and summary.get("format_version") == FORMAT_VERSION
                and summary.get("model_name") == MODEL_NAME and (run / BEST_LOSS_FILE).is_file()):
            yield run, summary


def select_checkpoint(run, summary, selector):
    """验证选择的权重文件确实存在。"""
    use_transition = selector == "transition_f1"
    file = BEST_TRANSITION_FILE if use_transition else BEST_LOSS_FILE
    epoch = summary["best_transition_f1_epoch" if use_transition else "best_epoch"]
    if epoch is None or not (run / file).is_file():
        raise ValueError(f"{run} 没有可用的 {file} 权重，请检查该次验证的转变样本。")
    return file, epoch


def restore_run(current):
    """读取训练时保存的模型、数据参数，仅允许评估修改运行资源。"""
    found = next(completed_runs(current["output_root"], current["evaluation_run_dir"]), None)
    if found is None:
        raise ValueError("找不到已完成的 WSP-Mamba 训练，请检查 EVALUATION_RUN_DIR。")
    run, summary = found
    settings = read_json(run / "config.json")
    operational = (
        "data_root", "device", "cpu_threads", "num_workers", "evaluation_batch_size",
        "progress_mode", "progress_refresh_seconds", "plain_log_interval_seconds",
        "evaluation_split", "evaluation_epoch", "mamba_backend",
    ) + PERFORMANCE_KEYS
    settings.update({name: current[name] for name in operational if name in current})
    validate_settings(settings)
    file, epoch = select_checkpoint(run, summary, current["evaluation_epoch"])
    reports = run / "transition_f1_evaluation" if file == BEST_TRANSITION_FILE else run
    return run, settings, file, epoch, reports


def restore_data(run, settings):
    """验证图温、标注与功率快照后，使用训练时冻结的窗口。"""
    split = read_json(run / "split_snapshot.json")
    validate_membership(split["records"])
    snapshot = read_json(run / "dataset_snapshot.json")
    sequences, fingerprint = load_sequences(split["records"], settings, (run / POWER_SNAPSHOT).read_bytes())
    if fingerprint != snapshot["fingerprint"]:
        raise ValueError("当前图温、标注或功率输入与训练快照不一致，请核对原数据版本。")
    return sequences, snapshot["windows"], read_json(run / "normalization.json")


def restore_model(run, settings, file, device):
    """从完整权重恢复网络及 BPM 参考向量。"""
    configuration = read_json(run / "model_config.json")
    options = configuration["model_settings"]
    if options != model_options(settings, options["reference_count"]):
        raise ValueError("保存的模型结构与训练配置不一致。")
    backend = select_scan_backend(settings["mamba_backend"], device)
    print(f"Mamba SSD 后端：{backend}", flush=True)
    model = WSPMamba(options, scan_backend=backend).to(device)
    load_weights(model, run / file, device)
    return model, make_criterion(configuration["class_weights"], device)


def forward_batch(model, batch, dataset, criterion, device):
    """原图批次走公共前向，冻结阶段直接读一次抽取的窗口特征。"""
    if isinstance(batch, dict):
        return _batch_forward(model, batch, dataset, criterion, device)
    features, labels, indices = batch
    settings = dataset.settings
    logits = model.classify_features(transfer_tensor(features, device, settings))
    loss = criterion(logits.float(), transfer_tensor(labels, device, settings))
    return loss, logits, labels.tolist(), indices.tolist(), 0, 0


def run_loader(model, loader, criterion, device, precision, phase, *, optimizer=None, epoch=None, epochs=None):
    """同一批次前向用于训练或验证。验证时只计算一次，完整结果供报告复用。"""
    training = optimizer is not None
    model.train(training)
    started = _phase_start(device)
    dataset = loader.dataset
    progress = PhaseProgress(len(loader), dataset.settings, phase, epoch, epochs)
    total_loss, nominal, encoded = 0.0, 0, 0
    labels, scores, indices = [], [], []
    try:
        with torch.set_grad_enabled(training):
            for number, batch in enumerate(loader, start=1):
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with precision.context():
                    losses, logits, targets, positions, sampled, actual = forward_batch(
                        model, batch, dataset, criterion, device,
                    )
                total_loss += _batch_loss_sum(losses)
                if training:
                    precision.backward_step(losses.mean(), model, optimizer)
                labels.extend(targets)
                indices.extend(positions)
                scores.append(torch.softmax(logits.detach().float(), dim=-1).cpu().numpy())
                nominal += sampled
                encoded += actual
                window = dataset.windows[positions[-1]]
                progress.update(number, total_loss / len(indices), window["sequence_id"], window["layer_id"])
    finally:
        progress.close()
    if not indices:
        raise ValueError(f"{phase}没有有效预测窗口。")
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.concatenate(scores, axis=0)
    predictions = scores.argmax(axis=1)
    metrics = _metrics(labels, predictions, dataset.settings["class_names"])
    _phase_metrics(metrics, total_loss, len(indices), device, started, nominal, encoded)
    return dict(metrics=metrics, labels=labels, scores=scores, predictions=predictions, indices=indices)


def save_reports(destination, dataset, result, phase):
    """调用公共结果报告和三页 Excel 总表。"""
    settings = dataset.settings
    names = settings["class_names"]
    save_evaluation_reports(destination, dataset, result, names, phase)
    if phase != "train":
        # 通用报告包含逐类柱图，WSP-Mamba 按本课题的输出约定保留矩阵和曲线。
        for folder in (destination, destination / "breakoff_layers"):
            (folder / "class_metrics.png").unlink(missing_ok=True)
        save_result_workbook(destination.parent, MODEL_NAME, result["checkpoint"],
                             result["model_epoch"], settings["future_frames"], names)


def main(current=None):
    current = get_settings() if current is None else current
    run, settings, checkpoint, epoch, report_root = restore_run(current)
    device = setup_environment(settings)
    model, criterion = restore_model(run, settings, checkpoint, device)
    precision = PrecisionPolicy(settings, device)
    sequences, windows, normalization = restore_data(run, settings)
    choice = current["evaluation_split"]
    phases = {"all": ("train", "val", "test"), "both": ("val", "test")}.get(choice, (choice,))
    for phase in phases:
        dataset = WindowDataset(sequences, windows[phase], settings, normalization)
        loader = make_loader(dataset, settings, device)
        result = run_loader(model, loader, criterion, device, precision, PHASE_TITLES[phase])
        result.update(model_epoch=epoch, checkpoint=checkpoint)
        destination = report_root / PHASE_DIRS[phase]
        save_reports(destination, dataset, result, phase)
        print(f"{PHASE_TITLES[phase]}完成，准确率 {result['metrics']['accuracy']:.2%}。报告 {destination}", flush=True)
    return report_root


if __name__ == "__main__":
    main()

