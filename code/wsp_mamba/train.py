"""训练 WSP-Mamba 基础模型，或在冻结编码器上训练 BPM 和分类头。"""

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .blocks import select_scan_backend
    from .config import FORMAT_VERSION, MODEL_NAME, get_settings, model_options, validate_settings
    from .dataset import CachedFeatureDataset, WindowDataset, build_windows, fit_normalization, load_sequences, make_loader, select_reference_windows
    from .evaluate import BEST_LOSS_FILE, BEST_TRANSITION_FILE, POWER_SNAPSHOT, completed_runs, run_loader, save_reports, select_checkpoint
    from .model import WSPMamba
else:
    from blocks import select_scan_backend
    from config import FORMAT_VERSION, MODEL_NAME, get_settings, model_options, validate_settings
    from dataset import CachedFeatureDataset, WindowDataset, build_windows, fit_normalization, load_sequences, make_loader, select_reference_windows
    from evaluate import BEST_LOSS_FILE, BEST_TRANSITION_FILE, POWER_SNAPSHOT, completed_runs, run_loader, save_reports, select_checkpoint
    from model import WSPMamba

from data_utils import load_split, read_json, write_csv, write_json
from training_acceleration import PrecisionPolicy, transfer_tensor
from training_console import PhaseProgress, print_epoch_summary
from wall_dataset import image_positions, require_classes, save_window_report
from wall_runtime import _save_weights, load_weights, make_criterion, setup_environment


def find_foundation(settings, snapshot):
    """寻找输入、参数、窗口及功率数据都与当前实验吻合的基础权重。"""
    input_keys = (
        "image_size", "window_size", "window_stride", "future_frames", "fps",
        "temperature_samples_per_image", "temperature_stat_exclude_value", "temperature_stat_exclude_atol",
        "startup_policy", "temperature_input_policy", "seed",
    )
    options = model_options(settings)
    for run, summary in completed_runs(settings["output_root"], settings["foundation_run_dir"]):
        if summary["train_stage"] != "foundation":
            continue
        saved = read_json(run / "config.json")
        if any(saved[key] != settings[key] for key in input_keys) or image_positions(saved) != image_positions(settings):
            continue
        if read_json(run / "model_config.json")["model_settings"] != options:
            continue
        if read_json(run / "dataset_snapshot.json") != snapshot:
            continue
        file, epoch = select_checkpoint(run, summary, settings["foundation_checkpoint"])
        return run, file, epoch
    raise ValueError("未找到输入和划分都匹配的基础训练。请检查 FOUNDATION_RUN_DIR 或先按当前配置训练 foundation。")


@torch.no_grad()
def cache_features(model, source, device, precision, title):
    """冻结后每窗只编码一次，为参考库和头部训练复用同一批特征。"""
    model.eval()
    settings = source.settings
    loader = make_loader(source, settings, device)
    features = torch.empty(len(source), model.encoder.feature_dim, dtype=torch.float32)
    progress = PhaseProgress(len(loader), settings, title)
    try:
        for number, batch in enumerate(loader, start=1):
            images = transfer_tensor(batch["images"], device, settings)
            histories = transfer_tensor(batch["temperatures"], device, settings)
            inverse = batch["inverse"]
            if inverse is not None:
                inverse = transfer_tensor(inverse, device, settings)
            with precision.context():
                vectors = model.encode_window(images, histories, inverse)
            features[batch["indices"]] = vectors.float().cpu()
            last = source.windows[batch["indices"][-1]]
            progress.update(number, 0.0, last["sequence_id"], last["layer_id"])
    finally:
        progress.close()
    if not torch.isfinite(features).all():
        raise ValueError("冻结编码器产生了非有限特征，请检查基础权重或混合精度。")
    return CachedFeatureDataset(source, features)


def validation_transitions(dataset, result):
    """按公共报告口径统计当前稳定预测窗的命中、漏报和误报。"""
    stable = dataset.settings["class_names"].index("stable")
    breakoff = dataset.settings["class_names"].index("breakoff")
    selected = np.asarray([dataset.windows[index]["current_label"] == stable for index in result["indices"]])
    actual = result["labels"][selected] == breakoff
    predicted = result["predictions"][selected] == breakoff
    positives = int(actual.sum())
    hits = int(np.count_nonzero(actual & predicted))
    false_positives = int(np.count_nonzero(~actual & predicted))
    denominator = positives + hits + false_positives
    return {
        "val_transition_windows": positives, "val_transition_hits": hits,
        "val_transition_false_positives": false_positives,
        "val_transition_f1": 2 * hits / denominator if positives else None,
    }


def plot_history(output, history):
    """保留训练、验证损失和 Macro-F1 曲线。"""
    from matplotlib import pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for phase in ("train", "val"):
        axes[0].plot(epochs, [row[f"{phase}_loss"] for row in history], label=phase)
        axes[1].plot(epochs, [row[f"{phase}_macro_f1"] for row in history], label=phase)
    for axis, title in zip(axes, ("Weighted loss", "Macro F1")):
        axis.set(xlabel="Epoch", title=title)
        if len(epochs) == 1:
            axis.set_xticks(epochs)
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=160)
    plt.close(figure)


def main(settings=None):
    settings = get_settings() if settings is None else dict(settings)
    validate_settings(settings)
    from openpyxl import Workbook  # 在开始长训练前检查评估总表依赖。
    del Workbook

    start_time = time.perf_counter()
    stage = settings["train_stage"]
    frozen = stage != "foundation"
    device = setup_environment(settings)
    backend = select_scan_backend(settings["mamba_backend"], device)
    precision = PrecisionPolicy(settings, device)
    print(f"WSP-Mamba 阶段 {stage} | Mamba SSD 后端 {backend}", flush=True)

    print("正在核对整墙划分、图温配对及功率记录。", flush=True)
    split = load_split(settings)
    settings["training_layer_selection"] = split["training_layer_selection"]
    power_payload = Path(settings["power_settings_file"]).read_bytes()
    sequences, fingerprint = load_sequences(split["records"], settings, power_payload)
    windows, skipped = build_windows(sequences, settings)
    require_classes(windows, len(settings["class_names"]))
    snapshot = dict(fingerprint=fingerprint, windows=windows)
    references = select_reference_windows(sequences, windows["train"], settings) if stage == "bpm" else []
    foundation = find_foundation(settings, snapshot) if frozen else None
    if foundation:
        normalization = read_json(foundation[0] / "normalization.json")
    else:
        normalization = fit_normalization(sequences, windows["train"], settings)

    model = WSPMamba(model_options(settings), scan_backend=backend)
    if not frozen:
        model.initialize_backbone(settings["atto_pretrained"], settings["atto_weights_file"])
    model.to(device)
    datasets = {split_name: WindowDataset(sequences, windows[split_name], settings, normalization)
                for split_name in ("train", "val")}

    output = Path(settings["output_root"]) / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{stage}")
    output.mkdir(parents=True, exist_ok=False)
    print(f"训练结果目录 {output} | 历史 {len(image_positions(settings))} 图 | 预测跨度 {settings['future_frames']} 帧", flush=True)
    for split_name, samples in windows.items():
        counts = {name: sum(window["label"] == index for window in samples)
                  for index, name in enumerate(settings["class_names"])}
        print(f"{split_name}：{len(samples)} 个窗口，类别数 {counts}。", flush=True)

    if frozen:
        base_run, file, base_epoch = foundation
        load_weights(model, base_run / file, device)
        model.freeze_encoder()
        print(f"已加载基础第 {base_epoch} 轮权重，正在缓存训练和验证窗口特征。", flush=True)
        datasets = {name: cache_features(model, dataset, device, precision, f"{name} 特征缓存")
                    for name, dataset in datasets.items()}
        # 两组冻结对照使用相同的随机种子、训练轮数、批次和优化器。
        torch.manual_seed(settings["seed"])
        vectors = datasets["train"].features[[row["training_window_index"] for row in references]] if references else None
        model.reset_head(vectors)
        write_json(output / "foundation.json", dict(run=str(base_run), checkpoint=file, model_epoch=base_epoch))
        if references:
            write_csv(output / "reference_manifest.csv", references, list(references[0]))
        print(f"参考库 {len(references)} 条向量。", flush=True)

    # 每次运行独立保存数据定义、类别权重以及可重建模型的参数。
    counts = np.bincount([row["label"] for row in windows["train"]], minlength=len(settings["class_names"]))
    class_weights = (counts.sum() / (len(counts) * counts)).tolist()
    write_json(output / "config.json", settings)
    write_json(output / "split_snapshot.json", split)
    write_json(output / "dataset_snapshot.json", snapshot)
    write_json(output / "normalization.json", normalization)
    write_json(output / "model_config.json", dict(model_settings=model_options(settings, len(references)),
                                                 class_weights=class_weights))
    (output / POWER_SNAPSHOT).write_bytes(power_payload)
    save_window_report(output, windows, skipped, training_order=settings["training_order"])
    write_json(output / "runtime_performance.json", {**precision.description(), "device": str(device), "mamba_ssd": backend})

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"总参数 {parameter_count:,} | 可训练参数 {sum(p.numel() for p in trainable):,}", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    criterion = make_criterion(class_weights, device)
    train_loader = make_loader(datasets["train"], settings, device, training=True)
    val_loader = make_loader(datasets["val"], settings, device)
    epochs = settings["head_epochs"] if frozen else settings["epochs"]

    history = []
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_validation, best_transition = None, None
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loader.batch_sampler.set_epoch(epoch)
        train = run_loader(model, train_loader, criterion, device, precision, "训练", optimizer=optimizer,
                           epoch=epoch, epochs=epochs)["metrics"]
        validation = run_loader(model, val_loader, criterion, device, precision, "验证")
        val = validation["metrics"]
        elapsed = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch, "train_loss": train["loss"], "val_loss": val["loss"],
            "train_macro_f1": train["macro_f1"], "val_macro_f1": val["macro_f1"], "val_accuracy": val["accuracy"],
            "train_seconds": train["seconds"], "validation_seconds": val["seconds"], "epoch_seconds": elapsed,
            "cumulative_seconds": time.perf_counter() - start_time,
            "estimated_seconds_to_max_epochs": elapsed * (epochs - epoch),
        }
        for phase, metrics in (("train", train), ("val", val)):
            for key in ("windows_per_second", "nominal_image_frames", "encoded_image_frames",
                        "image_encoding_reuse_fraction", "peak_gpu_allocated_gib", "peak_gpu_reserved_gib"):
                row[f"{phase}_{key}"] = metrics[key]
        row.update(validation_transitions(datasets["val"], validation))
        score = row["val_transition_f1"]
        if score is not None and (best_transition is None or (score, -val["loss"]) > best_transition[:2]):
            best_transition = (score, -val["loss"], epoch)
            _save_weights(model, output / BEST_TRANSITION_FILE)
        improved = val["loss"] < best_loss - 1e-8
        if improved:
            best_loss, best_epoch, stale = val["loss"], epoch, 0
            best_validation = {**validation, "model_epoch": epoch, "checkpoint": BEST_LOSS_FILE}
            _save_weights(model, output / BEST_LOSS_FILE)
            write_json(output / "best_validation_metrics.json", val)
        else:
            stale += 1
        history.append(row)
        write_csv(output / "history.csv", history, list(row))
        if frozen:
            print(f"第 {epoch}/{epochs} 轮 | train loss {train['loss']:.4f} | val loss {val['loss']:.4f}"
                  f" | val F1 {val['macro_f1']:.4f} | 最低验证损失第 {best_epoch} 轮", flush=True)
        else:
            print_epoch_summary(epoch, epochs, row, best_epoch, stale, settings["patience"], improved)
            if stale >= settings["patience"]:
                break

    summary = {
        "format_version": FORMAT_VERSION, "model_name": MODEL_NAME, "completed": True,
        "train_stage": stage, "epochs_completed": len(history), "best_epoch": best_epoch,
        "best_validation_loss": best_loss, "best_transition_f1": best_transition[0] if best_transition else None,
        "best_transition_f1_epoch": best_transition[2] if best_transition else None,
        "future_frames": settings["future_frames"], "parameters": parameter_count,
        "reference_count": len(references), "mamba_ssd": backend,
        "stopping_reason": "patience" if not frozen and stale >= settings["patience"] else "max_epochs",
    }
    # 权重与数据均已保存。先发布训练摘要，之后的出图可独立重跑。
    temporary_summary = output / "training_summary.json.tmp"
    write_json(temporary_summary, summary)
    temporary_summary.replace(output / "training_summary.json")
    del train_loader, val_loader
    print(f"训练权重已保存，最低验证损失在第 {best_epoch} 轮。正在整理报告。", flush=True)

    save_reports(output / "validation", datasets["val"], best_validation, "val")
    load_weights(model, output / BEST_LOSS_FILE, device)
    recheck = make_loader(datasets["train"], settings, device)
    train_result = run_loader(model, recheck, criterion, device, precision, "训练集重评")
    train_result.update(model_epoch=best_epoch, checkpoint=BEST_LOSS_FILE)
    save_reports(output, datasets["train"], train_result, "train")
    plot_history(output, history)
    print("训练及报告完成。用 evaluate.py 评估已保存的权重。", flush=True)
    return output


if __name__ == "__main__":
    main()

