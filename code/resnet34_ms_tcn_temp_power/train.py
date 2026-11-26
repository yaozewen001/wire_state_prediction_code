"""训练本模型，逐轮验证并保存两个选择指标对应的最佳权重。"""

import time  # 分别记录训练和验证耗时。
from datetime import datetime  # 每次训练生成独立目录。
from pathlib import Path  # 处理训练记录和权重路径。

import numpy as np  # 汇总训练类别数量和损失。
import torch  # 训练网络和优化器。

if __package__:
    from .config import FORMAT_VERSION, get_settings, model_options, validate_settings
    from .model import ResNet34TCNTempPower
    from .dataset import WindowDataset, build_windows, fit_normalization, load_sequences
    from .evaluate import (BEST_LOSS_FILE, BEST_TRANSITION_FILE, PHASE_TITLES, POWER_SNAPSHOT,
                           evaluate_loader, summarize_predictions)
    from .reports import REPORT_FORMAT_VERSION, plot_history, save_evaluation_reports, save_layer_event_report
else:
    from config import FORMAT_VERSION, get_settings, model_options, validate_settings
    from model import ResNet34TCNTempPower
    from dataset import WindowDataset, build_windows, fit_normalization, load_sequences
    from evaluate import (BEST_LOSS_FILE, BEST_TRANSITION_FILE, PHASE_TITLES, POWER_SNAPSHOT,
                           evaluate_loader, summarize_predictions)
    from reports import REPORT_FORMAT_VERSION, plot_history, save_evaluation_reports, save_layer_event_report

from data_utils import load_split, write_csv, write_json
from resnet_result_reports import require_result_workbook
from training_acceleration import PrecisionPolicy
from training_console import PhaseProgress, print_epoch_summary
from wall_dataset import require_classes, save_window_report
from wall_runtime import (_batch_forward, _batch_loss_sum, _phase_start, _save_weights, load_weights,
                          make_criterion, make_loader, setup_environment)


PERFORMANCE_METRICS = (
    "windows_per_second", "nominal_image_frames", "encoded_image_frames",
    "image_encoding_reuse_fraction", "peak_gpu_allocated_gib", "peak_gpu_reserved_gib",
)


def main():  # 在 IDE 运行本文件启动训练。
    settings = get_settings()
    started = time.perf_counter()
    validate_settings(settings)
    require_result_workbook(settings["model_name"])  # 提前检查 Excel 依赖。

    # 读取固定划分，准备训练和验证样本。
    split = load_split(settings)  # 使用当前固定整墙划分和已选训练层。
    settings["training_layer_selection"] = split["training_layer_selection"]
    settings["training_order"] = {
        "none": "wall_layer_time", "wall": "shuffled_walls_layer_time", "layer": "shuffled_layers_segment_time",
    }[settings["train_shuffle_unit"]]
    power_payload = Path(settings["power_settings_file"]).read_bytes()  # 一次性读表，后续训练和快照使用同一份字节。
    print("正在核对整墙清单、图温时间和功率记录。", flush=True)
    sequences, fingerprint = load_sequences(split["records"], settings, power_payload)
    windows, skipped = build_windows(sequences, settings)
    require_classes(windows, len(settings["class_names"]))
    normalization = fit_normalization(sequences, windows["train"], settings)
    output = Path(settings["output_root"]) / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + settings["model_name"] + "_v10")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", settings)
    write_json(output / "split_snapshot.json", split)
    write_json(output / "dataset_snapshot.json", {"fingerprint": fingerprint, "windows": windows})
    write_json(output / "normalization.json", normalization)
    (output / POWER_SNAPSHOT).write_bytes(power_payload)
    report = save_window_report(output, windows, skipped, training_order=settings["training_order"])
    report["signal_history"] = {
        "short_temperature_steps": settings["short_temperature_steps"],
        "long_temperature_steps": settings["long_temperature_steps"],
        "long_history_seconds": settings["long_history_us"] / 1_000_000,
        "power_steps": settings["power_steps"], "power_source": "setpoint", "power_snapshot": POWER_SNAPSHOT,
        "cutoff": "image_window_end", "scope": "current_eligible_segment",
    }
    write_json(output / "window_report.json", report)
    options = model_options(settings)
    counts = np.bincount([row["label"] for row in windows["train"]], minlength=len(settings["class_names"]))
    weights = (counts.sum() / (len(counts) * counts)).tolist()  # 仅按训练类别频数计算损失权重。
    write_json(output / "model_config.json", {"model_name": settings["model_name"], "model_settings": options, "class_weights": weights})
    print(f"训练结果目录 {output}", flush=True)
    print(f"图像 {options['image_steps']} 张 | 短温度 {settings['short_temperature_steps']} 点 | "
          f"长温度最多 {settings['long_temperature_steps']} 点 | 功率 {settings['power_steps']} 点", flush=True)
    for phase, stats in report.items():
        if phase in windows:
            print(f"{phase}：{stats['sequences']} 面墙，{stats['windows']} 个窗口，类别数 {stats['target_classes']}。", flush=True)
    device = setup_environment(settings)
    precision = PrecisionPolicy(settings, device)

    # 按本模型配置创建网络和优化器。
    model = ResNet34TCNTempPower(**options)
    model.initialize_backbone(settings["resnet_pretrained"], settings["resnet_weights_file"])  # 仅新训练初始化骨干。
    model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"结构 {settings['model_name']} | 参数 {parameter_count:,}", flush=True)
    criterion = make_criterion(weights, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    datasets = {phase: WindowDataset(sequences, windows[phase], settings, normalization) for phase in ("train", "val")}
    loaders = {
        "train": make_loader(datasets["train"], settings, device=device, shuffle=True),
        "val": make_loader(datasets["val"], settings, device=device, evaluation=True),
    }
    performance = {**precision.description(), "device": str(device),
                   "loaders": {phase: loader.performance for phase, loader in loaders.items()}}
    write_json(output / "runtime_performance.json", performance)
    history, best_loss, best_epoch, stale = [], float("inf"), 0, 0
    best_validation, best_transition = None, None

    # 每轮训练后立即验证，按验证结果保存权重。
    for epoch in range(1, settings["epochs"] + 1):
        epoch_start = time.perf_counter()
        loaders["train"].batch_sampler.set_epoch(epoch)
        training = train_one_epoch(model, loaders["train"], criterion, optimizer, device, precision, epoch)
        validation = evaluate_loader(model, loaders["val"], criterion, device, precision, "验证")
        train_stats, val_stats = training["metrics"], validation["metrics"]
        elapsed = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch, "train_loss": train_stats["loss"], "val_loss": val_stats["loss"],
            "train_macro_f1": train_stats["macro_f1"], "val_macro_f1": val_stats["macro_f1"], "val_accuracy": val_stats["accuracy"],
            "train_seconds": train_stats["seconds"], "validation_seconds": val_stats["seconds"], "epoch_seconds": elapsed,
            "cumulative_seconds": time.perf_counter() - started, "estimated_seconds_to_max_epochs": elapsed * (settings["epochs"] - epoch),
        }
        for phase, stats in (("train", train_stats), ("val", val_stats)):
            row.update({phase + "_" + key: stats[key] for key in PERFORMANCE_METRICS})
        checkpoint = None
        if settings["save_epoch_checkpoints"]:
            checkpoint = f"checkpoints/epoch_{epoch:03d}.pt"
            (output / "checkpoints").mkdir(exist_ok=True)
            _save_weights(model, output / checkpoint)
        events = save_layer_event_report(
            output / "validation_by_epoch" / f"epoch_{epoch:03d}", datasets["val"],
            {**validation, "model_epoch": epoch, "checkpoint": checkpoint}, settings["class_names"], "val",
        )  # 复用本轮验证预测，不再前向一次。
        score = events["transition_f1"]
        row["val_transition_f1"] = score
        if score is not None:
            rank = (score, -row["val_loss"])
            if best_transition is None or rank > best_transition[0]:
                _save_weights(model, output / BEST_TRANSITION_FILE)
                best_transition = (rank, epoch)
        improved = row["val_loss"] < best_loss - 1e-8
        if improved:
            best_loss, best_epoch, stale = row["val_loss"], epoch, 0
            best_validation = {**validation, "model_epoch": epoch, "checkpoint": BEST_LOSS_FILE}
            _save_weights(model, output / BEST_LOSS_FILE)
            write_json(output / "best_validation_metrics.json", val_stats)
        else:
            stale += 1
        history.append(row)
        write_csv(output / "history.csv", history, list(row))
        print_epoch_summary(epoch, settings["epochs"], row, best_epoch, stale, settings["patience"], improved)
        if stale >= settings["patience"]:
            break  # 早停仍按整体验证损失。


    # 汇总最佳轮次的验证报告，再用固定权重重评训练集。
    save_evaluation_reports(output / "validation", datasets["val"], best_validation, settings["class_names"], "val")
    load_weights(model, output / BEST_LOSS_FILE, device)
    loaders.clear()  # 释放训练读取进程，再对训练集做固定权重重评。
    loader = make_loader(datasets["train"], settings, device=device, evaluation=True)
    result = evaluate_loader(model, loader, criterion, device, precision, PHASE_TITLES["train"])
    result.update(model_epoch=best_epoch, checkpoint=BEST_LOSS_FILE)
    save_evaluation_reports(output, datasets["train"], result, settings["class_names"], "train")
    plot_history(output, history)
    summary = {
        "format_version": FORMAT_VERSION, "model_name": settings["model_name"], "completed": True,
        "best_epoch": best_epoch, "best_validation_loss": best_loss, "epochs_completed": len(history),
        "best_transition_f1": best_transition[0][0] if best_transition else None,
        "best_transition_f1_epoch": best_transition[1] if best_transition else None,
        "best_transition_f1_validation_loss": -best_transition[0][1] if best_transition else None,
        "parameters": parameter_count, "future_frames": settings["future_frames"],
        "report_format_version": REPORT_FORMAT_VERSION, "signal_history": report["signal_history"],
        "stopping_reason": "patience" if stale >= settings["patience"] else "max_epochs",
        "torch_version": str(torch.__version__), "device": str(device), "total_seconds": time.perf_counter() - started,
    }
    write_json(output / "training_summary.json", summary)  # 完整结束后才发布可评估的结果。
    print(f"训练完成，最低验证损失在第 {best_epoch} 轮。可在 IDE 运行 evaluate.py。", flush=True)
    return output


def train_one_epoch(model, loader, criterion, optimizer, device, precision, epoch):
    settings = loader.dataset.settings
    model.train()  # 切换到训练模式，骨干的归一化规则由模型自身决定。
    started = _phase_start(device)
    progress = PhaseProgress(len(loader), settings, "训练", epoch, settings["epochs"])
    total_loss, nominal, encoded = 0.0, 0, 0  # 累计本轮损失和读图指标。
    labels, scores, indices = [], [], []  # 保存本轮训练指标所需的输出。
    try:
        with torch.enable_grad():  # 训练批次需要保留前向计算图。
            for number, batch in enumerate(loader, start=1):
                optimizer.zero_grad(set_to_none=True)  # 清除上一批梯度。
                with precision.context():  # 训练和评估共用批次前向计算。
                    losses, logits, truth, batch_indices, batch_nominal, batch_encoded = _batch_forward(
                        model, batch, loader.dataset, criterion, device,
                    )
                loss_sum = _batch_loss_sum(losses)
                precision.backward_step(losses.mean(), model, optimizer)  # 反向传播并更新一次参数。
                total_loss += loss_sum
                nominal += batch_nominal
                encoded += batch_encoded
                labels.extend(truth)
                scores.extend(torch.softmax(logits.detach().float(), dim=1).cpu().numpy())
                indices.extend(batch_indices)
                last = loader.dataset.windows[indices[-1]]
                progress.update(number, total_loss / len(indices), last["sequence_id"], last["layer_id"])
    finally:
        progress.close()
    if not indices:
        raise ValueError("训练没有可用的预测窗口。")
    return summarize_predictions(loader, device, started, total_loss, nominal, encoded, labels, scores, indices)

if __name__ == "__main__":  # 允许 IDE 直接运行，导入时只定义函数。
    main()
