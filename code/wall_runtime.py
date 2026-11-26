"""各模型共用的训练、评估及逐帧预测流程；在各模型目录点击入口运行。"""

import inspect  # 检查权重读取接口是否支持受限加载。
import random  # 固定 Python 随机源。
import re  # 识别结果目录的预测跨度前缀。
import time  # 使用单调时钟记录训练耗时。
from collections import deque  # 在线只缓存当前层的有限局部历史。
from copy import copy  # 只在加载器副本上启用图像字节传输，不改变其他调用的数据接口。
from datetime import datetime  # 为每次运行创建独立结果目录。
from pathlib import Path  # 管理配置、图像和结果地址。

import numpy as np  # 组织输入数组和统计指标。
import torch  # 执行网络计算、反向传播和保存。
from PIL import Image  # 读取磁盘或采集端提供的图像。
from torch import nn  # 定义带权分类损失。
from torch.utils.data import DataLoader  # 加载按墙排列的局部窗口。

from training_console import PhaseProgress, print_epoch_summary  # 显示刷新不参与优化器更新。
from evaluation_reports import REPORT_FORMAT_VERSION, save_evaluation_reports  # 所有子集都在完成整墙预测之后筛选。
from layer_event_reports import save_layer_event_report  # 每轮复用已有验证输出生成事件表。
from resnet_result_reports import require_result_workbook  # 提前检查两个 ResNet 模型的 Excel 依赖。
from data_utils import OrderedLayerBatchSampler, uses_startup_context, uses_annotation_startup, first_window_end  # 共用策略，阶段信号不作为类别输入。
from training_acceleration import (PERFORMANCE_KEYS, PrecisionPolicy, WindowBatchCollator, execution_policy,  # 配置窗口批处理。
                                   phase_performance, transfer_tensor, validate_performance_settings)  # 保存实际精度、吞吐及显存峰值。


TRANSITION_F1_CHECKPOINT = "best_transition_f1.pt"  # 训练保存与独立评估共用的权重文件名。


def training_run_path(settings, suffix):
    """生成带预测跨度的训练结果路径，沿用原来的时间戳和运行后缀。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 固定宽度便于按时间排序。
    return Path(settings["output_root"]) / f"H{settings['future_frames']}_{timestamp}_{suffix}"


def training_run_sort_key(run):
    """新旧命名按同一规则排序，预测跨度不参与时间比较。"""
    name = run.name
    # 只识别带原时间戳的跨度前缀，手动命名的目录沿用原名称。
    timestamp_name = re.sub(r"^H[0-9]+_(?=[0-9]{8}_[0-9]{6}_[0-9]{6}_)", "", name)
    return timestamp_name, name  # 去掉前缀后同名时，以完整目录名确定顺序。


def sync_clock(device):  # GPU 异步计算完成后再读取时钟。
    if device.type == "cuda":  # CPU 不需要同步操作。
        torch.cuda.synchronize(device)  # 避免把尚未完成的 GPU 工作漏出耗时。
    return time.perf_counter()  # 返回不会受系统校时影响的秒数。


def duration(seconds):  # 将耗时转换成控制台容易阅读的形式。
    seconds = max(0, int(seconds))  # 预计剩余时间不显示负数。
    return "{}时{:02d}分{:02d}秒".format(seconds // 3600, seconds % 3600 // 60, seconds % 60)  # 保留小时。


def setup_environment(settings):  # 训练、评估及预测共用设备选择。
    random.seed(settings["seed"])  # 固定 Python 随机数。
    np.random.seed(settings["seed"])  # 固定 NumPy 随机数。
    torch.manual_seed(settings["seed"])  # 固定网络初始化。
    torch.set_num_threads(settings["cpu_threads"])  # 限制 CPU 线程数。
    if torch.cuda.is_available():  # CUDA 可用时固定所有设备随机源。
        torch.cuda.manual_seed_all(settings["seed"])  # 固定 GPU 随机数。
        torch.backends.cudnn.deterministic = True  # 优先使用可复现算法。
        torch.backends.cudnn.benchmark = False  # 不按每次输入重新选择卷积算法。
    requested = settings["device"]  # 获取用户指定的设备。
    if requested == "cuda" and not torch.cuda.is_available():  # 明确要求 GPU 时不静默退回 CPU。
        raise RuntimeError("当前解释器 CUDA 不可用，请选择 GPU 版 PyTorch 环境或调整 DEVICE。")  # 给出可处理的原因。
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested)  # 解析设备。
    print("计算设备：", torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU", flush=True)  # 显示实际设备。
    return device  # 供调用方移动模型和输入。


def ordered_collate(samples):  # 保留逐窗读取路径的样本列表。
    return samples  # 保持样本的原顺序。


def make_loader(dataset, settings, shuffle=False, device=None, evaluation=False):  # 训练可以重排墙或独立层块。
    shuffle_unit = settings.get("train_shuffle_unit", "wall") if shuffle and not evaluation else "none"  # 所有评估按清单顺序。
    validate_performance_settings(settings)  # 旧配置没有加速键时仍可加载。
    batch_size = settings.get("evaluation_batch_size", settings["batch_size"]) if evaluation else settings["batch_size"]  # 前向评估可以用独立批量。
    sampler = OrderedLayerBatchSampler(dataset.windows, batch_size,
                                       wall_shuffle_seed=settings["seed"] if shuffle_unit == "wall" else None,
                                       layer_shuffle_seed=settings["seed"] if shuffle_unit == "layer" else None)  # 每批仍同墙同层同段。
    generator = torch.Generator().manual_seed(settings["seed"])  # 固定加载进程随机源。
    policy = execution_policy(settings, getattr(dataset, "deterministic_image_inputs", False))  # 只对已确认安全的局部模型批处理。
    use_cuda = device.type == "cuda" if device is not None else settings["device"] != "cpu" and torch.cuda.is_available()  # CPU不锁页。
    workers = settings["num_workers"]  # 数据读取进程数独立于每次参数更新的窗口数。
    pin = use_cuda and settings.get("pin_memory", False)  # 预先锁页有助于CPU到GPU的异步传输。
    collate = ordered_collate  # 未启用张量批处理时逐窗前向。
    if policy["local_batch_enabled"]:  # 满足批处理条件时将窗口合成张量。
        collate = WindowBatchCollator(dataset.windows, policy["reuse_image_features"])  # 在锁页前构造最终批次张量。
        if hasattr(dataset, "packed_images"):  # 正式图像数据集可直接搬运原始字节。
            dataset = copy(dataset)  # 不修改原数据集，避免影响其他加载器。
            dataset.packed_images = True  # 移至ResNet前再归一化，与逐窗输入等价。
    options = {}  # 主进程读取时不能传入预取深度或保持进程参数。
    if workers:  # 至少有一个后台读取进程时才能提前加载。
        options.update(prefetch_factor=settings.get("prefetch_factor", 2), persistent_workers=settings.get("persistent_workers", False))  # 避免每轮重新启动工作进程。
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers, pin_memory=pin,  # 采样器保证同墙同层同段。
                        collate_fn=collate, generator=generator, **options)  # 训练、验证与测试共用有序加载。
    loader.performance = dict(policy, batch_size=batch_size, shuffle_walls=shuffle_unit == "wall", shuffle_unit=shuffle_unit,
                              num_workers=workers, pin_memory=pin,  # 记录实际打乱单位。
                              prefetch_factor=options.get("prefetch_factor"), persistent_workers=options.get("persistent_workers", False),  # 读取实际模式。
                              non_blocking_transfer=use_cuda and settings.get("non_blocking_transfer", False),  # CPU禁用CUDA传输。
                              image_storage_dtype="uint8" if getattr(dataset, "packed_images", False) else "float32")  # 仅存储和传输格式变化。
    return loader  # 两个执行路径使用同一个批次采样器。


def make_criterion(weights, device):  # 类别权重只来自训练集。
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device), reduction="none")  # 保留每个窗口损失。


def _save_weights(model, path):  # 写完参数再替换目标，防止留下半写入的权重。
    temporary = path.with_suffix(".pt.tmp")  # 临时文件与目标位于同一目录。
    torch.save(model.state_dict(), str(temporary))  # 只保存用于推理的模型参数。
    temporary.replace(path)  # 完整写入后才发布本轮权重。


def load_weights(model, path, device):  # 加载明确选中的新版模型参数。
    options = {"map_location": device}  # 支持在 CPU 读取 GPU 训练结果。
    if "weights_only" in inspect.signature(torch.load).parameters:  # 兼容不同 PyTorch 版本。
        options["weights_only"] = True  # 仅加载参数。
    state = torch.load(str(path), **options)  # 先核对历史权重的架构。
    if any(name.startswith("process_memory.") for name in state):  # 老实验的分类头也带额外输入。
        raise ValueError("该权重含旧墙内记忆参数。请用当前窗口模型重新训练。")
    model.load_state_dict(state, strict=True)  # 按参数名和形状严格恢复模型。


def local_model_options(saved_options):  # 兼容未启用墙内记忆的旧权重快照。
    if saved_options.get("memory_enabled", False):
        raise ValueError("该次训练使用了已移除的墙内记忆结构。请用当前窗口模型重新训练。")
    legacy_keys = {"memory_enabled", "memory_hidden_size", "memory_state_feedback", "temperature_samples_per_image"}
    return {name: value for name, value in saved_options.items() if name not in legacy_keys}


def _batch_forward(model, samples, dataset, criterion, device):  # 返回逐窗损失和三类分数。
    settings = dataset.settings
    if isinstance(samples, dict):  # ResNet 可以在同一个张量批次内计算多个窗口。
        images = transfer_tensor(samples["images"], device, settings)
        histories = transfer_tensor(samples["temperatures"], device, settings)
        labels = transfer_tensor(samples["labels"], device, settings)
        if samples["inverse"] is None:
            logits = model(images, histories)  # 每个窗口读取自己的全部历史。
        else:
            inverse = transfer_tensor(samples["inverse"], device, settings)
            logits = model.forward_shared_frames(images, inverse, histories)  # 恢复各窗口的时间顺序。
        losses = criterion(logits.float(), labels)  # 以FP32计算逐窗带权损失。
        return losses, logits, samples["labels"].tolist(), samples["indices"], samples["nominal_image_frames"], samples["encoded_image_frames"]

    losses, scores, labels, indices, frames = [], [], [], [], 0  # 逐窗路径沿用原批量归约。
    for images, histories, label, index in samples:
        logits = model(transfer_tensor(images.unsqueeze(0), device, settings),
                       transfer_tensor(histories.unsqueeze(0), device, settings))
        target = transfer_tensor(label.reshape(1), device, settings)  # 真实标签只参加损失计算。
        losses.append(criterion(logits.float(), target).mean())
        scores.append(logits.detach())
        labels.append(int(label))
        indices.append(int(index))
        frames += images.shape[0]
    return torch.stack(losses), torch.cat(scores, dim=0), labels, indices, frames, frames


def _batch_loss_sum(losses):  # 一批只同步一次损失。
    total = losses.detach().sum().item()
    if not np.isfinite(total):
        raise RuntimeError("损失出现NaN或无穷值，请检查输入、学习率或暂时关闭AMP。")
    return total


def _phase_start(device):  # 分开记录训练与各评估阶段的显存峰值。
    started = sync_clock(device)  # 先等上阶段GPU计算完成。
    if device.type == "cuda":  # CPU报告显存为0。
        torch.cuda.reset_peak_memory_stats(device)  # 清除上阶段峰值，当前保留的显存仍会统计。
    return started  # 返回本阶段开始时间。


def _phase_metrics(metrics, total_loss, count, device, started, nominal_frames, encoded_frames):  # 合并损失与实际运行统计。
    seconds = sync_clock(device) - started  # 包含读取、计算及本阶段统计时间。
    metrics.update(loss=total_loss / count, seconds=seconds)  # 按有效窗口数取平均。
    metrics.update(phase_performance(device, seconds, count, nominal_frames, encoded_frames))  # 保存速度、复用比例和显存峰值。


def train_one_epoch(model, loader, criterion, optimizer, device, class_names, epoch, epochs, timing=None, precision=None):  # 一批只更新一次参数。
    from metrics import classification_metrics  # 保留原分类统计口径。
    settings, dataset = loader.dataset.settings, loader.dataset  # 使用实际保存的配置。
    precision = precision or PrecisionPolicy(settings, device)  # 正式训练由入口跨轮次传入同一缩放器。
    model.train()  # 分类头正常启用随机失活，ResNet遵守BN配置。
    started = _phase_start(device)  # 记录训练开始及显存基线。
    totals, count, nominal_frames, encoded_frames = 0.0, 0, 0, 0  # 按窗口累计。
    labels_all, scores_all = [], []  # 保存训练标签和分数。
    progress = PhaseProgress(len(loader), settings, "训练", epoch, epochs)  # 终端仍显示一条活动进度。
    try:  # 中断或出错时也关闭进度条。
        for batch_number, samples in enumerate(loader, start=1):  # 有序采样器不跨墙、层或连续段。
            optimizer.zero_grad()  # 每批开始清空梯度。
            with precision.context():  # 自动混合精度只用于本批前向计算。
                losses, logits, labels, indices, nominal, encoded = _batch_forward(model, samples, dataset, criterion, device)
            totals += _batch_loss_sum(losses)  # 先检查有限损失，再反向更新参数。
            precision.backward_step(losses.mean(), model, optimizer)  # 维持批内逐窗平均损失；FP16缩放后先还原梯度再裁剪。
            count += len(indices)  # 每窗仍单独贡献一个训练标签。
            nominal_frames += nominal  # 未复用时本批会编码的图像数。
            encoded_frames += encoded  # 实际编码的图像数。
            labels_all.extend(labels)  # 原有训练统计。
            scores_all.extend(torch.softmax(logits.detach().float(), dim=1).cpu().numpy())  # 以FP32保存概率，兼容BF16。
            last = dataset.windows[indices[-1]]  # 进度条显示当前批次末窗口。
            progress.update(batch_number, totals / count, last["sequence_id"], last.get("layer_id", ""))  # 显示按时间刷新。
    finally:  # 出错和退出都清理控制台。
        progress.close()  # 保持日志整洁。
    if not count:  # 空训练集不能报告虚假指标。
        raise ValueError("训练集没有有效窗口。")  # 保持原错误处理。
    metrics, _ = classification_metrics(np.asarray(labels_all), np.asarray(scores_all), class_names)  # 在线训练指标口径不变。
    _phase_metrics(metrics, totals, count, device, started, nominal_frames, encoded_frames)  # 记录实测耗时与显存。
    return metrics  # 返回一轮训练结果。


def evaluate_loader(model, loader, criterion, device, class_names, phase="验证", precision=None):  # 完成全墙预测后才筛选报告。
    from metrics import classification_metrics  # 与原始总体指标一致。
    settings, dataset = loader.dataset.settings, loader.dataset  # 使用训练保存的输入规则。
    precision = precision or PrecisionPolicy(settings, device)  # 评估只使用前向精度，不缩放或更新梯度。
    model.eval()  # 固定参数，关闭随机失活。
    started = _phase_start(device)  # 记录评估耗时与显存峰值。
    totals, count, nominal_frames, encoded_frames = 0.0, 0, 0, 0  # 累计窗口数和图像数。
    labels_all, scores_all, indices = [], [], []  # 完整保留原预测顺序。
    progress = PhaseProgress(len(loader), settings, phase)  # 验证和测试也有独立进度。
    try:  # 评估报错时清理显示。
        with torch.no_grad():  # 评估不更新任何权重。
            for batch_number, samples in enumerate(loader, start=1):  # 不基于真实类别改变预测范围。
                with precision.context():  # 和训练使用相同前向精度，独立评估可由配置修改。
                    losses, logits, labels, batch_indices, nominal, encoded = _batch_forward(model, samples, dataset, criterion, device)
                totals += _batch_loss_sum(losses)  # 每批累计未来分类损失。
                count += len(batch_indices)  # 组尾不满批也计入。
                nominal_frames += nominal  # 汇总原窗口图像数。
                encoded_frames += encoded  # 汇总实际空间编码量。
                labels_all.extend(labels)  # 保存真实标签供离线评估。
                scores_all.extend(torch.softmax(logits.float(), dim=1).cpu().numpy())  # 使用FP32计算和保存三类概率。
                indices.extend(batch_indices)  # 每个预测与原窗口一一对应。
                last = dataset.windows[batch_indices[-1]]  # 与逐窗结果使用同一索引。
                progress.update(batch_number, totals / count, last["sequence_id"], last.get("layer_id", ""))  # 只按时间刷新。
    finally:  # 正常结束或中断均执行。
        progress.close()  # 关闭活动进度行。
    if not count:  # 集合为空时给出明确提示。
        raise ValueError("当前评估集合没有有效窗口，请检查层启动排除帧数和窗口大小。")  # 条件子集为空由报告模块登记。
    labels, scores = np.asarray(labels_all), np.asarray(scores_all)  # 恢复完整的逐窗输出。
    metrics, predictions = classification_metrics(labels, scores, class_names)  # 评估口径不变。
    _phase_metrics(metrics, totals, count, device, started, nominal_frames, encoded_frames)  # 不将报告出图耗时算作前向耗时。
    return {"metrics": metrics, "labels": labels, "scores": scores, "predictions": predictions, "indices": indices}  # 报告接口保持不变。


def model_options(settings, positions):  # 将公共设置与各模型的结构参数组合。
    options = {"num_classes": len(settings["class_names"]), "image_steps": len(positions),  # 当前局部图像窗口结构。
               "temperature_steps": settings["window_size"] * settings["temperature_samples_per_image"], "dropout": settings["dropout"]}  # 当前局部温度长度。
    options.update(image_feature_dim=settings.get("image_feature_dim", 64), classifier_hidden_dim=settings.get("classifier_hidden_dim", 32))  # 新模型显式保存128/64，旧v5保持原宽度。
    if settings["model_name"] == "cnn_tcn":  # TCN也明确保存两个时序分支的输出宽度。
        options.update(image_temporal_dim=settings.get("image_tcn_width", 64), temperature_temporal_dim=settings.get("temperature_tcn_width", 32))  # 不改变时间感受野。
    if settings["model_name"] == "cnn_lstm":  # 保持LSTM自己配置的层数。
        options.update(image_hidden_size=settings["image_lstm_hidden"], temperature_hidden_size=settings["temperature_lstm_hidden"], lstm_layers=settings["lstm_layers"])  # 不覆盖用户隐藏宽度。
    elif settings["model_name"] == "cnn_transformer":  # 保持 Transformer 自己的局部结构。
        options.update(image_model_dim=settings["image_transformer_dim"], temperature_model_dim=settings["temperature_transformer_dim"],  # 两个分支的特征维数。
                       transformer_heads=settings["transformer_heads"], transformer_layers=settings["transformer_layers"], transformer_ff_multiplier=settings["transformer_ff_multiplier"], image_frame_positions=positions)  # 注意力、位置与前馈配置。
    elif settings["model_name"] in ("resnet34_ms_tcn", "resnet34_convlstm"):  # 两种ResNet34共享温度分支和骨干设置。
        options.update(temperature_temporal_dim=settings["temperature_tcn_width"],  # 保存温度时序输出宽度。
                       temperature_kernel_sizes=list(settings["temperature_kernel_sizes"]),  # 两路温度TCN的时间核。
                       temperature_tcn_channels=list(settings["temperature_tcn_channels"]),  # 残差块通道列表。
                       freeze_image_backbone=settings["freeze_image_backbone"], freeze_backbone_batch_norm=settings["freeze_backbone_batch_norm"],
                       image_imagenet_normalization=settings["image_imagenet_normalization"])  # 恢复同样的图像预处理与BN策略。
        if settings["model_name"] == "resnet34_ms_tcn":  # 保持图像三路TCN基线参数。
            options.update(image_temporal_dim=settings["image_tcn_width"], image_kernel_sizes=list(settings["image_kernel_sizes"]),
                           image_tcn_channels=list(settings["image_tcn_channels"]))
        else:  # 保存单路ConvLSTM的空间图尺寸和输出宽度。
            options.update(image_temporal_dim=settings["image_convlstm_width"], image_convlstm_channels=settings["image_convlstm_channels"],
                           image_convlstm_grid_size=settings["image_convlstm_grid_size"])
    return options  # 同时保存到 model_config.json。


def train_main(model_class):  # 三种模型的训练入口共用这一流程。
    from config import get_settings, image_positions, validate_settings  # 运行目录决定具体模型配置。
    from dataset import WindowDataset, build_windows, fit_normalization, load_sequences, load_split, require_classes, save_window_report, write_csv, write_json  # 共享数据规则。
    from metrics import plot_history, save_predictions, future_state_report  # 保留原预测表并补充状态转换报告。
    run_started = time.perf_counter()  # 统计包含数据准备的总耗时。
    settings = get_settings()  # 读取本模型的配置。
    settings["format_version"] = 9  # 新训练记录v9规则，不改各模型已有超参数。
    settings["window_policy"] = "annotation_only_raw_temperature"  # 在训练快照明确记录799.9不作过滤。
    validate_settings(settings)  # 先检查可修改参数。
    validate_performance_settings(settings)  # 加速配置错误在读取墙数据前报告。
    require_result_workbook(settings["model_name"])  # 报表依赖缺失时在新训练启动前提示。
    legacy_shuffle = "wall" if settings.get("shuffle_train_walls", False) else "none"  # 保留其他模型原有选择。
    shuffle_unit = settings.get("train_shuffle_unit", legacy_shuffle)  # ResNet可单独选择按层重排。
    if shuffle_unit not in ("none", "wall", "layer"):  # 旧的同层号跨墙配置仅用于读取已训练结果。
        raise ValueError("训练请选择TRAIN_SHUFFLE_UNIT为layer、wall或none；同一批只使用一个墙层段。")
    settings["training_order"] = {"none": "wall_layer_time", "wall": "shuffled_walls_layer_time",
                                  "layer": "shuffled_layers_segment_time"}[shuffle_unit]  # 保存实际训练顺序。
    split_snapshot = load_split(settings)  # 继续使用原有按墙划分，不重新分墙。
    settings["training_layer_selection"] = split_snapshot["training_layer_selection"]  # 使用train_sequences.csv已经选定的监督层，完整冻结到本次配置。
    output = Path(settings["output_root"]) / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + settings["model_name"] + "_v9")  # v9保留799.9原值，状态只服从人工区间。
    output.mkdir(parents=True, exist_ok=False)  # 不覆盖任何已有训练。
    write_json(output / "config.json", settings)  # 保存本次配置。
    write_json(output / "split_snapshot.json", split_snapshot)  # 固定本次训练的墙归属。
    variant = "当前层局部窗口分类"  # 每个窗口独立计算。
    if settings["model_name"] == "resnet34_ms_tcn":  # 新模型明确显示本次确认的实际结构。
        variant = "ResNet34＋图像三路TCN＋温度两路TCN；局部双模态未来分类"  # 不把这个模型误称为原论文MSP-TCN。
    elif settings["model_name"] == "resnet34_convlstm":  # 运行日志区分窗口内ConvLSTM与跨层墙记忆。
        variant = "ResNet34＋图像单路ConvLSTM＋温度两路一维TCN；ConvLSTM状态限于本窗口"
    print("训练结果目录：{}\n模型：{}；变体：{}。".format(output, settings["model_name"], variant), flush=True)  # 明确研究模型变更。
    layer_selection = settings["training_layer_selection"]["layers_by_sequence"]  # 监督层只来自划分表的已确认结果。
    layer_scope = "训练、验证和测试均排除原始第1层" if settings.get("exclude_first_layer", False) else "验证和测试保留完整墙"  # 显示本次运行的实际层范围。
    print("训练监督层：train_sequences.csv列出{}面训练墙、{}个候选层；{}。".format(len(layer_selection), sum(len(item["train_layer_ids"]) for item in layer_selection.values()), layer_scope), flush=True)  # 候选层再按窗口规则处理。
    sequences, fingerprint = load_sequences(split_snapshot["records"], settings)  # 读取配对数据与监督标签。
    windows, skipped = build_windows(sequences, settings)  # 生成已知状态的有效窗口。
    report = save_window_report(output, windows, skipped, training_order=settings["training_order"])  # 区分静态清单与运行时墙序。
    startup_report = report["startup_exclusion"]  # 启动输入和监督是两个不同概念。
    print("启动策略：{} | 按标注排除unknown图温 {} 组 | 各层开头未确认 {} 张。".format(startup_report["startup_policy"], startup_report["unknown_image_temperature_groups_excluded"], startup_report["annotated_startup_frames"]), flush=True)  # 新版不再默认跳过50张。
    for split in ("train", "val", "test"):  # 显示不同集合规模。
        print("{}：{} 面墙，{} 个预测窗口，状态变化窗口 {} 个。".format(split, report[split]["sequences"], report[split]["windows"], report[split]["state_changes"]), flush=True)  # 区别墙数与窗口数。
    require_classes(windows, len(settings["class_names"]))  # 防止空集合或训练缺类。
    normalization = fit_normalization(sequences, windows["train"])  # 只使用训练观测拟合温度统计。
    write_json(output / "normalization.json", normalization)  # 后续评估及在线推理复用。
    write_json(output / "dataset_snapshot.json", {"fingerprint": fingerprint, "windows": windows})  # 保存可追溯输入清单。
    datasets = {split: WindowDataset(sequences, windows[split], settings, normalization) for split in ("train", "val")}  # 建立两个集合。
    options = model_options(settings, image_positions(settings))  # 按配置构建局部窗口模型。
    counts = np.bincount([item["label"] for item in windows["train"]], minlength=len(settings["class_names"]))  # 只统计训练未来类别。
    weights = (counts.sum() / (len(counts) * counts)).tolist()  # 保留原有逆频率损失权重。
    write_json(output / "model_config.json", {"model_name": settings["model_name"], "model_settings": options, "class_weights": weights})  # 保存结构和损失配置。
    device = setup_environment(settings)  # 选择运行设备。
    precision = PrecisionPolicy(settings, device)  # 一个缩放器贯穿所有训练轮次。
    loaders = {split: make_loader(dataset, settings, shuffle=(split == "train" and shuffle_unit != "none"),  # 只打乱训练。
                                  device=device, evaluation=(split == "val")) for split, dataset in datasets.items()}  # 验证使用固定墙序与评估批量。
    performance = dict(precision.description(), device=str(device), cpu_threads=settings["cpu_threads"], batch_size=settings["batch_size"],  # 记录实际资源配置。
                       gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,  # 便于对比不同服务器。
                       loaders={split: loader.performance for split, loader in loaders.items()})  # 记录真正启用的批处理与复用。
    write_json(output / "runtime_performance.json", performance)  # 不依赖训练是否已完整结束也能核对配置。
    loader_options = loaders["train"].performance  # 打印实际训练路径。
    print("运行方式：{} | 精度 {} | 图像编码复用 {} | 读取进程 {} | 锁页 {}".format(  # 启动时可核对A10配置。
        "同层窗口并行" if loader_options["local_batch_enabled"] else "按时间逐窗", performance["compute_dtype"],  # 未启用并行时逐窗计算。
        "开" if loader_options["reuse_image_features"] else "关", loader_options["num_workers"], "开" if loader_options["pin_memory"] else "关"), flush=True)  # 显示真正生效的值。
    if performance["amp_requested"] and not performance["amp_enabled"]:  # 避免用户以为已启用混合精度。
        print("AMP使用FP32：{}。".format(performance["amp_disabled_reason"]), flush=True)  # 解释CPU或旧环境回退原因。
    model = model_class(**options).to(device)  # 初始化本次模型。
    if settings["model_name"] == "resnet34_convlstm":  # 明确显示二维网格和窗内记忆。
        channels, grid = options["image_convlstm_channels"], options["image_convlstm_grid_size"]  # 当前隐藏图实际尺寸。
        parameters = sum(parameter.numel() for parameter in model.parameters())  # 统计模型的全部参数。
        print(f"结构：ResNet34 {options['image_feature_dim']}通道 | ConvLSTM {channels}×{grid}×{grid}（窗内记忆） | "
              f"图像/温度特征 {options['image_temporal_dim']}/{options['temperature_temporal_dim']}维 | "
              f"分类头 {options['classifier_hidden_dim']}维 | 参数 {parameters:,}", flush=True)  # 便于核对实验配置。
    else:  # 其他模型维持原有日志格式。
        print("结构：CNN {}维 | 图像时序 {}维 | 温度时序 {}维 | 分类头 {}维 | 参数 {:,}".format(options["image_feature_dim"], options.get("image_hidden_size", options.get("image_model_dim", options.get("image_temporal_dim"))), options.get("temperature_hidden_size", options.get("temperature_model_dim", options.get("temperature_temporal_dim"))), options["classifier_hidden_dim"], sum(p.numel() for p in model.parameters())), flush=True)  # 启动时显示实际尺寸。
    print("每窗{}张图＋{}个温度点 | 每批最多{}个连续窗口 | 本轮{}批 | 日志按时间刷新；每批更新参数。".format(options["image_steps"], options["temperature_steps"], settings["batch_size"], len(loaders["train"])), flush=True)  # 区分窗口、批次与日志。
    if "evaluation_batch_size" in settings:  # 旧模型沿用其单一批量设置。
        print("验证、测试和训练集重评每批最多{}个连续窗口。".format(settings["evaluation_batch_size"]), flush=True)  # 启动时可确认评估批量。
    criterion = make_criterion(weights, device)  # 按训练类别频数加权。
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])  # 不擅自修改训练超参数。
    best_validation = None  # 保存最佳轮次验证输出，结束后补图不再重复推理。
    best_transition_f1, best_transition_epoch, best_transition_loss = None, None, None  # 单独跟踪验证转变F1及同分用的损失。
    history, best_loss, best_epoch, stale, timing = [], float("inf"), 0, 0, {}  # 初始化训练与早停记录。
    train_sampler = loaders["train"].batch_sampler  # 每轮只更新训练采样器的墙顺序。
    for epoch in range(1, settings["epochs"] + 1):  # 最大轮数仍来自用户配置。
        train_sampler.set_epoch(epoch)  # 设置轮次后再开始读取和预取训练批次。
        if shuffle_unit == "layer":  # 墙可交错出现，按层记录实际次序。
            train_layer_order = train_sampler.layer_order()  # 与加载器使用同一个种子和轮次。
            first_wall, first_layer = train_layer_order[0]  # 显示该轮首层便于核对。
            last_wall, last_layer = train_layer_order[-1]  # 显示该轮末层。
            print("第{}轮训练层已打乱 | 首层 {} / {} | 末层 {} / {}。".format(
                epoch, first_wall, first_layer, last_wall, last_layer), flush=True)
        else:  # 固定墙序和墙重排都可以用墙列表完整表示。
            train_wall_order = train_sampler.wall_order()
        if shuffle_unit == "wall":  # 首墙和末墙在终端可直接核对。
            print("第{}轮训练墙已打乱 | 首墙 {} | 末墙 {}。".format(epoch, train_wall_order[0], train_wall_order[-1]), flush=True)  # 完整墙序另存历史。
        epoch_started = sync_clock(device)  # 记录整轮起点。
        training = train_one_epoch(model, loaders["train"], criterion, optimizer, device, settings["class_names"], epoch, settings["epochs"], timing, precision=precision)  # 使用同一精度策略训练全部选定窗口。
        validation = evaluate_loader(model, loaders["val"], criterion, device, settings["class_names"], precision=precision)  # 按固定顺序完成验证。
        elapsed = sync_clock(device) - epoch_started  # 包含训练和验证计算。
        timing["last_validation_seconds"] = validation["metrics"]["seconds"]  # 下一轮 ETA 复用已测验证耗时。
        row = {"epoch": epoch, "train_loss": training["loss"], "val_loss": validation["metrics"]["loss"],  # 未来分类主损失。
               "train_macro_f1": training["macro_f1"], "val_macro_f1": validation["metrics"]["macro_f1"], "val_accuracy": validation["metrics"]["accuracy"], "learning_rate": optimizer.param_groups[0]["lr"],  # 未来类别指标。
               "train_seconds": training["seconds"], "validation_seconds": validation["metrics"]["seconds"], "epoch_seconds": elapsed,  # 分阶段耗时。
               "cumulative_seconds": time.perf_counter() - run_started, "estimated_seconds_to_max_epochs": elapsed * (settings["epochs"] - epoch)}  # 假设不早停的预计时间。
        for prefix, stats in (("train", training), ("val", validation["metrics"])):  # 两个阶段分别记录效率。
            for name in ("windows_per_second", "nominal_image_frames", "encoded_image_frames", "image_encoding_reuse_fraction", "peak_gpu_allocated_gib", "peak_gpu_reserved_gib"):  # 留下实测吞吐和显存。
                row[prefix + "_" + name] = stats[name]  # 追加列不影响历史曲线使用的旧字段。
        if shuffle_unit == "layer":  # 同墙各层可以出现在本轮不同位置。
            row["train_layer_order"] = ";".join(f"{wall}/{layer}" for wall, layer in train_layer_order)
        else:  # 旧训练顺序和日志列保持兼容。
            row["train_wall_order"] = ";".join(train_wall_order)
        checkpoint_name = None  # 每轮候选权重的相对路径。
        if settings.get("save_epoch_checkpoints", False):  # 每轮评估后立即保留对应模型。
            checkpoint = output / "checkpoints" / f"epoch_{epoch:03d}.pt"  # 轮次固定对应一个文件。
            checkpoint.parent.mkdir(parents=True, exist_ok=True)  # 首轮创建目录。
            _save_weights(model, checkpoint)  # 可选保留每轮的独立权重。
            checkpoint_name = checkpoint.relative_to(output).as_posix()  # 支持上传后按相同相对路径读取。
        if settings.get("layer_event_reports", False):  # 用本轮已经算出的验证结果统计，不额外前向。
            epoch_dir = output / "validation_by_epoch" / f"epoch_{epoch:03d}"  # 独立记录各轮。
            replay = {**validation, "model_epoch": epoch, "checkpoint": checkpoint_name}  # 关联权重和报告。
            event = save_layer_event_report(epoch_dir, datasets["val"], replay, settings["class_names"], "val")
            save_predictions(epoch_dir / "predictions.csv", datasets["val"], replay, settings["class_names"])  # 保留阈值复算所需分数。
            transition_f1 = event["transition_f1"]  # 不适用或无真实转变时为空。
            row["val_transition_f1"] = transition_f1  # 每轮保存稳定转熔断F1，供复核选模。
            if transition_f1 is not None and (  # 先比转变F1，同分再比较整体损失。
                best_transition_f1 is None or (transition_f1, -row["val_loss"]) > (best_transition_f1, -best_transition_loss)
            ):
                _save_weights(model, output / TRANSITION_F1_CHECKPOINT)  # 只覆盖转变F1最优权重。
                best_transition_f1, best_transition_epoch, best_transition_loss = transition_f1, epoch, row["val_loss"]  # 记录权重对应轮次。
            if event["status"] == "ok":  # 一帧任务写入本轮首发覆盖、命中和误报。
                for key in ("first_events", "evaluable_first_events", "first_hits", "first_recall",
                            "transition_windows", "transition_hits", "negative_windows", "false_positive_windows", "false_positive_rate"):
                    row["val_" + key] = event[key]
                print("           层首发 {}/{}（无法评估 {}） | 稳定→熔断 {}/{} | 一帧错报 {}/{}".format(
                    event["first_hits"], event["evaluable_first_events"], event["unevaluable_first_events"],
                    event["transition_hits"], event["transition_windows"],
                    event["false_positive_windows"], event["negative_windows"]), flush=True)
        history.append(row)  # 保留所有完成轮次。
        write_csv(output / "history.csv", history, list(row))  # 每轮及时保存。
        improved = row["val_loss"] < best_loss - 1e-8  # 先记录本轮是否刷新最佳。
        if improved:  # 早停和整体最佳权重只看未来分类验证损失。
            best_validation = validation  # 保留最佳轮次预测，用于训练结束的验证报告。
            best_loss, best_epoch, stale = row["val_loss"], epoch, 0  # 更新最佳轮次。
            _save_weights(model, output / "best_model.pt")  # 覆盖保存整体损失最低的权重。
            write_json(output / "best_validation_metrics.json", validation["metrics"])  # 保存对应指标。
            save_predictions(output / "val_predictions.csv", datasets["val"], validation, settings["class_names"])  # 保存对应验证预测。
        else:  # 主任务验证损失没有改善。
            stale += 1  # 累计早停等待轮次。
        print_epoch_summary(epoch, settings["epochs"], row, best_epoch, stale, settings["patience"], improved)  # 每轮保留一条清晰汇总。
        print("           窗口/秒：训练 {:.2f} | 验证 {:.2f} | 局部图像少编码 {:.1%} | 训练显存峰值 {:.2f} GiB".format(  # 每轮一行效率摘要。
            training["windows_per_second"], validation["metrics"]["windows_per_second"], training["image_encoding_reuse_fraction"], training["peak_gpu_allocated_gib"]), flush=True)  # 峰值不包含其他训练进程。
        if stale >= settings["patience"]:  # 保留用户指定的早停规则。
            print("提前停止：已完成{}轮；最佳第{}轮；连续{}轮未来验证损失未刷新。".format(epoch, best_epoch, stale), flush=True)  # 说明停止原因。
            break  # 结束训练轮次。
    if best_validation is not None:  # 只输出最佳验证模型对应的报告。
        best_validation["model_epoch"] = best_epoch  # 图表和数据明确对应哪个最佳轮次。
        save_evaluation_reports(output / "validation", datasets["val"], best_validation, settings["class_names"], "val")  # 不读取测试预测。
        write_json(output / "validation" / "future_state_report.json", future_state_report(datasets["val"], best_validation, settings["class_names"]))  # 保存总体变化诊断。
    load_weights(model, output / "best_model.pt", device)  # 训练集最终报告统一使用最佳模型，不能使用最后一轮或训练中混合权重的预测。
    if "evaluation_batch_size" in settings or shuffle_unit != "none":  # 重评采用固定墙序和可配置的评估批量。
        loaders.clear()  # 释放优化和验证阶段的数据加载进程。
        loaders["train"] = make_loader(datasets["train"], settings, device=device, evaluation=True)  # 用相同训练窗口创建前向加载器。
        performance["loaders"]["train_reevaluation"] = loaders["train"].performance  # 在结果汇总中记录重评的实际批量。
        write_json(output / "runtime_performance.json", performance)  # 运行性能文件也保留重评配置。
    training_evaluation = evaluate_loader(model, loaders["train"], criterion, device, settings["class_names"], phase="训练集重评", precision=precision)  # 内部使用eval和no_grad，不反向传播或更新统计量。
    training_evaluation["model_epoch"] = best_epoch  # 与验证报告对应同一份best_model.pt。
    save_evaluation_reports(output, datasets["train"], training_evaluation, settings["class_names"], "train")  # 两张训练混淆矩阵直接放在本次结果根目录。
    plot_history(output / "training_curves.png", history)  # 绘制主损失和未来 F1。
    summary = {"format_version": 9, "model_name": settings["model_name"], "completed": True, "best_epoch": best_epoch, "best_validation_loss": best_loss,  # 保存v9人工标签与温度原值策略。
               "epochs_completed": len(history), "parameters": sum(item.numel() for item in model.parameters()), "torch_version": str(torch.__version__), "device": str(device),  # 模型及环境。
               "task": "future_state_classification", "training_order": settings["training_order"], "history_scope": "current_layer_only",  # 说明实际观测范围。
               "model_variant": "local_window_classifier",  # 与原纯基线区分。
               "future_frames": settings["future_frames"], "startup_policy": settings.get("startup_policy", "exclude"), "layer_start_rules": startup_report["per_wall_layer"],  # 保存实际标注起点，不要求用户填写张数。
               "total_seconds": time.perf_counter() - run_started, "training_seconds": sum(row["train_seconds"] for row in history), "validation_seconds": sum(row["validation_seconds"] for row in history),  # 总耗时。
               "training_evaluation_seconds": training_evaluation["metrics"]["seconds"], "report_format_version": REPORT_FORMAT_VERSION, "report_scopes": ["all_windows", "breakoff_layers"],  # 区分额外训练集重评耗时与正常优化时间。
               "final_training_accuracy": training_evaluation["metrics"]["accuracy"], "final_training_macro_f1": training_evaluation["metrics"]["macro_f1"],  # 固定最佳权重下的训练表现，不混用每轮过程指标。
               "runtime_performance": performance,  # 汇总与训练启动时记录的实际执行选项一致。
               "stopping_reason": "patience" if stale >= settings["patience"] else "max_epochs"}  # 记录停止原因。
    if settings.get("layer_event_reports", False):  # 记录可供选模的验证结果入口。
        summary["validation_events_by_epoch"] = "validation_by_epoch"
        summary["best_transition_f1"] = best_transition_f1  # 单独记录预警指标的最优分数。
        summary["best_transition_f1_epoch"] = best_transition_epoch  # 无可评估转变时留空。
        summary["best_transition_f1_validation_loss"] = best_transition_loss  # 保留同分时用于比较的损失。
    if settings.get("save_epoch_checkpoints", False):  # 指向实际保留的候选权重目录。
        summary["epoch_checkpoints"] = "checkpoints"
    write_json(output / "training_summary.json", summary)  # 完整结束后才能自动选择此结果。
    print("训练完成，最佳轮次 {}，总耗时 {}。下一步在 IDE 运行 evaluate.py。".format(best_epoch, duration(summary["total_seconds"])), flush=True)  # 提示下一步。


def choose_run(settings):  # 自动选择结构兼容的最近一次完整结果。
    from data_utils import read_json

    specified = settings["evaluation_run_dir"].strip()
    candidates = [Path(specified).expanduser().resolve()] if specified else [
        item.parent for item in Path(settings["output_root"]).glob("*/training_summary.json")
    ]
    accepted = []
    retired = False
    for path in candidates:
        summary_path = path / "training_summary.json"
        if not summary_path.is_file():
            continue
        summary = read_json(summary_path)
        if (summary.get("format_version") not in (5, 6, 7, 8, 9)
                or summary.get("model_name") != settings["model_name"]
                or not summary.get("completed") or not (path / "best_model.pt").is_file()):
            continue
        saved_settings = read_json(path / "config.json")
        options = read_json(path / "model_config.json")["model_settings"]
        if saved_settings.get("memory_enabled", False) or options.get("memory_enabled", False):
            retired = True  # 历史架构对应不同分类头，继续寻找可恢复的局部模型。
            continue
        accepted.append(path)
    if accepted:
        return max(accepted, key=lambda item: item.name)
    if retired:
        raise ValueError("选中的旧实验含墙内记忆。请在 IDE 用当前窗口模型重新训练。")
    raise ValueError("找不到本模型的完整训练结果，请先运行 train.py 或检查 EVALUATION_RUN_DIR。")


def evaluate_main(model_class):  # 按配置分别生成训练、验证、测试报告，不重新划分数据或重训。
    from config import get_settings, validate_settings  # 当前配置只选择结果、设备和显示方式。
    from dataset import WindowDataset, load_sequences, read_json, validate_membership, write_json  # 恢复本次训练的固定窗口。
    from metrics import future_state_report  # 保留原状态变化分析。
    current = get_settings()  # 读取用户选择。
    output = choose_run(current)  # v5/v6/v7/v8/v9均按各自保存的结构和启动策略恢复。
    settings = read_json(output / "config.json")  # 不用新默认值覆盖旧窗口或旧结构。
    for name in ("device", "batch_size", "evaluation_batch_size", "num_workers", "cpu_threads", "progress_mode", "progress_refresh_seconds", "plain_log_interval_seconds") + PERFORMANCE_KEYS:  # 只允许覆盖运行资源和显示。
        if name in current:  # 旧配置也能正常评估。
            settings[name] = current[name]  # 不改训练结构、标签和启动排除。
    validate_settings(settings)  # 检查恢复后的配置。
    validate_performance_settings(settings)  # 运行资源只由新的设置控制。
    split = read_json(output / "split_snapshot.json")  # 固定墙的集合归属。
    validate_membership(split["records"])  # 同组不能跨集合。
    snapshot = read_json(output / "dataset_snapshot.json")  # 使用训练时生成的监督窗口。
    normalization = read_json(output / "normalization.json")  # 不重新拟合验证或测试温度。
    options = read_json(output / "model_config.json")  # 必须按保存的特征宽度构造模型。
    sequences, fingerprint = load_sequences(split["records"], settings)  # 核对原始配对数据和标签。
    if fingerprint != snapshot["fingerprint"]:  # 不允许悄悄使用改变后的数据。
        raise ValueError("本次训练保存的数据指纹与当前配对、图像清单或标签不一致。")  # 继续保持原保护。
    selector = current.get("evaluation_split", "all")  # all生成训练矩阵和验证、测试报告。
    splits = {"all": ("train", "val", "test"), "both": ("val", "test"), "train": ("train",), "val": ("val",), "test": ("test",)}[selector]  # both保留旧配置兼容。
    if selector != "train":  # 仅重画训练混淆矩阵时无需 Excel 依赖。
        require_result_workbook(settings["model_name"])  # 评估推理之前检查报表依赖。
    chosen_epoch = current.get("evaluation_epoch")  # 可选择最低损失、最佳转变F1或指定轮次。
    if chosen_epoch is None:  # 默认保持验证损失最低的选模口径。
        checkpoint = output / "best_model.pt"
        report_root = output  # 既有默认报告位置。
        model_epoch = read_json(output / "training_summary.json").get("best_epoch")
    elif chosen_epoch == "transition_f1":  # 用验证转变F1最佳权重生成独立报告。
        checkpoint = output / TRANSITION_F1_CHECKPOINT  # 与训练使用同一个文件名。
        model_epoch = read_json(output / "training_summary.json").get("best_transition_f1_epoch")  # 读取真实保存轮次。
        if model_epoch is None or not checkpoint.is_file():  # 旧结果或缺少转变时明确提示。
            raise FileNotFoundError("本次训练没有保存稳定转熔断F1最佳权重：" + str(checkpoint))
        report_root = output / "transition_f1_evaluation"  # 与最低损失模型的报告分开保存。
    else:  # 指定轮次时将报告分开保存，保留最佳验证损失报告。
        if type(chosen_epoch) is not int or chosen_epoch < 1:  # 接受正整数轮次。
            raise ValueError('EVALUATION_EPOCH使用None、"transition_f1"或正整数轮次。')
        checkpoint = output / "checkpoints" / f"epoch_{chosen_epoch:03d}.pt"  # 指定轮次对应的冻结权重。
        if not checkpoint.is_file():  # 旧训练可能没有该轮候选。
            raise FileNotFoundError("没有保存该轮权重：" + str(checkpoint))
        report_root = output / "epoch_evaluations" / f"epoch_{chosen_epoch:03d}"
        model_epoch = chosen_epoch
    device = setup_environment(settings)  # 只设置一次运行环境。
    precision = PrecisionPolicy(settings, device)  # 独立评估只使用前向自动混合精度。
    for split_name in splits:  # 不将验证和测试预测混合。
        dataset = WindowDataset(sequences, snapshot["windows"][split_name], settings, normalization)  # 不重建或重新筛选监督窗口。
        model = model_class(**local_model_options(options["model_settings"])).to(device)  # 旧v5保持原尺寸，新v6恢复扩宽尺寸。
        load_weights(model, checkpoint, device)  # 载入指定模型权重。
        title = {"train": "训练集重评", "val": "验证", "test": "测试"}[split_name]  # 不能把训练集标成测试集。
        loader = make_loader(dataset, settings, device=device, evaluation=True)  # 评估使用保存的窗口范围及当前评估批量。
        result = evaluate_loader(model, loader, make_criterion(options["class_weights"], device), device, settings["class_names"], phase=title, precision=precision)  # 先完整预测墙。
        result["model_epoch"] = model_epoch  # 三个集合统一采用选定的同一轮。
        result["checkpoint"] = checkpoint.relative_to(output).as_posix()  # 报告区分最佳损失权重和候选权重。
        destination = report_root / {"train": ".", "val": "validation", "test": "evaluation"}[split_name]  # 训练图放在对应权重的报告根目录。
        save_evaluation_reports(destination, dataset, result, settings["class_names"], split_name)  # 训练输出两图，验证和测试输出各条件子集。
        if split_name != "train":  # 独立评估时只为验证与测试保存附加报告。
            write_json(destination / "future_state_report.json", future_state_report(dataset, result, settings["class_names"]))  # 补充总体状态变化统计。
            write_json(destination / "runtime_performance.json", {**precision.description(), "loader": loader.performance,  # 保存本次独立评估的实际执行方式。
                       "metrics": {name: result["metrics"][name] for name in ("seconds", "windows_per_second", "nominal_image_frames", "encoded_image_frames", "image_encoding_reuse_fraction", "peak_gpu_allocated_gib", "peak_gpu_reserved_gib")}})  # 与训练效率统计可对照。
        print("{}完成 | 准确率{:.2%} | Macro-F1 {:.4f} | 推理耗时{} | 报告：{}".format(title, result["metrics"]["accuracy"], result["metrics"]["macro_f1"], duration(result["metrics"]["seconds"]), destination), flush=True)  # 输出一次集合总结。


class WallFutureStatePredictor:  # 在线每来一张图和对应温度调用一次，不读取标签。
    def __init__(self, model_class, current_settings=None):  # 由各模型薄入口传入具体网络。
        from data_utils import read_json  # 仅读取训练元数据。
        from wall_dataset import image_positions  # 按训练快照恢复相同的选图规则。
        if current_settings is None:  # 旧入口仍可从当前运行目录读取配置。
            from config import get_settings
            current_settings = get_settings()
        current = current_settings
        self.run_dir = choose_run(current)  # 加载匹配且完整的v5/v6/v7/v8/v9结果，并按保存宽度恢复。
        self.settings = read_json(self.run_dir / "config.json")  # 恢复所有输入规则。
        for name in ("device", "cpu_threads"):  # 仅覆盖计算资源设置。
            self.settings[name] = current[name]  # 保持训练时的窗口设置。
        self.normalization = read_json(self.run_dir / "normalization.json")  # 复用训练温度统计。
        options = read_json(self.run_dir / "model_config.json")  # 读取完整模型结构。
        self.device = setup_environment(self.settings)  # 选择设备。
        self.model = model_class(**local_model_options(options["model_settings"])).to(self.device)  # 创建相同模型。
        load_weights(self.model, self.run_dir / "best_model.pt", self.device)  # 加载参数。
        self.model.eval()  # 在线关闭随机失活。
        self.positions = image_positions(self.settings)  # 恢复局部选图策略。
        self.samples_per_image = self.settings["temperature_samples_per_image"]  # 保留每图两个原始温度的接口。
        self.images = deque(maxlen=self.settings["window_size"])  # 局部图像不跨层。
        self.temperatures = deque(maxlen=self.settings["window_size"] * self.samples_per_image)  # 局部温度不跨层。
        self.reset()  # 初始墙状态为空。

    def reset(self, sequence_id=None):  # 每开始另一面墙必须清空局部窗口。
        self.images.clear()  # 清空局部图片。
        self.temperatures.clear()  # 清空局部温度。
        self.sequence_id = None if sequence_id is None else str(sequence_id)  # 可绑定本面墙标识。
        self.last_index, self.last_layer, self.last_segment = None, None, None  # 清空到达记录。
        self.last_timestamp = None  # 最近收到的图像采集时间。
        self.layer_position, self.segment_start_index = -1, None  # 每层启动帧计数与局部输出网格。
        self.processing_start_index = None  # 新版每层第一次收到允许预测信号时记录当前帧号。
        self.status = "startup"  # 调用方可区分暂未输出的原因。

    @torch.no_grad()  # 在线只更新过程状态，不计算梯度。
    def update(self, image, temperature, seq_idx, segment_id="0", layer_id="", temperature_available=True, timestamp_s=None, sequence_id=None, layer_frame_index=None, input_available=True, known_layer_end_index=None, processing_ready=None):  # 新版实时接口用阶段信号，回放按当前unknown标注给出该信号。
        from wall_dataset import preprocess_image  # 与训练采用相同图像预处理。
        if sequence_id is not None and str(sequence_id) != self.sequence_id:  # 明确换墙时自动清空状态。
            self.reset(sequence_id)  # 同一模型参数继续用于新墙。
        if type(seq_idx) is not int or seq_idx < 0:  # 保持原帧号接口。
            raise ValueError("seq_idx 必须是从 0 开始且在同一面墙内递增的整数。")  # 不凭空重编号。
        if self.last_index is not None and seq_idx <= self.last_index:  # 重复帧不能计入局部窗口。
            raise ValueError("同墙帧号必须递增；换墙请调用 reset 或传入新的 sequence_id。")  # 防止隐式混墙。
        timestamp = float(timestamp_s) if timestamp_s is not None else seq_idx / self.settings["fps"]  # 缺少实时时间时按帧率显示名义时间。
        if not np.isfinite(timestamp) or (self.last_timestamp is not None and timestamp < self.last_timestamp):  # 真实时间不能倒退。
            raise ValueError("timestamp_s 必须有限且沿同一面墙时间不倒退。")  # 要求调用方检查记录。
        layer, segment = str(layer_id), str(segment_id)  # 保留原始层和连续段编号。
        changed_layer = self.last_layer is None or layer != self.last_layer  # 开始新层时重新计数启动帧。
        if changed_layer:  # 新层重新等待启动结束信号。
            self.processing_start_index = None  # 同层连续段变化不能把已开始加工误当新启动。
        discontinuity = self.last_index is not None and (seq_idx != self.last_index + 1 or segment != self.last_segment or changed_layer)  # 局部输入不跨断点。
        if changed_layer or discontinuity:  # 换层和断点都会清空局部窗口。
            self.images.clear()  # 不把上一层图片带进局部窗口。
            self.temperatures.clear()  # 温度局部历史同步清空。
            self.segment_start_index = None  # 新段重新确定输出网格。
        self.layer_position = 0 if changed_layer else self.layer_position + 1  # 连续段切换不重置层内计数。
        if layer_frame_index is not None:  # 离线回放可直接使用原清单中的层内位置。
            if int(layer_frame_index) < self.layer_position:  # 层内位置不能倒退。
                raise ValueError("layer_frame_index 与当前层到达次序不一致。")  # 防止启动帧重复计数。
            self.layer_position = int(layer_frame_index)  # 兼容调用方明确记录的丢帧位置。
        self.last_index, self.last_layer, self.last_segment, self.last_timestamp = seq_idx, layer, segment, timestamp  # 所有到达帧都推进真实位置。
        from data_utils import resolve_layer_start  # 与离线取窗读取完全相同的逐墙逐层规则。
        annotation_start = uses_annotation_startup(self.settings)  # 新版不再读取50张或特殊层表。
        if annotation_start and processing_ready is not None and type(processing_ready) is not bool:  # 当前阶段只能是明确布尔值或未提供。
            raise ValueError("processing_ready应为True、False或None，不能传入真实状态类别。")  # 不将stable等标签直接传入模型。
        if annotation_start and processing_ready is True and self.processing_start_index is None:  # 首个已知当前标注只作为阶段事件。
            self.processing_start_index = seq_idx  # 此时只处理已收到的当前帧。
        if annotation_start and (processing_ready is False or self.processing_start_index is None):  # 在读图和温度标准化之前排除unknown。
            self.images.clear()  # 不允许unknown保留在局部历史中。
            self.temperatures.clear()  # 对应两点温度同时排除，不能与下一张图重新顺配。
            self.segment_start_index = None  # 下一张已知有效图片开始积累新的连续窗口。
            self.status = "unknown_excluded" if processing_ready is False else "awaiting_annotation_signal"  # 没有标注或外部信号时不能自行猜测启动结束。
            return None  # 等待有效输入。
        if not annotation_start and self.settings.get("layer_start_overrides") and self.sequence_id is None:  # 旧规则继续兼容。
            raise ValueError("存在逐墙逐层启动规则，请先 reset(墙文件夹名) 或传入 sequence_id。")  # 避免在线漏用人工指定的起点。
        startup, anchor = (0, "") if annotation_start else resolve_layer_start(self.settings, self.sequence_id or "", layer)  # 按标注模式无固定张数。
        if self.layer_position == startup and anchor and isinstance(image, (str, Path)) and Path(image).name != anchor:  # 回放时额外核对首图文件名。
            raise ValueError("{} 第{}层首张有效图应为{}，实际为{}。".format(self.sequence_id, layer, anchor, Path(image).name))  # 不把计数偏移带入模型。
        context_only = uses_startup_context(self.settings)  # 旧权重继续使用旧的输入排除方式。
        if self.layer_position < startup and not context_only:  # exclude模式仍然不读取启动输入。
            self.status = "startup"  # 继续等待有效历史。
            return None  # 等待启动排除段结束。
        if self.segment_start_index is None:  # 包括本段第一帧温度不可用的情况。
            self.segment_start_index = seq_idx  # 输出步长网格不因后续无效温度重新偏移。
        values = np.asarray(temperature, dtype=np.float64).reshape(-1)  # 保留全部温度点。
        if len(values) != self.samples_per_image:  # 不复制或平均温度补齐。
            raise ValueError("每张图需要 {} 个按时间排序的温度值。".format(self.samples_per_image))  # 与训练一致。
        if (self.settings.get("temperature_input_policy", "legacy_filtered") != "raw" and not temperature_available) or not input_available:  # raw策略只服从人工输入掩码，旧模型才使用温度可用性过滤。
            self.images.clear()  # 后续需重新积累连续有效的局部历史。
            self.temperatures.clear()  # 不把无效点填零。
            if annotation_start:  # 新版需要和离线有效输入段重新起窗保持一致。
                self.segment_start_index = None  # 坏图温两侧不拼接窗口。
            self.status = "unavailable"  # 调用方可以显示数据不可用。
            return None  # 下一张有效帧重新累计历史。
        if not np.isfinite(values).all():  # 可用温度必须为有限值。
            raise ValueError("温度非有限，请将 temperature_available 标为 False。")  # 不静默插值。
        if isinstance(image, (str, Path)):  # 支持磁盘回放。
            with Image.open(image) as opened:  # 图片读取后立即关闭。
                array = preprocess_image(opened, self.settings["image_size"])  # 与训练同样灰度化和缩放。
        else:  # 支持相机输出。
            if not isinstance(image, Image.Image):  # NumPy 图像必须保留原像素格式。
                raw = np.asarray(image)  # 整理为数组。
                if raw.dtype != np.uint8:  # 防止把 0 到 1 数组当作原图。
                    raise ValueError("在线图像使用 uint8 灰度或 RGB 数组。")  # 明确采集接口。
                image = Image.fromarray(raw)  # 转成统一图像对象。
            array = preprocess_image(image, self.settings["image_size"])  # 使用训练预处理。
        if min(array.shape) < 16 or (self.images and array.shape != self.images[-1].shape):  # 局部图像必须可堆叠。
            raise ValueError("图像尺寸过小或同层尺寸不一致，请统一原图或配置 IMAGE_SIZE。")  # 不隐藏尺寸变化。
        self.images.append(array)  # 缓存当前图像。
        self.temperatures.extend(values.tolist())  # 保留本层连续温度原值。
        length = self.settings["window_size"]  # 局部观察帧数。
        if context_only and self.layer_position < startup:  # 旧启动上下文模式先缓存图温，再等到可预测时刻。
            self.status = "startup_context"  # 表示图温已接收，但不将主动启动过程报告为三分类结果。
            return None  # 只由已知帧数决定，不偷看当前或未来真实标签。
        if known_layer_end_index is not None and seq_idx + self.settings["future_frames"] > int(known_layer_end_index):  # 控制器已知终点时不假设层外继续加工。
            self.status = "unavailable"  # 当前未来目标不属于本层。
            return None  # 不等待未来数据。
        if len(self.images) < length:  # 局部历史还未凑满。
            self.status = "warming_up"  # 区分启动排除和正常历史积累。
            return None  # 等待完整局部窗口。
        post_start = self.segment_start_index if annotation_start else max(self.segment_start_index, seq_idx + startup - self.layer_position)  # 新版从每个连续有效输入段的首张图积累完整窗口。
        first_end = first_window_end(self.segment_start_index, post_start, length, context_only)  # 启动上下文模式不再多等一个完整窗口。
        if (seq_idx - first_end) % self.settings["window_stride"]:  # 与离线局部监督网格一致。
            self.status = "waiting_stride"  # 当前窗口未到预测步长。
            return None  # 等待下一预测时刻。
        images = np.stack([self.images[position] for position in self.positions]).astype(np.float32) / 255.0  # 从局部历史选图。
        temperatures = ((np.asarray(self.temperatures, dtype=np.float64) - self.normalization["mean"]) / self.normalization["std"]).astype(np.float32)  # 与训练同样展开温度。
        logits = self.model(torch.from_numpy(images[None, :, None]).to(self.device), torch.from_numpy(temperatures[None, None]).to(self.device))  # 只用当前窗口预测未来状态。
        scores = torch.softmax(logits, dim=1)[0].cpu().numpy()  # 取得未来各状态概率。
        self.status = "ready"  # 已生成当前时刻的有效预测。
        target = seq_idx + self.settings["future_frames"]  # 不等待未来帧出现。
        result = {"layer_id": layer, "segment_id": segment, "end_seq_idx": seq_idx, "target_seq_idx": target, "end_time_s": timestamp,  # 保存真实当前时间。
                  "target_time_s": timestamp + self.settings["future_frames"] / self.settings["fps"], "predicted_class": self.settings["class_names"][int(scores.argmax())], "status": self.status,  # 未来时间是设定帧率下的名义目标。
                  "target_assumption": "current_layer_continues"}  # 未知层终点时，预测以当前层继续加工为前提。
        result.update({"score_" + name: float(scores[index]) for index, name in enumerate(self.settings["class_names"])})  # 保存未来概率。
        return result  # 返回采集程序或回放入口。


def predict_main(predictor_class, current=None):  # 用当前人工阶段信号回放，不输入真实状态。
    from data_utils import read_json, read_paired_frames, write_csv  # 读取配对并写出逐帧结果。
    if current is None:  # 保持旧模型入口从运行目录读取配置。
        from config import get_settings
        current = get_settings()
    predictor = predictor_class(current)  # 加载一次模型。
    if current["predict_sequence_dir"].strip():  # 用户选择待预测墙。
        folder = Path(current["predict_sequence_dir"]).expanduser().resolve()  # 使用指定地址。
    else:  # 默认回放固定测试集第一面墙。
        split = read_json(predictor.run_dir / "split_snapshot.json")  # 清单只用于找到文件夹。
        record = next(item for item in split["records"] if item["split"] == "test")  # 不读取任何标签内容。
        folder = Path(predictor.settings["data_root"]) / record["relative_folder"]  # 定位配对数据。
    paired, _ = read_paired_frames(folder, use_annotation_input_mask=uses_startup_context(predictor.settings) or uses_annotation_startup(predictor.settings), exclude_unknown_inputs=uses_annotation_startup(predictor.settings), temperature_input_policy=predictor.settings.get("temperature_input_policy", "legacy_filtered"))  # 与训练同时排除unknown图片及对应温度。
    annotation_start = uses_annotation_startup(predictor.settings)  # 旧模型按其保存的固定规则回放。
    if annotation_start:  # 这里只将当前帧是否已知转换成阶段布尔值，不提供实际类别。
        from data_utils import read_labels  # 复用同步与标签版本校验，不看未来标签来决定当前输出。
        phase_labels, _, _ = read_labels(folder, len(paired["paths"]), predictor.settings["class_names"], annotation_only=predictor.settings.get("temperature_input_policy") == "raw")  # 需要用户已完成unknown区间标注。
    rows, statuses = [], {}  # 保存有效预测和各类暂未输出原因。
    for index, path in enumerate(paired["paths"]):  # 每帧只调用一次状态推进。
        result = predictor.update(path, paired["values"][index], index, paired["segments"][index], paired["layer_ids"][index], paired["temperature_available"][index],  # 输入当前观测。
                                  timestamp_s=paired["timestamps_s"][index] if paired.get("has_recorded_timestamps", False) else None, sequence_id=folder.name, layer_frame_index=paired["layer_frame_positions"][index], input_available=paired["input_available"][index], processing_ready=bool(phase_labels[index] >= 0) if annotation_start else None)  # 只使用当前已知/未知阶段，不把三类真值作为特征。
        statuses[predictor.status] = statuses.get(predictor.status, 0) + 1  # 汇总启动、积累、等待和不可用帧。
        if result is not None:  # 模型先按当前观测完成预测，之后只核对离线清单边界。
            target = result["target_seq_idx"]  # 不读取未来图片、温度或标签。
            if target < len(paired["paths"]) and paired["layer_ids"][target] == paired["layer_ids"][index] and paired["segments"][target] == paired["segments"][index] and (not annotation_start or bool(np.asarray(paired["input_available"][index:target + 1]).all())):  # 预测完成后才事后核对目标位于同一有效输入段，不将未来标注输入网络。
                rows.append({"sequence_id": folder.name, **result})  # 仅保存当前层存在对应目标的回放记录。
        if (index + 1) % 500 == 0 or index + 1 == len(paired["paths"]):  # 长回放显示进度。
            print("回放 {}/{} 帧，输出 {} 次，当前状态 {}。".format(index + 1, len(paired["paths"]), len(rows), predictor.status), flush=True)  # 状态区分不输出原因。
    if not rows:  # 不把空结果误报为预测成功。
        raise ValueError("没有有效预测；各状态帧数：{}。请检查层启动排除帧数及局部历史长度。".format(statuses))  # 给出可核查统计。
    destination = predictor.run_dir / "predictions" / (folder.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".csv")  # 每次保存独立结果。
    write_csv(destination, rows, list(rows[0]))  # 保留未来类别、概率与真实当前时间。
    boundary = "换层重新累计当前层历史"  # 回放只使用当前层窗口。
    print("回放完成：{}\n各状态帧数：{}。新墙必须 reset 或传新的 sequence_id；{}。".format(destination, statuses, boundary), flush=True)  # 说明在线接口边界。
