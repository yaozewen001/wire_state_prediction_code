# -*- coding: utf-8 -*-
"""公共训练加速：只改变执行方式，不改变窗口、标签、网络参数和监督范围。"""

from contextlib import nullcontext  # CPU或关闭AMP时提供普通精度上下文。
import torch  # 使用DataLoader可识别的张量容器与CUDA混合精度。


PERFORMANCE_KEYS = (  # 独立评估只允许覆盖这些执行参数，不覆盖已保存的数据规则。
    "amp_enabled", "amp_dtype", "local_batch_enabled", "reuse_image_features",  # 精度和无记忆批处理。
    "pin_memory", "persistent_workers", "prefetch_factor", "non_blocking_transfer",  # 数据加载和传输。
)  # 未含窗口、提前帧数、选层、标签或网络通道配置。


def validate_performance_settings(settings):  # 兼容未保存新键的历史配置。
    for name in PERFORMANCE_KEYS:  # 检查全部新增开关。
        if name not in settings:  # 历史训练配置保持原有执行方式。
            continue  # 不把新的默认值强行写入旧快照。
        value = settings[name]  # 读取具体配置值。
        if name == "amp_dtype":  # auto优先使用CUDA原生支持的BF16，否则FP16。
            if value not in ("auto", "float16", "bfloat16"):  # 不接受任意字符串。
                raise ValueError("AMP_DTYPE使用auto、float16或bfloat16。")  # 给出可直接修改的选项。
        elif name == "prefetch_factor":  # 每个读取进程提前准备的批次数量。
            if type(value) is not int or value < 1:  # DataLoader要求正整数。
                raise ValueError("PREFETCH_FACTOR必须是正整数；NUM_WORKERS=0时不使用。")  # 主进程模式不传此参数。
        elif type(value) is not bool:  # 其余参数均为布尔开关。
            raise ValueError(name.upper() + "必须是True或False。")  # 避免字符串False被视为开启。


def execution_policy(settings, deterministic_images=True):  # 保证批处理和编码复用符合视觉骨干的统计方式。
    requested = bool(settings.get("local_batch_enabled", False))
    model = settings.get("model_name")
    fixed_bn = bool(settings.get("freeze_image_backbone", False) or settings.get("freeze_backbone_batch_norm", False))
    standard_resnet = model in ("resnet34_ms_tcn", "resnet34_convlstm")
    power_resnet = model in ("resnet34_ms_tcn_temp_power", "resnet34_lstm_temp_power", "resnet34_convlstm_temp_power")
    power_cnn = model in ("cnn_lstm_temp_power", "cnn_tcn_temp_power")  # 灰度 CNN 没有批归一化或随机层。
    enabled = requested and (power_cnn or power_resnet or (standard_resnet and fixed_bn))
    reuse = enabled and (power_cnn or fixed_bn) and deterministic_images and bool(settings.get("reuse_image_features", False))
    reason = ("灰度 CNN 局部窗口批处理" if power_cnn else "ResNet34 局部窗口批处理") if enabled else "按窗口分别计算"
    return {"local_batch_enabled": enabled, "reuse_image_features": reuse, "reason": reason}


class WindowBatchCollator:  # 顶层可序列化类，兼容Windows的spawn读取进程。
    def __init__(self, windows, reuse_images):  # 只保留索引元数据，不缓存网络特征或参数。
        self.windows, self.reuse_images = windows, bool(reuse_images)  # 每次调用只整理当前一个优化器批次。

    def __call__(self, samples):  # 在DataLoader读取进程内构造最终CPU张量，随后可统一锁页。
        indices = [int(sample[3]) for sample in samples]  # 保留报告所需的数据集索引顺序。
        windows = [self.windows[index] for index in indices]  # 只读取当前批次各窗口。
        keys = {(item["sequence_id"], str(item.get("layer_id", "")), str(item.get("segment_id", "")), item.get("segment_start_seq_idx")) for item in windows}  # 不混墙、不混层、不混连续段。
        if len(keys) != 1:  # 即使采样器被误用也不跨界拼批。
            raise ValueError("无记忆批处理同样禁止跨墙、层或连续段。")  # 不静默重排样本。
        steps = samples[0][0].shape[0]  # 当前窗口实际选取的图像张数。
        temperatures = torch.stack([sample[1] for sample in samples])  # 温度仍按每个窗口完整保留，不跨窗共享温度TCN输出。
        labels = torch.stack([sample[2] for sample in samples])  # 真值只作为独立损失目标。
        inverse = None  # 不复用时直接返回普通五维图像批次。
        if self.reuse_images:  # 仅对当前批次实际重复的历史图像复用空间编码。
            lookup, unique, mapping = {}, [], []  # 特征映射只活到本批结束，不跨参数更新缓存。
            for sample, window in zip(samples, windows):  # 各窗口仍保持自己的原时间位置。
                if len(window["image_indices"]) != steps or sample[0].shape[0] != steps:  # 不能用错误元数据匹配图像。
                    raise ValueError("窗口图像索引与实际输入张数不一致。")  # 保持来源可追溯。
                row = []  # 当前窗口的局部历史位置映射。
                for position, source_index in enumerate(window["image_indices"]):  # 只访问该窗口自己的历史，不添加未来图像。
                    key = (window["sequence_id"], int(source_index))  # 同一墙的原图片索引唯一标识一张观测。
                    if key not in lookup:  # 首次出现时才加入ResNet待编码列表。
                        lookup[key] = len(unique)  # 登记本批唯一图片位置。
                        unique.append(sample[0][position])  # 保留原有确定性预处理结果。
                    row.append(lookup[key])  # 重复图像引用同一个可反向传播的编码。
                mapping.append(row)  # 每个窗口分别保留完整的有序输入。
            images = torch.stack(unique)  # [本批唯一图片数,1,高,宽]，后续由DataLoader锁页。
            inverse = torch.tensor(mapping, dtype=torch.long)  # [窗口批量,每窗图数]，后续在GPU上恢复各历史。
        else:  # 用户关闭共享或未来加入随机增强时采用正常批处理。
            images = torch.stack([sample[0] for sample in samples])  # [窗口批量,每窗图数,1,高,宽]。
        return {"images": images, "temperatures": temperatures, "labels": labels, "indices": indices,  # 张量字典会被DataLoader递归锁页。
                "inverse": inverse, "nominal_image_frames": len(samples) * steps, "encoded_image_frames": len(images) if self.reuse_images else len(samples) * steps}  # 统计真正送入ResNet的图片数。


def transfer_tensor(value, device, settings):  # 锁页CPU张量才可能实现异步CPU到GPU传输。
    non_blocking = device.type == "cuda" and bool(settings.get("non_blocking_transfer", False))  # CPU不启用CUDA选项。
    return value.to(device, non_blocking=non_blocking)  # 调用方持有原CPU批次直至计算结束。


class PrecisionPolicy:  # 一个训练运行只创建一次，FP16动态缩放状态跨轮次保留。
    def __init__(self, settings, device):  # 模型参数本身仍保持FP32。
        self.device, self.requested = device, bool(settings.get("amp_enabled", False))  # 旧快照默认普通精度。
        self.enabled = self.requested and device.type == "cuda"  # 当前只为CUDA启用自动混合精度。
        self.disabled_reason = None if self.enabled else ("当前使用CPU" if self.requested else "配置关闭AMP")  # 记录实际禁用原因。
        requested_dtype = settings.get("amp_dtype", "auto")  # auto根据实际GPU能力选择。
        capability = torch.cuda.get_device_capability(device) if self.enabled else (0, 0)  # 只在实际CUDA设备上探测。
        supported = self.enabled and capability[0] >= 8 and bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())  # 只选择原生BF16，兼容旧版PyTorch。
        if self.enabled and requested_dtype == "bfloat16" and not supported:  # 显式请求不支持的精度时不悄悄替换。
            raise ValueError("当前GPU不支持BF16，请将AMP_DTYPE改为auto或float16。")  # CPU请求AMP则安全关闭。
        self.dtype = torch.bfloat16 if requested_dtype == "bfloat16" or (requested_dtype == "auto" and supported) else torch.float16  # 自动回退FP16。
        cudnn_version = torch.backends.cudnn.version() if self.enabled and torch.backends.cudnn.enabled else None  # 检查本机旧CUDA卷积栈。
        if self.enabled and self.dtype == torch.float16 and capability == (7, 5) and cudnn_version is not None and cudnn_version <= 8200:  # 本机Turing与cuDNN 8.2的FP16卷积实测会出现NaN。
            self.disabled_reason = "旧版Turing/cuDNN组合的FP16卷积兼容性"  # 保守避开已复现的数值问题，不影响A10的BF16。
            if requested_dtype == "float16":  # 显式选择不安全的路径时提示用户调整。
                raise ValueError(self.disabled_reason + "：请改用AMP_DTYPE=auto或升级PyTorch/CUDA环境。")  # 不静默覆盖显式要求。
            self.enabled = False  # 自动模式在该旧环境使用FP32。
        self.scaler = None  # FP32及BF16无需FP16动态梯度缩放。
        if self.enabled and self.dtype == torch.float16:  # FP16反向传播需要动态梯度缩放。
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):  # 兼容服务器PyTorch2.5。
                self.scaler = torch.amp.GradScaler("cuda")  # 同一个缩放器贯穿所有训练轮次。
            else:  # 兼容本地PyTorch1.10。
                self.scaler = torch.cuda.amp.GradScaler()  # 使用旧版等价接口。

    def context(self):  # 每次前向计算创建独立上下文。
        if not self.enabled:  # CPU或显式关闭AMP时不改变精度。
            return nullcontext()  # 训练与评估共用普通执行路径。
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):  # 新版优先使用设备明确的接口。
            return torch.amp.autocast("cuda", dtype=self.dtype)  # 卷积与线性层自动选择低精度。
        return torch.cuda.amp.autocast(dtype=self.dtype)  # 兼容旧版GPU环境。

    def backward_step(self, loss, model, optimizer):  # 梯度裁剪前必须恢复真实梯度量级。
        if self.scaler is not None:  # 仅FP16使用动态缩放。
            self.scaler.scale(loss).backward()  # 先缩放再反向传播，减轻小梯度下溢。
            self.scaler.unscale_(optimizer)  # 裁剪前撤销缩放。
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 保留原有梯度裁剪阈值。
            self.scaler.step(optimizer)  # 非有限梯度时由缩放器跳过错误更新。
            self.scaler.update()  # 根据梯度情况更新下一批缩放因子。
        else:  # BF16和FP32保持正常反向传播。
            loss.backward()  # 损失已经按原规则取批内逐窗口均值。
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 与FP16路径采用相同阈值。
            optimizer.step()  # 每批只更新一次参数。

    def description(self):  # 将实际运行精度写入JSON和终端。
        dtype = ("bfloat16" if self.dtype == torch.bfloat16 else "float16") if self.enabled else "float32"  # CPU不伪称AMP已开启。
        return {"amp_requested": self.requested, "amp_enabled": self.enabled, "compute_dtype": dtype, "amp_disabled_reason": self.disabled_reason,  # 参数和分类损失仍为FP32。
                "gradient_scaler": self.scaler is not None, "parameter_dtype": "float32", "loss_dtype": "float32"}  # 方便复现实验。


def phase_performance(device, seconds, count, nominal_frames, encoded_frames):  # 记录真实吞吐而非承诺固定倍数加速。
    return {"windows_per_second": count / max(seconds, 1e-9), "nominal_image_frames": nominal_frames, "encoded_image_frames": encoded_frames,  # 只统计局部窗口编码，旧记忆分支的额外编码不在此数中。
            "image_encoding_reuse_fraction": 1.0 - encoded_frames / max(nominal_frames, 1),  # 无共享时为0。
            "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0,  # 实际峰值张量显存。
            "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024 ** 3) if device.type == "cuda" else 0.0}  # 含PyTorch缓存分配器保留的显存。
