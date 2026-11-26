# -*- coding: utf-8 -*-
"""组装模型、定义各分支和前向计算。"""

import inspect  # 兼容不同PyTorch版本的安全权重加载参数。
from pathlib import Path  # 读取用户指定的离线预训练文件。

import torch  # 张量运算和参数加载。
from torch import nn  # 定义神经网络各层。
from torch.nn import functional as F  # 为因果卷积只在时间左侧补零。
from torchvision.models import resnet34  # 使用torchvision官方ResNet34结构。

try:  # 新版torchvision提供明确的预训练权重枚举。
    from torchvision.models import ResNet34_Weights  # 对应ImageNet预训练权重。
except ImportError:  # 允许已有torchvision 0.11环境执行结构检查。
    ResNet34_Weights = None  # 旧接口使用同一官方权重文件。

class ResNet34TCNTempPower(nn.Module):
    """ResNet34 → 三路图像 TCN。短、长温度各经一路 TCN。功率 → MLP。融合 → 三分类。"""

    def __init__(
        self, *, image_kernel_sizes, image_tcn_channels, num_classes, image_steps, short_temperature_steps, long_temperature_steps,
        power_steps, power_feature_dim, temperature_kernel_sizes, temperature_tcn_channels,
        temperature_tcn_width, image_tcn_width, classifier_hidden_dim, dropout,
        temperature_samples_per_image, freeze_image_backbone, freeze_backbone_batch_norm,
        image_imagenet_normalization,
    ):
        # 一、记录三段输入历史的长度，再搭建逐帧 ResNet34。
        if temperature_samples_per_image < 1 or short_temperature_steps % temperature_samples_per_image:
            raise ValueError("短温度长度必须由每张图的完整温度组组成。")
        self.history_lengths = (short_temperature_steps, long_temperature_steps, power_steps)  # 输入张量中三个独立片段的长度。
        temperature_steps = sum(self.history_lengths)  # 打包输入的总长度，用于核对形状。
        super().__init__()  # 初始化本模型的逐帧 ResNet34 骨干。
        if num_classes < 2 or image_steps < 1 or temperature_steps < 1:
            raise ValueError("类别至少 2 类，历史长度必须为正数。")
        self.image_steps, self.temperature_steps = int(image_steps), int(temperature_steps)  # 核对训练和回放的历史长度。
        self.freeze_image_backbone = bool(freeze_image_backbone)  # 保存是否微调视觉编码器。
        self.freeze_backbone_batch_norm = bool(freeze_backbone_batch_norm)  # 保存BN统计量策略。
        self.image_imagenet_normalization = bool(image_imagenet_normalization)  # 评估时恢复相同图像变换。
        self.backbone = resnet34(weights=None) if ResNet34_Weights is not None else resnet34(pretrained=False)  # 构造时不联网，评估只加载本次best_model.pt。
        self.backbone.fc = nn.Identity()  # 去掉ImageNet的1000类分类层，保留512维图像特征。
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))  # 保存预训练网络的三通道均值。
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))  # 与模型权重一并保存三通道标准差。

        # 二、搭建长短温度、功率分支和融合分类头。
        self.temperature_short = TemporalEncoder(1, temperature_tcn_channels, temperature_kernel_sizes[0], dropout)
        self.temperature_long = TemporalEncoder(1, temperature_tcn_channels, temperature_kernel_sizes[1], dropout)
        self.temperature_projector = nn.Sequential(  # 合并短、长温度的末端特征。
            nn.Linear(2 * temperature_tcn_channels[-1], temperature_tcn_width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(temperature_tcn_width, temperature_tcn_width),
        )
        self.power_encoder = nn.Sequential(  # 功率 MLP 独立读取同一窗口的设定值序列。
            nn.Linear(power_steps, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, power_feature_dim),
        )
        self.classifier = nn.Sequential(  # 默认 64 + 64 + 16 维拼接，分类头输出三类分数。
            nn.Linear(image_tcn_width + temperature_tcn_width + power_feature_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )
        self.backbone.requires_grad_(not self.freeze_image_backbone)  # 冻结模式仍加载相同的骨干结构。
        self.train(self.training)  # 立即应用固定 BN 统计的配置。

        # 三、搭建图像时序分支。
        self.image_encoder = MultiScaleTemporalEncoder(
            512, image_tcn_channels, image_kernel_sizes, image_tcn_width, dropout,
        )  # 三路分别使用 3、5、7 的时间卷积核，合并成一个向量。

    def forward(self, images, temperatures):  # 普通单窗与不复用的批处理均使用同一模型。
        if images.ndim != 5 or images.shape[2] != 1 or images.shape[1] != self.image_steps:  # 图像只包含当前窗口的历史。
            raise ValueError("图像需要[批量,历史图数,1,高,宽]，历史图数应与模型配置一致。")  # 与逐窗预测接口兼容。
        batch, steps, _, height, width = images.shape  # 保留原窗口维度。
        encoded = self.encode_frames(images.reshape(batch * steps, 1, height, width))  # 不共享时逐张编码。
        return self.classify_history(encoded.reshape(batch, steps, *encoded.shape[1:]), temperatures)  # 恢复各自的时序历史。

    def forward_shared_frames(self, images, inverse, temperatures):  # 同一优化器批次内共用确定性的ResNet编码。
        if self.training and not (self.freeze_image_backbone or self.freeze_backbone_batch_norm):  # 可更新BN统计时不能混合窗口。
            raise ValueError("复用图片编码前需要固定ResNet的BN统计量。")  # 公共加载器也会在此设置下退回逐窗。
        if inverse.ndim != 2 or inverse.shape[1] != self.image_steps or inverse.dtype != torch.long:  # 每个窗口的历史都要完整索引。
            raise ValueError("共享编码索引需要[批量,历史图数]的整数张量。")  # 拒绝错误的映射。
        encoded = self.encode_frames(images)  # 只编码本批唯一图片；不保存跨批缓存。
        history = encoded[inverse]  # 按原时间位置取回特征，梯度仍可回到同一张图片的编码。
        return self.classify_history(history, temperatures)  # 每个窗口单独输出一次分类。

    def classify_history(self, encoded, histories):
        if histories.shape != (encoded.shape[0], 1, self.temperature_steps):
            raise ValueError("历史输入需要 [批量, 1, 短温度点数＋长温度点数＋功率点数]。")
        short, long, power = torch.split(histories, self.history_lengths, dim=-1)  # 公共批次只打包传输，三路各自编码。
        short_features = self.temperature_short(short)  # 小核关注图像窗口附近的温度变化。
        long_features = self.temperature_long(long)  # 大核覆盖近 3.36 秒的温度历史。
        temperature = self.temperature_projector(torch.cat((short_features, long_features), dim=1))
        power_features = self.power_encoder(power.squeeze(1))  # 每窗一条功率历史。
        image_features = self._encode_image_history(encoded)  # 由 TCN 版或 LSTM 版提供。
        return self.classifier(torch.cat((image_features, temperature, power_features), dim=1))

    def encode_frames(self, images):  # 对当前批次的实际图片执行空间编码。
        if images.ndim != 4 or images.shape[1] != 1:  # 图像为[张数,1,高,宽]。
            raise ValueError("ResNet34图像编码需要[张数,1,高,宽]。")  # 不混入温度和状态标签。
        if images.dtype == torch.uint8:  # 加载进程可用原像素字节传输，减少共享内存和搬运量。
            images = images.float().div_(255.0)  # 与逐窗路径一致，转换到0到1后再进行ImageNet标准化。
        frame_images = images.repeat(1, 3, 1, 1)  # 灰度复制为三通道，不增加新传感信息。
        if self.image_imagenet_normalization:  # 图像预处理与温度标准化是独立的两件事。
            frame_images = (frame_images - self.image_mean) / self.image_std  # 输入原先已由公共数据接口除以255。
        return self._encode_frame_features(frame_images)  # 由图像分支决定返回向量或保留二维位置的特征图。

    def _encode_frame_features(self, images):
        return self.backbone(images)  # 每张图经全局池化形成一个 512 维向量。

    def _encode_image_history(self, encoded):
        return self.image_encoder(encoded.transpose(1, 2))  # 从逐帧向量转为 TCN 所需的通道优先格式。

    def train(self, mode=True):  # 公共训练器每轮调用model.train()时仍遵守BN配置。
        super().train(mode)  # 正常切换TCN、投影层和分类头的Dropout。
        if hasattr(self, "backbone"):  # 初始化尚未完成时不访问不存在的子模块。
            if self.freeze_image_backbone:  # 完全冻结视觉骨干时也固定其运行模式。
                self.backbone.eval()  # 不更新BN运行统计量。
            elif self.freeze_backbone_batch_norm:  # 微调卷积参数但固定BN运行统计量。
                for module in self.backbone.modules():  # 遍历ResNet34中的全部层。
                    if isinstance(module, nn.BatchNorm2d):  # 只影响二维批归一化层。
                        module.eval()  # BN的可训练仿射参数仍可更新。
        return self  # 保持PyTorch的链式调用接口。

    def initialize_backbone(self, pretrained=True, weights_file=""):  # 仅由train.py在新训练开始时调用一次。
        if not pretrained:  # 从头训练使用torchvision的随机初始化。
            print("ResNet34：从头训练，未加载预训练权重。", flush=True)  # 不把随机初始化伪称为预训练。
            return  # 评估和预测入口从不调用这个函数。
        if str(weights_file).strip():  # 离线环境可读取事先下载的官方权重文件。
            path = Path(weights_file).expanduser()  # 保留用户明确指定的位置。
            if not path.is_file():  # 不用随机权重替代缺失的预训练文件。
                raise FileNotFoundError("未找到ResNet34预训练权重：" + str(path))  # 用户修正config中的路径即可。
            options = {"map_location": "cpu"}  # 先在CPU加载官方骨干参数。
            if "weights_only" in inspect.signature(torch.load).parameters:  # 新PyTorch使用受限读取。
                options["weights_only"] = True  # 不执行任意对象反序列化。
            state = torch.load(str(path), **options)  # 只接受用户指定的预训练文件。
        else:  # 联网环境下载一次，后续复用torch缓存。
            try:  # 下载失败必须明确报错，不能偷偷改成从头训练。
                state = ResNet34_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True) if ResNet34_Weights is not None else torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/resnet34-b627a593.pth", map_location="cpu", progress=True, check_hash=True)  # 官方ImageNet权重。
            except Exception as error:  # 将网络错误转换成可操作的配置提示。
                raise RuntimeError("ResNet34预训练权重下载失败。请在config.py填写RESNET_WEIGHTS_FILE，或明确设置RESNET_PRETRAINED=False从头训练。") from error  # 保留原始错误原因。
        if isinstance(state, dict) and "state_dict" in state:  # 兼容只多包一层state_dict的本地权重。
            state = state["state_dict"]  # 后续仍进行严格的骨干结构匹配。
        if not isinstance(state, dict):  # 不能把完整序列化模型当作官方参数字典。
            raise ValueError("RESNET_WEIGHTS_FILE需要官方ResNet34的state_dict权重。")  # 不静默猜测模型类型。
        state = {name: value for name, value in state.items() if not name.startswith("fc.")}  # 只移除已替换的ImageNet分类头。
        self.backbone.load_state_dict(state, strict=True)  # 其余每一层都必须与官方ResNet34一致。
        print("ResNet34：已加载ImageNet预训练权重；{}。".format("仅训练时序与分类部分" if self.freeze_image_backbone else "微调视觉骨干"), flush=True)  # 提示实际训练策略。


class CausalConv1d(nn.Module):  # 时间位置只能读取自身与更早的特征。
    def __init__(self, in_channels, out_channels, kernel_size, dilation):  # 卷积核大小按分支配置。
        super().__init__()  # 注册模块。
        self.left_padding = (kernel_size - 1) * dilation  # 左侧补零，保证时间长度不变。
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)  # 不做右侧补零。

    def forward(self, values):  # 输入为[窗口批量,特征通道,时间长度]。
        return self.conv(F.pad(values, (self.left_padding, 0)))  # 禁止卷积读取该位置以后的输入。


class TemporalBlock(nn.Module):  # 每个残差块包含两次因果膨胀卷积。
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):  # 一个分支内的残差块采用同一核大小。
        super().__init__()  # 初始化父类。
        self.main = nn.Sequential(  # 主路径学习时间变化。
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),  # 第一层时间卷积。
            nn.ReLU(),  # 引入非线性。
            nn.Dropout(dropout),  # 限制过拟合。
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),  # 第二层时间卷积。
            nn.ReLU(),  # 引入非线性。
            nn.Dropout(dropout),  # 保持各分支相同的随机失活策略。
        )  # 两层均保持时间长度。
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)  # 残差路径只投影通道。

    def forward(self, values):  # 对整段输入进行因果计算。
        return F.relu(self.main(values) + self.skip(values))  # 残差相加后激活。


class TemporalEncoder(nn.Module):  # 一条TCN分支由多个残差块组成。
    def __init__(self, in_channels, channels, kernel_size, dropout):  # channels逐级指定输出通道。
        super().__init__()  # 注册网络。
        blocks = []  # 收集当前分支的残差块。
        for level, out_channels in enumerate(channels):  # 层数由通道列表决定。
            blocks.append(TemporalBlock(in_channels, out_channels, kernel_size, 2 ** level, dropout))  # 膨胀率依次为1、2、4等。
            in_channels = out_channels  # 下一块接收上一块输出。
        self.network = nn.Sequential(*blocks)  # 不同分支持有独立参数。

    def forward(self, values):  # 输入包含该窗口全部历史时刻。
        return self.network(values)[:, :, -1]  # 只取末端特征用于未来分类。


class MultiScaleTemporalEncoder(nn.Module):  # 并行处理同一历史，不是把多条TCN串联。
    def __init__(self, in_channels, channels, kernel_sizes, output_dim, dropout):  # 图像三路、温度两路使用相同接口。
        super().__init__()  # 注册全部分支和投影层。
        self.branches = nn.ModuleList([TemporalEncoder(in_channels, channels, size, dropout) for size in kernel_sizes])  # 每个核大小对应一条独立TCN。
        combined_dim = len(kernel_sizes) * channels[-1]  # 三路64维拼接为192维，两路64维拼接为128维。
        self.projector = nn.Sequential(  # 将不同尺度的特征统一到指定宽度。
            nn.Linear(combined_dim, output_dim),  # 汇总各尺度信息。
            nn.ReLU(),  # 学习非线性组合。
            nn.Dropout(dropout),  # 降低投影层过拟合。
            nn.Linear(output_dim, output_dim),  # 输出该模态的时序特征。
        )  # 温度和图像的投影层不共享参数。

    def forward(self, values):  # 所有分支观察同一段历史。
        features = torch.cat([branch(values) for branch in self.branches], dim=1)  # 在特征通道方向拼接。
        return self.projector(features)  # 返回[窗口批量,模态特征维数]。
