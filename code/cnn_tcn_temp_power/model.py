"""灰度 CNN、单路TCN、单路温度 TCN 和功率 MLP 的三分类网络。"""

import torch
from torch import nn
from torch.nn import functional as F

from image_encoder import make_image_cnn  # 沿用旧 CNN 基线的三层灰度空间编码器。


class CNNTCNTempPower(nn.Module):
    def __init__(
        self, *, num_classes, image_steps, image_feature_dim, image_temporal_dim,
        long_temperature_steps, power_steps, power_feature_dim, temperature_kernel_sizes,
        temperature_tcn_channels, classifier_hidden_dim, dropout, image_kernel_sizes, image_tcn_channels,
    ):
        super().__init__()
        self.image_steps = image_steps
        self.history_lengths = (long_temperature_steps, power_steps)  # 传输时打包的两种信号长度。
        self.image_cnn = make_image_cnn(image_feature_dim)  # 每张灰度图得到一个空间特征向量。
        self.image_tcn = TemporalEncoder(image_feature_dim, image_tcn_channels, image_kernel_sizes[0], dropout)
        self.temperature_tcn = TemporalEncoder(1, temperature_tcn_channels, temperature_kernel_sizes[0], dropout)
        self.power_encoder = nn.Sequential(  # 沿用 ResNet 对照的功率 MLP。
            nn.Linear(power_steps, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, power_feature_dim),
        )
        fused_dim = image_temporal_dim + temperature_tcn_channels[-1] + power_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),  # 返回交叉熵所需的三类原始分数。
        )

    def encode_frames(self, images):
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("CNN 接收 [图片数, 1, 高, 宽] 的灰度图像。")
        if images.dtype == torch.uint8:
            images = images.float().div_(255.0)  # 与逐窗读取路径的像素归一化一致。
        return self.image_cnn(images)

    def forward(self, images, histories):
        if images.ndim != 5 or images.shape[1] != self.image_steps:
            raise ValueError("图像接收 [批量, 历史张数, 1, 高, 宽]。")
        batch, steps, channels, height, width = images.shape
        encoded = self.encode_frames(images.reshape(batch * steps, channels, height, width))
        return self.classify_history(encoded.reshape(batch, steps, -1), histories)

    def forward_shared_frames(self, images, inverse, histories):
        if inverse.ndim != 2 or inverse.shape[1] != self.image_steps or inverse.dtype != torch.long:
            raise ValueError("共享图像索引需要 [批量, 历史张数] 的整数张量。")
        encoded = self.encode_frames(images)  # 一次编码本批次内所有唯一图片。
        return self.classify_history(encoded[inverse], histories)  # 恢复各窗口的时间顺序。

    def classify_history(self, encoded, histories):
        if histories.shape != (encoded.shape[0], 1, sum(self.history_lengths)):
            raise ValueError("信号需要 [批量, 1, 温度点数＋功率点数]。")
        temperature, power = torch.split(histories, self.history_lengths, dim=-1)
        image_features = self.image_tcn(encoded.transpose(1, 2))
        temperature_features = self.temperature_tcn(temperature)
        power_features = self.power_encoder(power.squeeze(1))
        fused = torch.cat((image_features, temperature_features, power_features), dim=1)
        return self.classifier(fused)


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
