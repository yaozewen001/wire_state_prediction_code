"""WSP-Mamba 从历史三模态输入到未来指定帧的三分类分数。"""

import torch
from torch import nn

if __package__:
    from .backbone import ConvNeXtV2Atto
    from .blocks import BreakOffPrecursorMatching, DualGatedFusion, ShiftedWindowMambaEncoder, temperature_tcn
else:
    from backbone import ConvNeXtV2Atto
    from blocks import BreakOffPrecursorMatching, DualGatedFusion, ShiftedWindowMambaEncoder, temperature_tcn


class WindowEncoder(nn.Module):
    """图像、长短温度和功率分支，产出 BPM 前的窗口向量。"""

    def __init__(self, options, scan_backend):
        super().__init__()
        image_width = options["swme_width"]
        self.history_sizes = (options["short_temperature_steps"], options["long_temperature_steps"],
                              options["power_steps"])
        self.image_imagenet_normalization = options["image_imagenet_normalization"]
        self.freeze_image_backbone = options["freeze_image_backbone"]
        self.atto = ConvNeXtV2Atto()
        self.image_projection = nn.Conv2d(self.atto.widths[-1], image_width, 1)
        self.swme = ShiftedWindowMambaEncoder(
            image_width, options["spatial_window_size"], options["spatial_shift_size"],
            options["mamba_state_dim"], options["mamba_head_dim"], scan_backend,
            options["shift_second_block"],
        )
        short_widths = options["short_temperature_channels"]
        long_widths = options["long_temperature_channels"]
        self.short_temperature = temperature_tcn(options["short_temperature_kernel"], short_widths, options["dropout"])
        self.long_temperature = temperature_tcn(options["long_temperature_kernel"], long_widths, options["dropout"])
        self.power = nn.Sequential(nn.Linear(options["power_steps"], 32), nn.GELU(),
                                   nn.Linear(32, options["power_feature_dim"]))
        self.use_dgf = options["use_dgf"]
        if self.use_dgf:
            self.fusion = DualGatedFusion(image_width, short_widths[-1], options["dgf_heads"], options["dropout"])
        else:
            # DGF 的对照仍使用图像与短温度，并给下游同样宽度的融合向量。
            self.fusion = nn.Linear(image_width + short_widths[-1], image_width)
        self.feature_dim = image_width + long_widths[-1] + options["power_feature_dim"]

        self.register_buffer("imagenet_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("imagenet_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
                             persistent=False)
        if self.freeze_image_backbone:
            self.atto.requires_grad_(False)
            self.atto.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_image_backbone:
            self.atto.eval()
        return self

    def encode_frames(self, images):
        """对本批唯一灰度图计算空间特征，保留末端特征网格。"""
        if images.dtype == torch.uint8:
            images = images.float().div(255.0)
        images = images.expand(-1, 3, -1, -1)
        if self.image_imagenet_normalization:
            images = (images - self.imagenet_mean) / self.imagenet_std
        return self.image_projection(self.atto(images))

    def forward(self, frame_grids, histories):
        short, long, power = histories.split(self.history_sizes, dim=-1)
        image = self.swme(frame_grids)  # [批量, 图像帧数, SWME 通道]
        short_tokens = self.short_temperature(short).transpose(1, 2)
        if self.use_dgf:
            image_temperature = self.fusion(image, short_tokens).mean(dim=1)
        else:
            image_temperature = self.fusion(torch.cat((image.mean(1), short_tokens.mean(1)), dim=-1))
        # 长温度末端的 TCN 感受野覆盖配置中的完整长历史。
        temperature = self.long_temperature(long)[..., -1]
        power_features = self.power(power.squeeze(1))
        return torch.cat((image_temperature, temperature, power_features), dim=-1)


class WSPMamba(nn.Module):
    """以相同接口执行基础分类、前兆匹配分类和同批图像编码复用。"""

    def __init__(self, options, scan_backend="torch"):
        super().__init__()
        self.options = dict(options)
        self.encoder = WindowEncoder(options, scan_backend)
        self.encoder_frozen = False
        self._build_head(options["reference_count"])

    def _build_head(self, reference_count):
        width = self.encoder.feature_dim
        self.bpm = None
        if reference_count:
            self.bpm = BreakOffPrecursorMatching(
                width, reference_count, self.options["bpm_match_dim"], self.options["bpm_feature_dim"],
            )
            width += self.options["bpm_feature_dim"]
        hidden = self.options["classifier_hidden_dim"]
        self.classifier = nn.Sequential(nn.Linear(width, hidden), nn.GELU(),
                                        nn.Dropout(self.options["dropout"]),
                                        nn.Linear(hidden, self.options["num_classes"]))

    def initialize_backbone(self, pretrained, weights_file):
        if pretrained:
            self.encoder.atto.load_pretrained(weights_file)

    def freeze_encoder(self):
        self.encoder_frozen = True
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def reset_head(self, references=None):
        """第二阶段从同一冻结编码器出发，重新初始化分类头。"""
        reference_count = 0 if references is None else len(references)
        self._build_head(reference_count)
        self.options["reference_count"] = reference_count
        self.to(next(self.encoder.parameters()).device)
        if references is not None:
            self.bpm.reference_bank.copy_(references)

    def encode_window(self, images, histories, inverse=None):
        """由全窗图像或本批唯一图像组装逐帧网格。"""
        if inverse is None:
            batch, frames = images.shape[:2]
            grids = self.encoder.encode_frames(images.flatten(0, 1))
            grids = grids.reshape(batch, frames, *grids.shape[1:])
        else:
            unique = self.encoder.encode_frames(images)
            grids = unique[inverse]  # 按每窗原始时间次序还原。
        return self.encoder(grids, histories)

    def classify_features(self, features):
        if self.bpm is not None:
            features = torch.cat((features, self.bpm(features)), dim=-1)
        return self.classifier(features)

    def forward(self, images, histories):
        return self.classify_features(self.encode_window(images, histories))

    def forward_shared_frames(self, images, inverse, histories):
        return self.classify_features(self.encode_window(images, histories, inverse))

