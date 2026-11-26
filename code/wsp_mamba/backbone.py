"""完整四阶段 ConvNeXt V2-Atto，返回全局池化前的逐帧特征图。"""

from pathlib import Path

import torch
from torch import nn


# Meta 发布的 ImageNet-1K Atto 权重，键名与下面的空间骨干对应。
ATTO_WEIGHTS_URL = (
    "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt"
)


class SpatialLayerNorm(nn.LayerNorm):
    """在每个空间位置沿通道做 LayerNorm，返回 NCHW。"""

    def forward(self, images):
        return super().forward(images.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class GlobalResponseNorm(nn.Module):
    """ConvNeXt V2 的 GRN，输入为 NHWC。"""

    def __init__(self, width):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, width))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, width))

    def forward(self, features):
        magnitude = torch.linalg.vector_norm(features, ord=2, dim=(1, 2), keepdim=True)
        response = magnitude / (magnitude.mean(dim=-1, keepdim=True) + 1e-6)
        return features + self.gamma * (features * response) + self.beta


class ConvNeXtV2Block(nn.Module):
    """一次深度卷积、通道扩张、GRN、通道投影和残差。"""

    def __init__(self, width):
        super().__init__()
        self.dwconv = nn.Conv2d(width, width, kernel_size=7, padding=3, groups=width)
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.pwconv1 = nn.Linear(width, 4 * width)
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm(4 * width)
        self.pwconv2 = nn.Linear(4 * width, width)

    def forward(self, features):
        local = self.dwconv(features).permute(0, 2, 3, 1)
        expanded = self.act(self.pwconv1(self.norm(local)))
        projected = self.pwconv2(self.grn(expanded)).permute(0, 3, 1, 2)
        return features + projected


class ConvNeXtV2Atto(nn.Module):
    """Atto 的四阶段分别为 2、2、6、2 块，末端固定为 320 通道。"""

    widths = (40, 80, 160, 320)
    depths = (2, 2, 6, 2)

    def __init__(self):
        super().__init__()
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, self.widths[0], 4, stride=4),
                          SpatialLayerNorm(self.widths[0], eps=1e-6)),
            *(nn.Sequential(SpatialLayerNorm(previous, eps=1e-6),
                            nn.Conv2d(previous, width, 2, stride=2))
              for previous, width in zip(self.widths, self.widths[1:])),
        ])
        self.stages = nn.ModuleList(
            nn.Sequential(*(ConvNeXtV2Block(width) for _ in range(depth)))
            for width, depth in zip(self.widths, self.depths)
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def load_pretrained(self, weights_file):
        path = Path(weights_file) if weights_file else None
        # 本地文件优先。留空时使用 PyTorch 的下载缓存。
        if path is not None:
            if not path.is_file():
                raise FileNotFoundError(f"找不到 ATTO_WEIGHTS_FILE：{path}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        else:
            try:
                checkpoint = torch.hub.load_state_dict_from_url(
                    ATTO_WEIGHTS_URL, map_location="cpu", weights_only=True,
                )
            except OSError as error:
                raise OSError("Atto 权重下载失败。可下载官方文件后填写 ATTO_WEIGHTS_FILE。") from error
        state = checkpoint.get("model", checkpoint)
        # 官方 ImageNet 末端的全局归一化和分类头由本任务的空间时序分支替代。
        state = {name: value for name, value in state.items()
                 if not name.startswith(("norm.", "head."))}
        self.load_state_dict(state, strict=True)

    def forward(self, images):
        for downsample, stage in zip(self.downsample_layers, self.stages):
            images = stage(downsample(images))
        return images

