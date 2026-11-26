"""SWME、温度 TCN、DGF 和 BPM 的网络层。"""

import math
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F


@lru_cache(maxsize=1)
def cuda_scan():
    """需要时读取官方 SSD 核心。包含 import 错误供明确指定 CUDA 时诊断。"""
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    except (ImportError, OSError, RuntimeError) as error:
        return None, error
    return mamba_chunk_scan_combined, None


def select_scan_backend(requested, device):
    """同一组 Mamba 参数可用官方 CUDA 核心或 PyTorch SSD 计算。"""
    if requested == "torch" or (requested == "auto" and device.type == "cpu"):
        return "torch"
    if device.type != "cuda":
        raise ValueError('MAMBA_BACKEND="cuda" 需要可用的 CUDA 设备。')
    kernel, error = cuda_scan()
    if kernel is not None:
        return "cuda"
    if requested == "cuda":
        raise RuntimeError("无法加载 Mamba CUDA 扫描。请核对 mamba-ssm 与 PyTorch/CUDA 的版本。") from error
    return "torch"


def torch_ssd_scan(values, dt, decay_rate, b, c, skip):
    """单状态分组的因果 SSD。按每对时刻的衰减量计算，支持自动求导。"""
    # 输入 values 为 [窗口, 序列长度, 头数, 每头通道]。B、C 在各头间共享。
    # 点积 C(t)·B(s) 后乘 s 到 t 的状态衰减，仅保留 s<=t 的项。
    with torch.autocast(device_type=values.device.type, enabled=False):
        values_fp32 = values.float()
        charge = values_fp32 * dt.unsqueeze(-1)
        prefix = (dt * decay_rate).transpose(1, 2).cumsum(dim=-1)
        log_decay = prefix.unsqueeze(-1) - prefix.unsqueeze(-2)
        length = values.shape[1]
        causal = torch.ones((length, length), dtype=torch.bool, device=values.device).tril()
        decay = log_decay.masked_fill(~causal, -torch.inf).exp()
        coefficients = torch.bmm(c.float(), b.float().transpose(1, 2)).unsqueeze(1)
        output = torch.matmul(coefficients * decay, charge.transpose(1, 2)).transpose(1, 2)
        return output + values_fp32 * skip.view(1, 1, -1, 1)


class MambaSequence(nn.Module):
    """Mamba-2 的投影、深度因果卷积、SSD、门控 RMS 归一化与输出投影。"""

    def __init__(self, width, state_dim, head_dim, backend):
        super().__init__()
        self.inner = 2 * width  # 通道扩张倍数为 2。
        self.state_dim = state_dim
        self.head_dim = head_dim
        self.heads = self.inner // head_dim
        self.backend = backend
        self.conv_width = self.inner + 2 * state_dim  # 卷积同时作用于内容、状态写入与读取分量。
        self.in_proj = nn.Linear(width, self.inner + self.conv_width + self.heads, bias=False)
        self.causal_conv = nn.Conv1d(self.conv_width, self.conv_width, kernel_size=4, padding=3,
                                    groups=self.conv_width)
        # 在常用步长范围内初始化；softplus 后得到正的步长。
        step = torch.empty(self.heads).uniform_(math.log(0.001), math.log(0.1)).exp()
        self.dt_bias = nn.Parameter(step + torch.log(-torch.expm1(-step)))
        self.a_log = nn.Parameter(torch.empty(self.heads).uniform_(1, 16).log())
        self.skip = nn.Parameter(torch.ones(self.heads))
        self.rms_weight = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, width, bias=False)

    def forward(self, sequence):
        gate, conv_input, dt = self.in_proj(sequence).split(
            (self.inner, self.conv_width, self.heads), dim=-1,
        )
        # 左右补零后仅保留前 L 个结果，当前位置只看自己和更早的位置。
        conv_output = self.causal_conv(conv_input.transpose(1, 2))[..., :sequence.shape[1]]
        projected = F.silu(conv_output.transpose(1, 2))
        values, b, c = projected.split((self.inner, self.state_dim, self.state_dim), dim=-1)
        values = values.reshape(*sequence.shape[:2], self.heads, self.head_dim)
        dt = F.softplus(dt.float() + self.dt_bias.float())
        decay_rate = -self.a_log.float().exp()
        if self.backend == "cuda":
            kernel, _ = cuda_scan()
            output = kernel(values, dt, decay_rate, b.unsqueeze(2), c.unsqueeze(2),
                            chunk_size=64, D=self.skip.float())
        else:
            output = torch_ssd_scan(values, dt, decay_rate, b, c, self.skip.float())
        # SSD 各头拼回通道，乘 SiLU 门，再以 FP32 计算 RMS 归一化。
        gated = output.reshape(*sequence.shape[:2], self.inner).float() * F.silu(gate.float())
        normalized = gated * torch.rsqrt(gated.square().mean(dim=-1, keepdim=True) + 1e-5)
        return self.out_proj((normalized * self.rms_weight).to(gate.dtype))


@lru_cache(maxsize=16)
def window_layout(height, width, size, shift):
    """把规则窗或移位窗分组，并记录恢复原始空间位置的逆索引。"""
    groups = {}
    for top in range(-shift, height, size):
        rows = range(max(0, top), min(top + size, height))
        for left in range(-shift, width, size):
            columns = range(max(0, left), min(left + size, width))
            # 边缘保留实际格子。不同面积的窗口分组，避免填充参加状态扫描。
            cells = [row * width + column for row in rows for column in columns]
            groups.setdefault(len(cells), []).append(cells)
    indices = tuple(torch.tensor(group, dtype=torch.long) for group in groups.values())
    restore = torch.cat([index.flatten() for index in indices]).argsort()
    return indices, restore


class WindowMambaBlock(nn.Module):
    """RW-M 与 SW-M 使用同一结构，各自持有独立权重。"""

    def __init__(self, width, window_size, shift, state_dim, head_dim, backend):
        super().__init__()
        self.window_size = window_size
        self.shift = shift
        self.norm = nn.LayerNorm(width)
        self.mamba = MambaSequence(width, state_dim, head_dim, backend)

    def forward(self, features):
        batch, frames, channels, height, width = features.shape
        indices, restore = window_layout(height, width, self.window_size, self.shift)
        flattened = features.flatten(3).transpose(2, 3)  # [批量, 帧, 空间格, 通道]
        output = []
        for index in indices:
            windows, cells = index.shape
            # 每个空间窗口独立扫描，序列先按帧、再按窗口内行列排序。
            patches = flattened[:, :, index.to(features.device), :]
            tokens = patches.permute(0, 2, 1, 3, 4).reshape(batch * windows, frames * cells, channels)
            updated = tokens + self.mamba(self.norm(tokens))
            restored = updated.reshape(batch, windows, frames, cells, channels).transpose(1, 2)
            output.append(restored.reshape(batch, frames, windows * cells, channels))
        grid = torch.cat(output, dim=2).index_select(2, restore.to(features.device))
        return grid.transpose(2, 3).reshape(batch, frames, channels, height, width)


class ShiftedWindowMambaEncoder(nn.Module):
    """SWME 先做 RW-M 再做 SW-M，最后逐帧汇聚空间特征。"""

    def __init__(self, width, window_size, shift, state_dim, head_dim, backend, shift_second):
        super().__init__()
        self.rw_m = WindowMambaBlock(width, window_size, 0, state_dim, head_dim, backend)
        # 消融时第二块仍有相同结构和独立权重，分窗位置改为规则网格。
        self.sw_m = WindowMambaBlock(width, window_size, shift if shift_second else 0,
                                    state_dim, head_dim, backend)

    def forward(self, features):
        return self.sw_m(self.rw_m(features)).mean(dim=(-1, -2))


class CausalConv1d(nn.Conv1d):
    """左侧补零，保证每个温度特征只读到该时刻。"""

    def __init__(self, input_width, output_width, kernel, dilation):
        super().__init__(input_width, output_width, kernel, dilation=dilation)
        self.left = (kernel - 1) * dilation

    def forward(self, values):
        return super().forward(F.pad(values, (self.left, 0)))


class TemporalResidualBlock(nn.Module):
    """两次相同膨胀尺度的因果卷积，保留时序长度。"""

    def __init__(self, input_width, width, kernel, dilation, dropout):
        super().__init__()
        self.convolution = nn.Sequential(
            CausalConv1d(input_width, width, kernel, dilation), nn.ReLU(), nn.Dropout(dropout),
            CausalConv1d(width, width, kernel, dilation), nn.ReLU(), nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(input_width, width, 1) if input_width != width else nn.Identity()

    def forward(self, values):
        return F.relu(self.convolution(values) + self.skip(values))


def temperature_tcn(kernel, widths, dropout):
    """每路温度是独立 TCN，依次使用 1、2、4 等膨胀率。"""
    return nn.Sequential(*(
        TemporalResidualBlock(previous, width, kernel, 2 ** level, dropout)
        for level, (previous, width) in enumerate(zip((1, *widths), widths))
    ))


class DualGatedFusion(nn.Module):
    """图像查短温度，随后在原视觉和交互特征上分别进行门控与残差。"""

    def __init__(self, image_width, temperature_width, heads, dropout):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            image_width, heads, dropout=dropout, kdim=temperature_width,
            vdim=temperature_width, batch_first=True,
        )
        self.interaction_mlp = nn.Sequential(
            nn.LayerNorm(image_width), nn.Linear(image_width, image_width * 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(image_width * 2, image_width),
        )
        self.image_gate = self._gate(image_width)
        self.interaction_gate = self._gate(image_width)

    @staticmethod
    def _gate(width):
        return nn.Sequential(nn.Linear(2 * width, width // 2), nn.ReLU(),
                             nn.Linear(width // 2, width), nn.Sigmoid())

    def forward(self, image, temperature):
        context, _ = self.cross_attention(image, temperature, temperature, need_weights=False)
        interaction = image + context
        interaction = interaction + self.interaction_mlp(interaction)
        gate_input = torch.cat((image, interaction), dim=-1)
        visual = image + self.image_gate(gate_input) * image
        mixed = interaction + self.interaction_gate(gate_input) * interaction
        return visual + mixed


class BreakOffPrecursorMatching(nn.Module):
    """BPM 为每条前兆参考生成一个响应，投影成补充分类特征。"""

    def __init__(self, window_dim, reference_count, match_dim, output_dim):
        super().__init__()
        # 参考向量保存在权重中，评估只需恢复训练产物。
        self.register_buffer("reference_bank", torch.zeros(reference_count, window_dim))
        self.query = nn.Linear(window_dim, match_dim)
        self.key = nn.Linear(window_dim, match_dim)
        self.value = nn.Linear(window_dim, match_dim)
        self.reference_weight = nn.Parameter(torch.ones(reference_count))
        self.response_projection = nn.Linear(reference_count, output_dim)
        self.scale = match_dim ** -0.5

    def forward(self, features):
        query = self.query(self.reference_bank).unsqueeze(0)
        key = self.key(features).unsqueeze(1)
        value = self.value(features).unsqueeze(1)
        # 前兆向量提供查询，当前窗口提供键和值；保留每条参考的加性匹配响应。
        responses = (torch.tanh(query + key) * value).sum(dim=-1) * self.scale
        return self.response_projection(responses * self.reference_weight)

