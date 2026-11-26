"""在 PyCharm、VS Code 中调整 WSP-Mamba 的训练、消融和评估。"""

import math
import sys
from pathlib import Path

# IDE 直接运行本目录的脚本时，也能找到上一级 code 的公共文件。
shared_code = str(Path(__file__).resolve().parents[1])
if shared_code not in sys.path:
    sys.path.insert(0, shared_code)

from data_config import get_data_settings, validate_data_settings
from training_acceleration import validate_performance_settings

MODEL_NAME = "wsp_mamba"
FORMAT_VERSION = 1

# ==================== 一、训练阶段与消融 ====================
TRAIN_STAGE = "foundation"  # foundation：训练基础模型；bpm：接入匹配；frozen_head：同预算无 BPM 对照。
FOUNDATION_RUN_DIR = ""  # 接入 BPM 或重训对照头所用的基础运行目录；留空按当前输入配置选最近一次。
FOUNDATION_CHECKPOINT = "loss"  # loss 选最低验证损失；transition_f1 选最佳稳定转熔断 F1。
REFERENCE_WINDOWS = [
    # 每条填（训练墙文件夹名, 层号, 目标熔断帧的原始编号）。
    # 编号取自 frame_manifest.csv 的 source_frame_number。
    # 程序按 FUTURE_FRAMES 往回找有效窗口，要求输入末帧稳定。
]
SHIFT_SECOND_BLOCK = True  # True 为 RW-M → SW-M；False 为同深度的 RW-M → RW-M。
USE_DGF = True  # False 时图像和短温度分别汇总、拼接并投影到相同维数。
ATTO_PRETRAINED = True  # 基础训练加载官方 ImageNet 权重。
ATTO_WEIGHTS_FILE = "weights/pretrained/convnextv2_atto_1k_224_ema.pt"  # 相对于项目根目录；留空可使用官方权重缓存或下载。
FREEZE_IMAGE_BACKBONE = False  # 基础训练默认微调 Atto；第二阶段自动冻结整个窗口编码器。

# ==================== 二、历史输入 ====================
WINDOW_SIZE = 20  # 历史跨度为连续 20 帧，稀疏选图照样使用同一段温度和功率。
WINDOW_STRIDE = 1  # 每向前一帧构造一个候选窗口。
FUTURE_FRAMES = 1  # 预测历史末帧之后第几帧，与对照模型设为同值。
IMAGE_SIZE = 640  # 正式训练分辨率为 640×640。
IMAGE_SAMPLING = "all"  # all、uniform 或 custom。
IMAGE_STEPS = 5  # 启用 uniform 时选几张图，20 帧历史跨度保持。
CUSTOM_FRAME_POSITIONS = [1, 5, 9, 13, 17, 20]  # custom 使用窗口内从 1 开始的帧号。
LONG_TEMPERATURE_STEPS = 168  # 50 Hz 下约 3.36 秒；25 Hz 的功率历史对应 84 点。
POWER_SETTINGS_FILE = ""  # 留空沿用 DATA_ROOT/power_settings.csv。
TEMPERATURE_STAT_EXCLUDE_VALUE = 799.9  # 只从拟合统计量中排除，保留该值入模。
TEMPERATURE_STAT_EXCLUDE_ATOL = 0.0001
IMAGE_IMAGENET_NORMALIZATION = True

# ==================== 三、子模块 ====================
SWME_WIDTH = 96  # 通过 1×1 卷积把 Atto 的 320 通道投影到此维数。
SPATIAL_WINDOW_SIZE = 2  # 在特征图上按 2×2 格分窗。
SPATIAL_SHIFT_SIZE = 1  # 第二块将窗口边界对角错开一格。
MAMBA_STATE_DIM = 64  # SSD 状态维，每头通道数和状态维是两个参数。
MAMBA_HEAD_DIM = 32  # 必须整除 SWME_WIDTH 的两倍。
MAMBA_BACKEND = "auto"  # auto 在可用时用官方 CUDA SSD；torch 可用普通 PyTorch；cuda 要求官方核。
SHORT_TEMPERATURE_KERNEL = 5
SHORT_TEMPERATURE_CHANNELS = [32, 64, 64]
LONG_TEMPERATURE_KERNEL = 13
LONG_TEMPERATURE_CHANNELS = [16, 32, 32]
DGF_HEADS = 4
POWER_FEATURE_DIM = 16
BPM_MATCH_DIM = 64
BPM_FEATURE_DIM = 32
CLASSIFIER_HIDDEN_DIM = 64
DROPOUT = 0.2

# ==================== 四、训练与运行资源 ====================
BATCH_SIZE = 4  # 一批只取同一墙、层和有效连续段内的窗口。
EVALUATION_BATCH_SIZE = 8  # 逐窗图像推理批量，独立于训练批量。
HEAD_BATCH_SIZE = 128  # 冻结后的缓存特征训练批量，两种头部方案共用。
EPOCHS = 100
PATIENCE = 10  # 基础阶段按验证损失早停。
HEAD_EPOCHS = 20  # BPM 与 frozen_head 都完整训练此轮数，匹配更新预算。
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
DEVICE = "auto"
CPU_THREADS = 8
NUM_WORKERS = 2 if sys.platform.startswith("linux") else 0
AMP_ENABLED = True
AMP_DTYPE = "auto"
REUSE_IMAGE_FEATURES = True  # 同一批次内只对重复图像运行一次 Atto。
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 1
NON_BLOCKING_TRANSFER = True
PROGRESS_MODE = "auto"
PROGRESS_REFRESH_SECONDS = 0.5
PLAIN_LOG_INTERVAL_SECONDS = 10.0

# ==================== 五、评估与回放 ====================
OUTPUT_ROOT = ""  # 默认使用项目 outputs/<数据集>/wsp_mamba。
EVALUATION_RUN_DIR = ""  # 可指定训练目录，留空选最近一次完成的 WSP-Mamba 运行。
EVALUATION_EPOCH = None  # None：最低验证损失；"transition_f1"：最高稳定转熔断 F1。
EVALUATION_SPLIT = "test"  # test、val、train、both（val+test）或 all。
PREDICT_SEQUENCE_ID = ""  # 单墙回放时留空选择测试集的第一面墙。


def project_path(path, project):
    """自定义相对路径均以项目根目录为基准。"""
    value = Path(path).expanduser()
    return str(value if value.is_absolute() else Path(project) / value)


def get_settings():
    """读取本文件的模型配置，并合入公共划分与数据配置。"""
    settings = get_data_settings()
    settings.update({name.lower(): value for name, value in globals().items() if name.isupper()})
    project = settings["project_dir"]
    dataset = settings["dataset_name"]
    settings["output_root"] = project_path(settings["output_root"] or f"outputs/{dataset}/{MODEL_NAME}", project)
    settings["power_settings_file"] = project_path(
        settings["power_settings_file"] or str(Path(settings["data_root"]) / "power_settings.csv"), project,
    )
    for key in ("atto_weights_file", "evaluation_run_dir", "foundation_run_dir"):
        if settings[key]:
            settings[key] = project_path(settings[key], project)
    settings.update(
        training_order="shuffled_layers_segment_time",
        exclude_first_layer=False,  # 当前层的有效范围由 unknown/ignore 标注决定。
        layer_event_reports=True,
        temperature_normalization_policy="train_unique_short_and_long_points_exclude_799_9_keep_inputs",
    )
    return settings


def validate_settings(settings):
    validate_data_settings(settings)
    validate_performance_settings(settings)
    if settings["model_name"] != MODEL_NAME or settings["format_version"] != FORMAT_VERSION:
        raise ValueError("请选择 WSP-Mamba 的训练目录和配置。")
    if settings["temperature_input_policy"] != "raw" or settings["startup_policy"] != "annotation_exclude":
        raise ValueError("WSP-Mamba 使用 raw 温度输入及 annotation_exclude 启动规则。")
    if settings["class_names"] != ["stable", "stubbing", "breakoff"]:
        raise ValueError("WSP-Mamba 的三分类类别顺序应为 stable、stubbing、breakoff。")
    if settings["train_stage"] not in ("foundation", "bpm", "frozen_head"):
        raise ValueError("TRAIN_STAGE 请选择 foundation、bpm 或 frozen_head。")
    if settings["foundation_checkpoint"] not in ("loss", "transition_f1"):
        raise ValueError("FOUNDATION_CHECKPOINT 请选择 loss 或 transition_f1。")
    positive_keys = (
        "image_size", "window_size", "window_stride", "future_frames", "long_temperature_steps",
        "batch_size", "evaluation_batch_size", "head_batch_size", "epochs", "patience", "head_epochs",
        "cpu_threads", "swme_width", "spatial_window_size", "spatial_shift_size", "mamba_state_dim",
        "mamba_head_dim", "dgf_heads", "power_feature_dim", "bpm_match_dim", "bpm_feature_dim",
        "classifier_hidden_dim",
    )
    for key in positive_keys:
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key.upper()} 必须为正整数。")
    for key in ("atto_pretrained", "freeze_image_backbone", "image_imagenet_normalization",
                "shift_second_block", "use_dgf"):
        if type(settings[key]) is not bool:
            raise ValueError(f"{key.upper()} 必须为 True 或 False。")
    if settings["image_size"] < 32:
        raise ValueError("IMAGE_SIZE 至少为 32，保证 Atto 四阶段都能下采样。")
    if settings["spatial_shift_size"] >= settings["spatial_window_size"]:
        raise ValueError("SPATIAL_SHIFT_SIZE 必须小于 SPATIAL_WINDOW_SIZE。")
    width = settings["swme_width"]
    if 2 * width % settings["mamba_head_dim"] or width % settings["dgf_heads"] or width < 2:
        raise ValueError("MAMBA_HEAD_DIM 须整除 2×SWME_WIDTH，DGF_HEADS 须整除 SWME_WIDTH。")
    if settings["mamba_backend"] not in ("auto", "torch", "cuda"):
        raise ValueError("MAMBA_BACKEND 请选择 auto、torch 或 cuda。")
    sampling = settings["image_sampling"]
    if sampling not in ("all", "uniform", "custom"):
        raise ValueError("IMAGE_SAMPLING 请选择 all、uniform 或 custom。")
    if sampling == "uniform" and (type(settings["image_steps"]) is not int
                                   or not 1 <= settings["image_steps"] <= settings["window_size"]):
        raise ValueError("IMAGE_STEPS 必须介于 1 和 WINDOW_SIZE 之间。")
    if sampling == "custom":
        frames = settings["custom_frame_positions"]
        if (not frames or any(type(p) is not int or not 1 <= p <= settings["window_size"] for p in frames)
                or list(frames) != sorted(set(frames))):
            raise ValueError("CUSTOM_FRAME_POSITIONS 填写从 1 开始、递增且不重复的窗内帧号。")
    ratio = settings["temperature_samples_per_image"]
    short = settings["window_size"] * ratio
    long = settings["long_temperature_steps"]
    if long < short:
        raise ValueError("LONG_TEMPERATURE_STEPS 至少等于短温度点数。")
    for branch, steps in (("short", short), ("long", long)):
        kernel = settings[f"{branch}_temperature_kernel"]
        channels = settings[f"{branch}_temperature_channels"]
        if (type(kernel) is not int or kernel < 1 or kernel % 2 == 0 or not channels
                or any(type(d) is not int or d < 1 for d in channels)):
            raise ValueError(f"{branch.upper()} 温度 TCN 使用正奇数卷积核和正整数通道。")
        coverage = 1 + 2 * (kernel - 1) * (2 ** len(channels) - 1)
        if branch == "long" and coverage < steps:
            raise ValueError(f"长温度末端感受野只有 {coverage} 点，请调整核大小、层数或输入长度。")
    settings["short_temperature_steps"] = short
    settings["long_history_us"] = round(long / (settings["fps"] * ratio) * 1_000_000)
    settings["power_steps"] = math.ceil(long / ratio)

    numeric = ("learning_rate", "weight_decay", "dropout", "temperature_stat_exclude_value",
               "temperature_stat_exclude_atol", "progress_refresh_seconds", "plain_log_interval_seconds")
    for key in numeric:
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key.upper()} 必须为有限数值。")
    if (settings["learning_rate"] <= 0 or settings["weight_decay"] < 0 or not 0 <= settings["dropout"] < 1
            or settings["temperature_stat_exclude_atol"] < 0 or settings["progress_refresh_seconds"] <= 0
            or settings["plain_log_interval_seconds"] <= 0):
        raise ValueError("请核对学习率、权重衰减、DROPOUT、温度容差和刷新间隔。")
    if type(settings["num_workers"]) is not int or settings["num_workers"] < 0:
        raise ValueError("NUM_WORKERS 必须为非负整数。")
    if settings["device"] not in ("auto", "cuda", "cpu") or settings["progress_mode"] not in ("auto", "bar", "plain", "off"):
        raise ValueError("DEVICE 和 PROGRESS_MODE 未使用文档支持的值。")
    if settings["evaluation_split"] not in ("all", "both", "train", "val", "test"):
        raise ValueError("EVALUATION_SPLIT 请选择 all、both、train、val 或 test。")
    if settings["evaluation_epoch"] not in (None, "transition_f1"):
        raise ValueError('EVALUATION_EPOCH 使用 None 或 "transition_f1"。')
    data = Path(settings["data_root"]).resolve()
    output = Path(settings["output_root"]).resolve()
    if output == data or data in output.parents:
        raise ValueError("OUTPUT_ROOT 必须在配对数据目录之外。")


def model_options(settings, reference_count=0):
    """仅保存重建网络所需的参数，便于从权重独立评估。"""
    names = (
        "swme_width", "spatial_window_size", "spatial_shift_size", "mamba_state_dim", "mamba_head_dim",
        "shift_second_block", "short_temperature_steps", "long_temperature_steps", "power_steps",
        "short_temperature_kernel", "long_temperature_kernel", "short_temperature_channels",
        "long_temperature_channels", "power_feature_dim", "dgf_heads", "use_dgf",
        "bpm_match_dim", "bpm_feature_dim", "classifier_hidden_dim", "dropout",
        "freeze_image_backbone", "image_imagenet_normalization",
    )
    return dict({key: settings[key] for key in names},
                num_classes=len(settings["class_names"]), reference_count=reference_count)

