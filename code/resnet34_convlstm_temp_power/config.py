"""本模型的训练、评估及回放设置。"""

import math  # 校验长短历史与功率采样长度。
import sys  # 根据操作系统设置数据读取进程数。
from pathlib import Path  # 定位公共代码和项目相对地址。

_shared_code_dir = str(Path(__file__).resolve().parents[1])  # IDE 直接运行也可找到 code 公共文件。
if _shared_code_dir not in sys.path:
    sys.path.insert(0, _shared_code_dir)

from resnet_config import resnet_window_settings, validate_resnet_window_settings
from training_acceleration import validate_performance_settings
from wall_dataset import image_positions

MODEL_NAME = "resnet34_convlstm_temp_power"  # 权重和结果目录使用此名称。
IMAGE_ENCODER = "convlstm"  # 图像时序分支。
FORMAT_VERSION = 10  # 三模态权重和快照格式。


# ==================== 一、训练 ====================
BATCH_SIZE = 40  # 每次更新最多处理同一墙层有效段的 40 个窗口。
EVALUATION_BATCH_SIZE = 48  # 验证、测试和训练集重评的批量。
EPOCHS = 100  # 最大训练轮数。
PATIENCE = 10  # 验证损失连续多少轮未改善时早停。
LEARNING_RATE = 0.0001  # AdamW 的学习率。
SAVE_EPOCH_CHECKPOINTS = False  # 默认只保存最低验证损失和最佳稳定转熔断 F1。
DEVICE = "auto"  # 自动选 GPU，亦可明确选择 "cpu" 或 "cuda"。
NUM_WORKERS = 2 if sys.platform.startswith("linux") else 0  # Windows IDE 默认使用主进程读图。
CPU_THREADS = 8  # PyTorch 在主进程执行 CPU 算子时使用的线程数。
AMP_ENABLED = True  # CUDA 可用时自动选择混合精度。
REUSE_IMAGE_FEATURES = True  # 复用同批内相同图片的 ResNet34 编码。

# ==================== 二、历史和数据 ====================
WINDOW_SIZE = 20  # 图像短窗长度，同时决定短温度读取 40 点。
WINDOW_STRIDE = 1  # 每向前一张图构造一个候选窗口。
FUTURE_FRAMES = 1  # 预测末帧之后第 1 张图的状态。
IMAGE_SIZE = None  # 保留原图；可填写 224 等整数调整网络输入尺寸。
LONG_TEMPERATURE_STEPS = 168  # 名义 50 Hz 下对应约 3.36 秒，同时决定功率窗长。
POWER_SETTINGS_FILE = ""  # 留空读取 DATA_ROOT/power_settings.csv；也可填写表格绝对路径。
RESNET_PRETRAINED = True  # 新训练时加载 ImageNet 预训练骨干。
RESNET_WEIGHTS_FILE = "weights/pretrained/resnet34-b627a593.pth"  # 相对项目根目录；从头训练时清空。
FREEZE_IMAGE_BACKBONE = False  # 默认允许微调 ResNet34。

# ==================== 三、结果与回放 ====================
OUTPUT_ROOT = ""  # 留空使用本模型独立的默认结果目录。
EVALUATION_RUN_DIR = ""  # 留空选择本模型最近一次完整训练。
EVALUATION_SPLIT = "all"  # 可选 all、both、train、val 或 test。
EVALUATION_EPOCH = None  # None 选最低验证损失；"transition_f1" 选最高转变 F1。
PREDICT_SEQUENCE_DIR = ""  # 留空回放固定测试集第一面墙，也可填写配对后的墙目录。

# ==================== 四、网络与运行细节 ====================
TRAIN_SHUFFLE_UNIT = "layer"  # 每轮打乱训练层块，同层连续段内保持时间顺序。
LAYER_EVENT_REPORTS = True  # 保留一帧稳定转熔断和层首发统计。
DROPOUT = 0.2  # 两版共享时序分支和分类头的失活比例。
WEIGHT_DECAY = 0.0001  # 沿用 AdamW 的权重衰减。
EXCLUDE_FIRST_LAYER = False  # 按现有标注屏蔽 unknown 和 ignore。
IMAGE_SAMPLING = "all"  # 默认使用窗口内全部图片。
IMAGE_STEPS = 20  # uniform 模式的选图数量。
CUSTOM_FRAME_POSITIONS = [1, 5, 9, 13, 17, 20]  # custom 模式从 1 开始编号。
FREEZE_BACKBONE_BATCH_NORM = True  # 固定 BN 运行统计，允许同批复用逐帧编码。
IMAGE_IMAGENET_NORMALIZATION = True  # 使用与预训练权重一致的图像归一化。
IMAGE_FEATURE_DIM = 512  # ResNet34 最后一组卷积输出的特征图通道数。
IMAGE_CONVLSTM_CHANNELS = 32  # ConvLSTM 的输入和隐藏特征图通道数。
IMAGE_CONVLSTM_GRID_SIZE = 6  # 进入 ConvLSTM 前，每帧保留 6×6 个空间位置。
IMAGE_CONVLSTM_WIDTH = 64  # 末时刻隐藏图全局池化并投影后的宽度。
TEMPERATURE_KERNEL_SIZES = [5, 13]  # 对应短温度、长温度。
TEMPERATURE_TCN_CHANNELS = [32, 64, 64]  # 膨胀率依次为 1、2、4。
TEMPERATURE_TCN_WIDTH = 64  # 两路温度合并后的维数。
POWER_FEATURE_DIM = 16  # 功率 MLP 的输出维数。
CLASSIFIER_HIDDEN_DIM = 64  # 拼接后分类头的中间层宽度。
TEMPERATURE_STAT_EXCLUDE_VALUE = 799.9  # 拟合温度统计时排除的数值。
TEMPERATURE_STAT_EXCLUDE_ATOL = 0.0001  # 兼容占位值的浮点误差。
AMP_DTYPE = "auto"  # CUDA 上优先选择支持的混合精度。
LOCAL_BATCH_ENABLED = True  # 本版所有分支只使用当前窗口。
PIN_MEMORY = True  # CUDA 时使用锁页传输。
PERSISTENT_WORKERS = True  # 有后台读取进程时跨轮保留。
PREFETCH_FACTOR = 1  # 控制原图模式下的主机内存使用。
NON_BLOCKING_TRANSFER = True  # CUDA 时可异步传输输入张量。
PROGRESS_MODE = "auto"  # IDE 终端显示进度条，日志文件按时间刷新。
PROGRESS_REFRESH_SECONDS = 0.5  # 终端最短刷新间隔。
PLAIN_LOG_INTERVAL_SECONDS = 10.0  # 重定向日志的刷新间隔。


def get_settings():  # 合并本模型参数和公共数据配置。
    settings = resnet_window_settings(globals(), "resnet34_temperature_power")
    project = Path(settings["project_dir"])  # 结果和自定义地址随项目移动。
    power_path = Path(settings["power_settings_file"]).expanduser() if settings["power_settings_file"] else Path(settings["data_root"]) / "power_settings.csv"
    output = Path(settings["output_root"]).expanduser() if settings["output_root"] else project / "outputs" / settings["dataset_name"] / MODEL_NAME
    settings.update(
        format_version=FORMAT_VERSION,
        image_encoder=IMAGE_ENCODER,
        output_root=str(output if output.is_absolute() else project / output),
        power_settings_file=str(power_path if power_path.is_absolute() else project / power_path),
        power_source="setpoint",
        temperature_normalization_policy="train_unique_short_and_long_points_exclude_799_9_keep_inputs",
    )
    return settings  # 训练时验证后保存解析好的路径和输入尺寸。


def validate_settings(settings):
    validate_resnet_window_settings(settings, MODEL_NAME, "image_convlstm_width")  # 共用数据和资源校验。
    validate_performance_settings(settings)  # 检查 AMP 与数据加载选项。
    if settings["image_encoder"] != IMAGE_ENCODER:
        raise ValueError("当前模型的图像分支与训练快照不匹配。")
    if settings["startup_policy"] != "annotation_exclude":
        raise ValueError('三模态模型要求 STARTUP_POLICY="annotation_exclude"。')
    if settings["train_shuffle_unit"] not in ("none", "wall", "layer"):
        raise ValueError("TRAIN_SHUFFLE_UNIT 请选择 none、wall 或 layer。")
    for key in ("image_convlstm_channels", "image_convlstm_grid_size"):
        if type(settings[key]) is not int or settings[key] < 1:  # 保证有效的通道数和空间网格。
            raise ValueError(f"{key.upper()} 必须是正整数。")
    short_steps = settings["window_size"] * settings["temperature_samples_per_image"]  # 按完整图像窗口计算。
    long_steps = settings["long_temperature_steps"]  # 长支路长度独立于图像窗口。
    if type(long_steps) is not int or long_steps < short_steps:
        raise ValueError("LONG_TEMPERATURE_STEPS 必须是至少等于短温度点数的整数。")
    if type(settings["power_feature_dim"]) is not int or settings["power_feature_dim"] < 1:
        raise ValueError("POWER_FEATURE_DIM 必须是正整数。")
    levels = len(settings["temperature_tcn_channels"])  # 每个残差块有两次同膨胀率卷积。
    for name, steps, kernel in zip(("短", "长"), (short_steps, long_steps), settings["temperature_kernel_sizes"]):
        coverage = 1 + 2 * (kernel - 1) * (2 ** levels - 1)  # 末端输出能覆盖的输入点数。
        if steps > coverage:
            raise ValueError(f"{name}温度配置为 {steps} 点，当前 TCN 末端只覆盖 {coverage} 点。请调整窗口、卷积核或残差块数量。")
    temperature_hz = settings["fps"] * settings["temperature_samples_per_image"]  # 当前名义采样率 50 Hz。
    settings["short_temperature_steps"] = short_steps
    settings["long_history_us"] = round(long_steps / temperature_hz * 1_000_000)  # 用真实时间戳裁剪，单位为微秒。
    if settings["long_history_us"] < 1:
        raise ValueError("长温度历史时长至少需要 1 微秒。")
    settings["power_steps"] = math.ceil(long_steps / settings["temperature_samples_per_image"])  # 功率按图像帧率取样。
    selector = settings.get("evaluation_epoch")  # 支持最低损失、转变 F1 或已保存的指定轮次。
    if selector not in (None, "transition_f1") and (type(selector) is not int or selector < 1):
        raise ValueError('EVALUATION_EPOCH 使用 None、"transition_f1" 或正整数轮次。')


def model_options(settings):
    names = (
        "short_temperature_steps", "long_temperature_steps", "power_steps", "power_feature_dim",
        "temperature_kernel_sizes", "temperature_tcn_channels", "temperature_tcn_width",
        "image_convlstm_width", "image_convlstm_channels", "image_convlstm_grid_size",
        "classifier_hidden_dim", "dropout", "temperature_samples_per_image",
        "freeze_image_backbone", "freeze_backbone_batch_norm", "image_imagenet_normalization",
    )  # 显式保存恢复权重需要的全部结构参数。
    options = {name: settings[name] for name in names}
    options.update(num_classes=len(settings["class_names"]), image_steps=len(image_positions(settings)))
    return options
