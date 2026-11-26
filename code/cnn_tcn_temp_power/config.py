"""编辑本模型的训练、评估和回放设置。"""

import math  # 校验输入历史和数值参数。
import sys  # 让 IDE 直接运行时也能定位公共模块。
from pathlib import Path

shared_code = str(Path(__file__).resolve().parents[1])
if shared_code not in sys.path:
    sys.path.insert(0, shared_code)

from data_config import get_data_settings, validate_data_settings
from resnet_config import validate_tcn_settings
from training_acceleration import validate_performance_settings
from wall_dataset import image_positions

MODEL_NAME = "cnn_tcn_temp_power"
FORMAT_VERSION = 1  # 含功率 CNN 独立保存快照及权重。

# ==================== 一、训练 ====================
BATCH_SIZE = 40  # 每次参数更新最多使用同墙、同层、同有效段的窗口数。
EVALUATION_BATCH_SIZE = 48  # 验证、测试和训练集重评批量。
EPOCHS = 100  # 最大训练轮数。
PATIENCE = 10  # 按整体验证损失早停。
LEARNING_RATE = 0.0001  # AdamW 学习率。
WEIGHT_DECAY = 0.0001  # AdamW 权重衰减。
DROPOUT = 0.2  # 时序模块和融合分类头的随机失活。
SAVE_EPOCH_CHECKPOINTS = False  # 默认保留最低验证损失和最佳稳定转熔断 F1 的权重。
DEVICE = "auto"  # 自动选 GPU，也可填写 "cpu" 或 "cuda"。
CPU_THREADS = 8  # PyTorch CPU 算子的线程数。
NUM_WORKERS = 2 if sys.platform.startswith("linux") else 0  # 数据读取进程数。
TRAIN_SHUFFLE_UNIT = "layer"  # 打乱各墙的训练层块，同层同段内保序。

# ==================== 二、历史输入 ====================
WINDOW_SIZE = 20  # 每个窗口覆盖最近 20 张原始图像。
WINDOW_STRIDE = 1  # 每向前一张图产生一个候选窗口。
FUTURE_FRAMES = 1  # 目标取窗口末尾之后第 1 张图的状态。
IMAGE_SIZE = None  # None 保留原图，亦可设置缩放后图像尺寸。
IMAGE_SAMPLING = "all"  # 可选 all、uniform 或 custom。
IMAGE_STEPS = 20  # uniform 模式使用。
CUSTOM_FRAME_POSITIONS = [1, 5, 9, 13, 17, 20]  # custom 模式使用，从 1 开始。
EXCLUDE_FIRST_LAYER = False  # 有效范围沿用逐帧标注中的 unknown 和 ignore。
LONG_TEMPERATURE_STEPS = 168  # 单路温度上限，50 Hz 下约 3.36 秒。也决定功率历史时长。
TEMPERATURE_STAT_EXCLUDE_VALUE = 799.9  # 拟合均值、标准差时排除此值，输入保留。
TEMPERATURE_STAT_EXCLUDE_ATOL = 0.0001  # 排除占位值的比较容差。
POWER_SETTINGS_FILE = ""  # 留空使用公共数据目录的 power_settings.csv。

# ==================== 三、网络 ====================
IMAGE_FEATURE_DIM = 128  # 三层灰度 CNN 输出的逐帧向量宽度。
IMAGE_TEMPORAL_DIM = 64  # 图像时序分支的输出宽度。
IMAGE_KERNEL_SIZES = (3,)  # 图像只用一路因果 TCN。
IMAGE_TCN_CHANNELS = (64, 64, IMAGE_TEMPORAL_DIM)  # 图像时序残差块宽度。
TEMPERATURE_KERNEL_SIZES = (13,)  # 单路长温度 TCN 使用一个核大小。
TEMPERATURE_TCN_CHANNELS = (32, 64, 64)  # 三个温度残差块，末端覆盖 169 个采样点。
POWER_FEATURE_DIM = 16  # 功率 MLP 输出宽度。
CLASSIFIER_HIDDEN_DIM = 64  # 拼接后分类头的中间层宽度。

# ==================== 四、结果和运行效率 ====================
OUTPUT_ROOT = ""  # 留空使用项目下本模型专属的结果目录。
EVALUATION_RUN_DIR = ""  # 留空自动选择本模型最近完成的训练目录。
EVALUATION_SPLIT = "both"  # 评估验证和测试。可选 test、val、train、all。
EVALUATION_EPOCH = None  # None 取最低验证损失，"transition_f1" 取最佳稳定转熔断 F1。
PREDICT_SEQUENCE_DIR = ""  # 留空回放测试集第一面墙，也可填写墙目录的绝对路径。
LAYER_EVENT_REPORTS = True  # 保留逐层首发、稳定转熔断和误报统计。
LOCAL_BATCH_ENABLED = True  # 同墙同层同有效段内张量批处理。
REUSE_IMAGE_FEATURES = True  # 同一批次内重复的图像共用一次 CNN 编码。
AMP_ENABLED = True  # CUDA 可用时启用混合精度。
AMP_DTYPE = "auto"  # 自动选 GPU 支持的精度。
PIN_MEMORY = True  # CUDA 环境锁页传输。
PERSISTENT_WORKERS = True  # 保留跨轮的读图进程。
PREFETCH_FACTOR = 1  # 限制原图模式的数据预取量。
NON_BLOCKING_TRANSFER = True  # 使用 CUDA 异步传输。
PROGRESS_MODE = "auto"  # 终端显示进度条，重定向日志按时间刷新。
PROGRESS_REFRESH_SECONDS = 0.5  # 终端刷新最短间隔。
PLAIN_LOG_INTERVAL_SECONDS = 10.0  # 普通日志刷新间隔。


def get_settings():
    settings = get_data_settings()  # 两套 CNN 沿用现有整墙划分及标签规则。
    settings.update({key.lower(): value for key, value in globals().items() if key.isupper()})
    project = Path(settings["project_dir"])
    data = Path(settings["data_root"])
    power = Path(POWER_SETTINGS_FILE).expanduser() if POWER_SETTINGS_FILE else data / "power_settings.csv"
    output = Path(OUTPUT_ROOT).expanduser() if OUTPUT_ROOT else project / "outputs" / settings["dataset_name"] / MODEL_NAME
    settings.update(power_settings_file=str(project / power), output_root=str(project / output),
                    power_source="setpoint", model_variant="grayscale_cnn_temperature_power",
                    temperature_normalization_policy="resnet_training_reference_exclude_799_9_keep_inputs")
    return settings


def validate_settings(settings):
    validate_data_settings(settings)  # 按公共配置核对划分、类别和采样比例。
    validate_performance_settings(settings)
    if settings.get("model_name") != MODEL_NAME or settings.get("format_version") != FORMAT_VERSION:
        raise ValueError("配置与当前模型或权重格式不匹配。")
    if settings["startup_policy"] != "annotation_exclude" or settings["temperature_input_policy"] != "raw":
        raise ValueError("本模型沿用 annotation_exclude 和 raw 数据规则。")
    names = ("batch_size", "evaluation_batch_size", "epochs", "patience", "cpu_threads", "window_size",
             "window_stride", "long_temperature_steps", "image_temporal_dim", "power_feature_dim", "classifier_hidden_dim")
    if any(type(settings[name]) is not int or settings[name] < 1 for name in names):
        raise ValueError("批量、轮数、窗口和网络宽度需要正整数。")
    for name in ("future_frames", "num_workers"):
        if type(settings[name]) is not int or settings[name] < 0:
            raise ValueError(name + "需要非负整数。")
    width = settings["image_feature_dim"]
    if type(width) is not int or width < 4 or width % 4:
        raise ValueError("IMAGE_FEATURE_DIM 需要至少为 4 的 4 的倍数。")
    size = settings["image_size"]
    if size is not None and (type(size) is not int or size < 16):
        raise ValueError("IMAGE_SIZE 使用 None 或至少 16 的整数。")
    if settings["image_sampling"] not in ("all", "uniform", "custom"):
        raise ValueError("IMAGE_SAMPLING 使用 all、uniform 或 custom。")
    if settings["image_sampling"] == "uniform" and (
        type(settings["image_steps"]) is not int or not 1 <= settings["image_steps"] <= settings["window_size"]
    ):
        raise ValueError("IMAGE_STEPS 需要介于 1 和 WINDOW_SIZE 之间。")
    if settings["image_sampling"] == "custom":
        positions = settings["custom_frame_positions"]
        if (not positions or any(type(value) is not int or not 1 <= value <= settings["window_size"] for value in positions)
                or list(positions) != sorted(set(positions))):
            raise ValueError("CUSTOM_FRAME_POSITIONS 需要严格递增的窗口内位置。")
    if settings["train_shuffle_unit"] not in ("none", "wall", "layer"):
        raise ValueError("TRAIN_SHUFFLE_UNIT 使用 none、wall 或 layer。")
    if settings["evaluation_split"] not in ("all", "both", "train", "val", "test"):
        raise ValueError("EVALUATION_SPLIT 使用 all、both、train、val 或 test。")
    epoch = settings["evaluation_epoch"]
    if epoch not in (None, "transition_f1") and (type(epoch) is not int or epoch < 1):
        raise ValueError("EVALUATION_EPOCH 使用 None、transition_f1 或已保存的正整数轮次。")
    if settings["device"] not in ("auto", "cuda", "cpu") or settings["progress_mode"] not in ("auto", "bar", "plain", "off"):
        raise ValueError("请检查 DEVICE 或 PROGRESS_MODE。")
    names = ("learning_rate", "dropout", "weight_decay", "temperature_stat_exclude_value",
             "temperature_stat_exclude_atol", "progress_refresh_seconds", "plain_log_interval_seconds")
    if any(isinstance(settings[name], bool) or not isinstance(settings[name], (int, float))
           or not math.isfinite(settings[name]) for name in names):
        raise ValueError("学习率、失活比例、权重衰减和时间间隔需要有限数值。")
    if (settings["learning_rate"] <= 0 or not 0 <= settings["dropout"] < 1 or settings["weight_decay"] < 0
            or settings["temperature_stat_exclude_atol"] < 0 or settings["progress_refresh_seconds"] <= 0
            or settings["plain_log_interval_seconds"] <= 0):
        raise ValueError("学习率、失活比例、权重衰减或间隔超出范围。")
    data = Path(settings["data_root"]).resolve()
    output = Path(settings["output_root"]).resolve()
    if output == data or data in output.parents:
        raise ValueError("OUTPUT_ROOT 需要位于数据目录之外。")
    ratio = settings["temperature_samples_per_image"]
    steps = settings["long_temperature_steps"]
    settings["short_temperature_steps"] = settings["window_size"] * ratio  # 仅用于与 ResNet 一致的温度统计。
    if steps < settings["short_temperature_steps"]:
        raise ValueError("单路温度长度需要覆盖图像窗口对应的温度点。")
    validate_tcn_settings(settings, "temperature", 1)
    temperature_coverage = 1 + 2 * (settings["temperature_kernel_sizes"][0] - 1) * (2 ** len(settings["temperature_tcn_channels"]) - 1)
    if steps > temperature_coverage:
        raise ValueError(f"温度 TCN 末端只覆盖 {temperature_coverage} 点，请调整核大小或通道列表。")
    validate_tcn_settings(settings, "image", 1)
    if settings["image_tcn_channels"][-1] != settings["image_temporal_dim"]:
        raise ValueError("图像 TCN 最后一个残差块宽度必须等于 IMAGE_TEMPORAL_DIM。")
    image_coverage = 1 + 2 * (settings["image_kernel_sizes"][0] - 1) * (2 ** len(settings["image_tcn_channels"]) - 1)
    if len(image_positions(settings)) > image_coverage:
        raise ValueError(f"图像 TCN 末端只覆盖 {image_coverage} 张图，请调整核大小或通道列表。")
    settings["long_history_us"] = round(steps / (settings["fps"] * ratio) * 1_000_000)
    settings["power_steps"] = math.ceil(steps / ratio)  # 功率按图像帧率采样，与长温度覆盖同一时长。


def model_options(settings):
    fields = ("image_feature_dim", "image_temporal_dim", "long_temperature_steps", "power_steps",
              "power_feature_dim", "temperature_kernel_sizes", "temperature_tcn_channels", "classifier_hidden_dim",
              "dropout", "image_kernel_sizes", "image_tcn_channels")
    return {**{name: settings[name] for name in fields}, "num_classes": len(settings["class_names"]),
            "image_steps": len(image_positions(settings))}
