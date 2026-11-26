# -*- coding: utf-8 -*-
"""各模型和数据划分共用的配置；在 IDE 中编辑本文件，再运行 split_data.py。"""

import math  # 校验帧率和划分比例，避免无效数值进入数据流程。
from pathlib import Path  # 使用相对项目位置，兼容 Windows 和 Linux。

# ==================== 一、数据地址与划分保存位置 ====================
PROJECT_DIR = Path(__file__).resolve().parents[1]  # 本文件位于项目的 code/ 中。
DATASET_NAME = "wall_dataset"  # 用于区分本批数据的训练结果目录。
DATA_ROOT = PROJECT_DIR / "data" / "datasets" / "wall_dataset"  # 指向已配对的墙数据，不是 wall_photo 或原温度目录。
SEED = 136  # 所有模型的训练种子及整墙划分种子。
HOLDOUT_SWAP_SEED = None  # 当前直接按SEED划分；此项保留用于恢复以前的验证测试对调方案。
SHUFFLE_TRAIN_WALLS = True  # 每轮重排训练墙；每面墙内部始终按原层、段和时间顺序。
BASE_SPLIT_DIR = PROJECT_DIR / "splits" / f"{DATASET_NAME}_seed{SEED}"  # 锁定已有训练墙的基准清单。
SPLIT_DIR = BASE_SPLIT_DIR if HOLDOUT_SWAP_SEED is None else BASE_SPLIT_DIR.with_name(  # 对调方案另存。
    f"{BASE_SPLIT_DIR.name}_holdout_swap_seed{HOLDOUT_SWAP_SEED}"
)  # 各模型共用当前选择的正式划分。

# ==================== 二、已配对数据的含义 ====================
FPS = 25  # 当前图像帧率；预测提前帧数按照图像帧解释。
TEMPERATURE_SAMPLES_PER_IMAGE = 2  # 当前每张图对应两个原始温度点。
STARTUP_POLICY = "annotation_exclude"  # 按标注排除unknown图温，再从连续有效帧积累窗口；不再填写固定张数或特殊层表。
CLASS_NAMES = ["stable", "stubbing", "breakoff"]  # 类别顺序固定，三个模型的输出编号保持一致。
TEMPERATURE_INPUT_POLICY = "raw"  # 799.9按原值入模；状态只依据人工区间，不用温度有效性或旧训练开关删样本。

# ==================== 三、按完整实验组划分 ====================
VALIDATION_FRACTION = 0.15  # 完整实验组约 15% 用于训练期间验证。
TEST_FRACTION = 0.15  # 完整实验组约 15% 留作最终测试，其余用于训练。
TRAIN_LAYER_POLICY = "from_pre_breakoff_layer_to_wall_end"  # 从首次有效熔断前一有效加工层起，一直监督到本墙结束；all_valid_layers表示训练全部有效层。
NO_BREAKOFF_TRAINING_POLICY = "all_valid_layers"  # 没有熔断的训练墙保留有效层作稳定/顶丝对照；exclude表示不监督该墙。


def get_data_settings():  # 给公共划分和各模型提供同一份数据参数。
    settings = {name.lower(): str(value) if isinstance(value, Path) else value  # 路径转文本，方便保存到 JSON。
                for name, value in globals().items() if name.isupper()}  # 只收集上面的大写配置。
    settings["class_names"] = list(CLASS_NAMES)  # 返回独立列表，防止调用方改动公共类别顺序。
    settings["format_version"] = 4  # 保留现有数据与训练快照的格式版本。
    return settings  # 不扫描数据，不写划分，也不启动训练。


def validate_data_settings(settings):  # 划分与模型入口使用相同的数据参数校验。
    swap_seed = settings.get("holdout_swap_seed")  # 兼容未配置对调种子的旧结果。
    if swap_seed is not None and (type(swap_seed) is not int or swap_seed < 0):  # 0是有效种子。
        raise ValueError("HOLDOUT_SWAP_SEED必须是非负整数或None。")  # 防止无意选择另一份划分。
    if settings.get("train_layer_policy", "from_pre_breakoff_layer_to_wall_end") not in ("from_pre_breakoff_layer_to_wall_end", "all_valid_layers", "first_breakoff_plus_previous"):  # 最后一种只用于读取旧训练结果。
        raise ValueError("新划分的TRAIN_LAYER_POLICY使用from_pre_breakoff_layer_to_wall_end或all_valid_layers。")  # 规则在划分阶段执行。
    if settings.get("no_breakoff_training_policy", "all_valid_layers") not in ("all_valid_layers", "exclude"):  # 明确无熔断对照墙的处理方式。
        raise ValueError("NO_BREAKOFF_TRAINING_POLICY使用all_valid_layers或exclude。")  # 不在训练时猜测要选哪层。
    if not str(settings.get("data_root", "")).strip():  # 不把空地址解释为当前工作目录。
        raise ValueError("请在 code/data_config.py 中填写 DATA_ROOT。")  # 给出唯一的数据配置位置。
    if "dataset_name" in settings and not str(settings["dataset_name"]).strip():  # 新配置必须有名称；旧 TCN 权重快照缺少此字段时仍可恢复。
        raise ValueError("请在 code/data_config.py 中填写 DATASET_NAME。")  # 避免结果目录含义不明。
    ratio = settings.get("temperature_samples_per_image", 1)  # 兼容旧的一图一温度配置。
    if type(ratio) is not int or ratio not in (1, 2):  # 当前数据读取接口只支持这两种明确的比例。
        raise ValueError("TEMPERATURE_SAMPLES_PER_IMAGE 只能为 1 或 2；当前墙数据填写 2。")  # 不自动猜测温度分组。
    fps = settings.get("fps", 50)  # 保存的旧训练配置没有此项时保持其历史解释。
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:  # 帧率必须是有限正数。
        raise ValueError("请在 code/data_config.py 中将 FPS 设置为有限正数，当前墙数据为 25。")  # 不重新测算设备帧率。
    policy = settings.get("startup_policy", "exclude")  # 旧快照缺少此项时严格恢复原来的排除方式。
    if policy not in ("annotation_exclude", "context_only", "exclude"):  # 后两种仅用于恢复旧结果。
        raise ValueError('STARTUP_POLICY 使用 "annotation_exclude"；其余模式仅保留旧版兼容。')  # 新训练完全排除unknown输入。
    skip = settings.get("layer_skip_frames", 0)  # 只校验旧快照中的固定张数，新配置不再提供该参数。
    if type(skip) is not int or skip < 0:  # 排除负数、小数和布尔值。
        raise ValueError("旧快照中的layer_skip_frames必须为非负整数。")  # 新版不再要求填写固定数量。
    overrides = settings.get("layer_start_overrides", {})  # 旧训练快照没有例外表时沿用其默认排除数量。
    if not isinstance(overrides, dict):  # 逐墙规则必须使用字典。
        raise ValueError("LAYER_START_OVERRIDES 必须是按墙名、层号组织的字典。")  # 提示正确配置位置。
    for wall, layer_rules in overrides.items():  # 逐项检查人工指定的墙与层。
        if not isinstance(wall, str) or not wall or not isinstance(layer_rules, dict):  # 墙标识用于查表，不作为路径执行。
            raise ValueError("启动例外需要非空墙名和对应的层号字典。")  # 避免拼错结构后静默使用默认值。
        for layer, rule in layer_rules.items():  # 层号使用与配对清单相同的字符串，例如"9"。
            if not isinstance(layer, str) or not layer or not isinstance(rule, dict) or set(rule) - {"skip_frames", "first_image"}:  # 拒绝未知键名。
                raise ValueError("启动例外的层号应为字符串，规则只填写 skip_frames 和 first_image。")  # 防止未生效的拼写错误。
            if type(rule.get("skip_frames")) is not int or rule["skip_frames"] < 0:  # 排除数量不能是负数、小数或布尔值。
                raise ValueError(wall + " 第" + layer + "层的 skip_frames 必须为非负整数。")  # 定位具体例外。
            anchor = rule.get("first_image", "")  # 可以仅指定数量，填写文件名则进行额外核对。
            if not isinstance(anchor, str) or (anchor and (Path(anchor).name != anchor or "/" in anchor or "\\" in anchor)):  # 锚点只能是单个原图文件名。
                raise ValueError("first_image 请填写图片文件名，不填写目录。")  # 不通过配置改动原始文件。
    names = settings["class_names"]  # 检查统一类别顺序。
    if not isinstance(names, list) or len(names) < 2 or any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):  # 至少需要两个不同的文字类别。
        raise ValueError("CLASS_NAMES 至少包含两个不同的类别名称。")  # 防止输出编号重复。
    if any(name not in ("stable", "stubbing", "breakoff") for name in names):  # unknown 和 ignore 不属于模型输出类别。
        raise ValueError("CLASS_NAMES 使用 stable、stubbing、breakoff 中的类别。")  # 明确本项目支持的状态。
    if type(settings.get("shuffle_train_walls", False)) is not bool:  # 旧快照没有开关时保持原训练设置。
        raise ValueError("SHUFFLE_TRAIN_WALLS必须是True或False。")  # 统一各模型的公共设置。
    if type(settings["seed"]) is not int:  # 统一固定可重复的随机种子。
        raise ValueError("SEED 必须是整数。")  # 不允许不同模块各自解释种子。
    fractions = (settings["validation_fraction"], settings["test_fraction"])  # 训练集为两个留出集合之外的完整组。
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1 for value in fractions) or sum(fractions) >= 1:  # 保证三集合都有合理比例。
        raise ValueError("验证和测试比例均须大于 0，且两者之和小于 1。")  # 不通过拆散同墙来凑集合。
    if not str(settings.get("split_dir", "")).strip():  # 划分保存位置不能缺失。
        raise ValueError("请在 code/data_config.py 中填写 SPLIT_DIR。")  # 指向公共配置。
    source, target = Path(settings["data_root"]).resolve(), Path(settings["split_dir"]).resolve()  # 解析实际位置。
    if target == source or source in target.parents:  # 不把划分清单写入输入数据目录内部。
        raise ValueError("SPLIT_DIR 请放在 DATA_ROOT 之外，例如项目的 splits/ 中。")  # 保持配对数据独立。
