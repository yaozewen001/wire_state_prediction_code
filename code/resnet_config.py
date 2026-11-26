"""四个 ResNet34 模型共用的配置转换和参数校验。"""

import math  # 检查数值范围。
from pathlib import Path  # 解析项目相对路径。

from data_config import PROJECT_DIR, get_data_settings, validate_data_settings  # 共用数据地址和划分。


def resnet_window_settings(values, model_variant):  # 两种ResNet34模型共用配置快照字段。
    settings = {name.lower(): str(value) if isinstance(value, Path) else value  # 将路径转为JSON可保存的文本。
                for name, value in values.items() if name.isupper()}  # 仅收集上方大写配置。
    settings.update(get_data_settings())  # 数据地址、类别和整墙划分来自公共data_config.py。
    settings.update(format_version=9, training_order="wall_layer_time", model_variant=model_variant,  # 与现有训练和评估快照格式兼容。
                    temperature_normalization_policy="train_unique_points_exclude_799_9_keep_inputs")  # 保存本模型独立的温度统计规则。
    if str(settings["resnet_weights_file"]).strip():  # 相对预训练路径固定相对于项目根目录，而不是IDE工作目录。
        path = Path(settings["resnet_weights_file"]).expanduser()  # 同时允许明确的绝对路径。
        settings["resnet_weights_file"] = str(path if path.is_absolute() else PROJECT_DIR / path)  # Linux服务器与Windows使用相同解析规则。
    return settings  # 训练、评估和回放共同使用这一入口。


def validate_resnet_window_settings(settings, model_name, image_width_key):  # 检查局部双分支模型的共同配置。
    validate_data_settings(settings)  # 检查数据地址、类别、帧率和划分比例。
    if settings.get("model_name") != model_name:  # 防止误用其他模型的配置和结果。
        raise ValueError(f"本目录使用 {model_name}，请检查运行配置。")
    if settings.get("temperature_input_policy") != "raw":  # 旧过滤策略可能先删除799.9及其配对图像。
        raise ValueError('请将code/data_config.py中的TEMPERATURE_INPUT_POLICY设为"raw"，确保799.9保留入模。')  # 明确唯一修改位置。
    if settings["image_feature_dim"] != 512:  # ResNet34最后一组残差块固定输出512通道。
        raise ValueError("IMAGE_FEATURE_DIM固定为512；调整融合维数请改{}。".format(image_width_key.upper()))  # 提示本模型的输出宽度配置。
    for name in ("batch_size", "epochs", "window_size", "window_stride", "patience", "cpu_threads", image_width_key, "temperature_tcn_width", "classifier_hidden_dim"):  # 这些参数必须为正整数。
        if type(settings[name]) is not int or settings[name] < 1:  # 排除布尔值、小数、零和负数。
            raise ValueError(name + "必须是正整数。")  # 指明具体错误键。
    if type(settings.get("exclude_first_layer", False)) is not bool:  # 旧训练快照缺少此项时仍按旧范围评估。
        raise ValueError("EXCLUDE_FIRST_LAYER必须是True或False。")  # 避免字符串被误判为开启。
    if settings.get("train_shuffle_unit", "none") not in ("none", "wall", "layer", "layer_number"):  # layer_number只用于读取已保存的评估配置。
        raise ValueError("TRAIN_SHUFFLE_UNIT使用none、wall或layer。")
    for name in ("layer_event_reports", "save_epoch_checkpoints"):  # 新报告和保存候选权重可单独开关。
        if type(settings.get(name, False)) is not bool:
            raise ValueError(name.upper() + "必须是True或False。")
    eval_batch_size = settings.get("evaluation_batch_size", settings["batch_size"])  # 旧配置沿用原批量。
    if type(eval_batch_size) is not int or eval_batch_size < 1:  # 与训练批量采用相同的正整数要求。
        raise ValueError("EVALUATION_BATCH_SIZE必须是正整数。")  # 提示应修改的配置项。
    if type(settings["future_frames"]) is not int or settings["future_frames"] < 0:  # 未来帧数允许0，但不能为负。
        raise ValueError("FUTURE_FRAMES必须是非负整数。")  # 不偷偷改变预测提前量。
    if settings["image_size"] is not None and (type(settings["image_size"]) is not int or settings["image_size"] < 32):  # 保证ResNet下采样后的空间尺寸可用。
        raise ValueError("IMAGE_SIZE填写None或至少32的整数。")  # None仍支持原图尺寸。
    validate_tcn_settings(settings, "temperature", 2)  # 两种图像分支共用两路温度TCN。
    mode = settings["image_sampling"]  # 选图规则只影响图像分支。
    if mode not in ("all", "uniform", "custom"):  # 防止拼错模式后静默均匀选图。
        raise ValueError("IMAGE_SAMPLING使用all、uniform或custom。")  # 列出全部可选值。
    if mode == "uniform" and (type(settings["image_steps"]) is not int or not 1 <= settings["image_steps"] <= settings["window_size"]):  # 均匀采样不能重复取同一位置。
        raise ValueError("IMAGE_STEPS必须介于1和WINDOW_SIZE之间。")  # all模式不使用该参数。
    if mode == "custom":  # 检查人工选图位置。
        values = settings["custom_frame_positions"]  # 保留从1开始的用户编号。
        if not isinstance(values, (list, tuple)) or not values or any(type(value) is not int or not 1 <= value <= settings["window_size"] for value in values) or list(values) != sorted(set(values)):  # 必须按时间严格递增。
            raise ValueError("CUSTOM_FRAME_POSITIONS需要窗口内从1开始、递增且不重复的位置。")  # 不跨窗或读取未来图像。
    for name in ("learning_rate", "weight_decay", "dropout", "temperature_stat_exclude_atol", "progress_refresh_seconds", "plain_log_interval_seconds"):  # 检查有限数值。
        value = settings[name]  # 布尔值不能代替浮点超参数。
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):  # 排除NaN、无穷值和字符串。
            raise ValueError(name + "必须是有限数值。")  # 在打开数据之前报错。
    if settings["learning_rate"] <= 0 or settings["weight_decay"] < 0 or not 0 <= settings["dropout"] < 1 or settings["temperature_stat_exclude_atol"] < 0:  # 检查合法范围。
        raise ValueError("学习率需大于0，权重衰减和温度容差需非负，DROPOUT需在[0,1)内。")  # 不自动替换用户配置。
    excluded_value = settings["temperature_stat_exclude_value"]  # 该值只用于计算训练温度统计。
    if isinstance(excluded_value, bool) or not isinstance(excluded_value, (int, float)) or not math.isfinite(excluded_value):  # 不允许用空值或无穷值进行匹配。
        raise ValueError("TEMPERATURE_STAT_EXCLUDE_VALUE必须是有限数值，当前填写799.9。")  # 明确当前数据的占位值。
    if settings["progress_refresh_seconds"] <= 0 or settings["plain_log_interval_seconds"] <= 0:  # 日志刷新间隔必须为正数。
        raise ValueError("进度和日志间隔必须大于0秒。")  # 避免刷屏。
    for name in ("resnet_pretrained", "freeze_image_backbone", "freeze_backbone_batch_norm", "image_imagenet_normalization"):  # 明确预训练和微调开关。
        if type(settings[name]) is not bool:  # 拒绝字符串True/False。
            raise ValueError(name + "必须是True或False。")  # IDE直接编辑布尔值。
    if not settings["resnet_pretrained"] and str(settings["resnet_weights_file"]).strip():  # 从头训练不能同时要求加载预训练文件。
        raise ValueError("使用RESNET_WEIGHTS_FILE时，请将RESNET_PRETRAINED设为True。")  # 防止权重路径被悄悄忽略。
    if settings["device"] not in ("auto", "cuda", "cpu") or settings["progress_mode"] not in ("auto", "bar", "plain", "off"):  # 公共运行器支持的选项。
        raise ValueError("DEVICE使用auto/cuda/cpu，PROGRESS_MODE使用auto/bar/plain/off。")  # 明确合法选项。
    if type(settings["num_workers"]) is not int or settings["num_workers"] < 0 or settings["evaluation_split"] not in ("all", "train", "val", "test", "both"):  # 检查加载和评估范围。
        raise ValueError("NUM_WORKERS为非负整数；EVALUATION_SPLIT使用all/train/val/test/both。")  # 不混用训练与测试集合。
    source = Path(settings["data_root"]).resolve()  # 原始模型数据只读。
    for name in ("split_dir", "output_root"):  # 生成文件不能写入数据目录。
        target = Path(settings[name]).resolve()  # 兼容相对路径。
        if target == source or source in target.parents:  # 阻止原始数据与结果混放。
            raise ValueError(name + "必须位于DATA_ROOT之外。")  # 保留原始配对数据。


def validate_tcn_settings(settings, modality, count):  # 校验某个模态的并行TCN参数。
    name = modality + "_kernel_sizes"  # 卷积核大小决定每条分支。
    sizes = settings[name]  # 同时接受配置和JSON快照。
    if (not isinstance(sizes, (list, tuple)) or len(sizes) != count
            or any(type(value) is not int or value < 1 or value % 2 == 0 for value in sizes)
            or len(set(sizes)) != count):  # 需要指定数量且互不相同的正奇数核。
        raise ValueError(name + "需要{}个不同的正奇数。".format(count))  # 提示具体配置键。
    name = modality + "_tcn_channels"  # 通道列表决定残差块数量。
    channels = settings[name]  # 每个数对应一个残差块的输出通道。
    if not isinstance(channels, (list, tuple)) or not channels or any(type(value) is not int or value < 1 for value in channels):  # 网络不能为空。
        raise ValueError(name + "必须是非空正整数列表。")  # 保持模型可恢复性。
