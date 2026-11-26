"""在现有有效窗口上构造短温度、长温度和功率三段输入。"""

import csv  # 读取支持 Excel 保存格式的功率表。
import io  # 直接读取训练时冻结的 CSV 内容。
from collections import defaultdict  # 按墙组织变更记录。
from datetime import datetime, timedelta, timezone  # 统一日期时间和在线相对秒数。
from numbers import Real  # 接受 Python 与 NumPy 的实数秒数。

import hashlib  # 固定图温、标签和功率表的输入版本。
from pathlib import Path  # 定位每面墙的采集清单。

import numpy as np  # 截取温度时间范围并拟合训练统计。
import torch  # 生成批次读取器使用的输入张量。

from data_utils import ordered_rows, read_csv  # 按公共格式读取配对清单。
from wall_dataset import WindowDataset as BaseWindowDataset
from wall_dataset import build_windows, load_sequences as load_base_sequences  # 先生成公共的历史窗口。



# ==================== 一、按采集时间组装三模态输入 ====================


def attach_signal_history(sequence, folder, power_table):
    wall = folder.name  # 功率表中的墙名使用数据文件夹名。
    manifest = folder / "frame_manifest.csv"
    temperatures = folder / "temperature.csv"
    frames = ordered_rows(read_csv(manifest), str(manifest))
    samples = ordered_rows(read_csv(temperatures), str(temperatures))
    pairing_path = folder / "pairing.csv"
    if pairing_path.is_file():  # 配对表保存未被表格软件缩写的完整采集时间。
        paired = ordered_rows(read_csv(pairing_path), str(pairing_path))
        fields = ("image_seq_idx", "sample_in_image", "temperature_index")
        if len(paired) != len(samples) or any(
            any(row.get(field) != sample.get(field) for field in fields)
            for row, sample in zip(paired, samples)
        ):
            raise ValueError(f"{wall} 的配对表与温度表采样编号不一致，请先核对数据整理结果。")
        samples = paired  # 只取时间，原温度数值仍来自 temperature.csv。
    try:
        image_times = np.asarray(
            [time_us(row.get("image_datetime") or row["source_datetime"]) for row in frames], dtype=np.int64,
        )
        temperature_times = np.asarray(
            [time_us(row["temperature_datetime"]) for row in samples], dtype=np.int64,
        ).reshape(sequence["values"].shape)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{wall} 的图像或温度采集时间缺失、格式错误或与配对数量不一致。") from error
    if len(image_times) != len(sequence["paths"]) or np.any(np.diff(image_times) < 0):
        raise ValueError(f"{wall} 的图像时间和图像清单数量不一致，或时间发生倒退。")
    sequence["image_times_us"] = image_times
    sequence["temperature_times_us"] = temperature_times
    sequence["power_schedule"] = bind_power_schedule(power_table, wall, sequence["layer_ids"], image_times)


def load_sequences(records, settings, power_payload):
    sequences, base_digest = load_base_sequences(records, settings)  # 复用数据、标签与有效帧的校验。
    power_table = parse_power_table(power_payload)
    digest = hashlib.sha256(base_digest.encode("ascii") + power_payload)  # 记录本次实际采用的功率表。
    for sequence in sequences.values():
        folder = Path(settings["data_root"]) / sequence["record"]["relative_folder"]
        attach_signal_history(sequence, folder, power_table)
        digest.update(sequence["image_times_us"].astype("<i8", copy=False).tobytes())
        digest.update(sequence["temperature_times_us"].astype("<i8", copy=False).tobytes())
    return sequences, digest.hexdigest()  # 完整配对时间也属于本次输入版本。


def temperature_indices(temperature_times_us, end_us, settings):
    times = temperature_times_us.reshape(-1)  # 调用者已限定在同一有效段的历史帧。
    short_start = max(0, len(times) - settings["short_temperature_steps"])  # 短路只读图像窗口内的温度组。
    short = np.flatnonzero(times[short_start:] <= end_us) + short_start
    lower = end_us - settings["long_history_us"]  # 长路以真实温度时间控制回看范围。
    long = np.flatnonzero((times > lower) & (times <= end_us))
    short = short[np.argsort(times[short], kind="stable")]  # 保留真实时间顺序，同一时间保持原配对次序。
    long = long[np.argsort(times[long], kind="stable")][-settings["long_temperature_steps"]:]
    return short, long


def pack_histories(values, indices, power_kw, settings, normalization):
    short, long = indices
    normalized = []
    for selected, size in zip((short, long), (settings["short_temperature_steps"], settings["long_temperature_steps"])):
        output = np.zeros(size, dtype=np.float32)  # 标准化后左侧补零，实测点右对齐。
        if selected.size:
            output[-selected.size:] = (values[selected] - normalization["mean"]) / normalization["std"]
        normalized.append(output)
    # 公共加载器只打包传输，模型入口会按保存的长度拆成三个独立输入。
    return np.concatenate((*normalized, power_kw.astype(np.float32)))[None, :]


def window_temperature(sequence, window, settings):
    start = window["effective_start_seq_idx"]  # 当前有效连续段起点，含短窗之前可读的历史。
    end = window["end_seq_idx"]
    end_us = sequence["image_times_us"][end]
    times = sequence["temperature_times_us"][start:end + 1]
    indices = temperature_indices(times, end_us, settings)
    return start, end_us, indices


def fit_normalization(sequences, training_windows, settings):
    used = {}  # 只标记训练窗口的实际采样点，包含长、短两路。
    ratio = settings["temperature_samples_per_image"]
    for window in training_windows:
        wall = window["sequence_id"]
        sequence = sequences[wall]
        if wall not in used:
            used[wall] = np.zeros(sequence["values"].size, dtype=bool)
        start, _, indices = window_temperature(sequence, window, settings)
        for selected in indices:
            used[wall][start * ratio + selected] = True  # 同一点出现在多个重叠窗口时只标记一次。
    samples = []
    for wall, mask in used.items():
        sequence = sequences[wall]
        values = sequence["values"].reshape(-1)[mask]
        source = sequence["temperature_source_indices"]
        if source is not None:
            ids = source.reshape(-1)[mask]  # 同一原始采样被重复配对时再次去重。
            known = np.flatnonzero(ids != "")
            _, first = np.unique(ids[known], return_index=True)
            values = values[np.concatenate((known[first], np.flatnonzero(ids == "")))]
        samples.append(values)
    all_points = np.concatenate(samples) if samples else np.empty(0)
    excluded = settings["temperature_stat_exclude_value"]
    atol = settings["temperature_stat_exclude_atol"]
    fitted = all_points[~np.isclose(all_points, excluded, rtol=0.0, atol=atol)]  # 仅统计时排除 799.9。
    if fitted.size == 0:
        raise ValueError("训练窗口中没有可用于标准化的温度点，请检查配对及 799.9 占位值。")
    std = float(fitted.std())  # 与旧版使用同样的总体标准差。
    return {
        "mean": float(fitted.mean()), "std": max(std, 1e-6),
        "unique_training_points": int(all_points.size), "statistics_points_used": int(fitted.size),
        "excluded_799_9_points": int(all_points.size - fitted.size), "excluded_value": excluded,
        "excluded_value_atol": atol, "std_ddof": 0, "std_floor_applied": std < 1e-6,
        "fit_scope": "unique_points_in_training_short_or_long_temperature_history",
        "input_799_9_policy": "keep_and_standardize", "temperature_padding": "zero_after_standardization",
        "power_input_units": "kW", "power_source": "setpoint",
    }


class WindowDataset(BaseWindowDataset):
    def __init__(self, sequences, windows, settings, normalization):
        super().__init__(sequences, windows, settings, normalization)
        self.power_offsets = power_offsets_us(settings)  # 全部窗口复用同一组相对时间位置。

    def __getitem__(self, index):
        images, _, label, index = super().__getitem__(index)  # 复用旧版的图像读取、压缩传输和标签接口。
        window = self.windows[index]
        sequence = self.sequences[window["sequence_id"]]
        start, end_us, indices = window_temperature(sequence, window, self.settings)
        values = sequence["values"][start:window["end_seq_idx"] + 1].reshape(-1)
        context = f"{window['sequence_id']} 第 {window['layer_id']} 层"
        power_kw = power_history(sequence["power_schedule"], end_us, sequence["image_times_us"][start], self.power_offsets, context)
        histories = pack_histories(values, indices, power_kw, self.settings, self.normalization)
        return images, torch.from_numpy(histories), label, index





# ==================== 二、功率设定时间线 ====================


_CLOCK_ORIGIN = datetime(1970, 1, 1)  # 无时区的采集表使用本地时钟。
_LOCAL_TIMEZONE = timezone(timedelta(hours=8))  # 带时区的时间统一转为当前数据采用的北京时间。


def time_us(value):
    if isinstance(value, Real) and not isinstance(value, bool):
        if not np.isfinite(value):
            raise ValueError("时间必须是有限秒数。")
        return round(float(value) * 1_000_000)  # 在线可使用同一时钟的相对秒数。
    moment = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).strip())
    if moment.tzinfo is not None:
        moment = moment.astimezone(_LOCAL_TIMEZONE).replace(tzinfo=None)
    delta = moment - _CLOCK_ORIGIN  # 整数微秒避免相同生效时间因浮点误差漏配。
    return delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds


def check_watts(value):
    watts = float(value)  # CSV 内容为文本，在线接口也支持数字。
    if not np.isfinite(watts) or watts < 0:
        raise ValueError("功率设定值必须是非负的有限瓦数。")
    return watts


def parse_power_table(payload):
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))  # 兼容 UTF-8 BOM。
    required = {"wall_id", "layer_id", "effective_at", "power_setpoint_w"}
    if not required.issubset(reader.fieldnames or ()):
        raise ValueError("power_settings.csv 需要 wall_id、layer_id、effective_at、power_setpoint_w 四列。")
    rows = defaultdict(list)  # 某面墙可登记多个层首或层内变更。
    for line, row in enumerate(reader, start=2):
        wall = (row["wall_id"] or "").strip()
        layer = (row["layer_id"] or "").strip()
        stamp = (row["effective_at"] or "").strip()
        try:
            if not wall or not layer:
                raise ValueError("墙名和层号均需填写。")
            moment = time_us(stamp) if stamp else None  # 空白在绑定该层时解析为首帧时间。
            watts = check_watts(row["power_setpoint_w"])
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(f"power_settings.csv 第 {line} 行（{wall} 第 {layer} 层）：{error}") from error
        rows[wall].append((layer, moment, watts))
    return dict(rows)


class PowerSchedule:
    def __init__(self, times=(), watts=()):
        self.times = list(times)  # 生效时间严格递增，在线可继续登记新的设定。
        self.watts = list(watts)  # 与时间一一对应，保持原始 W 值。

    def append(self, moment_us, watts):
        if self.times and moment_us <= self.times[-1]:
            raise ValueError("同一面墙的功率生效时间必须严格递增。")
        self.times.append(moment_us)
        self.watts.append(watts)

    def sample(self, moments_us, context):
        indices = np.searchsorted(self.times, moments_us, side="right") - 1  # 生效瞬间即可读取新值。
        if (indices < 0).any():
            raise ValueError(f"{context}缺少已生效的设定功率，请补齐功率记录。")
        return np.asarray(self.watts, dtype=np.float64)[indices]


def bind_power_schedule(table, wall, layer_ids, image_times_us):
    bounds = {}  # 层首和末帧以原始采集顺序为准，含 unknown 启动区间。
    for layer, moment in zip(layer_ids, image_times_us):
        if layer not in bounds:
            bounds[layer] = [moment, moment]
        bounds[layer][1] = moment
    events = []
    for layer, timestamp, watts in table.get(wall, ()):
        if layer not in bounds:
            raise ValueError(f"{wall} 的功率表第 {layer} 层在图像清单里不存在。")
        start, end = bounds[layer]
        moment = start if timestamp is None else timestamp  # 空白项从该层开始沿用。
        if not start <= moment <= end:
            raise ValueError(f"{wall} 第 {layer} 层的 effective_at 不在该层的图像采集时间内。")
        events.append((moment, layer, watts))
    schedule = PowerSchedule()
    for moment, layer, watts in sorted(events):
        if schedule.times and schedule.times[-1] == moment:
            raise ValueError(f"{wall} 第 {layer} 层有重复的功率生效时刻。")
        schedule.append(moment, watts)
    schedule.sample(image_times_us[:1], f"{wall} 第 {layer_ids[0]} 层")  # 在训练前检查整墙初始值。
    return schedule


def power_offsets_us(settings):
    steps = np.arange(settings["power_steps"] - 1, -1, -1)  # 窗口从旧到新，最后一个时刻是图像末帧。
    return np.rint(steps * (1_000_000 / settings["fps"])).astype(np.int64)


def power_history(schedule, end_us, segment_start_us, offsets_us, context):
    query = np.maximum(end_us - offsets_us, segment_start_us)  # 段首左侧按本段起点的设定值补齐。
    return schedule.sample(query, context) / 1000.0  # 功率以 kW 入模，恒功率数据也可直接使用。
