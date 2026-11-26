# -*- coding: utf-8 -*-
"""在 IDE 中运行：整理一面或多面墙，或将人工修改的区间标签同步为逐帧标签。"""

import bisect  # 按时间查找图像到达时已经取得的温度。
import csv  # 读取采集标注并输出模型所需表格。
import hashlib  # 核对复制结果，并记录标签同步关系。
import json  # 保存整理参数和结果摘要。
import math  # 判断温度和参数是否为有限数值。
import re  # 从采集文件名读取日期。
import shutil  # 只复制图片和来源记录，不移动原文件。
import tempfile  # 在临时目录中完成一面墙，再发布完整结果。
from collections import Counter  # 统计状态、未配对原因和可训练帧数。
from datetime import datetime, timedelta, timezone  # 处理时间匹配及跨午夜记录。
from pathlib import Path  # 避免依赖 IDE 的工作目录。

# ==================== 一、日常只修改这一段 ====================
PROJECT_DIR = Path(__file__).resolve().parents[1]  # 此文件应放在项目的 code/ 中。
MODE = "build"  # 新数据配对前，请填写来源目录、墙号和温度文件映射。
SOURCE_IMAGE_ROOT = PROJECT_DIR / "data" / "raw" / "wall_photo"  # 可改成自己的裁剪图像与标注目录。
TEMPERATURE_ROOT = PROJECT_DIR / "data" / "datasets" / "wall_temperature_csv"  # 当前原始温度 CSV 来源，与图片统一保存在本项目 data 下。
OUTPUT_DATA_ROOT = PROJECT_DIR / "data" / "datasets" / "wall_dataset"  # 整理后的墙数据总目录，不在名称中添加版本号。
REFRESH_WALL_IDS = []  # 留空时更新输出目录中所有已有墙。
REFRESH_MIN_VALID_TEMPERATURE = 800.0  # 仅兼容旧调用参数；本版不按阈值排除数据，799.9按原值输入。
REPAIR_WALL_IDS = []  # 旧修复模式仅处理本列表列出的墙。
REPAIR_BACKUP_ROOT = PROJECT_DIR / "records" / "dataset_repairs"  # 修复前完整备份CSV，并保存旧新编号对照；不备份或改写图片。
WALL_IDS = []  # 填写本次要配对的墙目录名。
TEMPERATURE_FILES = {}  # 键为墙目录名，值为该墙对应的源温度 CSV 文件名。
IMAGE_FPS = 25  # 用户确认的图像帧率，不重新估计实际帧率。
TEMPERATURE_HZ = 50  # 用户确认的原始温度采样率。
TEMPERATURE_SAMPLES_PER_IMAGE = 2  # 每张 25 fps 图像关联两条 50 Hz 原始温度，不平均。
TIME_OFFSET_SECONDS = 0.0  # 正值表示：图片时刻加上此值，再到温度时间轴查找。
MAX_TIME_ERROR_SECONDS = 0.08  # 两温度与图像匹配时刻的最大允许差；顺延超限时按时间重定位，图片仍保留。
INVALID_TEMPERATURE_VALUES = []  # 不按温度值过滤标签或窗口；799.9按原值保留。
VERIFY_COPIES = True  # 复制后核对图片哈希；不分析或改变像素、尺寸、帧率。
# ==================== 二、以下通常不需要修改 ====================

STATE_ALIASES = {"稳定": "stable", "稳态": "stable", "顶丝": "stubbing", "熔断": "breakoff", "待确认": "unknown", "unknow": "unknown", "忽略": "ignore", "-1": "ignore", "": "unknown"}  # unknow明确规范为unknown；其他不明拼写不猜类别。
VALID_STATES = {"stable", "stubbing", "breakoff", "unknown", "ignore"}  # unknown 和 ignore 不是训练类别。
MANIFEST_FIELDS = ["seq_idx", "image_name", "segment_id", "layer_id", "source_frame_number", "source_datetime", "use_for_training", "temperature_available", "temperature_start_seq_idx", "temperature_end_seq_idx"]  # 模型清单及追溯字段。
PAIR_FIELDS = ["seq_idx", "layer_id", "image_seq_idx", "image_name", "sample_in_image", "temperature_index", "temperature_c", "temperature_datetime", "image_datetime", "pairing_method", "time_error_s", "time_alignment_ok", "source_frame_number", "segment_id", "layer_anchor_datetime", "layer_anchor_first_temperature_index", "layer_anchor_second_temperature_index", "sample_available", "temperature_available", "source_state", "use_for_training"]  # 逐帧配对依据。
INTERVAL_FIELDS = ["start_seq_idx", "end_seq_idx", "state", "layer_id", "segment_id", "start_image", "end_image", "start_source_frame_number", "end_source_frame_number", "use_for_training", "temperature_available"]  # 前三列可编辑，其他列供追溯。


def read_csv(path, required):  # 按列名读取 UTF-8 CSV，避免靠列位置猜含义。
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:  # 兼容 Excel 的 UTF-8 BOM。
        reader = csv.DictReader(handle)  # 首行作为表头。
        missing = set(required) - set(reader.fieldnames or [])  # 检查必要列。
        if missing:  # 缺列时立即停止，不生成错误配对。
            raise ValueError(f"{path} 缺少列：{sorted(missing)}")  # 明确告诉用户修正哪个表。
        rows = list(reader)  # 一次读取一面墙的文字记录，不加载图像像素。
    if not rows:  # 空表不能形成数据集。
        raise ValueError(f"{path} 没有数据行。")  # 给出文件路径。
    return rows  # 保持原记录，调用方再明确排序。


def write_csv(path, rows, fields):  # 写入 CSV UTF-8；同步时使用原子替换。
    path = Path(path)  # 统一使用路径对象。
    path.parent.mkdir(parents=True, exist_ok=True)  # 创建必要目录。
    with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", dir=path.parent, prefix=".writing_", suffix=".csv", delete=False) as handle:  # 在同一目录暂存。
        temporary_path = Path(handle.name)  # 保存临时文件路径。
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")  # 只输出约定字段。
        writer.writeheader()  # 写表头。
        writer.writerows(rows)  # 写全部数据行。
    temporary_path.replace(path)  # 写完后才替换生成文件，不修改输入文件。


def write_json(path, value):  # 使用临时文件保存结构化记录。
    path = Path(path)  # 统一路径对象。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保父目录存在。
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".writing_", suffix=".json", delete=False) as handle:  # 暂存完整内容。
        temporary_path = Path(handle.name)  # 用于发布文件。
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)  # 不输出 NaN 或无穷温度。
    temporary_path.replace(path)  # 只替换本程序的生成记录。


def sha256(path):  # 分块计算哈希，避免将大图全部放进内存。
    digest = hashlib.sha256()  # 建立内容摘要。
    with Path(path).open("rb") as handle:  # 只读输入文件。
        for block in iter(lambda: handle.read(1024 * 1024), b""):  # 每次最多读取 1 MB。
            digest.update(block)  # 追加当前内容。
    return digest.hexdigest()  # 返回可用于核对的字符串。


def normalize_state(value):  # 统一状态名称，不根据温度修改图像真实状态。
    text = str(value).strip().lower()  # 清理 Excel 输入的空白。
    text = STATE_ALIASES.get(text, text)  # 中文名称转为固定英文名称。
    if text not in VALID_STATES:  # 拼写错误不能自动归入稳定。
        raise ValueError(f"不支持的状态 {value!r}，请使用稳定、顶丝、熔断、unknown 或 ignore。")  # 明确合法输入。
    return text  # 返回用于标签表的状态。


def flag(value, default=True):  # 读取标注中的是否参与训练开关。
    text = str(value).strip().lower()  # 兼容文字和数字开关。
    if not text:  # 空列按调用方明确给出的默认值解释。
        return default  # 来源没有该列时允许正常标注参与训练。
    if text in {"1", "1.0", "true", "yes", "是"}:  # 接受常见真值。
        return True  # 表示允许。
    if text in {"0", "0.0", "false", "no", "否"}:  # 接受常见假值。
        return False  # 表示排除。
    raise ValueError(f"无法识别开关 {value!r}，请填写 0 或 1。")  # 不静默误读。


def integer(value, description):  # 严格解析帧号及区间端点。
    try:  # 允许 CSV 中的整数字符串。
        return int(str(value).strip())  # 不接受带小数的错误帧号。
    except (TypeError, ValueError) as error:  # 将底层错误换成可理解的提示。
        raise ValueError(f"{description} 必须是整数，目前为 {value!r}。") from error  # 指明出错字段。


def image_datetime(row):  # 使用图像标注中已经保存的采集时间。
    value = str(row["source_datetime"]).strip()  # 不采用旧的绝对图片地址推断时间。
    try:  # 支持 YYYY-MM-DD HH:MM:SS.fff。
        result = datetime.fromisoformat(value)  # 把日期和时间一起解析，避免跨日混淆。
    except ValueError as error:  # 表格软件可能只保留分、秒和一位小数。
        image_name = str(row.get("output_image", ""))  # 从原始图片名称恢复时分秒和毫秒。
        image_match = re.fullmatch(r"frame_\d+_(\d{6})_(\d{3})\.[^.]+", image_name, flags=re.IGNORECASE)  # 只接受有完整时刻的文件名。
        date_match = re.fullmatch(r"wall_(\d{8})_\d{6}", str(row.get("wall_id", "")))  # 从该图片所属墙编号取得采集日期。
        short_match = re.fullmatch(r"(\d{1,2}):(\d{2})\.(\d)", value)  # 只恢复明确的分秒及十分之一秒记录。
        if not image_match or not date_match or not short_match:  # 不能唯一核实时拒绝猜测时间。
            raise ValueError(f"图片 {image_name} 的 source_datetime 无法安全恢复：{value!r}") from error  # 给出具体图片。
        result = datetime.strptime(date_match.group(1) + image_match.group(1) + image_match.group(2), "%Y%m%d%H%M%S%f")  # 直接使用原文件名中的时刻。
        recorded = int(short_match.group(1)) * 60 + int(short_match.group(2)) + int(short_match.group(3)) / 10  # 原表保留的分秒。
        actual = result.minute * 60 + result.second + result.microsecond / 1000000  # 文件名中对应的分秒。
        if min(abs(recorded - actual), abs(recorded - actual + 3600), abs(recorded - actual - 3600)) > 0.051:  # 允许四舍五入及跨小时。
            raise ValueError(f"图片 {image_name} 的文件名时间与标注时间不符：{value!r}") from error  # 冲突时停止。
    if result.tzinfo is not None:  # 当前采集表使用不带时区的同电脑时间。
        raise ValueError("source_datetime 请统一使用不带时区的本机采集时间。")  # 不擅自转换未知时区。
    return result  # 与温度时间比较。


def read_temperature(path):  # 原温度文件保持原样，按其中的 time 列建立时间轴。
    match = re.search(r"(\d{8})_(\d{6})", path.stem)  # 例如 serial_50hz_YYYYMMDD_HHMMSS。
    if not match:  # 只有时分秒的 CSV 必须能明确采集日期。
        raise ValueError(f"温度文件名需要包含 YYYYMMDD_HHMMSS：{path.name}")  # 防止跨日错误配对。
    day = datetime.strptime(match.group(1), "%Y%m%d").date()  # 取得第一天的日期。
    result, seen, previous = [], set(), None  # 保存采样记录、原始序号和上一时刻。
    for row in read_csv(path, ["index", "time", "value"]):  # 按已核对的实际表头读取。
        raw_index = integer(row["index"], "温度 index")  # 保留原始温度序号。
        if raw_index in seen:  # 重复序号不应被当作不同采样点。
            raise ValueError(f"{path.name} 存在重复 index={raw_index}。")  # 明确输入问题。
        seen.add(raw_index)  # 记录该采样点。
        text = row["time"].strip()  # 温度软件记录的时间。
        try:  # 同时支持只有时间和完整日期的表。
            current = datetime.fromisoformat(text) if "-" in text else datetime.combine(day, datetime.strptime(text, "%H:%M:%S.%f" if "." in text else "%H:%M:%S").time())  # 解析到毫秒。
            value = float(row["value"])  # 原始温度不进行平滑或插值。
        except ValueError as error:  # 缺值或坏时间不能被伪造为 0。
            raise ValueError(f"{path.name} 的 index={raw_index} 时间或温度无法解析。") from error  # 保留定位依据。
        if current.tzinfo is not None:  # 保证双方使用同一类时间。
            raise ValueError("温度 time 请使用与图片一致的本机时间格式。")  # 不自动猜时区。
        if previous is not None and current < previous:  # 检查跨午夜或意外倒序。
            if "-" not in text and previous - current > timedelta(hours=12):  # 午夜日期需要增加一天。
                day += timedelta(days=1)  # 更新后续采样日期。
                current += timedelta(days=1)  # 修正当前采样日期。
            else:  # 小幅倒序不能被解释为跨午夜。
                raise ValueError(f"{path.name} 的时间在 index={raw_index} 处倒退。")  # 不擅自重排坏记录。
        if previous is not None and current == previous:  # 同一时间出现多个样本时无法唯一配对。
            raise ValueError(f"{path.name} 存在重复时间 {current}。")  # 请求输入记录明确化。
        result.append({"index": raw_index, "time": current, "value": value})  # 保留原始温度及采样位置。
        previous = current  # 更新前一个时间。
    return result  # 不根据采集时间重新估计 50 Hz。


def layer_anchors(source, rows):  # 读取人眼确认的每层首图时间，各层独立锚定。
    path = source / "layers.csv"  # 裁剪标注程序已生成的层起止表。
    anchors = {}  # 以层号保存人工确认起点。
    if path.is_file():  # 优先使用明确登记的层边界。
        for row in read_csv(path, ["layer_id", "start_frame_number", "start_datetime"]):  # 不从温度突变自动猜层起点。
            layer = row["layer_id"].strip()  # 与逐图标注中的层号一致。
            if layer in anchors:  # 一层不能有两个互相冲突的首图起点。
                raise ValueError(f"{path} 层号 {layer} 重复。")  # 不随意覆盖人工记录。
            anchor_source = "layers.csv/start_datetime"  # 正常优先使用人工层表中的完整时间。
            try:  # 兼容层表时间被表格软件保存成分秒格式的情况。
                anchor_time = image_datetime({"source_datetime": row["start_datetime"], "output_image": row.get("start_image", layer)})  # 不猜测缺失的日期小时。
            except ValueError:  # 仅能从同一人工首图的逐图标注恢复完整时间。
                matches = [item for item in rows if item["layer_id"].strip() == layer and integer(item["source_frame_number"], "原始帧号") == integer(row["start_frame_number"], "层首帧号") and item["output_image"] == row.get("start_image", "")]  # 同时核对层号、帧号、文件名。
                if len(matches) != 1:  # 不能唯一确认时停止，绝不换用其他图片的时间。
                    raise ValueError(f"{path} 的层 {layer} 起点时间格式无效，且无法唯一找到对应首图；请恢复完整 start_datetime。")  # 提示修正原始层表。
                anchor_time = image_datetime(matches[0])  # 读取已确认首图的完整采集时间。
                anchor_source = "frame_annotations.csv/exact_layer_start_image"  # 在整理记录中保留实际时间来源。
                print(f"提示：{source.name} 层 {layer} 的层表时间格式无效，使用同一首图逐图标注中的完整时间 {anchor_time}。", flush=True)  # 不修改原始表格。
            anchors[layer] = {
                "frame": integer(row["start_frame_number"], "层首帧号"),  # 层首原始帧号。
                "time": anchor_time,  # 原人工首图的完整采集时间。
                "image": row.get("start_image", ""),  # 层首图片名称。
                "source": anchor_source,  # 记录时间来自层表还是同一首图的逐图标注。
            }  # 保存原始时间及帧号。
    for row in rows:  # 没有层表的来源也可以用人工标注序列的首图。
        layer = row["layer_id"].strip()  # 同一层只记录第一次出现的位置。
        if layer not in anchors:  # 层表缺少的层使用其第一张标注图。
            anchors[layer] = {"frame": integer(row["source_frame_number"], "层首帧号"), "time": image_datetime(row), "image": row["output_image"], "source": "first_annotated_image"}  # 不推断实际帧率。
    return anchors  # 各层的人工起点相互独立。


def pair_start_before(times, target):  # 按时间定位一组两个相邻温度样本。
    end = bisect.bisect_right(times, target) - 1  # 找不晚于匹配时刻的最近温度。
    return max(0, min(end - 1, len(times) - 2))  # 正常为最近两条；记录边界不足时仍保留真实采样并标为不可用。


def plan_wall(wall_id, source, temperature_path):  # 每层重新对齐，层内每图顺延两点，必要时重新按时间定位。
    rows = read_csv(source / "frame_annotations.csv", ["source_datetime", "source_frame_number", "output_image", "state", "layer_id", "segment_id"])  # 所有已标注图均进入复制计划。
    rows.sort(key=lambda row: integer(row["source_frame_number"], "source_frame_number"))  # 保持原始图片顺序。
    temperatures = read_temperature(temperature_path)  # 原始 50 Hz 数据不平滑、不插值。
    if len(temperatures) < 2:  # 无法从不足两条的记录构造两个真实温度。
        raise ValueError(f"{temperature_path} 不足两个温度点，请先补齐真实温度来源。")  # 不补造采样。
    times = [sample["time"] for sample in temperatures]  # 按实际记录时间查找。
    anchors = layer_anchors(source, rows)  # 人工确认的层首时间优先。
    paired, issues, seen_names, seen_frames = [], [], set(), set()  # 不再因为配对时间数量不等于二而丢弃图片。
    previous_time, previous_layer, previous_key, previous_frame, previous_start, segment_count = None, None, None, None, None, 0  # 管理层内顺延及真实断点。
    for row in rows:  # 每个标注行必须对应一个输出图像。
        name = row["output_image"].strip()  # 不使用已经过期的来源绝对路径。
        raw_frame = integer(row["source_frame_number"], "source_frame_number")  # 原始帧号可识别裁剪时省略的帧。
        if Path(name).name != name or name in {"", ".", ".."} or name in seen_names or raw_frame in seen_frames:  # 禁止重复或非法标注。
            raise ValueError(f"{wall_id} 有重复或非法图片名/帧号：{name}")  # 不重复复制。
        seen_names.add(name)  # 记录已核对的图像名。
        seen_frames.add(raw_frame)  # 记录原始帧号。
        if not (source / name).is_file():  # 缺少已标注裁剪文件时立即明确报错。
            raise FileNotFoundError(f"标注图片不存在：{source / name}")  # 不静默跳过标注图。
        current = image_datetime(row)  # 完整时间从原标注重建，修复派生表被 Excel 缩短的时间。
        if previous_time is not None and current <= previous_time:  # 采集顺序必须正确。
            raise ValueError(f"{name} 的采集时间重复或倒退。")  # 不猜测日期。
        target = current + timedelta(seconds=TIME_OFFSET_SECONDS)  # 用户确认按同电脑时间对应，默认偏移为零。
        layer, source_segment = row["layer_id"].strip(), row["segment_id"].strip()  # 层号和原连续段不由匹配误差决定。
        if not layer or not source_segment:  # 不允许空层号造成跨层取窗。
            raise ValueError(f"{name} 缺少 layer_id 或 segment_id。")  # 请求补齐原记录。
        anchor = anchors[layer]  # 当前层的人眼确认首图。
        anchor_target = anchor["time"] + timedelta(seconds=TIME_OFFSET_SECONDS)  # 各层从自己的时间起点定位温度。
        anchor_start = pair_start_before(times, anchor_target)  # 记录本层第一组真实温度序号。
        key = (layer, source_segment)  # 同层不同采集段仍有边界。
        gap = previous_frame is not None and raw_frame != previous_frame + 1  # 裁剪省略帧时不能继续压缩温度时间。
        manual_break = flag(row.get("manual_break_before", "0"), False)  # 保留人工标出的中断。
        if layer != previous_layer:  # 每层都必须重新锚定，绝不从上层顺延。
            start = anchor_start + (raw_frame - anchor["frame"]) * TEMPERATURE_SAMPLES_PER_IMAGE  # 根据层首帧号给出初始候选位置。
            method = "layer_time_anchor"  # 配对表可直接看到层首重对齐。
        elif key != previous_key or gap or manual_break:  # 省略图片或中断后按这一张真实时间重新定位。
            start = pair_start_before(times, target)  # 不把缺口两侧的温度压在一起。
            method = "gap_or_segment_time_anchor"  # 区别于普通层内顺延。
        else:  # 同层连续图片的正常情况。
            start = previous_start + TEMPERATURE_SAMPLES_PER_IMAGE  # 第一图两个点，第二图第三、第四点，依次顺延。
            method = "sequential_two"  # 不要求 40 ms 窗口内恰好两条。
        candidate_ok = 0 <= start and start + 1 < len(temperatures)  # 顺延不能超出温度记录。
        if candidate_ok:  # 检查顺延结果与图像时刻是否仍对得上。
            earliest_age = (target - times[start]).total_seconds()  # 较早温度与图像匹配时刻的差。
            latest_age = (target - times[start + 1]).total_seconds()  # 较晚温度不能来自图像之后。
            candidate_ok = earliest_age <= MAX_TIME_ERROR_SECONDS and latest_age >= 0  # 允许程序时间小偏差；偏差过大则重新找起点。
        if not candidate_ok:  # 时间漂移、采集间隔变化或越界时自动补偿。
            start = pair_start_before(times, target)  # 按当前图片时间重新找最近两条，可与上一图重复。
            method = "layer_time_reanchor" if layer != previous_layer else "time_reanchor"  # 记录补偿来源。
        selected = temperatures[start:start + TEMPERATURE_SAMPLES_PER_IMAGE]  # 始终保留两个实际相邻采样值。
        ages = [(target - sample["time"]).total_seconds() for sample in selected]  # 保存真实误差，不把时刻改成理想采样时刻。
        aligned = all(0 <= age <= MAX_TIME_ERROR_SECONDS for age in ages)  # 原温度范围不足或偏差过大时只标记，不删除图像。
        sample_rows = []  # 一个图像对应两行温度。
        for slot, sample in enumerate(selected):  # 按温度采集时间由早到晚保存。
            available = math.isfinite(sample["value"])  # 799.9按普通数值保留，不改写人工状态。
            sample_rows.append({"sample_in_image": slot, "temperature_index": sample["index"], "temperature_datetime": sample["time"].isoformat(sep=" "), "temperature_c": sample["value"], "time_error_s": round(ages[slot], 6), "sample_available": int(available)})  # 保留原温度序号，重复使用也可追溯。
        new_segment = key != previous_key or gap or manual_break  # 只有真实缺口或层、段变化才切分窗口。
        if new_segment:  # 时间重新匹配不会额外制造一段。
            segment_count += 1  # 修复旧规则造成的数百个人工碎段。
        segment = f"{source_segment}__P{segment_count:04d}"  # 每层及来源断点保持隔离。
        seq_idx = len(paired)  # 所有标注图都得到连续的模型序号。
        state = normalize_state(row["state"])  # 真实稳定、顶丝和熔断标签不因配对改变。
        item = {
            "seq_idx": seq_idx,  # 连续模型序号。
            "image_name": name,  # 原图文件名。
            "segment_id": segment,  # 连续采集段编号。
            "layer_id": layer,  # 人工确认的层号。
            "source_frame_number": raw_frame,  # 原始采集帧号。
            "source_datetime": current.isoformat(sep=" "),  # 完整图像采集时间。
            "image_datetime": current.isoformat(sep=" "),  # 图像时刻。
            "interval_start": (target - timedelta(seconds=1.0 / IMAGE_FPS)).isoformat(sep=" "),  # 名义图像时间区间起点。
            "interval_end": target.isoformat(sep=" "),  # 图像匹配时刻。
            "temperature_available": int(all(sample["sample_available"] for sample in sample_rows)),  # 本组数值是否有限；训练状态只看人工标注。
            "source_state": state,  # 图像真实状态。
            "use_for_training": int(flag(row.get("use_for_training", "1"))),  # 原标注是否允许用于训练。
            "temperature_start_seq_idx": seq_idx * TEMPERATURE_SAMPLES_PER_IMAGE,  # 本图第一条温度行号。
            "temperature_end_seq_idx": (seq_idx + 1) * TEMPERATURE_SAMPLES_PER_IMAGE - 1,  # 本图第二条温度行号。
            "temperature_samples": sample_rows,  # 两个真实温度采样记录。
            "pairing_method": method,  # 顺延或时间补偿方式。
            "time_alignment_ok": int(aligned),  # 配对时间误差是否合格。
            "layer_anchor_datetime": anchor["time"].isoformat(sep=" "),  # 该层人工确认的首图时间。
            "layer_anchor_image": anchor["image"],  # 该层首图名称。
            "layer_anchor_source": anchor["source"],  # 层首时间的来源。
            "layer_anchor_first_temperature_index": temperatures[anchor_start]["index"],  # 层首第一条原温度序号。
            "layer_anchor_second_temperature_index": temperatures[anchor_start + 1]["index"],  # 层首第二条原温度序号。
        }  # 一图两温度及层首依据。
        paired.append(item)  # 即使温度不可用，图像和真实状态仍保留。
        if not aligned:  # 记录时间不能可靠对应的图像，但它们仍会复制。
            issues.append({"source_frame_number": raw_frame, "image_name": name, "source_datetime": current.isoformat(sep=" "), "source_state": state, "layer_id": layer, "segment_id": segment, "interval_start": item["interval_start"], "interval_end": item["interval_end"], "candidate_temperature_count": 2, "reason": "time_out_of_tolerance_image_retained"})  # 训练标签会排除这些位置。
        previous_time, previous_layer, previous_key, previous_frame, previous_start = current, layer, key, raw_frame, start  # 下一图在当前层中继续顺延。
    if len(paired) != len(rows):  # 核对标注行数和计划复制数。
        raise RuntimeError("内部错误：计划复制图片数与标注数不一致。")  # 不发布漏图结果。
    return paired, issues, len(rows)  # 不再丢弃有标注的裁剪图像。


def temperature_rows(paired):  # 将每图的两个样本展开为两行温度，保持时间顺序。
    result = []  # 总行数严格为模型图像数的两倍。
    for image in paired:  # 按图像序号遍历。
        for sample in image["temperature_samples"]:  # 每图两条，各自保留原值。
            result.append({**image, **sample, "image_seq_idx": image["seq_idx"], "seq_idx": len(result)})  # seq_idx 是温度行序号，image_seq_idx 才是图片序号。
    return result  # temperature.csv 和 pairing.csv 使用同一顺序。


def make_intervals(paired):  # 将同层同段、同状态及同可用性的相邻帧合并。
    intervals = []  # 每一行对应一个可人工编辑的区间。
    for item in paired:  # 输入已经按采集顺序排列。
        key = (item["source_state"], item["layer_id"], item["segment_id"], item["use_for_training"], item["temperature_available"])  # 不跨有效性变化合并。
        last = intervals[-1] if intervals else None  # 取得前一区间。
        previous_key = (last["state"], last["layer_id"], last["segment_id"], last["use_for_training"], last["temperature_available"]) if last else None  # 比较区间条件。
        if key == previous_key:  # 状态和连续性均未变化。
            last.update(end_seq_idx=item["seq_idx"], end_image=item["image_name"], end_source_frame_number=item["source_frame_number"])  # 扩展区间末端。
        else:  # 新建区间，保留原图名用于对照。
            intervals.append({"start_seq_idx": item["seq_idx"], "end_seq_idx": item["seq_idx"], "state": item["source_state"], "layer_id": item["layer_id"], "segment_id": item["segment_id"], "start_image": item["image_name"], "end_image": item["image_name"], "start_source_frame_number": item["source_frame_number"], "end_source_frame_number": item["source_frame_number"], "use_for_training": item["use_for_training"], "temperature_available": item["temperature_available"]})  # 初始为单帧区间。
    return intervals  # 后续由人工修改前三列。


def expand_intervals(intervals, manifest):  # 从唯一人工修改来源生成模型逐帧标签。
    count = len(manifest)  # 模型图像清单的帧数。
    actual = [None] * count  # 检查每帧是否被恰好覆盖一次。
    for row in intervals:  # 按区间读取，不要求人工排序。
        start = integer(row["start_seq_idx"], "start_seq_idx")  # 起点包含在区间内。
        end = integer(row["end_seq_idx"], "end_seq_idx")  # 终点也包含在区间内。
        state = normalize_state(row["state"])  # 支持中文状态。
        if not 0 <= start <= end < count:  # 不允许负数、越界或反向区间。
            raise ValueError(f"标签区间 {start}～{end} 无效，有效序号为 0～{count - 1}。")  # 提示可填写范围。
        if manifest[start]["segment_id"] != manifest[end]["segment_id"] or len({item["segment_id"] for item in manifest[start:end + 1]}) != 1:  # 人工修改也不允许跨连续段。
            raise ValueError(f"标签区间 {start}～{end} 跨越连续段或层，请拆成多行。")  # 保留层内取窗口规则。
        for index in range(start, end + 1):  # 展开闭区间。
            if actual[index] is not None:  # 即使相同状态也不接受重叠。
                raise ValueError(f"标签区间在 seq_idx={index} 处重叠。")  # 不靠后写覆盖前写。
            actual[index] = state  # 保存人工确认的真实状态。
    missing = [index for index, state in enumerate(actual) if state is None]  # 查找遗漏帧。
    if missing:  # 未确定状态应明确写 unknown，避免漏填。
        raise ValueError(f"区间表漏标了 {len(missing)} 帧，例如 {missing[:5]}；请补齐或标为 unknown。")  # 不自动沿用旧标签。
    labels = []  # 模型仅需要序号和状态。
    for index, state in enumerate(actual):  # 图像真实状态与温度可用性分别处理。
        labels.append({"seq_idx": index, "state": state})  # 逐帧标签只展开人工区间；温度、时间对齐或旧训练开关不得覆盖真实状态。
    return labels  # 只有人工标注ignore时才排除该帧。


def save_sync_record(folder):  # 记录两份标签和清单的一致性，训练前可发现漏同步。
    annotations = folder / "annotations"  # 人工管理文件目录。
    write_json(annotations / "label_sync.json", {"updated_at_utc": datetime.now(timezone.utc).isoformat(), "interval_sha256": sha256(annotations / "label_intervals.csv"), "frame_labels_sha256": sha256(folder / "frame_labels.csv"), "manifest_sha256": sha256(folder / "frame_manifest.csv")})  # 标签变化后需要重新同步。


def build_wall(wall_id, source, temperature_path, destination):  # 首次整理不会覆盖已存在的墙目录。
    paired, unpaired, source_count = plan_wall(wall_id, source, temperature_path)  # 完成整墙配对后才开始输出。
    intervals = make_intervals(paired)  # 从已有状态生成初始区间表。
    labels = expand_intervals(intervals, paired)  # 仅展开人工区间生成逐帧标签。
    samples = temperature_rows(paired)  # 每图两行，20 张图对应 40 个温度点。
    with tempfile.TemporaryDirectory(prefix=".preparing_", dir=destination.parent) as staging_root:  # 临时目录只含本次新生成文件。
        folder = Path(staging_root) / wall_id  # 临时结果保持正式的墙名。
        (folder / "images").mkdir(parents=True)  # 自动建立模型要求的图片目录。
        (folder / "annotations").mkdir()  # 保存人工标签及来源快照。
        for index, row in enumerate(paired, start=1):  # 复制已经配对成功的图像。
            original, copied = source / row["image_name"], folder / "images" / row["image_name"]  # 保留图片原名。
            shutil.copy2(original, copied)  # 复制，不移动，不裁剪，不改变像素。
            if VERIFY_COPIES and sha256(original) != sha256(copied):  # 核对实际复制结果。
                raise OSError(f"图片复制核对失败：{row['image_name']}")  # 不发布损坏结果。
            if index % 250 == 0 or index == len(paired):  # 在 IDE 显示长复制任务进度。
                print(f"{wall_id}：已复制 {index}/{len(paired)} 张图片", flush=True)  # 显示处理范围。
        write_csv(folder / "frame_manifest.csv", paired, MANIFEST_FIELDS)  # 图片、层号及段号清单。
        write_csv(folder / "temperature.csv", samples, ["seq_idx", "image_seq_idx", "sample_in_image", "temperature_c", "temperature_index", "temperature_datetime"])  # 两条原值分别存储，不平均。
        write_csv(folder / "pairing.csv", samples, PAIR_FIELDS)  # 原始温度序号、时间及误差可逐项追溯。
        write_csv(folder / "frame_labels.csv", labels, ["seq_idx", "state"])  # 墙根部只放这一种模型标签。
        write_csv(folder / "annotations" / "label_intervals.csv", intervals, INTERVAL_FIELDS)  # 区间表放在子目录，不与模型读取冲突。
        write_csv(folder / "annotations" / "unpaired_frames.csv", unpaired, ["source_frame_number", "image_name", "source_datetime", "source_state", "layer_id", "segment_id", "interval_start", "interval_end", "candidate_temperature_count", "reason"])  # 即使没有排除项也保留表头。
        shutil.copy2(source / "frame_annotations.csv", folder / "annotations" / "source_frame_annotations.csv")  # 原标注保留一份快照。
        expected_names = {row["image_name"] for row in paired}  # 标注图像应全部出现在输出中。
        copied_names = {path.name for path in (folder / "images").iterdir() if path.is_file()}  # 不只检查总数，逐名核对。
        if copied_names != expected_names:  # 漏图或额外图片都不发布。
            raise RuntimeError("复制图像清单与全部标注不一致。")  # 明确保证标注图全部复制。
        layer_summary = []  # 报告每层独立温度起点，便于人工核对。
        for layer in dict.fromkeys(row["layer_id"] for row in paired):  # 按实际层顺序输出。
            group = [row for row in paired if row["layer_id"] == layer]  # 当前层的全部标注图。
            first = group[0]  # 人眼确认的当前层起点对应关系。
            layer_summary.append({"layer_id": layer, "first_image": first["image_name"], "confirmed_start_datetime": first["layer_anchor_datetime"], "anchor_source": first["layer_anchor_source"], "first_temperature_index": first["layer_anchor_first_temperature_index"], "second_temperature_index": first["layer_anchor_second_temperature_index"], "copied_images": len(group), "pairing_methods": dict(Counter(row["pairing_method"] for row in group))})  # 每层重新锚定，不继承上一层。
        temperature_uses = Counter(row["temperature_index"] for row in samples)  # 同一原始温度可被相邻图片重复使用。

        report = {
            "wall_id": wall_id,  # 当前墙编号。
            "image_fps": IMAGE_FPS,  # 用户确认的图像帧率。
            "temperature_hz": TEMPERATURE_HZ,  # 原始温度采样率。
            "temperature_match": "per_layer_time_anchor_sequential_two_with_time_resync",  # 本次实际使用的配对规则。
            "time_offset_seconds": TIME_OFFSET_SECONDS,  # 图像和温度时间轴的固定偏移。
            "max_time_error_seconds": MAX_TIME_ERROR_SECONDS,  # 允许的配对时间偏差。
            "source_image_folder": str(source),  # 原裁剪图片及标注目录。
            "source_temperature_file": str(temperature_path),  # 原始温度文件。
            "source_annotation_sha256": sha256(source / "frame_annotations.csv"),  # 原标注内容哈希。
            "source_temperature_sha256": sha256(temperature_path),  # 原温度内容哈希。
            "source_frames": source_count,  # 原标注图像总数。
            "copied_frames": len(paired),  # 本次复制图像总数。
            "images_excluded": 0,  # 图像排除数量，当前保持为零。
            "copy_name_check_passed": copied_names == expected_names,  # 复制后的文件名是否完整一致。
            "copy_hashes_verified": VERIFY_COPIES,  # 是否逐张核对图像哈希。
            "layer_anchors": layer_summary,  # 各层人工确认的起点及对应温度。
            "unique_source_temperature_points": len(temperature_uses),  # 使用的不同原温度采样点数量。
            "reused_temperature_assignments": sum(count - 1 for count in temperature_uses.values()),  # 允许重复使用的温度分配次数。
            "paired_frames": len(paired),  # 已建立温度分组的图像数。
            "paired_temperature_points": len(samples),  # 两倍于图像数的温度行数。
            "temperature_samples_per_image": TEMPERATURE_SAMPLES_PER_IMAGE,  # 每张图片对应的温度数。
            "time_aligned_frames": sum(row["time_alignment_ok"] for row in paired),  # 时间误差符合要求的图像数。
            "unpaired_frames": len(unpaired),  # 时间需要进一步核对的图像数，图片仍保留。
            "unpaired_reasons": dict(Counter(row["reason"] for row in unpaired)),  # 时间匹配问题及数量。
            "unavailable_temperature_frames": sum(not row["temperature_available"] for row in paired),  # 温度不可用于训练的图像组数。
            "training_label_counts": dict(Counter(row["state"] for row in labels)),  # 最终训练标签的类别数量。
            "continuous_segments": len({row["segment_id"] for row in paired}),  # 按真实层和断点保留的连续段数量。
            "max_observed_time_error_seconds": max(abs(row["time_error_s"]) for row in samples),  # 实际观察到的最大绝对时间误差。
            "created_at_utc": datetime.now(timezone.utc).isoformat(),  # 生成报告时的 UTC 时间。
        }  # 记录真实整理结果，不声称训练效果。
        write_json(folder / "preparation_report.json", report)  # 保存整理报告。
        save_sync_record(folder)  # 写入标签同步标识。
        if destination.exists():  # 再次检查，避免覆盖另一操作刚生成的墙。
            raise FileExistsError(f"输出已存在，拒绝覆盖：{destination}")  # 重做时只处理已生成结果，不删除来源图片及标注。
        folder.rename(destination)  # 所有文件完成后发布整墙目录。
    print(f"完成：{destination}\n配对 {len(paired)} 帧，未配对 {len(unpaired)} 帧，训练标签 {report['training_label_counts']}", flush=True)  # 用户可立即打开查看。


def sync_labels(wall_id, folder):  # 只更新生成的逐帧标签，不读取旧标注覆盖人工修改。
    manifest = read_csv(folder / "frame_manifest.csv", MANIFEST_FIELDS)  # 获取不可跨越的边界和测温可用性。
    if [integer(row["seq_idx"], "seq_idx") for row in manifest] != list(range(len(manifest))):  # 同步必须基于原配对序列。
        raise ValueError(f"{wall_id} 的图片清单序号不连续，请勿手动改动清单。")  # 防止标签错位。
    interval_path = folder / "annotations" / "label_intervals.csv"  # 唯一人工修改来源。
    intervals = read_csv(interval_path, ["start_seq_idx", "end_seq_idx", "state"])  # 只把前三列解释为标注指令。
    labels = expand_intervals(intervals, manifest)  # 先检查完整、无重叠、不跨段。
    for row in intervals:  # 更新辅助显示列，避免端点改动后仍显示旧图片名。
        start, end = integer(row["start_seq_idx"], "start_seq_idx"), integer(row["end_seq_idx"], "end_seq_idx")  # 已由展开函数验证。
        row.update(state=normalize_state(row["state"]), layer_id=manifest[start]["layer_id"], segment_id=manifest[start]["segment_id"], start_image=manifest[start]["image_name"], end_image=manifest[end]["image_name"], start_source_frame_number=manifest[start]["source_frame_number"], end_source_frame_number=manifest[end]["source_frame_number"], use_for_training="以 frame_manifest 为准", temperature_available="以 frame_manifest 为准")  # 有效性仍由逐帧清单决定。
    write_csv(interval_path, sorted(intervals, key=lambda row: integer(row["start_seq_idx"], "start_seq_idx")), INTERVAL_FIELDS)  # 保存规范顺序和新的辅助字段。
    write_csv(folder / "frame_labels.csv", labels, ["seq_idx", "state"])  # 原子替换派生标签文件。
    save_sync_record(folder)  # 只有同步成功才更新哈希标识。
    print(f"已同步 {wall_id}：{len(labels)} 帧，标签数量 {dict(Counter(row['state'] for row in labels))}。请使用新的标签版本重新划分并训练。", flush=True)  # 不修改已有权重和结果。


def validate_paths():  # 检查配置及输出与来源的隔离关系。
    if TEMPERATURE_SAMPLES_PER_IMAGE != 2 or not math.isclose(TEMPERATURE_HZ / IMAGE_FPS, 2.0):  # 本次只明确支持图像 25 / 温度 50 的一比二接口。
        raise ValueError("本整理版本要求温度采样率为图像帧率的两倍，每图两个温度。")  # 不自动改动采样率或补值。
    if MODE not in {"build", "sync_labels", "repair_deleted", "refresh_existing"}:  # 不接受拼错的操作模式。
        raise ValueError('MODE 只能填写 "refresh_existing"、"build"、"sync_labels" 或 "repair_deleted"。')  # 明确合法选项。
    if MODE == "refresh_existing":  # 统一更新使用已有墙，不依赖旧的首次整理名单或已移动的原图目录。
        if not isinstance(REFRESH_WALL_IDS, list):  # 留空必须写成列表，不能填写字符串。
            raise ValueError("REFRESH_WALL_IDS 必须为列表；[] 表示全部已有墙。")  # 给出可直接修改的配置说明。
        if not OUTPUT_DATA_ROOT.is_dir():  # 统一更新只能处理已配对的数据目录。
            raise FileNotFoundError(f"已有墙目录不存在：{OUTPUT_DATA_ROOT}")  # 不创建空数据集。
        selected = REFRESH_WALL_IDS or sorted(path.name for path in OUTPUT_DATA_ROOT.iterdir() if path.is_dir() and path.name.startswith("wall_"))  # 只扫描直接位于数据根目录下的墙。
        if not math.isfinite(REFRESH_MIN_VALID_TEMPERATURE):  # 阈值必须可用于可靠比较。
            raise ValueError("REFRESH_MIN_VALID_TEMPERATURE 必须为有限数值。")  # 不推测其他阈值。
    else:  # 保留原来三个模式各自明确选择墙的规则。
        selected = REPAIR_WALL_IDS if MODE == "repair_deleted" else WALL_IDS  # 旧修复模式仍只处理指定墙。
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):  # 三种模式均需明确选择墙。
        raise ValueError("当前模式的墙名单必须为非空、没有重复编号的列表。")  # 不默认扫描全部墙。
    for wall_id in selected:  # 限制所有墙编号为一个文件夹名。
        if not isinstance(wall_id, str) or Path(wall_id).name != wall_id or wall_id in {"", ".", ".."}:  # 防止写入其他目录。
            raise ValueError(f"非法墙编号：{wall_id!r}")  # 指明配置错误。
    for source_root in (SOURCE_IMAGE_ROOT.resolve(), TEMPERATURE_ROOT.resolve()):  # 原始图像和温度都不允许成为输出目录。
        output_root = OUTPUT_DATA_ROOT.resolve()  # 解析最终绝对输出位置。
        if output_root == source_root or source_root in output_root.parents or output_root in source_root.parents:  # 禁止包含和被包含关系。
            raise ValueError("OUTPUT_DATA_ROOT 必须与原图及原温度目录分开，不能相互包含。")  # 防止误写输入。
    for value, name in ((IMAGE_FPS, "IMAGE_FPS"), (TEMPERATURE_HZ, "TEMPERATURE_HZ"), (MAX_TIME_ERROR_SECONDS, "MAX_TIME_ERROR_SECONDS")):  # 检查名义采样参数。
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:  # 帧率和允许误差都必须为有限正数。
            raise ValueError(name + " 必须为有限正数。")  # 不推断实际采样率。
    if not math.isfinite(TIME_OFFSET_SECONDS) or any(not math.isfinite(value) for value in INVALID_TEMPERATURE_VALUES):  # 偏移允许正负，阈值必须有限。
        raise ValueError("时间偏移和无效温度值必须是有限数值。")  # 避免无效比较。


def main():  # 在 PyCharm 或 VS Code 中右键运行此文件。
    validate_paths()  # 不合适的参数在任何复制操作前拦截。
    wall_root = OUTPUT_DATA_ROOT  # 各墙直接放在 wall_dataset/墙编号/，不创建中间 walls 层。
    if MODE == "refresh_existing":  # 已有数据的删图、温度改值与区间改标统一入口。
        from dataset_repair import refresh_batch  # 复用原索引核验、备份及回滚逻辑，不增加新的运行脚本。
        selected = REFRESH_WALL_IDS or sorted(path.name for path in wall_root.iterdir() if path.is_dir() and path.name.startswith("wall_"))  # 默认选中全部已有墙。
        refresh_batch(wall_root, selected, TEMPERATURE_ROOT, TEMPERATURE_FILES, REPAIR_BACKUP_ROOT, INVALID_TEMPERATURE_VALUES, REFRESH_MIN_VALID_TEMPERATURE)  # 保留原采样编号，只同步当前源温度数值与人工标注。
        return  # 不复制原图、不启动划分或模型训练。
    if MODE == "repair_deleted":  # 已有原配对但派生图片被删时，不调用首次配对流程。
        from dataset_repair import repair_batch  # 修复逻辑与首次整理分开，三个模型继续共用同一数据接口。
        repair_batch(wall_root, REPAIR_WALL_IDS, REPAIR_BACKUP_ROOT)  # 原seq区间作为标注依据，图片名核对原温度。
        return  # 不从原图补回删除帧，不重新按时间匹配，不启动训练。
    if MODE == "sync_labels":  # 后续只修改区间标签时走此分支。
        for wall_id in WALL_IDS:  # 只同步明确选择的墙。
            sync_labels(wall_id, wall_root / wall_id)  # 不再从来源 state 导入标签。
        return  # 不复制图片，不匹配温度，不启动训练。
    jobs = []  # 先检查所有选中任务，避免配置缺失时只做了一半。
    for wall_id in WALL_IDS:  # 依次处理配置清单中的墙。
        source = SOURCE_IMAGE_ROOT / wall_id  # 本墙裁剪图片和标注目录。
        destination = wall_root / wall_id  # 本墙模型输入目录。
        if destination.exists():  # 初次整理模式不覆盖既有数据。
            raise FileExistsError(f"已存在 {destination}；改标签请设置 MODE='sync_labels'，重新配对请使用另一个空输出目录。")  # 说明后续操作。
        name = TEMPERATURE_FILES.get(wall_id, "")  # 只使用用户明确指定的温度文件。
        if not name or Path(name).name != name:  # 不模糊匹配邻近日期的实验。
            raise ValueError(f"请在 TEMPERATURE_FILES 中填写 {wall_id} 对应的温度文件名。")  # 可逐墙补充映射。
        temperature_path = TEMPERATURE_ROOT / name  # 拼出原始温度完整路径。
        if not (source / "frame_annotations.csv").is_file() or not temperature_path.is_file():  # 输入不存在时不生成半成品。
            raise FileNotFoundError(f"请检查图像标注 {source / 'frame_annotations.csv'} 和温度文件 {temperature_path}。")  # 输出两个需要检查的地址。
        jobs.append((wall_id, source, temperature_path, destination))  # 记录通过路径检查的任务。
    metadata_path = OUTPUT_DATA_ROOT / "dataset_metadata.json"  # 记录该数据版本的图像帧率。
    metadata = {
        "image_fps": IMAGE_FPS,  # 用户确认的图像帧率。
        "temperature_hz": TEMPERATURE_HZ,  # 原始温度采样率。
        "paired_temperature_hz": TEMPERATURE_HZ,  # 配对保留的名义温度采样率。
        "temperature_samples_per_image": TEMPERATURE_SAMPLES_PER_IMAGE,  # 每张图片对应的温度数。
        "temperature_layout": "long_grouped_by_image_v2",  # 温度文件的分组格式。
        "temperature_match": "per_layer_time_anchor_sequential_two_with_time_resync",  # 本次实际使用的配对规则。
        "time_offset_seconds": TIME_OFFSET_SECONDS,  # 图像和温度时间轴的固定偏移。
        "max_time_error_seconds": MAX_TIME_ERROR_SECONDS,  # 允许的配对时间偏差。
        "invalid_temperature_values": INVALID_TEMPERATURE_VALUES,  # 仪器测温无效占位值列表。
    }  # 每图两条原温度，显式记录分组布局。
    existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else None  # 保留已有数据集元信息。
    comparable_metadata = {key: value for key, value in metadata.items() if key != "invalid_temperature_values"}  # 此旧过滤标记不改变图温配对方式。
    existing_comparable = {key: value for key, value in (existing_metadata or {}).items() if key != "invalid_temperature_values"}  # 仅比较影响配对的字段。
    if existing_metadata is not None and existing_comparable != comparable_metadata:  # 其他配对规则不同仍停止。
        existing_walls = [path for path in wall_root.iterdir() if path.is_dir() and (path / "images").is_dir()]  # 不把残留的元信息视为一面墙。
        if existing_walls:  # 仍有已生成墙时先明确处理它们。
            raise ValueError("当前目录还有按旧规则生成的墙，请先处理这些生成结果；保留 wall_photo 和原温度文件。")  # 不自动覆盖任何现有墙。
        print("输出目录已无生成的墙，将更新残留的 dataset_metadata.json 配对规则。", flush=True)  # 用户删除首面生成墙后可以直接重跑。
    wall_root.mkdir(parents=True, exist_ok=True)  # 现在才创建输出目录。
    if existing_metadata is None or existing_comparable != comparable_metadata:  # 已有兼容元信息保持原样。
        write_json(metadata_path, metadata)  # 只为新数据集或允许更新的空数据集写入。
    print(f"开始整理 {len(jobs)} 面墙：图像 {IMAGE_FPS} 帧/秒，原温度 {TEMPERATURE_HZ} 点/秒，每层从人工确认首图时间重新对齐，层内顺延两个温度，允许重用并按时间补偿。", flush=True)  # 不进行帧率测量。
    for job in jobs:  # 按用户指定顺序处理。
        build_wall(*job)  # 每面墙单独发布完整结果。
    print(f"整理结束。请先打开结果查看：{wall_root}\n本程序没有划分数据、训练模型或修改原图/原温度。训练新墙数据时请在模型 config.py 中设置 FPS={IMAGE_FPS}、TEMPERATURE_SAMPLES_PER_IMAGE=2，并填写新的 DATA_ROOT。", flush=True)  # 引导用户下一步。


if __name__ == "__main__":  # 导入工具函数时不会处理实验文件。
    main()  # 在 IDE 点击运行即可，无需命令行参数。
