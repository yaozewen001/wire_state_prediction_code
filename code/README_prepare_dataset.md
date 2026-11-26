# 墙数据整理与统一更新

## v9：人工标注是唯一状态依据（2026-11-03）

- `frame_labels.csv`严格展开`annotations/label_intervals.csv`，不按温度数值、`temperature_available`、`time_alignment_ok`或旧`use_for_training`把状态改成ignore。
- 各模型共用`data_config.py`的`TEMPERATURE_INPUT_POLICY="raw"`。799.9按原值进入标准化、局部温度网络，不插值、不填零、不另加缺测通道，不因799.9拆分连续段或丢弃窗口。
- 仅按人工unknown、ignore排除输入；真实缺图、层和墙边界继续保持。NaN、无穷等无法计算的数值直接报错，不静默删样本或改标签。
- 清单中旧的温度有效性和来源训练开关可保留追溯，新训练不使用它们过滤。标准化只拟合训练历史，包含实际输入的799.9原值。
- 本地逐帧标签修复后，刷新公共划分记录并保留原墙归属；上传服务器必须同步新代码、新标签和新划分。旧模型不能自动补学此前被排除的样本，需要重新训练；新结果目录使用v9。
- 原始图像、人工区间、温度数值和配对编号保持不变。若只改标注，可在IDE运行prepare_dataset.py的sync_labels模式，不必重新按时间配对。


## 1. 已有墙更新后的运行顺序

在 PyCharm 或 VS Code 中运行 `code/prepare_dataset.py`。已有墙删除图片、修改区间标注或修改源温度数值后，统一切换到 `MODE = "refresh_existing"`。

1. 保存源温度 CSV 和各墙的 `annotations/label_intervals.csv`，关闭正在编辑这些文件的 Excel 窗口。
2. 运行 `code/prepare_dataset.py`，确认控制台显示“统一更新完成”。
3. 核对 `code/data_config.py` 的 `DATA_ROOT`，运行唯一的 `code/split_data.py`。
4. 配置所选模型的 `config.py`，运行该模型目录的 `train.py`。发布的六个模型共用数据与划分。

`split_data.py` 只管理划分，不能替代步骤 2 的温度与标注更新。新数据需要重新训练；旧模型权重不会自动改变。

## 2. 日常配置

所有配置位于 `code/prepare_dataset.py` 开头，不需要直接运行 `dataset_repair.py`。

| 配置 | 当前值或填写方式 | 何时修改 |
|---|---|---|
| `MODE` | `"refresh_existing"` | 已有墙统一更新时保持此值 |
| `OUTPUT_DATA_ROOT` | `PROJECT_DIR / "data" / "datasets" / "wall_dataset"` | 已配对墙移动到其他位置时 |
| `TEMPERATURE_ROOT` | `PROJECT_DIR / "data" / "datasets" / "wall_temperature_csv"` | 最终源温度 CSV 所在位置变化时 |
| `REFRESH_WALL_IDS` | `[]` 为全部已有墙 | 仅处理部分墙时填写明确墙名列表 |
| `TEMPERATURE_FILES` | 墙名 → 温度文件名 | 建立或核对明确的温度来源时 |
| `REPAIR_BACKUP_ROOT` | `PROJECT_DIR / "records" / "dataset_repairs"` | 改变自动备份和报告地址时 |
| `INVALID_TEMPERATURE_VALUES` | `[799.9]` | 设备无效占位值改变时 |

更新全部已有墙：

```python
MODE = "refresh_existing"  # 统一更新已有墙。
REFRESH_WALL_IDS = []  # 选中OUTPUT_DATA_ROOT下全部已有wall_...目录。
```

只更新一面墙：

```python
MODE = "refresh_existing"  # 统一更新已有墙。
REFRESH_WALL_IDS = ["wall_example_01"]  # 只处理明确填写的墙。
```

统一更新不使用旧的 `WALL_IDS`、`REPAIR_WALL_IDS` 或旧 `wall_photo` 来源目录。实际保留图片以已有墙的 `images/` 为准。

温度来源通过 `TEMPERATURE_FILES` 与本墙 `preparation_report.json` 核对，只取历史地址中的文件名，在当前 `TEMPERATURE_ROOT` 查找。两种记录指向不同采集时间时停止。统一命名后优先读取 `serial_50hz_YYYYMMDD_HHMMSS.csv`，不按目录顺序或修改时间猜测来源。

## 3. 统一更新的行为

| 输入或变化 | 处理方式 |
|---|---|
| `images/` 中删除图片 | 从派生清单去掉缺图，不从原始采集目录补回 |
| 保留图片的顺序 | 按运行前 `frame_manifest.csv` 中的 `seq_idx` 保持原顺序 |
| 新状态标注 | 读取运行前区间表的前三列：`start_seq_idx`、`end_seq_idx`、`state` |
| 层号、原帧号、起止图片名等辅助列 | 根据原清单与保留图片重新生成，不覆盖人工新状态 |
| 最新温度数值 | 用每图原来的两个 `temperature_index` 查源温度的 `index`，写入当前 `value` |
| 图温对应关系 | 保留原采样编号，不按剩余图片行数重新顺排 |
| 配对时刻与每层锚点 | 保留原配对时刻、时间误差和层首锚点，不重新按舍入时间定位 |
| 删除留下的缺口与换层 | 保留连续段边界，窗口不能跨过缺图位置 |
| 重复运行 | 没有新结构变化时保留现有编号和段号，不反复添加段号后缀 |

例如，A 图使用源采样 100、101，B 图使用 102、103。删除 A 后，B 仍使用 102、103，仅更新这两条采样的温度数值。

部分源温度时间保留到 0.1 秒，会有多个采样时间相同。统一更新通过原采样编号查值，不凭空补造毫秒，也不覆盖旧配对中已保存的完整时刻。若源温度行编号被改动、所需原编号丢失，或需要改变同步关系，必须另行核验，不能使用本模式猜测对应。

源温度 CSV、实际保留图片及原始标注快照不被本模式修改。`pairing.csv/source_state` 保留原标注快照；当前训练状态以更新后的区间表和逐帧标签为准。

## 4. 标注编号和检查规则

运行前继续使用现有清单中的 `seq_idx` 标注，允许删除位置留下编号缺口，不要自行将剩余区间整体前移。成功后模型编号从 0 连续排列，区间随图片转换。之后继续标注时使用新编号；旧新编号对照在备份的 `index_map.csv`。

| 检查问题 | 行为 |
|---|---|
| 保留图片被不同状态重复标注 | 报告 CSV 行号和原编号，不自动选择类别 |
| 保留图片漏标 | 报告范围；不确定需由人工明确填写 `unknown` |
| 区间倒置、越界、未知拼写 | 停止；仅明确兼容 `unknow` 为 `unknown` |
| 前三列完全为空 | 忽略空行，旧辅助列不会恢复已清空的标注 |
| 同状态重复覆盖 | 合并同一状态，不重复生成图片或标签 |
| 新增图片不在原清单中 | 停止，不能猜其温度 |
| 旧温度表与配对表记录不一致 | 停止，不能随意选一份覆盖 |
| 源采样编号重复、倒序或缺失 | 停止并指出来源及编号 |

全部选中墙先进行只读预检查。任何一面输入检查失败，整批墙数据不写入，控制台和批量报告列出问题。修正后再次运行相同入口。

全部检查通过后才逐墙备份与发布。写入期间文件占用或被另外编辑时停止后续墙，本墙写失败时尝试恢复原文件。此前成功的墙在报告中明确登记，不会声称整批回滚。遇到未全部完成提示时，不要继续划分或训练。

## 5. 输出和备份

| 文件 | 用途 |
|---|---|
| `frame_manifest.csv` | 当前图片顺序、层、连续段与温度有效性 |
| `temperature.csv` | 模型温度输入，每图两行 |
| `pairing.csv` | 图像、原采样编号、原时刻和最新温度值的追溯 |
| `annotations/label_intervals.csv` | 转换后的当前编号、真实状态和辅助列，后续人工标注入口 |
| `frame_labels.csv` | 根据区间生成的训练标签 |
| `annotations/label_sync.json` | 标签和清单同步哈希 |
| `preparation_report.json` | 源温度版本及更新统计，保留原层首锚点 |

备份位于 `records/dataset_repairs/refresh_运行时间/`：

```text
refresh_运行时间/
├─ refresh_report.json        全部是否完成、每墙错误或更新统计
└─ wall_YYYYMMDD_HHMMSS/
   ├─ original/               写入前的派生CSV，保留原目录结构
   ├─ index_map.csv           旧新seq_idx、原图和原采样编号对照
   └─ repair_report.json      本墙更新统计
```


## 6. 其他模式与首次整理

| 模式 | 适用情况 | 墙选择 |
|---|---|---|
| `refresh_existing` | 已有墙删图、改源温度或改标签后的统一更新，默认模式 | `REFRESH_WALL_IDS`，空列表为全部已有墙 |
| `sync_labels` | 仅改区间标签，未删图片且不需要更新温度 | `WALL_IDS` |
| `repair_deleted` | 旧的只修复缺图入口，不读取新源温度值 | `REPAIR_WALL_IDS` |
| `build` | 首次建立新墙配对 | `WALL_IDS`、`TEMPERATURE_FILES` |

`build` 使用 `SOURCE_IMAGE_ROOT/墙名/` 的裁剪图、`frame_annotations.csv` 和可选 `layers.csv`，输出新墙。已有同名输出时拒绝覆盖，不要为了更新而删除已配对墙再build，以免丢失后来人工标注。

首次配对使用已确认的图像 25 fps、温度 50 Hz，每图两个原值，不重新测算帧率。每层按人工确认首图时间独立锚定，层内顺延两个采样；候选晚于图像或误差超过 `MAX_TIME_ERROR_SECONDS` 时重新按时间定位。`TIME_OFFSET_SECONDS` 默认 0，允许误差默认 0.08 秒；这些参数不用于 `refresh_existing` 重新改变既有对应。

首次build需要可靠的采样时间，仍拒绝重复时间。0.1秒舍入记录不能当作已经恢复原始毫秒精度；已有墙的统一更新直接沿用原配对的编号和时刻。
