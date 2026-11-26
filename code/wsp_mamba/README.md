# WSP-Mamba

用当前和过去的图像、温度、激光功率设定值，预测未来指定帧的稳定、顶丝或熔断状态。

- **Atto → SWME**。ConvNeXt V2-Atto 保留四阶段，在 640×640 输入下输出 20×20×320 特征图。通道投影至 96，依次经过 **RW-M** 与 **SW-M** 后逐帧汇聚。边缘窗口使用实际格子，避免把相对边缘拼入同一窗。
- **短温度 TCN → DGF**。短温度保留 40 个时间位置。图像查询短温度，视觉和交互特征分别经过门控与残差，然后沿时间汇总。
- **长温度 TCN、功率 MLP**。独立编码最多 168 个温度点和 84 个功率设定值。和 DGF 汇总拼成 144 维基础窗口向量。
- **BPM → 分类头**。每条训练参考单独产生匹配响应，汇成 32 维补充特征。接入 BPM 后分类头读取拼接的 176 维向量。

## 在 IDE 中怎么运行

1. 共用项目的 `code/split_data.py`、已配对墙数据、功率表和 `data_config.py`。训练按已保存的整墙划分和训练层清单读窗，验证和测试使用各自完整墙的有效窗口。
2. 在本目录 `config.py` 选择 `FUTURE_FRAMES`。首次对照用 `IMAGE_SAMPLING = "all"`。比较少帧时保留 `WINDOW_SIZE = 20`，设 `IMAGE_SAMPLING = "uniform"`，再用 `IMAGE_STEPS = 5` 或 `6`。温度和功率的历史范围沿用相同口径。
3. 用 `TRAIN_STAGE = "foundation"` 在 IDE 运行 `train.py`。`ATTO_WEIGHTS_FILE` 已指向项目根目录下的 `weights/pretrained/convnextv2_atto_1k_224_ema.pt`。上传时按相同位置放置这份权重。留空时可用 PyTorch 缓存或下载 [Atto 官方权重](https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt)。临时验证代码时可将 `ATTO_PRETRAINED` 设为 `False`。
4. 在 `REFERENCE_WINDOWS` 中填写（训练墙文件夹名, 层号, 目标熔断帧的原始编号）。编号取自 `frame_manifest.csv` 的 `source_frame_number`。程序按 `FUTURE_FRAMES` 定位历史末帧，筛选末帧稳定、目标帧熔断的有效训练窗口。基础训练完成后，设 `TRAIN_STAGE = "bpm"` 并运行 `train.py`。建库后将两端的实际帧号写入 `reference_manifest.csv`。仍可用 `FOUNDATION_RUN_DIR` 指定基础权重的运行目录。两阶段共用同一提前量、实际选图位置、图温功率版本和窗口清单。
5. 在 IDE 运行 `evaluate.py`，默认评估测试集。`EVALUATION_RUN_DIR` 可指定某次运行。`EVALUATION_EPOCH = "transition_f1"` 选择另一份权重，设 `EVALUATION_SPLIT = "both"` 可在同一总表汇总该权重的验证和测试。`predict.py` 支持对运行快照中的一面墙做离线回放。
  
`MAMBA_BACKEND = "auto"` 在可用的 CUDA 环境优先用 [Mamba 官方 SSD 核](https://github.com/state-spaces/mamba)。普通 PyTorch 路径也执行同一因果 SSD，可用于本地核对。云端若需优化，可按官方环境说明安装 `mamba-ssm`，再通过启动行核对所用后端。CPU 运行 640×640 批量训练耗时较长。

## 消融

| 要验证的部分 | 配置 |
| --- | --- |
| 移位窗口 | `SHIFT_SECOND_BLOCK = False`。两块均为规则分窗，保留两套独立权重。 |
| 图温双门控融合 | `USE_DGF = False`。图像与短温度分别汇总，拼接后投影至同一融合维度。 |
| 前兆匹配 | `TRAIN_STAGE = "bpm"` 对 `TRAIN_STAGE = "frozen_head"`。两组从同一基础权重出发，冻结编码器，就其固定特征训练相同的轮数和批次。 |

每组用同一预测跨度与划分比较。验证损失和稳定转熔断 F1 各选一份权重。运行目录包含 `history.csv`、训练曲线、混淆矩阵、验证和测试的总体及熔断层评估、逐窗结果、层事件统计与 `评估总表.xlsx`。换提前量或改编码结构后，可以按新的配置训练基础模型并建立对应的参考库。

