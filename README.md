# WSP-Mamba 送丝状态预测

使用历史熔池图像、温度和激光功率设定值，预测未来指定帧的稳定、顶丝或熔断状态。本仓库提供主模型、五个对照模型以及共用的数据准备、整墙划分和评估程序。

| 模型 | 代码 |
| --- | --- |
| WSP-Mamba | [code/wsp_mamba](code/wsp_mamba) |
| CNN＋LSTM | [code/cnn_lstm_temp_power](code/cnn_lstm_temp_power) |
| CNN＋TCN | [code/cnn_tcn_temp_power](code/cnn_tcn_temp_power) |
| ResNet34＋LSTM | [code/resnet34_lstm_temp_power](code/resnet34_lstm_temp_power) |
| ResNet34＋多尺度 TCN | [code/resnet34_ms_tcn_temp_power](code/resnet34_ms_tcn_temp_power) |
| ResNet34＋ConvLSTM | [code/resnet34_convlstm_temp_power](code/resnet34_convlstm_temp_power) |

## 在 IDE 中运行

1. 按[环境说明](environment/README.md)安装 PyTorch、torchvision 和普通依赖，并在 PyCharm 或 VS Code 里选择对应的 Python 解释器。
2. 使用自己的图像、温度与标注，按[数据整理说明](code/README_prepare_dataset.md)整理成配对数据。每面墙提供 `images/`、`frame_manifest.csv`、`temperature.csv`、`pairing.csv` 和人工标注。数据根目录还需要 `power_settings.csv`。本仓库提供源码和格式说明。
3. 在 [code/data_config.py](code/data_config.py) 中设置 `DATA_ROOT`，在 IDE 运行 [code/split_data.py](code/split_data.py) 生成训练、验证和测试划分。
4. 在所选模型的 `config.py` 设置预测跨度和训练参数，然后运行同目录的 `train.py`。训练完通过 `evaluate.py` 加载权重评估。WSP-Mamba 的两阶段训练、BPM 参考窗和消融配置见[模型说明](code/wsp_mamba/README.md)。

功率表示设定值。下面是填写格式示例，示例墙号与数值仅用于说明。

```csv
wall_id,layer_id,effective_at,power_setpoint_w
wall_example_01,1,1,1000
```

## 预训练权重与 Mamba 后端

- WSP-Mamba 的 ConvNeXt V2-Atto 可使用[官方预训练权重](https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt)。将权重放到 `weights/pretrained/convnextv2_atto_1k_224_ema.pt`，或将 `ATTO_WEIGHTS_FILE` 留空，由 PyTorch 使用缓存或下载。ResNet34 模型使用 torchvision 的 ImageNet 权重选项。
- `MAMBA_BACKEND = "auto"` 会在可用时选择 Mamba-2 的 CUDA SSD 算子。用户也可用 `"torch"` 明确选择 PyTorch 实现，或用 `"cuda"` 要求 CUDA 算子。
- Windows 上已完成安装及算子验证的组合为 Python 3.10、PyTorch 2.4.1+cu124、CUDA Toolkit 12.4、MSVC 14.38 x64、Triton 3.1.0、causal-conv1d 1.4.0 与 mamba-ssm 2.2.2，显卡为 RTX 4090。可参考[社区 Windows 移植说明](https://github.com/1552186151/mamba2_for_windows)。验证范围为 Mamba CUDA 前向与 SSD 算子前向、反向。

第三方项目及许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
