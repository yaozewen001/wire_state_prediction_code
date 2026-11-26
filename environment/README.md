# 训练环境检查与依赖安装

本说明适用于项目的 CNN 和 ResNet34 模型。到云服务器后，在 IDE 选择实际训练用的解释器，直接运行项目的 `code/setup_environment.py`。默认只检查，逐项显示当前版本与最低版本的比较。

## 1. 云服务器选择

| 项目 | 项目最低版本与原参考镜像 |
|---|---|
| GPU | NVIDIA RTX 3090 / RTX 4090，单卡 24 GB |
| 系统 | Ubuntu 22.04；本地 Windows 也可使用同样的 Python 和主要依赖版本 |
| Python | **3.10.0 起**，由云镜像或 IDE 解释器提供 |
| PyTorch | **2.5.1 起**，选择与驱动匹配的构建 |
| torchvision | **0.20.1 起**，与 PyTorch 成对安装 |
| CUDA 驱动 | 使用云平台配套驱动，确保所选镜像可以实际调用 GPU |
| 其他依赖 | 普通依赖检查以本目录 `requirements.txt` 所列版本为下限 |

PyTorch 2.5.1 和 torchvision 0.20.1 是原参考组合。PyTorch 官方明确提供 2.5.1＋torchvision 0.20.1 的 cu121 安装组合。若平台只有同版本 cu124 镜像，也可先做实际 GPU 检查并记录构建版本；同一组对比实验应保持环境一致。

如果改用 RTX 5090 等 Blackwell 显卡，应重新选择支持该架构的新版 PyTorch 与 CUDA 12.8 或之后的匹配构建，不直接套用 cu121。本安装脚本不会自动替换 Python、PyTorch、CUDA 驱动或整个云镜像。

### 1.1 主选与备用版本速查（2026-10-19 记录）

找不到完全一致的云镜像时，可按下表选择候选组合。适用范围为 **Ubuntu 22.04、NVIDIA RTX 3090/4090、当前三个基准模型**。表格记录当时考虑的具体组合。选择更新镜像后可直接运行默认检查。

| 定位 | Python | PyTorch | torchvision | 可选 CUDA 构建 | 当前最低版本检查 |
|---|---|---|---|---|---|
| 主选 | **3.10.x** | **2.5.1** | **0.20.1** | **12.1（优先）**、12.4 | 达到下限，继续检查依赖和 GPU |
| 主选的 Python 备用 | 3.11.x | 2.5.1 | 0.20.1 | 12.1、12.4 | 达到下限 |
| 备用 A | 3.10.x 或 3.11.x | 2.5.0 | 0.20.0 | 12.1、12.4 | 低于下限 |
| 备用 B | 3.10.x 或 3.11.x | 2.4.1 | 0.19.1 | 12.1、12.4 | 低于下限 |
| 备用 C | 3.10.x 或 3.11.x | 2.6.0 | 0.21.0 | 12.4、12.6 | 达到下限 |

上述组合依据 [PyTorch 官方安装列表](https://pytorch.org/get-started/previous-versions/)、[torchvision 的 Python 版本对应表](https://github.com/pytorch/vision/blob/v0.21.0/README.md#installation) 和 [torch 2.6.0 的 Python 声明](https://pypi.org/project/torch/2.6.0/) 记录。表中只列出本项目优先考虑的 CUDA 构建，未穷举所有官方构建。PyTorch 和 torchvision 必须按同一行搭配；同一行的多个 CUDA 版本表示择一使用。2.6.0 不套用表中其他行的 cu121。

**备用是选型候选，不表示这些组合都已在目标服务器逐一训练验证。** 官方安装组合成立、基础算子可运行、三个模型的真实数据流程可运行，是三个不同的检查层次。选定后先用真实数据进行少量批次的前向、反向、验证及权重保存/恢复检查，再开展完整训练；三个基准及其对比实验使用同一组已确认的环境。后续外部论文实现和 Mamba 等额外扩展仍需分别检查依赖。

这里的 CUDA 构建指 `torch.version.cuda` 对应的运行时版本，例如 `cu121` 是 CUDA 12.1。云平台还需提供兼容驱动；`nvidia-smi` 显示的 CUDA 版本不能直接当作当前 PyTorch 构建版本。本机 PyTorch 1.10.1＋CUDA 11.3 可保留用于既有环境的流程检查，不作为本表的新服务器首选或备用环境。本机跑通后，服务器仍需独立检查。

## 2. 在新环境中的操作顺序

1. 在 VS Code 或 PyCharm 连接服务器，选择训练时实际用的 Python 解释器。
2. 打开项目的 `code/setup_environment.py`，默认 `MODE = "check"`，直接在 IDE 运行。核心版本相等或更高时显示通过，随后核对普通依赖是否可导入，并用小型 PyTorch 和 torchvision 算子实测 CUDA。
3. Python 太旧时切换解释器或云镜像。PyTorch 或 torchvision 太旧或缺失时，安装相互匹配且支持服务器驱动的组合。普通依赖不足时，按检查给出的 `environment/requirements.txt` 在该解释器里补装。
4. 显示“环境检查通过”后，本目录的 `environment_check.json` 会记录实际版本、解释器、CUDA 构建和 GPU。确认项目的 `data_config.py` 和已有划分，再运行所选模型的 `train.py`。

如需按固定清单安装普通依赖，可以显式设置 `MODE = "install"`。此模式使用当前解释器执行 pip，并锁住已有 PyTorch 和 torchvision 的完整构建版本。只检查 CPU 时可以设 `REQUIRE_CUDA = False`。

### 镜像缺少 PyTorch 时

PyTorch 不必通过浏览器手动下载。若镜像没有对应版本，可在 **IDE 的终端**中执行下面这条环境安装命令，或直接换成已包含对应版本的镜像：

```text
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

此命令只用于已选择好的 Python 3.10 环境。普通基准训练使用官方预编译安装包时，通常无需再单独安装完整 CUDA Toolkit 或 cuDNN；GPU 驱动仍由服务器提供。Mamba 等需要额外编译扩展的情况另行处理。当前任务没有音频输入，不要求安装 torchaudio。

## 3. 文件说明

| 文件 | 用途 | 用户操作 |
|---|---|---|
| `README.md` | 目标环境、准备流程与兼容范围 | 查阅 |
| `requirements.txt` | 固定当前基础依赖及通用数据工具 | 按需补装，或在 install 模式使用 |
| `requirements_models.txt` | 可选 timm 与现成分割模型库 | 需要时把安装脚本的可选开关改 True |
| `constraints.txt` | 定义底座下限及普通依赖的安装约束 | 默认不修改 |
| `../code/setup_environment.py` | IDE 检查最低版本和实际算子 | 直接运行 |
| `environment_check.json` | 成功检查后生成的一份实际环境记录 | 实验结果归档时保留；不必手工编辑 |

当前模型目录的 `requirements.txt` 也已统一引用本目录的基础依赖和约束，避免三个模型各维护一套版本。

## 4. 普通依赖版本

| 包 | 固定版本 | 用途 |
|---|---|---|
| NumPy | 1.26.4 | 温度、标签和数组处理 |
| Pillow | 10.4.0 | 图像读取与缩放 |
| scikit-learn | 1.5.2 | 分类评估 |
| Matplotlib | 3.9.2 | 训练曲线、混淆矩阵 |
| SciPy | 1.13.1 | scikit-learn 等库的基础依赖 |
| pandas | 2.2.3 | 后续表格分析工具，当前基线不直接依赖 |
| tqdm | 4.67.1 | 后续进度显示工具 |
| PyYAML | 6.0.2 | 后续读取 YAML 配置 |
| openpyxl | 3.1.5 | 写入 ResNet34 的 Excel 评估总表 |

这些主要版本支持 Python 3.10，声明依赖范围相容。其他传递依赖由 pip 在约束范围内解析；实际版本记录在 `environment_check.json`，这不是包含所有操作系统库的完整镜像锁文件。

## 5. 后续模型兼容范围

| 模型或实验 | 是否沿用当前底座 | 额外说明 |
|---|---|---|
| CNN＋TCN、CNN＋LSTM、CNN＋轻量 Transformer | 是 | 当前三个基准均使用 PyTorch 基础模块 |
| 时间注意力、门控融合 | 是 | 修改网络结构，不需要更换训练框架 |
| ConvLSTM | 是 | 可以用 PyTorch 实现，具体第三方代码另查依赖 |
| Swin-Tiny＋TCN | 是 | torchvision 0.20.1 自带 `swin_t`；也可按需安装 timm 1.0.15 |
| U-Net | 是 | 可以用纯 PyTorch 实现；现成实现可选 segmentation-models-pytorch 0.4.0 |
| GAF 温度图 | 是 | GAF 是数据转换；可用 NumPy 实现，若使用其他转换库再固定其版本 |
| 辅助监督预训练、自监督预训练 | 通常是 | 训练策略可沿用 PyTorch；外部仓库的额外依赖在接入时确认 |
| CNN＋Mamba | 单独适配后使用 | 涉及 Linux、GPU、CUDA Toolkit、编译器、PyTorch/CUDA/ABI 匹配，不在默认安装清单中 |

需要可选图像或分割库时，将 `INSTALL_OPTIONAL_MODELS=True` 后重新运行安装脚本。可选清单固定为 `timm==1.0.15`、`segmentation-models-pytorch==0.4.0`，其声明依赖与当前底座相容；不代表任意编码器或外部论文实现均已实测通过。

模型版本兼容不等于具体实现可直接替换：仍需保持图像通道、输入形状、历史长度、标签定义、预处理和训练/预测接口一致。任意未来模型或新的第三方包不能预先保证全部兼容，接入时检查，不为了当前不用的模块升级整个基准环境。

## 6. 选帧和温度配对的边界

`prepare_dataset.py` 已按图像清单和采集时间固定每张图的两个温度样本。训练不重新按文件夹排列或选图后的行号匹配温度。

`WINDOW_SIZE=50` 固定本层的历史范围；图像 `all` 读全部 50 张，`uniform` 或 `custom` 只选其中若干张。所选图片仍按从早到晚的顺序进入模型；自定义位置必须递增，不能把图片任意倒序。

当前采用两条分支各自编码同一段历史再融合：例如图像选择 6 张，温度分支仍读取这 50 图历史对应的 100 个温度点。因此两条分支输入长度可以不同，选帧不触发重新配对。若以后要让温度也只保留被选图的 12 个点，需要另行修改明确的温度采样策略，不能只改图像张数。

## 7. 正式训练前的配置

| 配置位置 | 需要确认 |
|---|---|
| IDE 解释器与本环境说明 | Python、PyTorch、CUDA 和实际 GPU 可用 |
| `code/data_config.py` | 已配对数据地址、公共划分、25 fps、每图两个温度、类别与种子 |
| 各模型 `config.py` | 历史范围、选图策略、窗口步长、未来预测距离 |
| 各模型 `config.py` | `BATCH_SIZE`、`EPOCHS`、`LEARNING_RATE`、`IMAGE_SIZE`、`DEVICE`、读取进程数 |

数据和环境确定后，日常主要调整历史范围、选图、步长和预测距离，但训练批量大小、轮数、学习率仍需明确设置。更换 GPU 不会自动决定最佳批量大小。三个基准对比应使用同一份墙名单和相同输入定义。

环境脚本只检查依赖和小型算子，不读取实验数据、不生成划分、不执行模型训练，也不替代真实数据少量批次的前向、反向和权重保存/恢复检查。目标服务器环境尚未由本项目实际完整训练验证。

## 8. 官方依据

- [PyTorch 官方历史版本及安装组合](https://pytorch.org/get-started/previous-versions/)
- [torchvision 0.20 的 Swin-Tiny 接口](https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.swin_t.html)
- [timm 1.0.15 官方依赖声明](https://github.com/huggingface/pytorch-image-models/blob/v1.0.15/pyproject.toml)
- [segmentation-models-pytorch 0.4.0 官方依赖声明](https://github.com/qubvel-org/segmentation_models.pytorch/blob/v0.4.0/pyproject.toml)
- [Mamba 官方安装条件](https://github.com/state-spaces/mamba#installation)
- [PyTorch 2.7 的 Blackwell 支持说明](https://pytorch.org/blog/pytorch-2-7/)
