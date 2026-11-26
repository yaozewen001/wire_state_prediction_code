# -*- coding: utf-8 -*-
"""在 IDE 检查最低版本、依赖导入和 GPU 算子。普通依赖安装可单独启用。"""

import importlib  # 安装完成后按模块名检查是否能真正导入。
import json  # 将检查通过的实际环境保存为一份记录。
import platform  # 记录操作系统，区分 Windows 本地和 Linux 服务器。
import subprocess  # 使用当前 IDE 解释器执行 pip，避免装到另一个 Python 中。
import sys  # 获取正在使用的解释器及 Python 版本。
import tempfile  # 安装期间临时约束 PyTorch 的完整 CUDA 构建，不留下额外配置文件。
from datetime import datetime, timezone  # 记录环境检查时间。
from pathlib import Path  # 按项目目录定位依赖清单，不依赖 IDE 工作目录。

# ==================== 一、在 IDE 中运行前确认这几项 ====================
MODE = "check"  # check 只检查。install 使用固定清单安装普通依赖后再检查。
INSTALL_OPTIONAL_MODELS = False  # 当前三个基准保持 False；需要 timm 或现成 U-Net 库时再改 True。
REQUIRE_CUDA = True  # 云 GPU 训练保持 True。仅检查 CPU 时设为 False。
MIN_PYTHON_VERSION = "3.10.0"  # 项目支持检查的 Python 下限。
MIN_TORCH_VERSION = "2.5.1"  # PyTorch 最低公开版本，保留完整 CUDA 构建显示。
MIN_TORCHVISION_VERSION = "0.20.1"  # 视觉库下限，实际配对由算子检查验证。
PROJECT_DIR = Path(__file__).resolve().parents[1]  # 本文件位于项目的 code/ 中。
ENVIRONMENT_DIR = PROJECT_DIR / "environment"  # 所有环境说明、依赖和检查结果集中放在此目录。


def package_version(name):  # 按 pip 发行包名读取实际安装版本。
    from importlib import metadata  # Python 版本核对后再读取包记录。

    try:
        return metadata.version(name)  # 保留 +cu124、+cpu 等构建标记供终端显示。
    except metadata.PackageNotFoundError:
        return None  # 由检查入口汇总缺少的软件包。


def compare_versions(actual, minimum):  # 按发布版本数值比较，预发布版也遵守版本顺序。
    try:
        from packaging.version import Version
    except ModuleNotFoundError:  # 初始环境可以复用 pip 自带的版本比较工具。
        from pip._vendor.packaging.version import Version

    current = Version(actual.split("+", 1)[0])  # CUDA 构建后缀参与显示，公开版本参与比较。
    required = Version(minimum)
    return (current > required) - (current < required)


def print_version_check(name, actual, minimum):  # 核心版本逐项显示，缺失时也保留明确的最低要求。
    if actual is None:
        print(f"{name:<12} 未安装 | 最低 {minimum} | 不通过", flush=True)
        return False
    order = compare_versions(actual, minimum)
    operator = {-1: "<", 0: "=", 1: ">"}[order]
    status = "通过" if order >= 0 else "低于要求"
    print(f"{name:<12} {actual} {operator} {minimum} | {status}", flush=True)
    return order >= 0


def check_prerequisites():  # 在导入 PyTorch 或安装依赖前检查三个核心版本。
    python_version = platform.python_version()
    if sys.version_info[:2] < (3, 8):  # 太旧的解释器只显示 Python 检查结果。
        print(f"Python       {python_version} < {MIN_PYTHON_VERSION} | 低于要求", flush=True)
        raise RuntimeError(f"请先在 IDE 或云镜像选择 Python {MIN_PYTHON_VERSION} 或更新版本。")
    python_ok = print_version_check("Python", python_version, MIN_PYTHON_VERSION)
    torch_ok = print_version_check("PyTorch", package_version("torch"), MIN_TORCH_VERSION)
    vision_ok = print_version_check("torchvision", package_version("torchvision"), MIN_TORCHVISION_VERSION)
    fixes = []
    if not python_ok:
        fixes.append(f"请在 IDE 或云镜像选择 Python {MIN_PYTHON_VERSION} 或更新版本。")
    if not (torch_ok and vision_ok):
        fixes.append(f"请按 {ENVIRONMENT_DIR / 'README.md'} 安装相互匹配的 PyTorch 和 torchvision，或选择预装镜像。")
    if fixes:
        raise RuntimeError("\n".join(fixes))


def selected_requirement_files():  # 根据开关决定安装基础依赖还是额外模型库。
    files = [ENVIRONMENT_DIR / "requirements.txt"]  # 三个基准必须的依赖及通用数据工具。
    if INSTALL_OPTIONAL_MODELS:  # 不提前安装 Mamba 或其他编译扩展。
        files.append(ENVIRONMENT_DIR / "requirements_models.txt")  # timm 和现成分割模型库仅按需安装。
    return files  # 同一组文件也用于安装后的版本检查。


def required_versions(files):  # 从清单读取固定版本，避免在 Python 中维护第二套版本号。
    versions = {}  # 按发行包名保存固定版本。
    for path in files:  # 基础包与可选包使用相同解析规则。
        for line in path.read_text(encoding="utf-8-sig").splitlines():  # 兼容中文注释与 BOM。
            value = line.split("#", 1)[0].strip()  # 注释不作为依赖内容。
            if not value:  # 跳过空行和纯注释。
                continue  # 不产生空包名。
            if value.count("==") != 1:  # 本项目清单只允许明确固定版本。
                raise ValueError(f"依赖清单应为 包名==版本：{path.name} 中的 {value}")  # 不静默接受未固定版本。
            name, version = value.split("==")  # 取得发行包名和目标版本。
            if name in versions and versions[name] != version:  # 防止两份清单互相冲突。
                raise ValueError(f"依赖清单中 {name} 的固定版本冲突。")  # 在安装之前拦截。
            versions[name] = version  # 同名同版本只检查一次。
    return versions  # 不联网，也不修改文件。


def install_command(files, core_constraint_path=None):  # 生成参数列表，不拼接 shell 命令，支持路径空格和中文。
    command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]  # 始终安装到当前 IDE 解释器。
    for path in files:  # 把依赖文件明确传给 pip。
        command.extend(["--requirement", str(path)])  # 路径作为独立参数传递。
    command.extend(["--constraint", str(ENVIRONMENT_DIR / "constraints.txt")])  # 可选模型库也不能自动升级 PyTorch 底座。
    if core_constraint_path is not None:  # 安装时额外固定当前已经装好的完整 PyTorch 构建版本。
        command.extend(["--constraint", str(core_constraint_path)])  # 防止 +cu121 被替换为同公开版本的其他构建。
    return command  # 由主函数决定是否实际安装。


def check_dependencies(files, exact=False):  # 检查模式接受更高版本，安装模式核对固定清单。
    versions = required_versions(files)  # 普通依赖要求以清单为准。
    failures = []
    for name, expected in versions.items():
        actual = package_version(name)
        if actual is None:
            failures.append(f"{name} 未安装 | {'固定' if exact else '最低'} {expected}")
            continue
        order = compare_versions(actual, expected)
        if order < 0 or (exact and order != 0):
            operator = "<" if order < 0 else ">"
            failures.append(f"{name} {actual} {operator} {expected}")
    if failures:
        requirements = "\n".join(str(path) for path in files)
        raise RuntimeError("普通依赖版本未达要求：\n" + "\n".join(failures)
                           + "\n请在当前 IDE 解释器按以下清单补装：\n" + requirements)

    import_names = {"Pillow": "PIL", "scikit-learn": "sklearn", "PyYAML": "yaml"}
    for name in versions:  # 实际导入可发现二进制和传递依赖问题。
        module = import_names.get(name, name.replace("-", "_"))
        importlib.import_module(module)
    if exact:  # 安装模式保留 pip 的全环境依赖核对。
        subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    print(f"普通依赖检查通过 | {len(versions)} 项", flush=True)
    return versions  # 保存报告时列出项目所需版本。


def check_torch_runtime():  # 确认 torch、torchvision 和所选 GPU 可以执行实际算子。
    import torch  # 基础依赖准备好后再导入，避免旧 NumPy 等包干扰检查。
    import torchvision  # 检查与 torch 成对安装的视觉库。
    from torchvision.ops import nms  # 检查 torchvision 编译扩展，及早发现 CPU/GPU 构建混装。
    cuda_available = torch.cuda.is_available()  # 不能只根据 nvidia-smi 或安装包名称认定 GPU 可用。
    if REQUIRE_CUDA and not cuda_available:  # 租 GPU 时禁止检查通过后悄悄转到 CPU 训练。
        raise RuntimeError("当前解释器的 CUDA 不可用。请检查 GPU 实例、驱动及 PyTorch GPU 安装版本；不要直接开始正式训练。")  # 不在脚本里安装驱动。
    device = torch.device("cuda" if cuda_available else "cpu")  # CPU 检查模式也使用实际可用的设备。
    values = torch.ones((8, 8), device=device)  # 使用很小的张量，避免占用训练显存。
    product = values @ values  # 执行一个真实计算，检查 GPU 架构和运行时是否匹配。
    if not torch.isfinite(product).all().item():  # 触发结果回读并检查有限数值。
        raise RuntimeError("PyTorch 小张量计算结果异常。")  # 不把算子执行失败记录为通过。
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 1.1, 1.1]], device=device)  # 测试 torchvision 的小输入。
    scores = torch.tensor([0.9, 0.8], device=device)  # 不下载权重，不读取实验图像。
    if nms(boxes, scores, 0.5).numel() < 1:  # 在同一设备检查视觉库扩展是否工作。
        raise RuntimeError("torchvision 编译扩展检查失败。")  # 常见原因是安装包构建不匹配。
    return {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__), "cuda_runtime": torch.version.cuda, "cuda_available": cuda_available, "device": str(device), "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None}  # 如实记录实际环境。


def main():  # 在 IDE 中运行本文件即可，不需要命令行参数。
    if MODE not in ("install", "check"):  # 防止拼错模式后误安装。
        raise ValueError('MODE 只能为 "install" 或 "check"。')  # 提示合法值。
    print(f"当前解释器 {sys.executable}\n模式 {MODE}", flush=True)  # 显示实际检查的 IDE 解释器。
    check_prerequisites()  # Python 或 PyTorch 不符合时在任何安装之前停止。
    files = selected_requirement_files()  # 选择当前需要的依赖清单。
    if MODE == "install":  # 只有此模式会联网安装普通 Python 包。
        required_versions(files)  # 安装前检查清单及版本冲突。
        core_versions = {name: package_version(name) for name in ("torch", "torchvision")}  # 保留当前镜像的完整版本和 CUDA 构建后缀。
        with tempfile.TemporaryDirectory(prefix="wire_state_dependencies_") as temporary_dir:  # 临时约束使用系统临时目录，不污染实验数据。
            core_constraints = Path(temporary_dir) / "pytorch_build_constraints.txt"  # 仅在此次安装中有效。
            core_constraints.write_text("\n".join(f"{name}=={version}" for name, version in core_versions.items()) + "\n", encoding="utf-8")  # 在公共版本之外固定完整构建。
            subprocess.run(install_command(files, core_constraints), check=True)  # pip 的下载和错误信息直接显示在 IDE 控制台。
        if any(package_version(name) != version for name, version in core_versions.items()):  # 安装后核对镜像的 PyTorch 底座仍然一致。
            raise RuntimeError("安装后 PyTorch 构建版本出现变化，请检查当前环境；本次不记录成功。")  # 不把底座变化静默当成正常安装。
    versions = check_dependencies(files, exact=(MODE == "install"))  # 检查实际包和导入。
    runtime = check_torch_runtime()  # 小张量计算检查，不启动任何模型训练。
    from importlib import metadata  # 记录检查通过的实际包版本。

    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "optional_model_libraries": INSTALL_OPTIONAL_MODELS,
        "minimum_core_versions": {  # 记录实际使用的检查下限。
            "python": MIN_PYTHON_VERSION,
            "torch": MIN_TORCH_VERSION,
            "torchvision": MIN_TORCHVISION_VERSION,
        },
        "required_versions": versions,
        "runtime": runtime,
        "installed_packages": {  # 记录直接和传递依赖的实际版本。
            distribution.metadata["Name"]: distribution.version
            for distribution in metadata.distributions()
            if distribution.metadata.get("Name")
        },
    }
    ENVIRONMENT_DIR.mkdir(parents=True, exist_ok=True)  # 只在项目环境目录保存检查结果。
    report_path = ENVIRONMENT_DIR / "environment_check.json"  # 每次成功检查更新同一份文件，避免生成大量记录目录。
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")  # 失败时不会写出新的成功报告。
    print(f"环境检查通过。\n实际计算设备：{runtime['device']} / {runtime['gpu_name']}\n实际 PyTorch CUDA：{runtime['cuda_runtime']}\n检查记录：{report_path}\n下一步：确认 data_config.py 和已有划分，再运行所选模型的 train.py。", flush=True)  # 此通过只代表依赖与算子，不代表完成训练验证。


if __name__ == "__main__":  # 导入工具函数不会安装软件。
    try:  # 在 IDE 中提供清楚的错误信息和失败退出状态。
        main()  # 只有用户直接运行此文件才执行安装或检查。
    except (OSError, RuntimeError, ValueError, ImportError, subprocess.CalledProcessError) as error:  # 保留失败状态，不报告虚假的安装成功。
        print(f"环境准备未完成：{error}", file=sys.stderr, flush=True)  # 显示具体原因。
        raise SystemExit(1) from error  # IDE 看到非零退出码，便于识别失败。
