"""调用公共评估报表，绘制本模型的训练曲线。"""

from evaluation_reports import REPORT_FORMAT_VERSION, _metrics, save_evaluation_reports
from layer_event_reports import save_layer_event_report


def plot_history(output, history):
    from matplotlib import pyplot as plt  # 复用评估模块已配置的离线绘图后端。

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row["epoch"] for row in history]
    for phase in ("train", "val"):
        axes[0].plot(epochs, [row[phase + "_loss"] for row in history], marker="o", label=phase)
        axes[1].plot(epochs, [row[phase + "_macro_f1"] for row in history], marker="o", label=phase)
    for axis, title in zip(axes, ("Weighted loss", "Macro F1")):
        axis.set(xlabel="Epoch", title=title)
        axis.legend()
        axis.grid(alpha=0.2)
        if len(epochs) == 1:
            axis.set_xticks(epochs)
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=160)
    plt.close(figure)
