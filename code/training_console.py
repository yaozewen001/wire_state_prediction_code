"""训练、验证、测试共用的终端进度条。"""

import sys
import time
from tqdm import tqdm


def clock_text(seconds):
    if seconds is None:
        return "待测"

    seconds = max(0, int(seconds))
    return "{:02d}:{:02d}:{:02d}".format(
        seconds // 3600,
        seconds % 3600 // 60,
        seconds % 60
    )


class PhaseProgress:
    def __init__(self, total, settings, phase, epoch=None, epochs=None):
        self.total = int(total)
        self.phase = phase
        self.started = time.perf_counter()
        self.count = 0

        self.desc = (
            phase if epoch is None
            else "{} {:02d}/{}".format(phase, epoch, epochs)
        )

        # 强制显示进度条，不再检测终端类型
        self.bar = tqdm(
            total=self.total,
            desc=self.desc,
            unit="批",
            file=sys.stdout,
            disable=False,
            dynamic_ncols=True,
            mininterval=0.5,
            miniters=1,
            leave=False,
            ascii=False,
            bar_format=(
                "{desc} {percentage:3.0f}%|{bar:12}| "
                "{n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}]{postfix}"
            )
        )

    def update(self, completed, loss, wall, layer):
        self.count = int(completed)

        short_wall = str(wall).removeprefix("wall_")

        detail = "loss={:.4f} | 墙:{} 层:{}".format(
            loss,
            short_wall,
            layer if layer is not None else "未分层"
        )

        self.bar.set_postfix_str(detail, refresh=False)
        self.bar.update(max(0, self.count - self.bar.n))

    def close(self):
        self.bar.close()


def print_epoch_summary(
    epoch, epochs, row, best_epoch,
    stale, patience, improved
):
    flag = (
        "刷新最佳" if improved
        else "等待 {}/{}".format(stale, patience)
    )

    message = (
        "第 {}/{} 轮 | train loss {:.4f} | "
        "val loss {:.4f} | val F1 {:.4f} | "
        "val acc {:.2%} | {} | 最佳第{}轮\n"
        "训练{} 验证{} 本轮{} 累计{} | "
        "跑满最大轮数预计还需{}（早停前估计）"
    ).format(
        epoch,
        epochs,
        row["train_loss"],
        row["val_loss"],
        row["val_macro_f1"],
        row["val_accuracy"],
        flag,
        best_epoch,
        clock_text(row["train_seconds"]),
        clock_text(row["validation_seconds"]),
        clock_text(row["epoch_seconds"]),
        clock_text(row["cumulative_seconds"]),
        clock_text(row["estimated_seconds_to_max_epochs"])
    )

    print(message, flush=True)