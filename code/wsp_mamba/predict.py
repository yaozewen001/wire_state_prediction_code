"""回放一面已配对的墙，导出逐窗未来状态和三类概率。"""

if __package__:
    from .config import get_settings
    from .dataset import WindowDataset, make_loader
    from .evaluate import restore_data, restore_model, restore_run, run_loader
else:
    from config import get_settings
    from dataset import WindowDataset, make_loader
    from evaluate import restore_data, restore_model, restore_run, run_loader

from data_utils import write_csv
from training_acceleration import PrecisionPolicy
from wall_runtime import setup_environment


def main(current=None):
    current = get_settings() if current is None else current
    run, settings, checkpoint, epoch, report_root = restore_run(current)
    device = setup_environment(settings)
    model, criterion = restore_model(run, settings, checkpoint, device)
    sequences, windows, normalization = restore_data(run, settings)
    wall = current["predict_sequence_id"].strip() or windows["test"][0]["sequence_id"]
    selected = [window for phase in windows.values() for window in phase if window["sequence_id"] == wall]
    if not selected:
        raise ValueError(f"{wall} 在该运行快照中没有有效窗口。请检查 PREDICT_SEQUENCE_ID。")
    dataset = WindowDataset(sequences, selected, settings, normalization)
    loader = make_loader(dataset, settings, device)
    result = run_loader(model, loader, criterion, device, PrecisionPolicy(settings, device), "单墙回放")
    names = settings["class_names"]
    rows = []
    for index, prediction, scores in zip(result["indices"], result["predictions"], result["scores"]):
        window = selected[index]
        row = {name: window[name] for name in ("sequence_id", "layer_id", "end_seq_idx", "target_seq_idx",
                                               "end_time_s", "target_time_s", "target_state")}
        row.update(predicted_state=names[prediction], future_frames=settings["future_frames"], checkpoint_epoch=epoch)
        row.update({f"score_{name}": float(score) for name, score in zip(names, scores)})
        rows.append(row)
    report_root.mkdir(parents=True, exist_ok=True)
    output = report_root / f"predictions_{wall}.csv"
    write_csv(output, rows, list(rows[0]))
    print(f"单墙回放结果 {output}", flush=True)
    return output


if __name__ == "__main__":
    main()

