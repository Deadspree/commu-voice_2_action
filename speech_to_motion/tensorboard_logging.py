"""Native training events and one-time import of existing JSONL training history."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path


class TrainingLogger:
    def __init__(self, run_dir, start_epoch=0, start_step=0):
        self.root = Path(run_dir) / "tensorboard"
        self.start_epoch, self.start_step = start_epoch, start_step

    def __enter__(self):
        from torch.utils.tensorboard import SummaryWriter
        self.stack = ExitStack()
        try:
            self.epochs = self.stack.enter_context(SummaryWriter(
                str(self.root / "epochs"), purge_step=self.start_epoch + 1, flush_secs=10))
            self.batches = self.stack.enter_context(SummaryWriter(
                str(self.root / "batches"), purge_step=self.start_step, flush_secs=10))
        except BaseException:
            self.stack.close()
            raise
        return self

    def batch(self, step, loss, optimizer):
        self.batches.add_scalar("loss/train", loss, step)
        for index, group in enumerate(optimizer.param_groups):
            self.batches.add_scalar(f"learning_rate/{'decoder' if index == 0 else 'encoder'}", group["lr"], step)

    def epoch(self, metrics):
        write_epoch(self.epochs, metrics)
        self.epochs.flush()
        self.batches.flush()

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


def write_epoch(writer, metrics):
    step = metrics["epoch"] + 1
    for key, tag in (("train_loss", "loss/train"), ("val_loss", "loss/validation"),
                     ("best_val", "loss/best_validation"), ("best_train", "loss/best_training")):
        if key in metrics:
            writer.add_scalar(tag, metrics[key], step)
    for split in ("train", "val"):
        report = metrics.get(f"{split}_metrics")
        if report:
            write_angle_metrics(writer, report, split, step)


def write_angle_metrics(writer, report, split, step):
    """Show degree errors and per-joint tolerance accuracy, not classification accuracy."""
    from .common import JOINT_LIMITS, JOINT_NAMES
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import numpy as np
    writer.add_scalar(f"mae_degrees/{split}", report["mae_degrees"], step)
    writer.add_scalar(f"rmse_degrees/{split}", report["rmse_degrees"], step)
    writer.add_scalar(f"joint_limit_violation_percent/{split}",
                      report["joint_limit_violation_fraction"] * 100, step)
    for joint, value in report["per_joint_mae_degrees"].items():
        writer.add_scalar(f"per_joint_mae_degrees/{split}/{joint}", value, step)
    for joint, value in report.get("per_joint_rmse_degrees", {}).items():
        writer.add_scalar(f"per_joint_rmse_degrees/{split}/{joint}", value, step)
    for joint, value in report.get("per_joint_rmse_percent_of_range", {}).items():
        writer.add_scalar(f"per_joint_rmse_percent_of_range/{split}/{joint}", value, step)
    if "per_joint_rmse_degrees" in report:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.axis("off")
        values = [[name, f"[{lo:g}, {hi:g}]", f"{report['per_joint_rmse_degrees'][name]:.3f}",
                   f"{report['per_joint_rmse_percent_of_range'][name]:.2f}%"]
                  for name, (lo, hi) in zip(JOINT_NAMES, JOINT_LIMITS)]
        table = ax.table(cellText=values, colLabels=["Joint", "Range (degrees)", "RMSE (degrees)", "% of range"],
                         cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax.set_title(f"{split}: pose-estimation-style errors (lower is better)")
        fig.tight_layout()
        writer.add_figure(f"rmse_by_joint/{split}", fig, step, close=True)
    for tolerance, value in report["accuracy_percent"].items():
        writer.add_scalar(f"accuracy_percent/{split}/within_{tolerance}_deg", value, step)
    matrix = np.asarray(report["accuracy_matrix_percent"])
    fig, ax = plt.subplots(figsize=(7, 8))
    heat = ax.imshow(matrix, vmin=0, vmax=100, cmap="YlGn")
    ax.set_xticks(range(len(report["tolerances_degrees"])),
                  [f"≤ {t}°" for t in report["tolerances_degrees"]])
    ax.set_yticks(range(len(JOINT_NAMES)), JOINT_NAMES)
    ax.set_title(f"{split}: predictions within angle tolerance (%)")
    ax.set_xlabel("Absolute error against pose-estimator targets")
    for row in range(len(JOINT_NAMES)):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if matrix[row, column] > 65 else "black")
    fig.colorbar(heat, ax=ax, label="Joint predictions (%)")
    fig.tight_layout()
    writer.add_figure(f"accuracy_matrix/{split}", fig, step, close=True)


def import_history(log_root):
    """Backfill older runs once; leave native event streams alone on later calls."""
    from torch.utils.tensorboard import SummaryWriter
    imported = []
    for path in sorted(Path(log_root).glob("*/metrics.jsonl")):
        destination = path.parent / "tensorboard" / "epochs"
        if any(destination.glob("events.out.tfevents.*")):
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        with SummaryWriter(str(destination)) as writer:
            for metrics in rows:
                write_epoch(writer, metrics)
        imported.append(path.parent.name)
    return imported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", default="outputs/speech_to_motion")
    args = parser.parse_args()
    print("Imported history:", import_history(args.log_root))


if __name__ == "__main__":
    main()
