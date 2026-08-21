"""Check the angle distribution of each joint in the CommU dataset.

Reads angle_joints.csv (columns: index, image_name, joint_1..joint_14) and plots
a histogram for each joint, overlaid with the joint's physical [Min, Max] range
(from DataLoaderConfig.joint_limits). This helps spot under-represented regions
of each joint's range, which is important for training a pose regressor.

Usage:
  python dataset/commu_pose_dataset/check_distribution.py
  python dataset/commu_pose_dataset/check_distribution.py --save
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Allow importing the dataloader config for the joint limits.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pose_estimation.configs.dataloader_config import DEFAULT_DATALOADER_CONFIG

# Joint names, in order (index 0..13). Matches CS_JointLimitDefinition.cs.
JOINT_NAMES = [
    "BodyYaw",
    "BodyPitch",
    "RightArmPitch",
    "RightArmRoll",
    "LeftArmPitch",
    "LeftArmRoll",
    "FacePitch",
    "FaceYaw",
    "FaceRoll",
    "EyePitch",
    "RightEyeYaw",
    "LeftEyeYaw",
    "Eyelid",
    "Mouth",
]

DEFAULT_CSV = Path(__file__).resolve().parent / "angle_joints.csv"


def load_joint_data(csv_path: Path) -> list[list[float]]:
    """Return a list of 14 joint-angle lists, one per joint."""
    joints: list[list[float]] = [[] for _ in range(14)]
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or len(row) < 16:
                continue
            # Skip a header row if present (first cell is non-numeric).
            try:
                float(row[0])
            except ValueError:
                continue
            for j in range(14):
                joints[j].append(float(row[2 + j]))
    return joints


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-joint angle distributions.")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV), help="Path to angle_joints.csv")
    parser.add_argument("--save", action="store_true", help="Save the figure to a PNG instead of showing it.")
    parser.add_argument("--out", type=str, default="joint_distribution.png", help="Output path when --save is used.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    joints = load_joint_data(csv_path)
    limits = DEFAULT_DATALOADER_CONFIG.joint_limits

    n = len(JOINT_NAMES)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for j in range(n):
        ax = axes[j]
        lo, hi = limits[j]
        data = joints[j]
        ax.hist(data, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        # Mark the physical range boundaries.
        ax.axvline(lo, color="red", linestyle="--", linewidth=1.5, label=f"min {lo:.0f}")
        ax.axvline(hi, color="green", linestyle="--", linewidth=1.5, label=f"max {hi:.0f}")
        ax.set_title(f"{j}: {JOINT_NAMES[j]}  [{lo:.0f}, {hi:.0f}]")
        ax.set_xlabel("angle (deg)")
        ax.set_ylabel("count")
        ax.legend(fontsize=7)
        # Show coverage of the physical range.
        if data:
            cov = (max(data) - min(data)) / (hi - lo) * 100.0 if hi > lo else 0.0
            ax.text(
                0.02, 0.95, f"n={len(data)}  cov={cov:.0f}%",
                transform=ax.transAxes, fontsize=8, va="top",
            )

    # Hide any unused subplot axes.
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("CommU joint angle distributions (physical range in red/green)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if args.save:
        out = Path(args.out)
        fig.savefig(out, dpi=150)
        print(f"Saved figure to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()