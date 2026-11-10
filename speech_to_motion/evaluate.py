"""Evaluate stitched clips once per timestamp against pseudo-labels and mean pose."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .audio import decode_audio
from .common import JOINT_LIMITS, JOINT_NAMES, read_json, write_json
from .dataset import MotionDataset
from .inference import predict_audio
from .prepare_dataset import read_motion
from .train import checkpoint_model, load_checkpoint
from .metrics import AngleMetrics
from .tensorboard_logging import write_angle_metrics
from .normalization import training_mean


def evaluate(checkpoint, cfg, split, device):
    dataset = MotionDataset(cfg, split)
    if dataset.summary["dataset_id"] != checkpoint["dataset_id"]:
        raise ValueError("Evaluation dataset differs from checkpoint dataset")
    model = checkpoint_model(checkpoint).to(device).eval()
    errors, baseline_errors = np.zeros(14), np.zeros(14)
    velocity_error = np.zeros(14)
    frames = transitions = violations = 0
    clips = []
    accuracy = AngleMetrics({"mean": [0] * 14, "std": [1] * 14})
    for record in dataset.records:
        audio = decode_audio(dataset.root / record["audio"], cfg["data"]["sample_rate"], cfg["data"]["ffmpeg"])
        times, prediction = predict_audio(model, audio, cfg, checkpoint["normalization"], device,
                                           record["start_seconds"], record["end_seconds"])
        source_times, angles = read_motion(dataset.root / record["motion"])
        target = np.stack([np.interp(times, source_times, angles[:, j]) for j in range(14)], axis=1)
        error = np.abs(prediction - target)
        accuracy.update(torch.from_numpy(prediction)[None], torch.from_numpy(target)[None],
                        torch.ones(1, len(times), dtype=torch.bool))
        errors += error.sum(0)
        baseline_errors += np.abs(target - training_mean(checkpoint["normalization"])).sum(0)
        velocity_error += np.abs(np.diff(prediction, axis=0) - np.diff(target, axis=0)).sum(0) * cfg["data"]["motion_fps"]
        limits = np.asarray(JOINT_LIMITS)
        violations += int(((prediction < limits[:, 0]) | (prediction > limits[:, 1])).sum())
        frames += len(times)
        transitions += max(0, len(times) - 1)
        clips.append({"id": record["id"], "frames": len(times), "mae_degrees": float(error.mean())})
        print(f"{len(clips)}/{len(dataset.records)} clips", flush=True)
    return {**accuracy.compute(), "split": split, "dataset_id": checkpoint["dataset_id"], "frames": frames,
            "mae_degrees": float(errors.sum() / (frames * 14)),
            "per_joint_mae_degrees": dict(zip(JOINT_NAMES, (errors / frames).tolist())),
            "mean_pose_mae_degrees": float(baseline_errors.sum() / (frames * 14)),
            "velocity_mae_degrees_per_second": float(velocity_error.sum() / (max(transitions, 1) * 14)),
            "joint_limit_violation_fraction": violations / (frames * 14), "clips": clips,
            "target_note": "Pose-estimator pseudo-labels; metrics do not establish true physical pose accuracy"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--data-root")
    parser.add_argument("--prepared-dir")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tensorboard-dir", help="Write full-clip metrics and accuracy matrix here")
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    cfg = checkpoint["config"]
    for key, value in (("root", args.data_root), ("prepared_dir", args.prepared_dir), ("ffmpeg", args.ffmpeg)):
        if value:
            cfg["data"][key] = value
    result = evaluate(checkpoint, cfg, args.split, torch.device(args.device))
    write_json(args.output, result)
    if args.tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter
        with SummaryWriter(args.tensorboard_dir) as writer:
            write_angle_metrics(writer, result, args.split, checkpoint["epoch"] + 1)
    print(f"MAE: {result['mae_degrees']:.4f} degrees; mean-pose baseline: {result['mean_pose_mae_degrees']:.4f}")


if __name__ == "__main__":
    main()
