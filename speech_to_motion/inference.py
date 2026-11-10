"""Predict timestamped CommU joint angles from audio with overlap-blended windows."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .audio import decode_audio
from .common import JOINT_LIMITS, JOINT_NAMES, write_json
from .dataset import window_starts
from .runtime import predict_audio, checkpoint_model, load_checkpoint
from .normalization import denormalize_angles


def export_csv(path, times, angles, metadata, overwrite=False):
    path = Path(path)
    metadata_path = path.with_suffix(".metadata.json")
    if not overwrite and (path.exists() or metadata_path.exists()):
        raise FileExistsError(f"Output already exists: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "timestamp_seconds", *JOINT_NAMES])
        for index, (timestamp, values) in enumerate(zip(times, angles)):
            writer.writerow([index, f"{timestamp:.9f}", *[f"{v:.6f}" for v in values]])
    write_json(metadata_path, {**metadata, "joint_names": JOINT_NAMES, "angle_units": "degrees",
                               "frame_count": len(times), "joint_limits": JOINT_LIMITS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--clamp", action="store_true", help="Clamp exported angles to Unity joint limits")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    device = torch.device(args.device)
    model = checkpoint_model(checkpoint).to(device).eval()
    cfg = checkpoint["config"]
    audio = decode_audio(args.audio, cfg["data"]["sample_rate"], args.ffmpeg)
    times, angles = predict_audio(model, audio, cfg, checkpoint["normalization"], device)
    if args.clamp:
        limits = np.asarray(JOINT_LIMITS)
        angles = angles.clip(limits[:, 0], limits[:, 1])
    export_csv(args.output, times, angles,
               {"source_audio": str(Path(args.audio).resolve()), "checkpoint": str(Path(args.checkpoint).resolve()),
                "fps": cfg["data"]["motion_fps"], "predictions": "clamped" if args.clamp else "raw_unclamped",
                "duration_seconds": len(audio) / cfg["data"]["sample_rate"],
                "context": "offline bidirectional; overlapping windows blended"}, args.overwrite)
    print(args.output)


if __name__ == "__main__":
    main()
