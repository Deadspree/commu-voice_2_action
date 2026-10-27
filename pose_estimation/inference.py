"""Inference script for the CommU DINOv3 joint-angle regressor.

Loads a trained checkpoint and predicts joint angles from images or every
frame of a video. Video output includes timestamps for Unity CSV playback.

Usage examples:
  python pose_estimation/inference.py --image path/to/image.png
  python pose_estimation/inference.py --image path/to/image.png --checkpoint pose_estimation.pt
  python pose_estimation/inference.py --image path/to/folder --output-mode distribution
  python pose_estimation/inference.py --video path/to/video.mp4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms as T

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from commu_dataloader import CommuDataset
from configs.dataloader_config import DataLoaderConfig
from configs.model_config import ModelConfig
from models.dino_regressor import DINORegressor

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

DEFAULT_CHECKPOINT = "pose_estimation/checkpoints/pose_estimation.pt"


def build_inference_transform(image_size: tuple[int, int]) -> T.Compose:
    """Build the deterministic inference transform (val/test preprocessing)."""
    cfg = DataLoaderConfig()
    return T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=list(cfg.mean), std=list(cfg.std)),
        ]
    )


def load_image(path: Path, transform) -> torch.Tensor:
    """Load a single image and apply the inference transform."""
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)  # add batch dim


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    output_mode: str,
    config: Optional[DataLoaderConfig] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the model and return (predicted_angles_deg, sigma_deg or None).

    Predictions are denormalized from [-1, 1] back to degrees. For the
    "distribution" output mode, the predicted sigma is also returned in
    degrees.
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    if output_mode == "point":
        pred = model(image_tensor)
        angles = CommuDataset.denormalize_joints(pred, config)
        return angles.squeeze(0).cpu(), None

    mu, log_sigma = model(image_tensor)
    angles = CommuDataset.denormalize_joints(mu, config)
    sigma = torch.exp(log_sigma)
    # Sigma is in normalized units; scale by the joint range to get degrees.
    limits = torch.tensor(
        (config or DataLoaderConfig()).joint_limits,
        dtype=torch.float32,
        device=sigma.device,
    )
    span = limits[:, 1] - limits[:, 0]
    sigma_deg = sigma * (span / 2.0)
    return angles.squeeze(0).cpu(), sigma_deg.squeeze(0).cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the CommU DINOv3 joint-angle regressor.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=str, help="Path to an image file or a directory of images.")
    source.add_argument("--video", type=str, help="Video to predict frame by frame, preserving timestamps.")
    parser.add_argument("--output", type=str, help="Video joint CSV destination; defaults to VIDEO_STEM_joints.csv beside the video.")
    parser.add_argument("--batch-size", type=int, default=8, help="Video inference batch size (default: 8).")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing video prediction CSV.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help=f"Path to a saved checkpoint (.pt). Defaults to '{DEFAULT_CHECKPOINT}'.")
    parser.add_argument("--output-mode", type=str, default=None, choices=["point", "distribution"], help="Output mode. Defaults to the mode stored in the checkpoint.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--backbone-weights", type=str, default=None, help="Path or URL to pretrained DINOv3 backbone weights.")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def resolve_video_timestamps(raw: list[float], fps: float) -> tuple[list[float], int]:
    """Repair bad decoder timestamps locally without shifting later valid frames."""
    result = []
    repairs = 0
    index = 0
    while index < len(raw):
        previous = result[-1] if result else -1.0 / fps
        value = raw[index]
        if math.isfinite(value) and value >= 0 and value > previous:
            result.append(value)
            index += 1
            continue
        end = index + 1
        while end < len(raw) and not (math.isfinite(raw[end]) and raw[end] >= 0 and raw[end] > previous):
            end += 1
        step = (raw[end] - previous) / (end - index + 1) if end < len(raw) else 1.0 / fps
        result.extend(previous + step * (j + 1) for j in range(end - index))
        repairs += end - index
        index = end
    return result, repairs


def predict_video(model, video_path: Path, output_path: Path, transform, device,
                  output_mode: str, config: DataLoaderConfig, batch_size: int,
                  checkpoint_path: Path, overwrite: bool = False) -> None:
    """Export frame_index,timestamp_seconds,14 raw joint angles for Unity playback."""
    import cv2

    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if output_path.resolve() == video_path.resolve():
        raise ValueError("Output must not overwrite the input video")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; use --overwrite to replace it")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    fps = capture.get(cv2.CAP_PROP_FPS)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 0
    batch, timestamps = [], []

    try:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid frame rate for timestamp fallback")
        # Read only timing first, so a broken timestamp can be interpolated
        # against its neighbors instead of causing cumulative timing drift.
        raw_timestamps = []
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            raw_timestamps.append(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        frame_times, fallback_count = resolve_video_timestamps(raw_timestamps, fps)
        capture.release()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Cannot reopen video: {video_path}")
        with partial.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame_index", "timestamp_seconds", *JOINT_NAMES])

            def flush_batch():
                nonlocal count
                if not batch:
                    return
                angles, _ = predict(model, torch.stack(batch), device, output_mode, config)
                angles = angles.reshape(-1, len(JOINT_NAMES))
                if not torch.isfinite(angles).all():
                    raise ValueError(f"Non-finite predictions near frame {count}")
                for timestamp, values in zip(timestamps, angles.tolist()):
                    writer.writerow([count, f"{timestamp:.9f}", *[f"{v:.6f}" for v in values]])
                    count += 1
                batch.clear()
                timestamps.clear()
                if count % (batch_size * 4) == 0:
                    print(f"Predicted {count}/{total or '?'} frames", flush=True)

            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                index = count + len(batch)
                if index >= len(frame_times):
                    raise ValueError("Decoded frame count changed between timing and inference passes")
                timestamp = frame_times[index]
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                batch.append(transform(image))
                timestamps.append(timestamp)
                if len(batch) == batch_size:
                    flush_batch()
            flush_batch()
        if count == 0:
            raise ValueError("No video frames decoded")
        if count != len(frame_times):
            raise ValueError("Decoded frame count changed between timing and inference passes")
        if total > 0 and count != total:
            print(f"Warning: decoded {count} frames; container reported {total}. "
                  "Frame-count metadata can be approximate; verify the source if unexpected.", flush=True)
        last_interval = frame_times[-1] - frame_times[-2] if count > 1 else 1.0 / fps
        metadata = {
            "source_video": str(video_path.resolve()), "checkpoint": str(checkpoint_path.resolve()),
            "fps": fps, "frame_count": count, "reported_frame_count": total,
            "width": width, "height": height,
            "duration_seconds": frame_times[-1] + last_interval,
            "timestamp_fallback_frames": fallback_count, "joint_names": JOINT_NAMES,
            "angle_units": "degrees", "predictions": "raw_unclamped", "output_mode": output_mode,
            "joint_limits": config.joint_limits,
        }
        partial.replace(output_path)
        output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved {count} frames to {output_path} ({metadata['duration_seconds']:.3f}s)", flush=True)
    finally:
        capture.release()


def main() -> None:
    args = parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    saved_config = ckpt.get("config", {})

    # Prefer the output mode stored in the checkpoint; fall back to the CLI flag.
    output_mode = args.output_mode or saved_config.get("output_mode", "point")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Output mode: {output_mode}")

    dataloader_cfg = DataLoaderConfig()
    model_cfg = ModelConfig()

    model = DINORegressor(
        num_joints=model_cfg.num_joints,
        pretrained_backbone=model_cfg.pretrained_backbone,
        backbone_weights=args.backbone_weights or model_cfg.backbone_weights,
        freeze_backbone=model_cfg.freeze_backbone,
        dropout=model_cfg.dropout,
        hidden_dim=model_cfg.hidden_dim,
        backbone_name=model_cfg.backbone_name,
        output_mode=output_mode,
        config=model_cfg,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    print("Loaded model weights from checkpoint.")

    transform = build_inference_transform(dataloader_cfg.image_size)

    if args.video:
        video_path = Path(args.video)
        output_path = Path(args.output) if args.output else video_path.with_name(video_path.stem + "_joints.csv")
        predict_video(model, video_path, output_path, transform, device, output_mode,
                      dataloader_cfg, args.batch_size, checkpoint_path, args.overwrite)
        return

    image_path = Path(args.image)
    if image_path.is_dir():
        images = sorted(
            [p for p in image_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
        )
        if not images:
            raise FileNotFoundError(f"No images found in directory: {image_path}")
    else:
        images = [image_path]

    print(f"\n=== Predictions ({len(images)} image(s)) ===")
    for img in images:
        tensor = load_image(img, transform)
        angles, sigma = predict(model, tensor, device, output_mode, config=dataloader_cfg)
        print(f"\n{img.name}:")
        for idx, name in enumerate(JOINT_NAMES):
            val = angles[idx].item()
            if sigma is not None:
                print(f"  {name:<14} {val:>8.3f} deg  (±{sigma[idx].item():.3f})")
            else:
                print(f"  {name:<14} {val:>8.3f} deg")


if __name__ == "__main__":
    main()
