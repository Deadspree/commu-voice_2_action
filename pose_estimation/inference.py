"""Inference script for the CommU DINOv3 joint-angle regressor.

Loads a trained checkpoint and runs inference on a single image (or a
directory of images), returning the predicted joint angles in degrees.

Usage examples:
  python pose_estimation/inference.py --image path/to/image.png
  python pose_estimation/inference.py --image path/to/image.png --checkpoint pose_estimation.pt
  python pose_estimation/inference.py --image path/to/folder --output-mode distribution
"""

from __future__ import annotations

import argparse
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
    )
    span = limits[:, 1] - limits[:, 0]
    sigma_deg = sigma * (span / 2.0)
    return angles.squeeze(0).cpu(), sigma_deg.squeeze(0).cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the CommU DINOv3 joint-angle regressor.")
    parser.add_argument("--image", type=str, required=True, help="Path to an image file or a directory of images.")
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