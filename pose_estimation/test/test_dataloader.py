from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from commu_dataloader import CommuDataset
from configs.dataloader_config import DataLoaderConfig


def inspect_dataset_samples(
    num_samples: int = 5,
    split: str = "train",
    show_images: bool = True,
    config: DataLoaderConfig | None = None,
) -> None:
    """Inspect a small number of samples from the dataset without writing any files.

    This is useful for quick debugging and sanity checks while keeping the
    existing folder structure unchanged.
    """
    data_root = Path(__file__).resolve().parents[1] / ".." / "dataset" / "commu_pose_dataset"
    cfg = config or DataLoaderConfig()
    dataset = CommuDataset(data_root=data_root, split=split, image_size=(224, 224), config=cfg)

    print("Using config:")
    print(f"  train_ratio={cfg.train_ratio}, val_ratio={cfg.val_ratio}, test_ratio={cfg.test_ratio}")
    print(f"  affine_translate={cfg.train_affine_translate}, affine_scale={cfg.train_affine_scale}")
    print(f"  color_jitter=brightness={cfg.train_color_jitter_brightness}, contrast={cfg.train_color_jitter_contrast}, saturation={cfg.train_color_jitter_saturation}, hue={cfg.train_color_jitter_hue}")
    print(f"  blur_sigma={cfg.train_gaussian_blur_sigma}, mean={cfg.mean}, std={cfg.std}")

    if len(dataset) < num_samples:
        raise ValueError(f"Requested {num_samples} samples but dataset only has {len(dataset)}")

    print(f"Inspecting {num_samples} samples from split='{split}'")
    for idx in range(num_samples):
        image, joints = dataset[idx]
        assert isinstance(image, torch.Tensor), f"Image at index {idx} is not a tensor"
        assert image.shape[0] == 3, f"Expected 3 channels, got shape {image.shape}"
        assert joints.shape == torch.Size([14]), f"Expected 14 joint angles, got {joints.shape}"

        print(
            f"sample {idx}: image_shape={tuple(image.shape)} "
            f"dtype={image.dtype} joints_shape={tuple(joints.shape)} "
            f"joints={joints[:5].tolist()}"
        )

        if show_images:
            image_np = image.permute(1, 2, 0).cpu().numpy()
            image_np = np.clip(image_np, 0.0, 1.0)
            image_np = (image_np * 255).astype(np.uint8)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            cv2.imshow(f"sample {idx}", image_np)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    config = DataLoaderConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        train_affine_translate=(0.03, 0.03),
        train_affine_scale=(0.9, 1.1),
        train_color_jitter_brightness=0.15,
        train_color_jitter_contrast=0.15,
        train_color_jitter_saturation=0.15,
        train_color_jitter_hue=0.03,
        train_gaussian_blur_sigma=(0.2, 1.2),
    )
    inspect_dataset_samples(num_samples=5, split="val", show_images=True, config=config)
