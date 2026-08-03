from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class DataLoaderConfig:
    """Configuration for data loading, augmentation, and dataset splitting."""

    data_root: str = "dataset/commu_pose_dataset"
    image_size: Tuple[int, int] = (224, 224)
    num_joints: int = 14
    split_seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    train_affine_translate: Tuple[float, float] = (0.05, 0.05)
    train_affine_scale: Tuple[float, float] = (0.95, 1.05)
    train_color_jitter_brightness: float = 0.2
    train_color_jitter_contrast: float = 0.2
    train_color_jitter_saturation: float = 0.2
    train_color_jitter_hue: float = 0.05
    train_gaussian_blur_sigma: Tuple[float, float] = (0.1, 2.0)

    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def resolved_data_root(self) -> Path:
        return Path(self.data_root).expanduser().resolve()


DEFAULT_DATALOADER_CONFIG = DataLoaderConfig()
