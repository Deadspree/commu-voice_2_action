from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

try:
    from .configs.dataloader_config import DataLoaderConfig, DEFAULT_DATALOADER_CONFIG
except ImportError:  # pragma: no cover - fallback for direct script execution
    from configs.dataloader_config import DataLoaderConfig, DEFAULT_DATALOADER_CONFIG


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "dataset" / "commu_pose_dataset"
DEFAULT_IMAGE_SIZE: tuple[int, int] = (224, 224)
DEFAULT_NUM_JOINTS = 14
DEFAULT_SEED = 42


class CommuDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch dataset for CommU robot pose estimation.

    The dataset expects a directory tree containing:
    - images/: RGB images referenced by labels.csv
    - labels.csv: one row per sample with the format:
      <sample_index>,<image_filename>,<joint_angle_1>,...,<joint_angle_14>

    The dataset is split into train/validation/test subsets before any
    augmentation is applied. The split is deterministic thanks to a fixed seed.
    """

    def __init__(
        self,
        data_root: Optional[Union[str, Path]] = None,
        split: str = "train",
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        split_seed: int = DEFAULT_SEED,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        transform: Optional[T.Compose] = None,
        config: Optional[DataLoaderConfig] = None,
    ) -> None:
        self.config = config or DEFAULT_DATALOADER_CONFIG
        self.data_root = Path(data_root or self.config.resolved_data_root() or DEFAULT_DATA_ROOT).expanduser().resolve()
        self.split = split.lower()
        self.split_seed = split_seed if split_seed != DEFAULT_SEED else self.config.split_seed
        self.train_ratio = train_ratio if train_ratio != 0.8 else self.config.train_ratio
        self.val_ratio = val_ratio if val_ratio != 0.1 else self.config.val_ratio
        self.test_ratio = test_ratio if test_ratio != 0.1 else self.config.test_ratio
        self.image_size = self._normalize_image_size(image_size or self.config.image_size or DEFAULT_IMAGE_SIZE)
        self.images_dir = self.data_root / "images"
        self.labels_path = self._resolve_labels_path(self.data_root)

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.images_dir}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file does not exist: {self.labels_path}")

        self.samples = self._load_samples()
        self.transform = transform or self._build_default_transform(self.split, self.image_size, self.config)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, joint_values = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # Per-joint min-max normalization to [-1, 1] using the physical joint
        # limits. Joints have very different ranges (e.g. FaceYaw is +/-10 deg
        # while RightArmPitch is -180..+5 deg), so a single global mean/std
        # would compress small-range joints and let large-range joints dominate
        # the loss. Mapping each joint's [Min, Max] to [-1, 1] equalizes all
        # outputs to a common scale. Denormalization back to degrees is:
        #   angle = (norm + 1) * (max - min) / 2 + min
        joint_tensor = torch.tensor(joint_values, dtype=torch.float32)
        joint_tensor = self._normalize_joints(joint_tensor)
        return image, joint_tensor

    def _normalize_joints(self, joints: torch.Tensor) -> torch.Tensor:
        """Map each joint's [Min, Max] physical range to [-1, 1]."""
        limits = torch.tensor(
            self.config.joint_limits, dtype=torch.float32, device=joints.device
        )
        mins = limits[:, 0]
        maxs = limits[:, 1]
        # (angle - min) / (max - min) -> [0, 1], then * 2 - 1 -> [-1, 1].
        return 2.0 * (joints - mins) / (maxs - mins) - 1.0

    @staticmethod
    def denormalize_joints(norm_joints: torch.Tensor, config: Optional[DataLoaderConfig] = None) -> torch.Tensor:
        """Inverse of _normalize_joints: map [-1, 1] back to degrees."""
        cfg = config or DEFAULT_DATALOADER_CONFIG
        limits = torch.tensor(cfg.joint_limits, dtype=torch.float32, device=norm_joints.device)
        mins = limits[:, 0]
        maxs = limits[:, 1]
        return (norm_joints + 1.0) * (maxs - mins) / 2.0 + mins

    @staticmethod
    def _resolve_labels_path(data_root: Path) -> Path:
        candidates = [data_root / "labels.csv", data_root / "angle_joints.csv"]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _load_samples(self) -> list[Tuple[Path, list[float]]]:
        rows = self._read_label_rows(self.labels_path)
        if not rows:
            raise ValueError(f"No label rows found in {self.labels_path}")

        samples: list[Tuple[Path, list[float]]] = []
        for image_name, joint_values in rows:
            image_path = self.images_dir / image_name
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            samples.append((image_path, joint_values))

        return self._split_samples(samples)

    def _read_label_rows(self, labels_path: Path) -> list[Tuple[str, list[float]]]:
        rows: list[Tuple[str, list[float]]] = []
        with labels_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue

                first = row[0].strip().lower()
                if first in {"image", "image_name", "filename", "file_name", "index"}:
                    continue
                if len(row) < 16:
                    raise ValueError(
                        f"Expected at least 16 columns in labels.csv, got {len(row)}: {row}"
                    )

                image_name = row[1].strip()
                joint_values = [float(value) for value in row[2 : 2 + DEFAULT_NUM_JOINTS]]
                rows.append((image_name, joint_values))

        return rows

    def _split_samples(self, samples: Sequence[Tuple[Path, list[float]]]) -> list[Tuple[Path, list[float]]]:
        total = len(samples)
        if not 0.0 < self.train_ratio <= 1.0:
            raise ValueError("train_ratio must be greater than 0 and less than or equal to 1")
        if not 0.0 < self.val_ratio <= 1.0:
            raise ValueError("val_ratio must be greater than 0 and less than or equal to 1")
        if not 0.0 < self.test_ratio <= 1.0:
            raise ValueError("test_ratio must be greater than 0 and less than or equal to 1")

        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

        generator = torch.Generator().manual_seed(self.split_seed)
        indices = torch.randperm(total, generator=generator).tolist()

        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)

        if self.split == "train":
            selected_indices = indices[:train_end]
        elif self.split == "val":
            selected_indices = indices[train_end:val_end]
        elif self.split == "test":
            selected_indices = indices[val_end:]
        else:
            raise ValueError("split must be one of: 'train', 'val', 'test'")

        return [samples[idx] for idx in selected_indices]

    @staticmethod
    def _normalize_image_size(image_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        if isinstance(image_size, int):
            return (image_size, image_size)
        if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
            return int(image_size[0]), int(image_size[1])
        raise ValueError("image_size must be an int or a tuple/list of two ints")

    @staticmethod
    def _build_default_transform(split: str, image_size: Tuple[int, int], config: Optional[DataLoaderConfig] = None) -> T.Compose:
        """Build a split-specific transform pipeline.

        Only the training split receives augmentation transforms. Validation and
        test splits use the deterministic preprocessing path only.
        """
        cfg = config or DEFAULT_DATALOADER_CONFIG
        if split == "train":
            # Use RandomResizedCrop (random crop + resize) when enabled,
            # otherwise a plain Resize (same as val/test). RRC is a strong
            # augmentation; disable it (train_use_rrc=False) if you want the
            # model to see the full image, e.g. when the pretrained weights
            # are the real bottleneck rather than augmentation.
            resize_or_crop = (
                T.RandomResizedCrop(
                    size=image_size,
                    scale=cfg.train_rrc_scale,
                    ratio=cfg.train_rrc_ratio,
                    interpolation=InterpolationMode.BILINEAR,
                )
                if cfg.train_use_rrc
                else T.Resize(image_size, interpolation=InterpolationMode.BILINEAR)
            )
            transform = T.Compose(
                [
                    resize_or_crop,
                    T.RandomHorizontalFlip(p=cfg.train_hflip_prob),
                    T.RandomAffine(
                        degrees=0,
                        translate=cfg.train_affine_translate,
                        scale=cfg.train_affine_scale,
                    ),
                    T.ColorJitter(
                        brightness=cfg.train_color_jitter_brightness,
                        contrast=cfg.train_color_jitter_contrast,
                        saturation=cfg.train_color_jitter_saturation,
                        hue=cfg.train_color_jitter_hue,
                    ),
                    T.GaussianBlur(kernel_size=(3, 3), sigma=cfg.train_gaussian_blur_sigma),
                    T.ToTensor(),
                    T.Normalize(mean=list(cfg.mean), std=list(cfg.std)),
                ]
            )
        else:
            transform = T.Compose(
                [
                    T.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
                    T.ToTensor(),
                    T.Normalize(mean=list(cfg.mean), std=list(cfg.std)),
                ]
            )

        return transform


def create_train_dataloader(
    data_root: Optional[Union[str, Path]] = None,
    image_size: Optional[Union[int, Tuple[int, int]]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    split_seed: int = DEFAULT_SEED,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for the training split."""
    dataset = CommuDataset(
        data_root=data_root,
        split="train",
        image_size=image_size,
        split_seed=split_seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def create_val_dataloader(
    data_root: Optional[Union[str, Path]] = None,
    image_size: Optional[Union[int, Tuple[int, int]]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    split_seed: int = DEFAULT_SEED,
) -> DataLoader:
    """Create a DataLoader for the validation split."""
    dataset = CommuDataset(
        data_root=data_root,
        split="val",
        image_size=image_size,
        split_seed=split_seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def create_test_dataloader(
    data_root: Optional[Union[str, Path]] = None,
    image_size: Optional[Union[int, Tuple[int, int]]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    split_seed: int = DEFAULT_SEED,
) -> DataLoader:
    """Create a DataLoader for the test split."""
    dataset = CommuDataset(
        data_root=data_root,
        split="test",
        image_size=image_size,
        split_seed=split_seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


__all__ = [
    "CommuDataset",
    "create_train_dataloader",
    "create_val_dataloader",
    "create_test_dataloader",
]
