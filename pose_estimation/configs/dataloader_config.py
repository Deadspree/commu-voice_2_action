from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class DataLoaderConfig:
    """Configuration for data loading, augmentation, and dataset splitting."""

    data_root: str = "dataset/commu_pose_dataset"
    image_size: Tuple[int, int] = (384, 384)
    num_joints: int = 14
    # Per-joint min-max normalization to [-1, 1] using the physical joint
    # limits (from CS_JointLimitDefinition.cs). Joints have very different
    # ranges (e.g. FaceYaw is +/-10 deg while RightArmPitch is -180..+5 deg),
    # so a single global mean/std would compress small-range joints and let
    # large-range joints dominate the loss. Mapping each joint's [Min, Max] to
    # [-1, 1] equalizes all outputs to a common scale. Denormalization back to
    # degrees is: angle = (norm + 1) * (max - min) / 2 + min.
    joint_limits: Tuple[Tuple[float, float], ...] = (
        (-20.0, 20.0),    # 0  BodyYaw
        (-40.0, 40.0),    # 1  BodyPitch
        (-180.0, 0.0),    # 2  RightArmPitch
        (-30.0, 15.0),    # 3  RightArmRoll
        (-180.0, 0.0),    # 4  LeftArmPitch
        (-15.0, 30.0),    # 5  LeftArmRoll
        (-12.0, 12.0),    # 6  FacePitch
        (-10.0, 10.0),    # 7  FaceYaw
        (-20.0, 20.0),    # 8  FaceRoll
        (-20.0, 20.0),    # 9  EyePitch
        (-30.0, 30.0),    # 10 RightEyeYaw
        (-30.0, 30.0),    # 11 LeftEyeYaw
        (0.0, 30.0),      # 12 Eyelid
        (-40.0, 40.0),    # 13 Mouth
    )
    split_seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    train_affine_translate: Tuple[float, float] = (0.05, 0.05)
    train_affine_scale: Tuple[float, float] = (0.95, 1.05)
    train_color_jitter_brightness: float = 0.15
    train_color_jitter_contrast: float = 0.15
    train_color_jitter_saturation: float = 0.15
    train_color_jitter_hue: float = 0.03
    train_gaussian_blur_sigma: Tuple[float, float] = (0.1, 1.5)
    # RandomResizedCrop: the single most effective augmentation for small
    # datasets. It crops a random region (scale 0.5-1.0 of the image) and
    # resizes it to the target size, forcing the model to recognize the robot
    # from varied crops and scales instead of memorizing whole images.
    # NOTE: scale was loosened from (0.5, 1.0) to (0.7, 1.0) because the
    # train_loss was running well above val_loss (underfitting the augmented
    # data). A milder crop lets the model fit the training distribution better.
    train_rrc_scale: Tuple[float, float] = (0.7, 1.0)
    train_rrc_ratio: Tuple[float, float] = (0.75, 1.333)
    # Enable/disable RandomResizedCrop in the training transform. When False,
    # the training pipeline uses a plain Resize (like val/test) instead of RRC.
    # Disable RRC if you want the model to see the full image (e.g. when the
    # pretrained weights are the real bottleneck, not augmentation).
    train_use_rrc: bool = False
    # Horizontal flip is disabled by default because the robot's arm joints are
    # asymmetric (LeftArm vs RightArm). Enable only if your data is symmetric.
    train_hflip_prob: float = 0.0

    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def resolved_data_root(self) -> Path:
        return Path(self.data_root).expanduser().resolve()


DEFAULT_DATALOADER_CONFIG = DataLoaderConfig()
