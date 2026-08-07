from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Default location of the pretrained DINOv3 backbone weights, relative to this
# config file (pose_estimation/configs/ -> pose_estimation/weights/).
DEFAULT_BACKBONE_WEIGHTS = str(
    Path(__file__).resolve().parents[1] / "weights" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the DINO regression model."""

    num_joints: int = 14
    pretrained_backbone: bool = True
    backbone_weights: str | None = DEFAULT_BACKBONE_WEIGHTS
    freeze_backbone: bool = True
    dropout: float = 0.1
    hidden_dim: int = 256
    backbone_name: str = "dinov3_vitb16"


DEFAULT_MODEL_CONFIG = ModelConfig()
