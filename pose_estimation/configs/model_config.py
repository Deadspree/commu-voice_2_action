from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the DINO regression model."""

    num_joints: int = 14
    pretrained_backbone: bool = True
    backbone_weights: str | None = None
    freeze_backbone: bool = True
    dropout: float = 0.1
    hidden_dim: int = 256
    backbone_name: str = "dinov3_vitb16"


DEFAULT_MODEL_CONFIG = ModelConfig()
