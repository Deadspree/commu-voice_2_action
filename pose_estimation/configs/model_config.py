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
    dropout: float = 0.3
    hidden_dim: int = 512
    backbone_name: str = "dinov3_vitb16"
    # Number of final transformer blocks to keep trainable when the backbone
    # is partially frozen. 0 = freeze everything (full freeze). When > 0, only
    # the last N blocks (plus the final norm) are unfrozen, which adapts the
    # top features to the domain while preserving the pretrained lower layers.
    # This dramatically reduces overfitting compared to unfreezing the whole
    # backbone on a small dataset.
    freeze_last_n: int = 0
    # Feature mode for the regression head:
    #   "cls"       - use only the CLS token (768-dim). Original behavior.
    #   "cls_patch" - concatenate the CLS token with the global-average-pooled
    #                 patch tokens (768 + 768 = 1536-dim). Adds spatial context,
    #                 which helps pose estimation. Recommended for frozen backbones.
    feature_mode: str = "cls_patch"


DEFAULT_MODEL_CONFIG = ModelConfig()
