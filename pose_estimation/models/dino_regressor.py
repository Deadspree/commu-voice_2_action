from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional

import torch
from torch import nn

try:
    from ..configs.model_config import ModelConfig, DEFAULT_MODEL_CONFIG
except ImportError:  # pragma: no cover - fallback for direct script execution
    from configs.model_config import ModelConfig, DEFAULT_MODEL_CONFIG


class DINORegressor(nn.Module):
    """DINOv3 backbone + lightweight regression head for joint-angle prediction."""

    def __init__(
        self,
        *,
        num_joints: int = 14,
        pretrained_backbone: bool = True,
        backbone_weights: Optional[str] = None,
        freeze_backbone: bool = True,
        dropout: float = 0.1,
        hidden_dim: int = 256,
        backbone_name: str = "dinov3_vitb16",
        config: Optional[ModelConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config or DEFAULT_MODEL_CONFIG
        self.num_joints = num_joints if num_joints != 14 else self.config.num_joints
        self.backbone_name = backbone_name if backbone_name != "dinov3_vitb16" else self.config.backbone_name
        self.feature_dim = 768

        self.backbone = self._build_backbone(
            pretrained_backbone=pretrained_backbone if pretrained_backbone is not True else self.config.pretrained_backbone,
            backbone_weights=backbone_weights if backbone_weights is not None else self.config.backbone_weights,
        )
        self.feature_dim = self._infer_feature_dim()

        self.regression_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim if hidden_dim != 256 else self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout if dropout != 0.1 else self.config.dropout),
            nn.Linear(hidden_dim if hidden_dim != 256 else self.config.hidden_dim, self.num_joints),
        )

        if freeze_backbone if freeze_backbone is not True else self.config.freeze_backbone:
            self.freeze_backbone()

    def _build_backbone(self, *, pretrained_backbone: bool, backbone_weights: Optional[str]) -> nn.Module:
        project_root = Path(__file__).resolve().parents[2]
        dinov3_repo_dir = project_root / "third_party" / "dinov3"
        if dinov3_repo_dir.exists():
            sys.path.insert(0, str(dinov3_repo_dir))

        if not pretrained_backbone:
            return self._build_dino_backbone()

        if backbone_weights is not None:
            local_weights_path = Path(backbone_weights).expanduser()
            if local_weights_path.exists():
                backbone = self._build_dino_backbone()
                state_dict = torch.load(local_weights_path, map_location="cpu")
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                if isinstance(state_dict, dict):
                    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
                incompatible = backbone.load_state_dict(state_dict, strict=False)
                if incompatible.missing_keys:
                    warnings.warn(f"Missing keys while loading DINOv3 weights: {incompatible.missing_keys}")
                if incompatible.unexpected_keys:
                    warnings.warn(f"Unexpected keys while loading DINOv3 weights: {incompatible.unexpected_keys}")
                return backbone

        try:
            return torch.hub.load(
                str(dinov3_repo_dir),
                self.backbone_name,
                source="local",
                weights=backbone_weights,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback for missing checkpoints
            warnings.warn(
                f"Unable to load pretrained DINOv3 backbone from {dinov3_repo_dir}: {exc}. "
                "Falling back to a randomly initialized backbone.",
            )
            return self._build_dino_backbone()

    def _build_dino_backbone(self) -> nn.Module:
        try:
            from dinov3.models.vision_transformer import DinoVisionTransformer
        except Exception as exc:  # pragma: no cover - defensive import guard
            raise RuntimeError(f"Could not import DINOv3 backbone implementation: {exc}") from exc

        return DinoVisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            pos_embed_rope_base=100.0,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=768,
            depth=12,
            num_heads=12,
            ffn_ratio=4.0,
            qkv_bias=True,
            drop_path_rate=0.0,
            layerscale_init=1e-5,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
        )

    def _infer_feature_dim(self) -> int:
        if hasattr(self.backbone, "embed_dim"):
            return int(self.backbone.embed_dim)
        if hasattr(self.backbone, "num_features"):
            return int(self.backbone.num_features)
        return self.feature_dim

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected image tensor input")

        features = self.backbone.forward_features(x)
        if isinstance(features, dict):
            embedding = features.get("x_norm_clstoken")
        else:
            embedding = features

        if embedding is None:
            raise ValueError("DINOv3 backbone did not return a usable embedding")
        return self.regression_head(embedding)
