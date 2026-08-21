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
        output_mode: str = "point",
        feature_mode: Optional[str] = None,
        freeze_last_n: Optional[int] = None,
        config: Optional[ModelConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config or DEFAULT_MODEL_CONFIG
        self.num_joints = num_joints if num_joints != 14 else self.config.num_joints
        self.backbone_name = backbone_name if backbone_name != "dinov3_vitb16" else self.config.backbone_name
        self.output_mode = output_mode.lower()
        if self.output_mode not in {"point", "distribution"}:
            raise ValueError("output_mode must be either 'point' or 'distribution'")
        self.feature_mode = (feature_mode or self.config.feature_mode).lower()
        if self.feature_mode not in {"cls", "cls_patch"}:
            raise ValueError("feature_mode must be either 'cls' or 'cls_patch'")
        self.feature_dim = 768

        self.backbone = self._build_backbone(
            pretrained_backbone=pretrained_backbone if pretrained_backbone is not True else self.config.pretrained_backbone,
            backbone_weights=backbone_weights if backbone_weights is not None else self.config.backbone_weights,
        )
        self.feature_dim = self._infer_feature_dim()

        # In "cls_patch" mode we concatenate the CLS token with the
        # global-average-pooled patch tokens, doubling the input dimension.
        head_input_dim = self.feature_dim
        if self.feature_mode == "cls_patch":
            head_input_dim = self.feature_dim * 2

        hidden = hidden_dim if hidden_dim != 256 else self.config.hidden_dim
        if self.output_mode == "point":
            self.regression_head = nn.Sequential(
                nn.Linear(head_input_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout if dropout != 0.1 else self.config.dropout),
                nn.Linear(hidden, self.num_joints),
            )
        else:
            self.regression_head = nn.Sequential(
                nn.Linear(head_input_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout if dropout != 0.1 else self.config.dropout),
                nn.Linear(hidden, self.num_joints * 2),
            )

        # Determine how much of the backbone to freeze.
        #   freeze_last_n == 0  -> freeze the entire backbone (full freeze)
        #   freeze_last_n > 0   -> freeze all but the last N blocks
        #   freeze_last_n is None -> fall back to the config value
        if freeze_last_n is None:
            freeze_last_n = self.config.freeze_last_n
        if freeze_backbone if freeze_backbone is not True else self.config.freeze_backbone:
            self.freeze_backbone(freeze_last_n=freeze_last_n)

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
            hub_kwargs = {"source": "local"}
            # Only pass weights when explicitly provided. Passing weights=None
            # overrides the hub function's default (Weights.LVD1689M) with None,
            # which breaks the URL construction.
            if backbone_weights is not None:
                hub_kwargs["weights"] = backbone_weights
            return torch.hub.load(
                str(dinov3_repo_dir),
                self.backbone_name,
                **hub_kwargs,
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

        backbone = DinoVisionTransformer(
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
        # The constructor leaves cls_token/storage_tokens/mask_token as
        # uninitialized (torch.empty) memory. Without init_weights() these can
        # contain NaN, which poisons the forward pass and the loss.
        backbone.init_weights()
        return backbone

    def _infer_feature_dim(self) -> int:
        if hasattr(self.backbone, "embed_dim"):
            return int(self.backbone.embed_dim)
        if hasattr(self.backbone, "num_features"):
            return int(self.backbone.num_features)
        return self.feature_dim

    def freeze_backbone(self, freeze_last_n: int = 0) -> None:
        """Freeze the backbone, optionally keeping the last N blocks trainable.

        Args:
            freeze_last_n: Number of final transformer blocks to keep trainable.
                0 (default) freezes the entire backbone. When > 0, the last N
                blocks and the final norm layer remain trainable while all
                earlier layers are frozen.
        """
        if freeze_last_n <= 0:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            return

        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None or len(blocks) == 0:
            # Fallback: if we can't identify blocks, freeze everything.
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            return

        n_blocks = len(blocks)
        freeze_until = max(0, n_blocks - freeze_last_n)

        # Freeze all parameters first.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        # Unfreeze the last N blocks.
        for block in blocks[freeze_until:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

        # Unfreeze the final norm layer (applied after the last block).
        norm = getattr(self.backbone, "norm", None)
        if norm is not None:
            for parameter in norm.parameters():
                parameter.requires_grad = True

    def _build_embedding(self, features) -> torch.Tensor:
        """Build the regression-head input from the backbone features dict.

        - "cls":       return the CLS token only.
        - "cls_patch": concatenate the CLS token with the global-average-pooled
                       patch tokens to add spatial context.
        """
        if isinstance(features, dict):
            cls_token = features.get("x_norm_clstoken")
            patch_tokens = features.get("x_norm_patchtokens")
        else:
            cls_token = features
            patch_tokens = None

        if cls_token is None:
            raise ValueError("DINOv3 backbone did not return a usable CLS token")

        if self.feature_mode == "cls_patch":
            if patch_tokens is None:
                raise ValueError("DINOv3 backbone did not return patch tokens for cls_patch mode")
            # Global average pooling over the spatial (patch) dimension.
            pooled_patch = patch_tokens.mean(dim=1)
            return torch.cat([cls_token, pooled_patch], dim=-1)

        return cls_token

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected image tensor input")

        features = self.backbone.forward_features(x)
        embedding = self._build_embedding(features)

        outputs = self.regression_head(embedding)
        if self.output_mode == "point":
            return outputs

        mu, log_sigma = torch.chunk(outputs, 2, dim=-1)
        return mu, log_sigma
