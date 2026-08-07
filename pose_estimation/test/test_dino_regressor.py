from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pose_estimation.models.dino_regressor import DINORegressor


def test_dino_regressor_output_shape():
    model = DINORegressor(num_joints=14, pretrained_backbone=False, freeze_backbone=True)
    model.eval()

    batch = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        outputs = model(batch)

    assert outputs.shape == (2, 14)


def test_dino_regressor_distribution_output_shape():
    model = DINORegressor(
        num_joints=14,
        pretrained_backbone=False,
        freeze_backbone=True,
        output_mode="distribution",
    )
    model.eval()

    batch = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        mu, log_sigma = model(batch)

    assert mu.shape == (2, 14)
    assert log_sigma.shape == (2, 14)
