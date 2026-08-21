"""Evaluation script for the CommU DINOv3 joint-angle regressor.

Loads a trained checkpoint and evaluates it on the held-out test split,
reporting loss and RMSE. For the "distribution" output mode it also reports
mean uncertainty (average predicted sigma) across joints.

Usage examples:
  python pose_estimation/evaluate.py --checkpoint checkpoints/best.pt
  python pose_estimation/evaluate.py --checkpoint checkpoints/best.pt --output-mode distribution
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from commu_dataloader import CommuDataset, create_test_dataloader
from configs.dataloader_config import DataLoaderConfig
from configs.model_config import ModelConfig
from models.dino_regressor import DINORegressor


def gaussian_nll_loss(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Gaussian negative log-likelihood (must match train.py)."""
    sigma = torch.exp(log_sigma)
    variance = sigma ** 2
    squared_error = (target - mu) ** 2
    nll = 0.5 * (
        torch.log(torch.tensor(2.0 * torch.pi, device=mu.device))
        + 2.0 * log_sigma
        + squared_error / variance
    )
    return nll.mean()


def build_loss(output_mode: str, loss_name: str) -> nn.Module:
    if output_mode == "point":
        if loss_name == "l1":
            return nn.L1Loss()
        return nn.MSELoss()
    if output_mode == "distribution":
        if loss_name != "nll":
            raise ValueError("distribution output mode requires loss='nll'")
        return gaussian_nll_loss
    raise ValueError(f"Unknown output_mode: {output_mode}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    device: torch.device,
    output_mode: str,
    config: Optional[DataLoaderConfig] = None,
) -> dict[str, float]:
    """Evaluate on a loader and return aggregate metrics.

    The dataloader returns per-joint min-max normalized targets in [-1, 1]
    (each joint's physical [Min, Max] mapped to [-1, 1]). Predictions and
    targets are denormalized back to degrees before computing RMSE so the
    reported metrics are in degrees. The loss is computed on the normalized
    values (matching training).

    In addition to the overall loss/RMSE, per-joint RMSE (and per-joint mean
    sigma for the "distribution" mode) are accumulated so each joint can be
    reported individually.
    """
    model.eval()
    total_loss = 0.0
    total_sq_err = 0.0
    total_count = 0
    total_sigma = 0.0
    sigma_count = 0
    num_batches = 0

    # Per-joint accumulators (joints are the last dimension of the targets).
    per_joint_sq_err: torch.Tensor | None = None
    per_joint_count: torch.Tensor | None = None
    per_joint_sigma: torch.Tensor | None = None

    for images, joints in loader:
        images = images.to(device, non_blocking=True)
        joints = joints.to(device, non_blocking=True)

        if output_mode == "point":
            pred = model(images)
            loss = loss_fn(pred, joints)
            # Denormalize to degrees for RMSE reporting.
            pred_deg = CommuDataset.denormalize_joints(pred, config)
            joints_deg = CommuDataset.denormalize_joints(joints, config)
            total_sq_err += ((pred_deg - joints_deg) ** 2).sum().item()
            total_count += joints.numel()
            sq_err = (pred_deg - joints_deg) ** 2
        else:
            mu, log_sigma = model(images)
            loss = loss_fn(mu, log_sigma, joints)
            # Denormalize to degrees for RMSE reporting.
            mu_deg = CommuDataset.denormalize_joints(mu, config)
            joints_deg = CommuDataset.denormalize_joints(joints, config)
            total_sq_err += ((mu_deg - joints_deg) ** 2).sum().item()
            total_count += joints.numel()
            total_sigma += torch.exp(log_sigma).sum().item()
            sigma_count += log_sigma.numel()
            sq_err = (mu_deg - joints_deg) ** 2

        # Accumulate per-joint squared error (sum over batch dim).
        joint_sq = sq_err.sum(dim=0)
        if per_joint_sq_err is None:
            per_joint_sq_err = joint_sq.detach().cpu()
            per_joint_count = torch.zeros_like(per_joint_sq_err)
        else:
            per_joint_sq_err += joint_sq.detach().cpu()
        per_joint_count += torch.ones_like(per_joint_sq_err) * joints.shape[0]

        if output_mode == "distribution":
            joint_sigma = torch.exp(log_sigma).sum(dim=0)
            if per_joint_sigma is None:
                per_joint_sigma = joint_sigma.detach().cpu()
            else:
                per_joint_sigma += joint_sigma.detach().cpu()

        total_loss += loss.item()
        num_batches += 1

    metrics = {"loss": total_loss / max(num_batches, 1)}
    if total_count > 0:
        metrics["rmse"] = (total_sq_err / total_count) ** 0.5
    if sigma_count > 0:
        metrics["mean_sigma"] = total_sigma / sigma_count

    # Per-joint RMSE (and mean sigma) keyed by joint index.
    if per_joint_sq_err is not None and per_joint_count is not None:
        per_joint_rmse = (per_joint_sq_err / per_joint_count.clamp(min=1)) ** 0.5
        metrics["per_joint_rmse"] = per_joint_rmse.tolist()
        if per_joint_sigma is not None:
            metrics["per_joint_sigma"] = (per_joint_sigma / per_joint_count.clamp(min=1)).tolist()
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the CommU DINOv3 joint-angle regressor.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved checkpoint (.pt).")
    parser.add_argument("--data-root", type=str, default=None, help="Path to the dataset root.")
    parser.add_argument("--output-mode", type=str, default="point", choices=["point", "distribution"])
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "l1", "nll"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--backbone-weights", type=str, default=None, help="Path or URL to pretrained DINOv3 backbone weights.")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def main() -> None:
    args = parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    saved_config = ckpt.get("config", {})

    # Prefer the output mode stored in the checkpoint; fall back to the CLI flag.
    output_mode = saved_config.get("output_mode", args.output_mode)
    loss_name = saved_config.get("loss", args.loss)
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Output mode: {output_mode}, loss: {loss_name}")

    dataloader_cfg = DataLoaderConfig(data_root=args.data_root or "dataset/commu_pose_dataset")
    model_cfg = ModelConfig()

    test_loader = create_test_dataloader(
        data_root=dataloader_cfg.data_root,
        image_size=dataloader_cfg.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_seed=args.seed,
    )
    print(f"Test batches: {len(test_loader)}")

    model = DINORegressor(
        num_joints=model_cfg.num_joints,
        pretrained_backbone=model_cfg.pretrained_backbone,
        backbone_weights=args.backbone_weights or model_cfg.backbone_weights,
        freeze_backbone=model_cfg.freeze_backbone,
        dropout=model_cfg.dropout,
        hidden_dim=model_cfg.hidden_dim,
        backbone_name=model_cfg.backbone_name,
        output_mode=output_mode,
        config=model_cfg,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    print("Loaded model weights from checkpoint.")

    loss_fn = build_loss(output_mode, loss_name)
    metrics = evaluate(
        model,
        test_loader,
        loss_fn,
        device,
        output_mode,
        config=dataloader_cfg,
    )

    print("\n=== Test results ===")
    print(f"  loss      : {metrics['loss']:.4f}")
    print(f"  rmse      : {metrics.get('rmse', float('nan')):.4f}")
    if "mean_sigma" in metrics:
        print(f"  mean sigma: {metrics['mean_sigma']:.4f}")

    # Per-joint breakdown.
    per_joint_rmse = metrics.get("per_joint_rmse")
    if per_joint_rmse:
        per_joint_sigma = metrics.get("per_joint_sigma")
        # Each joint has a different physical range, so report the RMSE both in
        # degrees and as a percentage of that joint's full [Min, Max] range.
        # This makes errors comparable across joints with very different ranges
        # (e.g. FaceYaw +/-10 deg vs RightArmPitch -180..0 deg).
        limits = dataloader_cfg.joint_limits
        print("\n=== Per-joint results ===")
        print(f"  {'Joint':<10}{'Range':>12}{'RMSE':>10}{'% of range':>12}{'Mean sigma':>14}")
        for idx, rmse in enumerate(per_joint_rmse, start=1):
            lo, hi = limits[idx - 1]
            span = hi - lo
            pct = (rmse / span * 100.0) if span > 0 else float("nan")
            sigma_str = f"{per_joint_sigma[idx - 1]:.4f}" if per_joint_sigma else "-"
            print(
                f"  {'Joint ' + str(idx):<10}"
                f"{f'[{lo:.0f}, {hi:.0f}]':>12}"
                f"{rmse:>10.4f}"
                f"{pct:>11.2f}%"
                f"{sigma_str:>14}"
            )


if __name__ == "__main__":
    main()
