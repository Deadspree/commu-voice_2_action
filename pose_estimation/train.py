"""Training script for the CommU DINOv3 joint-angle regressor.

Supports two output modes:
  - "point":        deterministic regression, trained with MSE (or L1).
  - "distribution": probabilistic regression, outputs (mu, log_sigma) per
                    joint and is trained with a Gaussian negative log-likelihood.

Usage examples:
  python pose_estimation/train.py --output-mode point --epochs 50
  python pose_estimation/train.py --output-mode distribution --epochs 50 --loss nll
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

POSE_ESTIMATION_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DIR = POSE_ESTIMATION_DIR / "checkpoints"
DEFAULT_RUN_DIR = POSE_ESTIMATION_DIR / "run"

from commu_dataloader import CommuDataset, create_train_dataloader, create_val_dataloader
from configs.dataloader_config import DataLoaderConfig
from configs.model_config import ModelConfig
from models.dino_regressor import DINORegressor


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def gaussian_nll_loss(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Gaussian negative log-likelihood per joint, averaged over batch & joints.

    NLL = 0.5 * (log(2*pi) + 2*log_sigma + ((target - mu) / sigma)^2)

    log_sigma is unconstrained; sigma = exp(log_sigma) is always positive.
    """
    sigma = torch.exp(log_sigma)
    variance = sigma ** 2
    squared_error = (target - mu) ** 2
    nll = 0.5 * (torch.log(torch.tensor(2.0 * torch.pi, device=mu.device)) + 2.0 * log_sigma + squared_error / variance)
    return nll.mean()


def build_loss(output_mode: str, loss_name: str) -> nn.Module:
    """Return the loss module matching the model's output mode."""
    if output_mode == "point":
        if loss_name == "l1":
            return nn.L1Loss()
        return nn.MSELoss()
    if output_mode == "distribution":
        if loss_name != "nll":
            raise ValueError("distribution output mode requires loss='nll'")
        return gaussian_nll_loss
    raise ValueError(f"Unknown output_mode: {output_mode}")


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    output_mode: str,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
) -> float:
    """Run one training epoch and return the average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, joints in loader:
        images = images.to(device, non_blocking=True)
        joints = joints.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            if output_mode == "point":
                pred = model(images)
                loss = loss_fn(pred, joints)
            else:
                mu, log_sigma = model(images)
                loss = loss_fn(mu, log_sigma, joints)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    device: torch.device,
    output_mode: str,
    use_amp: bool = False,
    config: Optional[DataLoaderConfig] = None,
) -> dict[str, float]:
    """Evaluate on a loader and return average loss and RMSE (point mode only).

    The dataloader returns per-joint min-max normalized targets in [-1, 1].
    The loss is computed on the normalized values (matching training), but the
    RMSE is denormalized back to degrees so the reported metric is in the same
    units as the standalone evaluate.py.
    """
    model.eval()
    total_loss = 0.0
    total_sq_err = 0.0
    total_count = 0
    num_batches = 0

    for images, joints in loader:
        images = images.to(device, non_blocking=True)
        joints = joints.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            if output_mode == "point":
                pred = model(images)
                loss = loss_fn(pred, joints)
                # Denormalize to degrees for RMSE reporting.
                pred_deg = CommuDataset.denormalize_joints(pred, config)
                joints_deg = CommuDataset.denormalize_joints(joints, config)
                total_sq_err += ((pred_deg - joints_deg) ** 2).sum().item()
                total_count += joints.numel()
            else:
                mu, log_sigma = model(images)
                loss = loss_fn(mu, log_sigma, joints)
                # Denormalize to degrees for RMSE reporting.
                mu_deg = CommuDataset.denormalize_joints(mu, config)
                joints_deg = CommuDataset.denormalize_joints(joints, config)
                total_sq_err += ((mu_deg - joints_deg) ** 2).sum().item()
                total_count += joints.numel()

        total_loss += loss.item()
        num_batches += 1

    metrics = {"loss": total_loss / max(num_batches, 1)}
    if total_count > 0:
        metrics["rmse"] = (total_sq_err / total_count) ** 0.5
    return metrics


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    epoch: int,
    best_val_loss: float,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": config,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)


def build_optimizer(
    model: nn.Module,
    *,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> tuple[torch.optim.Optimizer, list, list, float, float]:
    """Group model parameters into backbone/head and return an AdamW optimizer.

    The backbone is a pretrained DINOv3 and should be fine-tuned gently (low
    LR), while the head can train faster. Frozen backbone params have
    requires_grad=False, so they are simply excluded from gradients.
    """
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    return optimizer, backbone_params, head_params, backbone_lr, head_lr


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    kind: str,
    epochs: int,
    warmup_epochs: int,
    plateau_patience: int,
) -> object | None:
    """Build an LR scheduler.

    - "cosine":  linear warmup (optional) followed by cosine annealing to ~0.
                  Recommended for long runs; prevents late-training oscillation.
    - "plateau": ReduceLROnPlateau on validation loss.
    - "none":    constant LR (returns None).
    """
    if kind == "none":
        return None

    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=plateau_patience, min_lr=1e-7,
        )

    # cosine (default)
    warmup = max(0, warmup_epochs)
    total = max(1, epochs - warmup)

    def lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return (epoch + 1) / warmup
        progress = min((epoch - warmup) / total, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CommU DINOv3 joint-angle regressor.")
    parser.add_argument("--data-root", type=str, default=None, help="Path to the dataset root.")
    parser.add_argument("--output-mode", type=str, default="point", choices=["point", "distribution"])
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "l1", "nll"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=None, help="Learning rate for the backbone. Defaults to --lr if not set. Use a low value (e.g. 1e-5) when unfreezing the backbone.")
    parser.add_argument("--head-lr", type=float, default=None, help="Learning rate for the regression head. Defaults to --lr if not set. When unfreezing the backbone, lower this (e.g. 1e-4) so the head stays stable while the backbone features shift.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau", "none"], help="LR schedule: 'cosine' anneals to 0 (default), 'plateau' reduces on validation-loss plateau, 'none' keeps LR constant.")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Linear LR warmup over the first N epochs (cosine schedule only).")
    parser.add_argument("--plateau-patience", type=int, default=10, help="Epochs without val-loss improvement before 'plateau' reduces LR.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (autocast + GradScaler).")
    parser.add_argument("--compile", action="store_true", help="Compile the model with torch.compile.")
    parser.add_argument("--validate-every", type=int, default=1, help="Run validation every N epochs.")
    parser.add_argument("--backbone-weights", type=str, default=None, help="Path or URL to pretrained DINOv3 backbone weights.")
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR), help="Directory to save checkpoints.")
    parser.add_argument("--run-dir", type=str, default=None, help="Directory to save TensorBoard logs. Defaults to pose_estimation/run/<name> (or pose_estimation/run/run_<timestamp> if no --name is given).")
    parser.add_argument("--name", type=str, default=None, help="Run name used to prefix checkpoint files (e.g. 'myrun' -> myrun_best.pt, myrun_last.pt). Defaults to best.pt/last.pt.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from.")
    parser.add_argument("--load-weights", type=str, default=None, help="Path to a checkpoint to load ONLY model weights from (no optimizer state, no epoch). Useful for fine-tuning from a pretrained checkpoint.")
    parser.add_argument("--unfreeze-backbone", action="store_true", help="Unfreeze the backbone so its weights are trained too. By default the backbone is frozen.")
    parser.add_argument("--freeze-last-n", type=int, default=None, help="When unfreezing, keep only the last N transformer blocks trainable (e.g. 4). 0 or unset freezes the whole backbone. Reduces overfitting on small datasets.")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N batches during training.")
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

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # Configs
    dataloader_cfg = DataLoaderConfig(data_root=args.data_root or "dataset/commu_pose_dataset")
    model_cfg = ModelConfig()

    # Data
    train_loader = create_train_dataloader(
        data_root=dataloader_cfg.data_root,
        image_size=dataloader_cfg.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_seed=args.seed,
        shuffle=True,
    )
    val_loader = create_val_dataloader(
        data_root=dataloader_cfg.data_root,
        image_size=dataloader_cfg.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_seed=args.seed,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = DINORegressor(
        num_joints=model_cfg.num_joints,
        pretrained_backbone=model_cfg.pretrained_backbone,
        backbone_weights=args.backbone_weights or model_cfg.backbone_weights,
        freeze_backbone=model_cfg.freeze_backbone and not args.unfreeze_backbone,
        dropout=model_cfg.dropout,
        hidden_dim=model_cfg.hidden_dim,
        backbone_name=model_cfg.backbone_name,
        output_mode=args.output_mode,
        freeze_last_n=args.freeze_last_n,
        config=model_cfg,
    ).to(device)

    loss_fn = build_loss(args.output_mode, args.loss)

    # Independent learning rates for the backbone and the regression head.
    # When unfreezing the backbone, keep the head LR low (e.g. 1e-4) so the
    # head stays stable while the backbone's features shift underneath it.
    head_lr = args.head_lr if args.head_lr is not None else args.lr
    backbone_lr = args.backbone_lr if args.backbone_lr is not None else head_lr
    optimizer, backbone_params, head_params, backbone_lr, head_lr = build_optimizer(
        model,
        head_lr=head_lr,
        backbone_lr=backbone_lr,
        weight_decay=args.weight_decay,
    )
    print(f"Optimizer: backbone_lr={backbone_lr}, head_lr={head_lr}, "
          f"backbone_params={len(backbone_params)}, head_params={len(head_params)}")

    scheduler = build_scheduler(
        optimizer,
        kind=args.scheduler,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        plateau_patience=args.plateau_patience,
    )
    print(f"Scheduler: {args.scheduler or 'none'}" + (f" (warmup {args.warmup_epochs} epochs)" if args.warmup_epochs > 0 else ""))

    # Mixed precision: use a GradScaler only for fp16 (CUDA). On MPS/bf16 the
    # scaler is unnecessary, so we keep it None unless AMP is enabled on CUDA.
    scaler = None
    if args.amp:
        if device.type == "cuda":
            scaler = torch.amp.GradScaler("cuda")
        elif device.type == "mps":
            scaler = torch.amp.GradScaler("mps")

    if args.compile:
        model = torch.compile(model)
        print("Model compiled with torch.compile")

    start_epoch = 0
    best_val_loss = float("inf")
    checkpoint_dir = Path(args.checkpoint_dir)

    # Optional run-name prefix for checkpoint files.
    prefix = f"{args.name}_" if args.name else ""
    best_ckpt_path = checkpoint_dir / f"{prefix}best.pt"
    last_ckpt_path = checkpoint_dir / f"{prefix}last.pt"

    # TensorBoard log directory. Defaults to pose_estimation/run/<name> when a
    # run name is given, otherwise pose_estimation/run/run_<timestamp>.
    if args.run_dir:
        tb_log_dir = Path(args.run_dir)
    elif args.name:
        tb_log_dir = DEFAULT_RUN_DIR / args.name
    else:
        tb_log_dir = DEFAULT_RUN_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard logs -> {tb_log_dir}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    if args.load_weights:
        ckpt = torch.load(args.load_weights, map_location=device)
        # Load only the model weights. Ignore optimizer state, epoch, and any
        # other training metadata so we start training fresh from these weights.
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [load-weights] Missing keys: {missing}")
        if unexpected:
            print(f"  [load-weights] Unexpected keys: {unexpected}")
        print(f"Loaded model weights from {args.load_weights} (optimizer state NOT loaded)")

    config_snapshot = {
        "output_mode": args.output_mode,
        "loss": args.loss,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "head_lr": head_lr,
        "backbone_lr": backbone_lr,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "plateau_patience": args.plateau_patience,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "freeze_last_n": args.freeze_last_n,
        "model": model_cfg.__dict__,
        "dataloader": dataloader_cfg.__dict__,
    }

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, args.output_mode,
            scaler=scaler, use_amp=args.amp,
        )
        elapsed = time.time() - epoch_start

        if (epoch + 1) % args.validate_every == 0:
            val_metrics = evaluate(model, val_loader, loss_fn, device, args.output_mode, use_amp=args.amp, config=dataloader_cfg)
            val_loss = val_metrics["loss"]
            rmse_str = f", rmse={val_metrics.get('rmse', float('nan')):.4f}" if "rmse" in val_metrics else ""
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f}{rmse_str} | "
                f"{elapsed:.1f}s"
            )

            # Log to TensorBoard.
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            if "rmse" in val_metrics:
                writer.add_scalar("rmse/val", val_metrics["rmse"], epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    best_ckpt_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val_loss,
                    config_snapshot,
                )
                print(f"  -> saved best checkpoint (val_loss={best_val_loss:.4f})")

            # Plateau scheduler keys off validation loss.
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
        else:
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"{elapsed:.1f}s"
            )
            # Log train loss even on non-validation epochs.
            writer.add_scalar("loss/train", train_loss, epoch)

        # Cosine / LambdaLR steps exactly once per epoch. Plateau is stepped
        # on validation loss inside the validate branch above.
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        save_checkpoint(
            last_ckpt_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_loss,
            config_snapshot,
        )

    # Persist the run config alongside the checkpoints.
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_snapshot, handle, indent=2, default=str)

    # Persist the run config alongside the TensorBoard logs too.
    with (tb_log_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_snapshot, handle, indent=2, default=str)

    writer.close()
    print(f"Training finished. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
