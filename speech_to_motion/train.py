"""Train the decoder, then selectively unfreeze HuBERT from the best baseline."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .common import dataset_signature, load_config, seed_everything, write_json
from .dataset import MotionDataset, collate_windows
from .extract_features import read_cache_index
from .losses import motion_loss
from .metrics import AngleMetrics
from .models.speech_to_motion import SpeechToMotion
from .tensorboard_logging import TrainingLogger
from .runtime import autocast_context, load_checkpoint, checkpoint_model


def move_batch(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else
            [t.to(device) for t in v] if isinstance(v, list) and v and torch.is_tensor(v[0]) else v
            for k, v in batch.items()}


def make_optimizer(model, training, stage):
    groups = [{"params": list(model.decoder.parameters()),
               "lr": training["decoder_lr"] if stage == "frozen" else training["finetune_decoder_lr"]}]
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    if encoder:
        groups.append({"params": encoder, "lr": training["encoder_lr"]})
    return torch.optim.AdamW(groups, weight_decay=training["weight_decay"])


@torch.no_grad()
def validate(model, loader, device, precision, velocity_weight, normalization):
    model.eval()
    total, frames = 0.0, 0
    angles = AngleMetrics(normalization)
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast_context(device, precision):
            prediction = model(batch)
            loss = motion_loss(prediction.float(), batch["target"], batch["mask"], velocity_weight)
        count = int(batch["mask"].sum())
        angles.update(prediction, batch["target"], batch["mask"])
        total += float(loss) * count
        frames += count
    return total / frames, angles.compute()


def train(cfg, stage, run_name, init_checkpoint=None, resume=None, online=False):
    seed_everything(cfg["seed"])
    settings = cfg["training"]
    device = torch.device(settings["device"])
    if stage == "finetune" and not (init_checkpoint or resume):
        raise ValueError("Fine-tuning requires --init-checkpoint or --resume")
    if init_checkpoint and resume:
        raise ValueError("Choose --init-checkpoint or --resume")
    run_dir = Path(settings["output_dir"]) / run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise FileExistsError(f"Run exists: {run_dir}; use a new name or --resume")
    if resume and not (run_dir / "checkpoints" / "best.pt").is_file():
        raise ValueError("Resume requires the existing run directory including best.pt; "
                         "copy the complete run when moving GPUs, or use --init-checkpoint")
    index = read_cache_index(cfg) if stage == "frozen" and not online else None
    cache = index["cache_key"] if index else None
    training_only = cfg["data"].get("training_only", False)
    train_data = MotionDataset(cfg, "train", cache)
    val_data = None if training_only else MotionDataset(cfg, "val", cache)
    checkpoint = load_checkpoint(resume or init_checkpoint) if (resume or init_checkpoint) else None
    if cache and checkpoint and checkpoint["unfreeze_last_n"] > 0:
        raise ValueError("A fine-tuned encoder cannot use pretrained feature caches; pass --online")
    if checkpoint:
        if checkpoint["dataset_id"] != train_data.summary["dataset_id"]:
            raise ValueError("Checkpoint split/normalization does not match prepared dataset")
        for section in ("encoder", "decoder"):
            if checkpoint["config"][section] != cfg[section]:
                raise ValueError(f"Checkpoint {section} differs from config")
        model = checkpoint_model(checkpoint, cfg)
    else:
        encoder_cfg = dict(cfg)
        if index:
            encoder_cfg = {**cfg, "encoder": {**cfg["encoder"], "revision": index["resolved_revision"]}}
        model = SpeechToMotion(encoder_cfg)
    if index:
        resolved = getattr(model.encoder.model.config, "_commit_hash", None)
        # Reconstructed configs may store the resolved revision only in the checkpoint.
        resolved = resolved or (checkpoint or {}).get("encoder_revision")
        if resolved and resolved != index["resolved_revision"]:
            raise ValueError("Checkpoint encoder differs from cached feature encoder")
    last_n = 0 if stage == "frozen" else settings["unfreeze_last_n"]
    if stage == "finetune" and last_n < 1:
        raise ValueError("Fine-tuning requires unfreeze_last_n >= 1")
    model.encoder.set_trainable(last_n, settings["gradient_checkpointing"])
    model.to(device)
    optimizer = make_optimizer(model, settings, stage)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and settings["precision"] == "fp16")
    epochs = settings["frozen_epochs"] if stage == "frozen" else settings["finetune_epochs"]
    generator = torch.Generator().manual_seed(cfg["seed"])
    start_epoch, best, stale = 0, float("inf"), 0
    if resume:
        if checkpoint["stage"] != stage or checkpoint["unfreeze_last_n"] != last_n:
            raise ValueError("Resume must use the same stage and freeze policy")
        if (checkpoint["config"]["seed"] != cfg["seed"] or
                dataset_signature(checkpoint["config"]) != dataset_signature(cfg)):
            raise ValueError("Resume requires the same seed and window settings")
        allowed = {"device", "workers", "output_dir", "frozen_epochs", "finetune_epochs", "patience"}
        if any(checkpoint["config"]["training"][k] != v for k, v in settings.items() if k not in allowed):
            raise ValueError("Training settings changed; use --init-checkpoint for a new experiment")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best = checkpoint.get("best_score", checkpoint.get("best_val", float("inf")))
        stale = checkpoint["stale"]
        torch.set_rng_state(checkpoint["rng_state"])
        generator.set_state(checkpoint["loader_rng_state"])
        if device.type == "cuda" and checkpoint["cuda_rng_states"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
    loader_kwargs = dict(batch_size=settings["batch_size"], num_workers=settings["workers"],
                         collate_fn=collate_windows, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs) if val_data is not None else None
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    import yaml
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    write_json(run_dir / "normalization.json", train_data.normalization)
    write_json(run_dir / "dataset_summary.json", train_data.summary)
    print(f"stage={stage}, windows={len(train_data)}, trainable encoder layers={last_n}, "
          f"parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)
    accumulation = settings["accumulation_steps"]
    nominal_frames = (settings["batch_size"] * cfg["data"]["window_seconds"] *
                      cfg["data"]["motion_fps"] * accumulation)
    with TrainingLogger(run_dir, start_epoch, start_epoch * len(train_loader)) as logger:
        for epoch in range(start_epoch, epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum, count = 0.0, 0
            train_angles = AngleMetrics(train_data.normalization)
            for step, batch in enumerate(train_loader):
                batch = move_batch(batch, device)
                # Weight each accumulation group by actual valid frames, including
                # its short final batch. Accumulate unnormalized frame sums first.
                with autocast_context(device, settings["precision"]):
                    prediction = model(batch)
                    loss = motion_loss(prediction.float(), batch["target"], batch["mask"],
                                       settings["velocity_weight"])
                frames = int(batch["mask"].sum())
                train_angles.update(prediction, batch["target"], batch["mask"])
                scaler.scale(loss * (frames / nominal_frames)).backward()
                loss_sum += float(loss.detach()) * frames
                logger.batch(epoch * len(train_loader) + step, float(loss.detach()), optimizer)
                count += frames
                if step % accumulation == 0:
                    group_frames = 0
                group_frames += frames
                if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                    scaler.unscale_(optimizer)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(nominal_frames / group_frames)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            metrics = {"epoch": epoch, "train_loss": loss_sum / count,
                       "train_metrics": train_angles.compute()}
            if val_loader is not None:
                val, val_angles = validate(model, val_loader, device, settings["precision"],
                                           settings["velocity_weight"], val_data.normalization)
                metrics.update(val_loss=val, val_metrics=val_angles)
            selection = metrics["train_loss"] if training_only else metrics["val_loss"]
            if not torch.isfinite(torch.tensor(selection)):
                raise RuntimeError("Nonfinite training/evaluation loss; inspect data and precision")
            improved = selection < best
            best, stale = (selection, 0) if improved else (best, stale + 1)
            metrics["best_train" if training_only else "best_val"] = best
            with (run_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
            logger.epoch(metrics)
            payload = {"schema_version": 1, "model": model.state_dict(), "config": cfg,
                       "hubert_config": model.encoder.model.config.to_dict(),
                       "encoder_revision": (index or {}).get("resolved_revision") or
                           (checkpoint or {}).get("encoder_revision") or
                           getattr(model.encoder.model.config, "_commit_hash", None) or cfg["encoder"]["revision"],
                       "normalization": train_data.normalization, "dataset_id": train_data.summary["dataset_id"],
                       "stage": stage, "unfreeze_last_n": last_n, "epoch": epoch,
                       "best_score": best, "selection_metric": "train_loss" if training_only else "val_loss",
                       **({"best_train": best} if training_only else {"best_val": best}),
                       "stale": stale, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                       "rng_state": torch.get_rng_state(), "loader_rng_state": generator.get_state(),
                       "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
            for name in (["last.pt", "best.pt"] if improved else ["last.pt"]):
                destination = run_dir / "checkpoints" / name
                temporary = destination.with_suffix(".tmp")
                torch.save(payload, temporary)
                temporary.replace(destination)
            if not training_only and settings["patience"] is not None and stale >= settings["patience"]:
                print("Early stopping", flush=True)
                break
    return run_dir / "checkpoints" / "best.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="speech_to_motion/configs/baseline.yaml")
    parser.add_argument("--stage", choices=["frozen", "finetune"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--online", action="store_true", help="Run frozen HuBERT on audio instead of cache")
    parser.add_argument("--data-root")
    parser.add_argument("--prepared-dir")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--unfreeze-last-n", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    for section, key, value in (("data", "root", args.data_root), ("data", "prepared_dir", args.prepared_dir),
                                ("training", "device", args.device), ("training", "batch_size", args.batch_size),
                                ("training", "accumulation_steps", args.accumulation_steps),
                                ("training", "unfreeze_last_n", args.unfreeze_last_n)):
        if value is not None:
            cfg[section][key] = value
    if cfg["training"]["batch_size"] < 1 or cfg["training"]["accumulation_steps"] < 1:
        parser.error("batch-size and accumulation-steps must be positive")
    print(train(cfg, args.stage, args.run_name, args.init_checkpoint, args.resume, args.online))


if __name__ == "__main__":
    main()
