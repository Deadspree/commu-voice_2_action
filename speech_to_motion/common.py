"""Shared schema and configuration; independent of the pose estimator's GPU stack."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np

JOINT_NAMES = [
    "BodyYaw", "BodyPitch", "RightArmPitch", "RightArmRoll", "LeftArmPitch",
    "LeftArmRoll", "FacePitch", "FaceYaw", "FaceRoll", "EyePitch", "RightEyeYaw",
    "LeftEyeYaw", "Eyelid", "Mouth",
]
JOINT_LIMITS = [(-20, 20), (-40, 40), (-180, 0), (-30, 15), (-180, 0),
                (-15, 30), (-12, 12), (-10, 10), (-20, 20), (-20, 20),
                (-30, 30), (-30, 30), (0, 30), (-40, 40)]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_config(path):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    data, model, train = cfg["data"], cfg["decoder"], cfg["training"]
    if data.get("normalization", "standardization") not in ("joint_limits", "standardization"):
        raise ValueError("data.normalization must be joint_limits or standardization")
    if data["sample_rate"] != 16000:
        raise ValueError("HuBERT requires a 16000 Hz sample rate")
    for key in ("window_seconds", "train_stride_seconds", "eval_stride_seconds", "motion_fps"):
        if data[key] <= 0:
            raise ValueError(f"data.{key} must be positive")
    if data["window_seconds"] < 0.1:
        raise ValueError("Use windows of at least 0.1 seconds")
    for key in ("train_stride_seconds", "eval_stride_seconds"):
        if data[key] > data["window_seconds"]:
            raise ValueError(f"data.{key} must not leave gaps between windows")
    if model["width"] % model["heads"]:
        raise ValueError("Decoder width must be divisible by its attention heads")
    if train["batch_size"] < 1 or train["accumulation_steps"] < 1:
        raise ValueError("Batch size and accumulation steps must be positive")
    return cfg


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataset_signature(cfg):
    # Paths are deliberately excluded so a prepared dataset can move to RunPod.
    return {k: cfg["data"][k] for k in (
        "sample_rate", "motion_fps", "window_seconds", "train_stride_seconds", "eval_stride_seconds")}


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def active_splits(cfg):
    return ("train",) if cfg["data"].get("training_only", False) else ("train", "val", "test")
