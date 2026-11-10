"""Persistent inference independent of the training CLI."""
from contextlib import nullcontext
import math
import threading

import numpy as np
import torch

from .dataset import window_starts
from .models.speech_to_motion import SpeechToMotion
from .normalization import denormalize_angles


def autocast_context(device, precision):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 unavailable; set training.precision to fp16 or fp32")
    if precision not in ("bf16", "fp16"):
        raise ValueError("precision must be fp32, fp16 or bf16")
    return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Unsupported speech-to-motion checkpoint schema")
    return checkpoint


def checkpoint_model(checkpoint, cfg=None):
    model = SpeechToMotion(cfg or checkpoint["config"], checkpoint["hubert_config"])
    model.load_state_dict(checkpoint["model"])
    return model


@torch.inference_mode()
def predict_audio(model, audio, cfg, normalization, device, start=0.0, end=None):
    model.eval()
    data = cfg["data"]
    sr, fps = data["sample_rate"], data["motion_fps"]
    end = min(len(audio) / sr, end if end is not None else len(audio) / sr)
    if end - start < 0.1:
        raise ValueError("Need at least 0.1 seconds of audio")
    times = np.arange(start, end - 1e-8, 1 / fps)
    accumulated = np.zeros((len(times), 14), dtype=np.float64)
    weights = np.zeros(len(times), dtype=np.float64)
    for window_start in window_starts(start, end, data["window_seconds"], data["train_stride_seconds"]):
        first, last = round(window_start * sr), round(min(window_start + data["window_seconds"], end) * sr)
        indices = np.flatnonzero((times >= window_start - 1e-8) &
                                 (times < min(window_start + data["window_seconds"], end) - 1e-8))
        if not len(indices):
            continue
        local_times = torch.tensor(np.maximum(0, times[indices] - first / sr), dtype=torch.float32, device=device)
        batch = {"audio": [torch.tensor(audio[first:last], device=device)], "times": [local_times],
                 "mask": torch.ones((1, len(indices)), device=device, dtype=torch.bool)}
        with autocast_context(device, cfg["training"]["precision"]):
            prediction = model(batch)[0].float().cpu().numpy()
        # Nonzero endpoints ensure clip boundaries remain covered.
        blend = np.maximum(0.05, np.sin(np.pi * (times[indices] - first / sr) / ((last - first) / sr)) ** 2)
        accumulated[indices] += prediction * blend[:, None]
        weights[indices] += blend
    if np.any(weights == 0):
        raise RuntimeError("Window scheduling left uncovered motion timestamps")
    angles = accumulated / weights[:, None]
    angles = denormalize_angles(angles, normalization)
    if not np.isfinite(angles).all():
        raise RuntimeError("Model produced nonfinite joint angles")
    return times, angles.astype(np.float32)


class MotionRuntime:
    """Load a checkpoint once and serialize inference across conversation turns."""
    def __init__(self, checkpoint_path, device="auto", cpu_threads=4):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cpu" and cpu_threads:
            torch.set_num_threads(cpu_threads)
        checkpoint = load_checkpoint(checkpoint_path)
        checkpoint.pop("optimizer", None)
        self.config = checkpoint["config"]
        self.normalization = checkpoint["normalization"]
        self.model = checkpoint_model(checkpoint).to(self.device).eval()
        self.lock = threading.Lock()

    def predict(self, audio, sample_rate):
        from scipy.signal import resample_poly
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all() or sample_rate <= 0:
            raise ValueError("Expected finite mono audio and a positive sample rate")
        if len(audio) / sample_rate < 0.1:
            raise ValueError("At least 0.1 seconds of audio is required")
        if sample_rate != 16000:
            divisor = math.gcd(int(sample_rate), 16000)
            audio = resample_poly(audio, 16000 // divisor, int(sample_rate) // divisor).astype(np.float32)
        with self.lock:
            return predict_audio(self.model, audio, self.config, self.normalization, self.device)
