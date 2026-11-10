"""Decode AAC/WAV/MP4 through the project's existing FFmpeg installation."""
import json
import subprocess

import numpy as np


def audio_duration(path, ffprobe="ffprobe"):
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=duration:format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    info = json.loads(result.stdout)
    if not info.get("streams"):
        raise ValueError(f"No audio stream: {path}")
    duration = info["streams"][0].get("duration", info.get("format", {}).get("duration"))
    duration = float(duration)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid audio duration: {path}")
    return duration


def decode_audio(path, sample_rate=16000, ffmpeg="ffmpeg"):
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn",
         "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1"],
        check=True, capture_output=True,
    )
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Empty or nonfinite audio: {path}")
    return audio
