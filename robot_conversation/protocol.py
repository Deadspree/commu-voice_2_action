"""Version 1 JSON messages; mono PCM16 audio and flattened frame-major angles."""
import base64
import json
import re

import numpy as np

from speech_to_motion.common import JOINT_NAMES

VERSION = 1
MAX_MESSAGE_BYTES = 4 * 1024 * 1024


def encode_pcm(audio):
    audio = np.asarray(audio)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("Expected finite mono audio")
    samples = np.clip(np.rint(audio * 32768), -32768, 32767).astype("<i2")
    return base64.b64encode(samples.tobytes()).decode("ascii")


def decode_pcm(encoded, sample_rate, max_seconds=20, min_seconds=0.1):
    if type(sample_rate) is not int or not 8000 <= sample_rate <= 96000:
        raise ValueError("Invalid PCM sample rate")
    if not isinstance(encoded, str) or len(encoded) > max_seconds * sample_rate * 2 * 4 / 3 + 8:
        raise ValueError("Audio exceeds recording limit")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid base64 PCM") from error
    if len(raw) % 2 or len(raw) < max(2, sample_rate * min_seconds * 2) or len(raw) > sample_rate * max_seconds * 2:
        raise ValueError("PCM duration is outside the configured limits")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768


def parse_request(raw):
    if not isinstance(raw, str) or len(raw.encode()) > MAX_MESSAGE_BYTES:
        raise ValueError("Expected a bounded JSON text message")
    message = json.loads(raw)
    if not isinstance(message, dict) or type(message.get("version")) is not int or message.get("version") != VERSION:
        raise ValueError("Unsupported protocol version")
    if message.get("type") not in ("audio_start", "audio_chunk", "audio_end", "text", "cancel", "playback_done", "demo"):
        raise ValueError("Unknown message type")
    if not isinstance(message.get("turn_id"), str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", message["turn_id"]):
        raise ValueError("Invalid turn ID")
    if message["type"] == "text":
        if not isinstance(message.get("text"), str) or not 1 <= len(message["text"].strip()) <= 2000:
            raise ValueError("Text must contain 1 to 2000 characters")
    return message


def packet(kind, turn_id="", **values):
    return {"version": VERSION, "type": kind, "turn_id": turn_id, **values}


def response_chunk(turn_id, sequence, text, audio, sample_rate, times, angles):
    times, angles = np.asarray(times), np.asarray(angles)
    duration = len(audio) / sample_rate
    if (times.ndim != 1 or not len(times) or angles.shape != (len(times), 14)
            or not np.isfinite(angles).all() or not np.isfinite(times).all()
            or times[0] < 0 or times[-1] >= duration + 1e-4 or np.any(np.diff(times) <= 0)):
        raise ValueError("Invalid motion timeline")
    return packet("response_chunk", turn_id, sequence=sequence, text=text,
                  sample_rate=sample_rate, sample_count=len(audio), audio_pcm16=encode_pcm(audio),
                  joint_names=JOINT_NAMES, timestamps=times.tolist(), angles=angles.reshape(-1).tolist())
