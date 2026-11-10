"""Preserve every output sample, retaining context solely for motion inference."""
from dataclasses import dataclass
import numpy as np


@dataclass
class MotionSegment:
    audio: np.ndarray
    context: np.ndarray
    offset: float


class ResponseBuffer:
    rate = 24000

    def __init__(self, cfg):
        self.size = round(cfg['segment_seconds'] * self.rate)
        self.left = round(cfg['context_seconds'] * self.rate)
        self.right = round(cfg['lookahead_seconds'] * self.rate)
        self.pending = np.empty(0, dtype=np.float32)
        self.previous = np.empty(0, dtype=np.float32)

    def append(self, raw, final=False):
        if len(raw) % 2:
            raise ValueError('Gemini returned incomplete PCM16 samples')
        samples = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768
        self.pending = np.concatenate((self.pending, samples))
        result = []
        while len(self.pending) >= self.size + self.right or (final and len(self.pending)):
            count = min(self.size, len(self.pending))
            audio = self.pending[:count].copy()
            context = np.concatenate((self.previous, self.pending[:count + self.right]))
            # HuBERT needs >= 0.1 seconds even for a very short final tail.
            if len(context) < self.rate // 10:
                context = np.pad(context, (0, self.rate // 10 - len(context)))
            result.append(MotionSegment(audio, context, len(self.previous) / self.rate))
            self.previous = np.concatenate((self.previous, audio))[-self.left:] if self.left else np.empty(0, dtype=np.float32)
            self.pending = self.pending[count:]
        return result


async def predict_segment(predictor, segment, previous, cfg):
    times, angles = await predictor.predict(segment.context, 24000)
    duration = len(segment.audio) / 24000
    timeline = np.arange(0, duration - 1e-10, 1 / 30)
    cropped = np.column_stack([np.interp(timeline + segment.offset, times, angles[:, j]) for j in range(14)])
    # Boundary correction fades out, while PCM itself is never crossfaded or duplicated.
    blend = cfg['blend_seconds']
    if previous is not None and blend > 0:
        cropped += np.maximum(0, 1 - timeline / blend)[:, None] * (previous - cropped[0])
    return timeline, cropped.astype(np.float32)
