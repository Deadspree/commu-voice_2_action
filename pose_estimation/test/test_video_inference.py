"""Check frame/timestamp alignment without loading the DINO backbone."""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import DataLoaderConfig, JOINT_NAMES, predict_video, resolve_video_timestamps


class Capture:
    def __init__(self, timestamps):
        self.timestamps = timestamps
        self.index = -1
        self.released = False

    def isOpened(self):
        if self.released:
            self.index = -1
            self.released = False
        return True

    def get(self, prop):
        return {
            cv2.CAP_PROP_FPS: 20.0,
            cv2.CAP_PROP_FRAME_COUNT: len(self.timestamps),
            cv2.CAP_PROP_FRAME_WIDTH: 8,
            cv2.CAP_PROP_FRAME_HEIGHT: 8,
            cv2.CAP_PROP_POS_MSEC: self.timestamps[self.index] * 1000,
        }.get(prop, 0)

    def read(self):
        self.index += 1
        return (True, np.zeros((8, 8, 3), dtype=np.uint8)) if self.index < len(self.timestamps) else (False, None)

    def release(self):
        self.released = True


class MidpointModel(torch.nn.Module):
    def forward(self, images):
        return torch.zeros((images.shape[0], 14))


def export(tmp_path, monkeypatch, timestamps):
    source = tmp_path / "input.mp4"
    source.touch()
    dest = tmp_path / "angles.csv"
    capture = Capture(timestamps)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    predict_video(MidpointModel(), source, dest, transforms.ToTensor(), torch.device("cpu"),
                  "point", DataLoaderConfig(), 2, tmp_path / "model.pt")
    return dest, capture


def test_video_exports_every_frame_including_partial_batch(tmp_path, monkeypatch):
    dest, capture = export(tmp_path, monkeypatch, [0.0, 0.04, 0.13])
    with dest.open() as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["frame_index", "timestamp_seconds", *JOINT_NAMES]
        rows = list(reader)
    assert [int(row["frame_index"]) for row in rows] == [0, 1, 2]
    assert [float(row["timestamp_seconds"]) for row in rows] == pytest.approx([0, .04, .13])
    assert [float(rows[0][name]) for name in JOINT_NAMES] == pytest.approx([
        (low + high) / 2 for low, high in DataLoaderConfig().joint_limits
    ])
    metadata = json.loads(dest.with_suffix(".metadata.json").read_text())
    assert metadata["frame_count"] == 3
    assert metadata["timestamp_fallback_frames"] == 0
    assert capture.released


def test_video_falls_back_when_decoder_timestamps_repeat(tmp_path, monkeypatch):
    dest, _ = export(tmp_path, monkeypatch, [0, 0, 0])
    rows = list(csv.DictReader(dest.open()))
    assert [float(row["timestamp_seconds"]) for row in rows] == pytest.approx([0, .05, .10])
    assert json.loads(dest.with_suffix(".metadata.json").read_text())["timestamp_fallback_frames"] == 2


def test_video_does_not_replace_an_existing_export(tmp_path, monkeypatch):
    dest, _ = export(tmp_path, monkeypatch, [0])
    before = dest.read_bytes()
    with pytest.raises(FileExistsError):
        export(tmp_path, monkeypatch, [0, .05])
    assert dest.read_bytes() == before


def test_backwards_timestamp_does_not_shift_the_remaining_video():
    times, repairs = resolve_video_timestamps([0, .1, .09, .2, .25], 20)
    assert times == pytest.approx([0, .1, .15, .2, .25])
    assert repairs == 1
