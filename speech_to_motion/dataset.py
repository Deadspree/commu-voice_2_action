"""Windowed dataset with timestamp interpolation and content-identified caches."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio import decode_audio
from .common import dataset_signature, digest, file_hash, read_json, read_records
from .prepare_dataset import read_motion
from .normalization import normalize_angles


def window_starts(start, end, window, stride):
    if end <= start or min(window, stride) <= 0 or stride > window:
        raise ValueError("Invalid window bounds or stride")
    last = max(start, end - window)
    values = list(np.arange(start, last + 1e-8, stride))
    if last - values[-1] > 1e-6:
        values.append(last)
    return [float(x) for x in values]


class MotionDataset(Dataset):
    def __init__(self, cfg, split, cache_key=None):
        self.cfg, self.cache_key = cfg, cache_key
        data = cfg["data"]
        self.root, self.prepared = Path(data["root"]), Path(data["prepared_dir"])
        self.summary = read_json(self.prepared / "summary.json")
        if self.summary.get("training_only", False) != data.get("training_only", False):
            raise ValueError("Prepared dataset training-only mode differs from config")
        if data.get("training_only", False) and split != "train":
            raise ValueError("Training-only dataset has no validation or test split")
        if self.summary["signature"] != dataset_signature(cfg):
            raise ValueError("Prepared dataset settings changed; prepare again in a new directory")
        self.normalization = read_json(self.prepared / "normalization.json")
        if self.normalization.get("method", "standardization") != data.get("normalization", "standardization"):
            raise ValueError("Prepared target normalization differs from config; use the matching prepared directory")
        self.records = read_records(self.prepared / "splits" / f"{split}.jsonl")
        if not self.records:
            raise ValueError(f"Empty {split} split")
        for record in self.records:
            for field in ("audio", "motion"):
                if file_hash(self.root / record[field]) != record[f"{field}_sha256"]:
                    raise ValueError(f"Source {field} changed after preparation: {record['id']}")
        stride = data["train_stride_seconds"] if split == "train" else data["eval_stride_seconds"]
        self.windows = [(i, start) for i, r in enumerate(self.records)
                        for start in window_starts(r["start_seconds"], r["end_seconds"],
                                                   data["window_seconds"], stride)]
        self._audio = lru_cache(maxsize=4)(self._decode)
        self._motion = lru_cache(maxsize=16)(lambda path: read_motion(path))

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_audio", None)
        state.pop("_motion", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._audio = lru_cache(maxsize=4)(self._decode)
        self._motion = lru_cache(maxsize=16)(read_motion)

    def _decode(self, path):
        data = self.cfg["data"]
        return decode_audio(path, data["sample_rate"], data["ffmpeg"])

    def __len__(self):
        return len(self.windows)

    def cache_path(self, index):
        record_index, start = self.windows[index]
        r = self.records[record_index]
        identity = digest({"audio": r["audio_sha256"], "start": start,
                           "end": min(start + self.cfg["data"]["window_seconds"], r["end_seconds"]),
                           "settings": dataset_signature(self.cfg), "version": 1})
        return self.prepared / "hubert_features" / self.cache_key / f"{identity}.pt"

    def __getitem__(self, index):
        record_index, start = self.windows[index]
        record, data = self.records[record_index], self.cfg["data"]
        end = min(start + data["window_seconds"], record["end_seconds"])
        sr = data["sample_rate"]
        # Align target times to the actual rounded audio sample boundary.
        first, last = round(start * sr), round(end * sr)
        times = np.arange(0, (last - first) / sr - 1e-8, 1 / data["motion_fps"])
        source_times, angles = self._motion(str(self.root / record["motion"]))
        targets = np.stack([np.interp(first / sr + times, source_times, angles[:, j])
                            for j in range(14)], axis=1).astype(np.float32)
        item = {"target": torch.tensor(normalize_angles(targets, self.normalization), dtype=torch.float32),
                "times": torch.tensor(times, dtype=torch.float32), "id": record["id"], "start": first / sr}
        if self.cache_key is not None:
            path = self.cache_path(index)
            if not path.is_file():
                raise FileNotFoundError(f"Missing feature cache {path}; run extract_features first")
            features = torch.load(path, map_location="cpu", weights_only=True)
            item.update(features=features["features"].float(), feature_times=features["times"])
        else:
            audio = self._audio(str(self.root / record["audio"]))
            if last > len(audio) + 1:
                raise ValueError(f"Decoded audio is shorter than the prepared timeline: {record['id']}")
            item["audio"] = torch.from_numpy(audio[first:last].copy())
        return item


def collate_windows(items):
    from torch.nn.utils.rnn import pad_sequence
    batch = {"target": pad_sequence([i["target"] for i in items], batch_first=True),
             "times": [i["times"] for i in items], "ids": [i["id"] for i in items]}
    batch["mask"] = torch.arange(batch["target"].shape[1])[None, :] < torch.tensor(
        [len(i["target"]) for i in items])[:, None]
    if "audio" in items[0]:
        batch["audio"] = [i["audio"] for i in items]
    else:
        batch["features"] = [i["features"] for i in items]
        batch["feature_times"] = [i["feature_times"] for i in items]
    return batch
