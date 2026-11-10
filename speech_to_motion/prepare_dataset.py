"""Validate existing Lucy pairs and split source-video groups by aligned duration."""
import argparse
from collections import defaultdict
from pathlib import Path
import random
import subprocess

import numpy as np

from .audio import audio_duration
from .normalization import joint_limit_normalization
from .common import (JOINT_LIMITS, JOINT_NAMES, dataset_signature, digest, file_hash,
                     load_config, read_json, write_json)


def read_motion(path):
    import csv
    with Path(path).open(newline="") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames != ["frame_index", "timestamp_seconds", *JOINT_NAMES]:
            raise ValueError(f"Unexpected joint CSV schema: {path}")
        values = np.array([[float(r[k]) for k in rows.fieldnames] for r in rows], dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"Need at least two finite motion rows: {path}")
    if np.any(np.diff(values[:, 0]) != 1) or np.any(np.diff(values[:, 1]) <= 0):
        raise ValueError(f"Nonconsecutive frames or nonincreasing timestamps: {path}")
    if values[0, 1] < 0:
        raise ValueError(f"Negative motion timestamp: {path}")
    return values[:, 1], values[:, 2:].astype(np.float32)


def split_groups(records, ratios, seed, trials=2000):
    ratios = np.asarray(ratios, dtype=float)
    if len(ratios) != 3 or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1):
        raise ValueError("Provide three positive split ratios summing to one")
    groups = defaultdict(list)
    for record in records:
        groups[record["source_group"]].append(record)
    if len(groups) < 3:
        raise ValueError("At least three original source-video groups are required")
    keys = sorted(groups)
    durations = {k: sum(r["duration_seconds"] for r in groups[k]) for k in keys}
    target = sum(durations.values()) * ratios
    rng = random.Random(seed)
    best, best_score = None, float("inf")
    # Seeded multi-start greedy assignment. Small target splits receive equal
    # relative importance; every candidate keeps complete source groups intact.
    for _ in range(trials):
        order = rng.sample(keys, len(keys))
        totals = np.zeros(3)
        assignment = [[] for _ in range(3)]
        for index, key in enumerate(order):
            empty = [i for i in range(3) if not assignment[i]]
            choices = empty if len(order) - index == len(empty) else range(3)
            def score(i):
                after = totals.copy()
                after[i] += durations[key]
                return np.square((after - target) / target).sum()
            selected = min(choices, key=score)
            assignment[selected].append(key)
            totals[selected] += durations[key]
        score_value = np.square((totals - target) / target).sum()
        if score_value < best_score:
            best, best_score = assignment, score_value
    return {name: sorted([r for k in group_keys for r in groups[k]], key=lambda r: r["id"])
            for name, group_keys in zip(("train", "val", "test"), best)}


def prepare(cfg, overwrite=False):
    data = cfg["data"]
    root, output = Path(data["root"]), Path(data["prepared_dir"])
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Prepared directory is not empty: {output}; use --overwrite deliberately")
    provenance = read_json(data["provenance"])
    sources = {Path(r["path"]).stem: r for r in provenance["clips"]}
    records, rejected = [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            source = sources.get(folder.name)
            if not source:
                raise ValueError("No source provenance for this clip")
            group = source.get("url") or source.get("video_fn")
            if not group:
                raise ValueError("No original source-video identity")
            manifest = read_json(folder / "manifest.json")
            if manifest.get("status") != "completed":
                raise ValueError("Lucy generation is not completed")
            # This audio was extracted from prepared.mp4, whose timeline starts at zero.
            if manifest.get("timeline", {}).get("prepared_start_seconds") != 0:
                raise ValueError("Unsupported/missing prepared timeline origin")
            audio, motion = folder / "source_audio.m4a", folder / "robot_with_audio_joints.csv"
            metadata = read_json(folder / "robot_with_audio_joints.metadata.json")
            if metadata.get("joint_names") != JOINT_NAMES or metadata.get("angle_units") != "degrees":
                raise ValueError("Joint metadata schema/units do not match CommU")
            timestamps, angles = read_motion(motion)
            if metadata["frame_count"] != len(timestamps):
                raise ValueError("Joint metadata row count mismatch")
            seconds = audio_duration(audio, data["ffprobe"])
            # Only supervise timestamps supported by both modalities. No extrapolation.
            start, end = float(timestamps[0]), min(seconds, float(timestamps[-1]))
            if end - start < 0.1:
                raise ValueError("Less than 0.1 seconds of shared audio/motion support")
            limits = np.asarray(JOINT_LIMITS)
            record = {"id": folder.name, "source_group": group, "speaker": source.get("speaker"),
                      "audio": str(audio.relative_to(root)), "motion": str(motion.relative_to(root)),
                      "start_seconds": start, "end_seconds": end, "duration_seconds": end - start,
                      "audio_seconds": seconds, "audio_sha256": file_hash(audio),
                      "motion_sha256": file_hash(motion),
                      "out_of_limits_fraction": float(((angles < limits[:, 0]) |
                                                        (angles > limits[:, 1])).mean())}
            records.append(record)
        except (ValueError, KeyError, OSError, TypeError, subprocess.CalledProcessError) as error:
            rejected.append({"id": folder.name, "reason": str(error)})
    if not records:
        raise ValueError("No valid audio/motion pairs")
    training_only = data.get("training_only", False)
    splits = ({"train": records, "val": [], "test": []} if training_only else
              split_groups(records, data["ratios"], cfg["seed"]))
    # Match the uniform motion grid actually used for training, once per clip,
    # rather than over-counting overlapping training windows.
    samples = []
    for record in splits["train"]:
        times, angles = read_motion(root / record["motion"])
        grid = np.arange(record["start_seconds"], record["end_seconds"], 1 / data["motion_fps"])
        samples.append(np.stack([np.interp(grid, times, angles[:, j]) for j in range(14)], axis=1))
    samples = np.concatenate(samples)
    if data.get("normalization", "standardization") == "joint_limits":
        normalization = joint_limit_normalization(samples.mean(0).tolist())
    else:
        normalization = {"mean": samples.mean(0).tolist(), "std": samples.std(0).clip(1.0).tolist(),
                         "joint_names": JOINT_NAMES, "angle_units": "degrees"}
    summary = {"seed": cfg["seed"], "target_ratios": data["ratios"],
               "training_only": training_only,
               "alignment": "prepared audio origin = motion origin; truncate to shared timestamp support",
               "signature": dataset_signature(cfg), "rejected": rejected,
               "splits": {k: {"clips": len(v), "groups": len({r['source_group'] for r in v}),
                              "seconds": sum(r["duration_seconds"] for r in v)} for k, v in splits.items()}}
    summary["dataset_id"] = digest({"splits": splits, "normalization": normalization})
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = output / "splits" / f"{name}.jsonl"
        path.parent.mkdir(exist_ok=True)
        import json
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_json(output / "normalization.json", normalization)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="speech_to_motion/configs/baseline.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--provenance")
    parser.add_argument("--prepared-dir")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for key, value in (("root", args.data_root), ("provenance", args.provenance),
                       ("prepared_dir", args.prepared_dir)):
        if value:
            cfg["data"][key] = value
    import json
    print(json.dumps(prepare(cfg, args.overwrite), indent=2))


if __name__ == "__main__":
    main()
