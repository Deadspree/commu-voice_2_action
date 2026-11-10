"""Prepare physical-limit targets using existing splits and immutable audio caches."""
import argparse
import os
from pathlib import Path
import shutil
import tempfile

from .common import digest, read_json, read_records, write_json
from .normalization import joint_limit_normalization, training_mean


def migrate(source, destination, training_only=False):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Destination exists, preserved: {destination}")
    old_summary = read_json(source / "summary.json")
    splits = {name: read_records(source / "splits" / f"{name}.jsonl") for name in ("train", "val", "test")}
    source_normalization = read_json(source / "normalization.json")
    if training_only:
        if source_normalization.get("method") != "joint_limits":
            raise ValueError("Use a physical-limit normalized source to keep normalization unchanged")
        records = sorted([r for rows in splits.values() for r in rows], key=lambda r: r["id"])
        if not records or len({r["id"] for r in records}) != len(records):
            raise ValueError("Source manifests are empty or contain duplicate clip IDs")
        splits = {"train": records, "val": [], "test": []}
        normalization = source_normalization
    else:
        normalization = joint_limit_normalization(training_mean(source_normalization).tolist())
    summary = {**old_summary, "dataset_id": digest({"splits": splits, "normalization": normalization}),
               "normalization": "joint_limits", "migrated_from_dataset_id": old_summary["dataset_id"]}
    if training_only:
        summary.update(training_only=True, target_ratios=[1.0, 0.0, 0.0],
                       splits={name: {"clips": len(rows), "groups": len({r["source_group"] for r in rows}),
                                      "seconds": sum(r["duration_seconds"] for r in rows)}
                               for name, rows in splits.items()})
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".minmax-", dir=destination.parent))
    linked = copied = 0
    try:
        (temporary / "splits").mkdir()
        import json
        for name, rows in splits.items():
            (temporary / "splits" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        write_json(temporary / "normalization.json", normalization)
        write_json(temporary / "summary.json", summary)
        # Audio features do not depend on target normalization. Hard-link their
        # immutable files; each dataset keeps an independent, matching registry.
        feature_root = source / "hubert_features"
        if feature_root.is_dir():
            for path in sorted(feature_root.rglob("*")):
                if not path.is_file() or path.suffix not in (".pt", ".json"):
                    continue
                target = temporary / "hubert_features" / path.relative_to(feature_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                if path.name == "index.json":
                    index = read_json(path)
                    if index["dataset_id"] != old_summary["dataset_id"]:
                        raise ValueError(f"Stale feature registry: {path}")
                    write_json(target, {**index, "dataset_id": summary["dataset_id"],
                                        "complete": not training_only})
                else:
                    try:
                        os.link(path, target)
                        linked += 1
                    except OSError:
                        shutil.copy2(path, target)
                        copied += 1
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"destination": str(destination), "dataset_id": summary["dataset_id"],
            "linked_feature_files": linked, "copied_feature_files": copied,
            "splits": summary["splits"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="dataset/speech_to_motion")
    parser.add_argument("--destination", default="dataset/speech_to_motion_minmax")
    parser.add_argument("--training-only", action="store_true", help="Merge all clips into train; preserve normalization")
    args = parser.parse_args()
    import json
    print(json.dumps(migrate(args.source, args.destination, args.training_only), indent=2))


if __name__ == "__main__":
    main()
