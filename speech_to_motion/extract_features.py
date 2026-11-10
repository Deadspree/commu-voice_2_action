"""Cache each exact training/evaluation window using a frozen HuBERT encoder."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .common import active_splits, digest, load_config, read_json, write_json
from .dataset import MotionDataset, collate_windows
from .models.hubert_encoder import HubertEncoder


def cache_key(cfg):
    return digest({"encoder": cfg["encoder"], "preprocessing_version": 1})[:24]


def cache_index_path(cfg):
    return Path(cfg["data"]["prepared_dir"]) / "hubert_features" / cache_key(cfg) / "index.json"


def read_cache_index(cfg):
    path = cache_index_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(f"Run extract_features first; missing {path}")
    index = read_json(path)
    if not index.get("complete", True):
        raise ValueError("Window coverage changed; run extract_features to fill missing training windows")
    if index["dataset_id"] != read_json(Path(cfg["data"]["prepared_dir"]) / "summary.json")["dataset_id"]:
        raise ValueError("Feature index belongs to a different prepared dataset; extract again")
    return index


def extract(cfg, device, batch_size=16, overwrite=False):
    path = cache_index_path(cfg)
    old = read_json(path) if path.is_file() else None
    # Resume using the exact resolved encoder, even if the remote main branch moved.
    revision = old["resolved_revision"] if old and not overwrite else cfg["encoder"]["revision"]
    encoder = HubertEncoder(cfg["encoder"]["name"], revision).to(device).eval()
    resolved = getattr(encoder.model.config, "_commit_hash", None) or revision
    summary = read_json(Path(cfg["data"]["prepared_dir"]) / "summary.json")
    if old and old["dataset_id"] != summary["dataset_id"] and not overwrite:
        raise ValueError("Prepared data changed; use --overwrite to rebuild this feature cache")
    # Immutable resolved-model namespaces prevent mixing old/new features when
    # main moves or extraction is interrupted partway through --overwrite.
    key = digest({"requested": cache_key(cfg), "resolved": resolved,
                  "model": encoder.model.config.to_dict()})[:24]
    counts = {}
    for split in active_splits(cfg):
        dataset = MotionDataset(cfg, split)
        dataset.cache_key = key
        pending = [i for i in range(len(dataset)) if overwrite or not dataset.cache_path(i).is_file()]
        # Online dataset must load audio; cache_path still uses the separate lookup dataset.
        online = MotionDataset(cfg, split)
        loader = DataLoader(torch.utils.data.Subset(online, pending), batch_size=batch_size,
                            collate_fn=collate_windows, num_workers=0)
        offset = 0
        with torch.inference_mode():
            for batch in loader:
                encoded = encoder([a.to(device) for a in batch["audio"]])
                for features, times in encoded:
                    destination = dataset.cache_path(pending[offset])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix(".tmp")
                    torch.save({"features": features.cpu().half(), "times": times.cpu()}, temporary)
                    temporary.replace(destination)
                    offset += 1
                print(f"{split}: {offset}/{len(pending)} pending windows", flush=True)
        counts[split] = len(dataset)
    write_json(path, {"dataset_id": summary["dataset_id"], "resolved_revision": resolved,
                      "cache_key": key,
                      "encoder": cfg["encoder"], "hubert_config": encoder.model.config.to_dict(),
                      "windows": counts})
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="speech_to_motion/configs/baseline.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-root")
    parser.add_argument("--prepared-dir")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.prepared_dir:
        cfg["data"]["prepared_dir"] = args.prepared_dir
    print(extract(cfg, args.device, args.batch_size, args.overwrite))


if __name__ == "__main__":
    main()
