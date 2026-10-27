"""Atomic run state and input fingerprints. Never store credentials."""
import hashlib
import json
from pathlib import Path


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, state):
    path = Path(path)
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(state, indent=2) + "\n")
    partial.replace(path)
