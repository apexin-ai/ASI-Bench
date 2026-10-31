"""Download the pinned MatGL CHGNet checkpoint at image build time."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import urllib.request


def fetch(lock_path: Path, destination: Path) -> None:
    lock = json.loads(lock_path.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{lock['repository']}/resolve/{lock['revision']}/"
    for name, expected in lock["files"].items():
        data = urllib.request.urlopen(base + name, timeout=120).read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {name}")
        (destination / name).write_bytes(data)


if __name__ == "__main__":
    fetch(Path(sys.argv[1]), Path(sys.argv[2]))
