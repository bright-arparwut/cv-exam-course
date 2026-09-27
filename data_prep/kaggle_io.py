from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable


def download(slug: str, dest: Path) -> Path:
    """Download and unzip a Kaggle dataset into dest; skip if dest already has files."""
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        print(f"[kaggle] {dest} already populated, skipping download")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"], check=True)
    return dest


def find_class_root(root: Path, classes: Iterable[str]) -> Path:
    root = Path(root)
    wanted = set(classes)
    def junk(p: Path) -> bool:  # macOS zip metadata and hidden folders hold no real images
        return any(part == "__MACOSX" or part.startswith(".") for part in p.relative_to(root).parts)

    candidates = [root, *sorted(p for p in root.rglob("*") if p.is_dir() and not junk(p))]
    matches = [d for d in candidates if wanted <= {c.name for c in d.iterdir() if c.is_dir()}]
    if not matches:
        raise FileNotFoundError(f"no folder under {root} contains all class folders {sorted(wanted)}")
    train_matches = [m for m in matches if "train" in str(m.relative_to(root)).lower()]
    return (train_matches or matches)[0]
