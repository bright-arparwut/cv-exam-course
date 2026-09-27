"""Drill 02 reference: split a folder-per-class train set into train/ and val/ for Ultralytics."""
import random
import shutil
from pathlib import Path


def make_val_split(src, dst, val_frac=0.1, seed=0):
    shutil.rmtree(dst, ignore_errors=True)  # stale files from an older split would leak into training
    rng = random.Random(seed)
    counts = {"train": 0, "val": 0}
    for cls_dir in sorted(p for p in Path(src).iterdir() if p.is_dir()):
        files = sorted(f for f in cls_dir.iterdir() if f.is_file() and not f.name.startswith("."))
        rng.shuffle(files)
        n_val = max(1, int(len(files) * val_frac))
        for split, subset in (("val", files[:n_val]), ("train", files[n_val:])):
            out = Path(dst) / split / cls_dir.name
            out.mkdir(parents=True, exist_ok=True)
            for f in subset:
                shutil.copy(f, out / f.name)
            counts[split] += len(subset)
    return counts


if __name__ == "__main__":
    print(make_val_split("data/intel/train", "work/intel_cls"))
