"""Build an Ultralytics-ready classification folder from a folder-per-class train set.

Ultralytics classify needs root/train/<class>/ and root/val/<class>/. This builds that
from e.g. data/intel/train using symlinks, so no images are copied and data/intel
(the exam-style layout used by the mock) is left untouched.

Usage:
    python -m data_prep.yolo_layout                          # 90/10 split -> data/intel_cls
    python -m data_prep.yolo_layout --full --out data/intel_full   # train on every image
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

from data_prep.split import list_images


def make_yolo_cls_dir(src, dst, val_frac: float = 0.1, seed: int = 0, full: bool = False) -> dict:
    """Symlink src/<class>/* into dst/train/<class>/ and dst/val/<class>/.

    full=False: ~val_frac of each class goes to val (at least 1), the rest to train.
    full=True:  every image goes to train AND val (val accuracy is then meaningless -
                use it for a final run once your settings are chosen).
    """
    src, dst = Path(src).resolve(), Path(dst).resolve()
    if dst == src or src in dst.parents:
        raise ValueError(f"output {dst} must not be the source folder or inside it")
    shutil.rmtree(dst, ignore_errors=True)  # always rebuild: stale links would leak into training

    rng = random.Random(seed)
    counts = {"train": 0, "val": 0}
    classes = sorted(d.name for d in src.iterdir() if d.is_dir() and not d.name.startswith("."))
    for cls in classes:
        files = list_images(src / cls)
        if full:
            splits = {"train": files, "val": files}
        else:
            files = files[:]
            rng.shuffle(files)
            n_val = max(1, int(len(files) * val_frac))
            splits = {"val": files[:n_val], "train": files[n_val:]}
        for split, subset in splits.items():
            out = dst / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for f in subset:
                (out / f.name).symlink_to(f)
            counts[split] += len(subset)
    return {**counts, "classes": classes}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="data/intel/train")
    p.add_argument("--out", default="data/intel_cls")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--full", action="store_true", help="use every image for train and val")
    a = p.parse_args(argv)
    info = make_yolo_cls_dir(a.src, a.out, val_frac=a.val_frac, full=a.full)
    print(f"[yolo_layout] {a.out}: train={info['train']} val={info['val']} classes={info['classes']}")
    print(f"[yolo_layout] train with: model.train(data=\"{a.out}\", ...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
