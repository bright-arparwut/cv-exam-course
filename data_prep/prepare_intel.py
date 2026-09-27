"""Week 1 dataset: Intel Image Classification (6 scene classes).

Usage (on Lightning AI, with ~/.kaggle/kaggle.json in place):
    python -m data_prep.prepare_intel
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_prep.kaggle_io import download, find_class_root
from data_prep.split import make_private_split

SLUG = "puneet6060/intel-image-classification"
CLASSES = ("buildings", "forest", "glacier", "mountain", "sea", "street")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", default="data/raw/intel")
    p.add_argument("--out", default="data/intel")
    p.add_argument("--hidden", default="data/_hidden/intel")
    p.add_argument("--max-per-class", type=int, default=1000, help="cap per class to keep mocks fast")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)

    raw = download(SLUG, Path(a.raw))
    src = find_class_root(raw, CLASSES)
    print(f"[intel] using class folders in {src}")
    info = make_private_split(
        src, a.out, a.hidden, seed=42, max_per_class=a.max_per_class, overwrite=a.overwrite
    )
    print(f"[intel] train={info['train']} test={info['test']} classes={info['classes']}")
    print(f"[intel] hidden labels -> {a.hidden}/test_labels.csv (do not open before grading)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
