"""Tiny step runner for the practice files.

You won't have this on test day - it only walks you through the steps.
Usage (from anywhere):
    python weeks/week01/practice/day1_train.py      # run from step 1
    python weeks/week01/practice/day1_train.py 4    # skip to step 4
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repo root


def run(steps, argv=None):
    os.chdir(ROOT)  # so data/... paths work and runs/ lands in the ignored root folder
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    argv = sys.argv[1:] if argv is None else argv
    start = int(argv[0]) if argv else 1
    ctx = {}
    for n, step in enumerate(steps, 1):
        if n < start:
            continue
        title = (step.__doc__ or step.__name__).strip().splitlines()[0]
        print(f"\n=== Step {n}: {title} ===")
        t0 = time.perf_counter()
        try:
            step(ctx)
        except NotImplementedError:
            print(f"--> TODO: write step {n} in {Path(sys.argv[0]).name}, delete its "
                  f"`raise NotImplementedError` line, then run again with: {n}")
            return
        print(f"[ok] step {n} passed ({time.perf_counter() - t0:.1f}s)")
    print("\nAll steps done. Commit: git add weeks/week01/practice && git commit -m 'week1' && git push")
