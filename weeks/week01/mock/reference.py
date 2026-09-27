"""Week 1 mock reference solution. Used to measure the baseline; learner opens it only after grading."""
import random
import shutil
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

DATA = Path("data/intel")
WORK = Path("work/intel_cls")

# 1. train/val split (Ultralytics needs root/train/<cls> and root/val/<cls>)
shutil.rmtree(WORK, ignore_errors=True)  # never train on files left over from an older split
rng = random.Random(0)
for cls_dir in sorted(p for p in (DATA / "train").iterdir() if p.is_dir()):
    files = sorted(f for f in cls_dir.iterdir() if not f.name.startswith("."))
    rng.shuffle(files)
    n_val = max(1, int(0.1 * len(files)))
    for split, subset in (("val", files[:n_val]), ("train", files[n_val:])):
        out = WORK / split / cls_dir.name
        out.mkdir(parents=True, exist_ok=True)
        for f in subset:
            shutil.copy(f, out / f.name)

# 2. train
model = YOLO("yolo11n-cls.pt")
model.train(data=str(WORK), epochs=5, imgsz=160, batch=64, seed=0)
best = YOLO(Path(model.trainer.save_dir) / "weights" / "best.pt")

# 3. predict -> CSV
rows = []
test_files = sorted(p for p in (DATA / "test").iterdir() if not p.name.startswith("."))
for r in best.predict(source=[str(p) for p in test_files], imgsz=160, stream=True, verbose=False):
    rows.append({"image_id": Path(r.path).stem, "label": r.names[r.probs.top1]})
pd.DataFrame(rows, columns=["image_id", "label"]).to_csv("submission.csv", index=False)
print(f"wrote {len(rows)} rows (expected {len(test_files)})")
