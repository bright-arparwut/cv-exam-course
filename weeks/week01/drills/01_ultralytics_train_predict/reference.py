"""Drill 01 reference: train YOLO classify on a train/val folder, predict a folder, get class names."""
from pathlib import Path

import torch

# Ultralytics only uses the Mac GPU when asked: prefer MPS (M1), then CUDA, else CPU
DEVICE = "mps" if torch.backends.mps.is_available() else (0 if torch.cuda.is_available() else "cpu")
from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")
model.train(data="work/intel_cls", epochs=3, imgsz=160, batch=64, seed=0, device=DEVICE)

best = YOLO(Path(model.trainer.save_dir) / "weights" / "best.pt")
for r in best.predict(source="data/intel/test", imgsz=160, device=DEVICE, stream=True, verbose=False):
    top1 = r.probs.top1                     # int class index
    print(Path(r.path).stem, r.names[top1], float(r.probs.top1conf))
