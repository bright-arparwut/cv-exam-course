"""Drill 01 reference: train YOLO classify on a train/val folder, predict a folder, get class names."""
from pathlib import Path

from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")
model.train(data="work/intel_cls", epochs=3, imgsz=160, batch=64, seed=0)

best = YOLO(Path(model.trainer.save_dir) / "weights" / "best.pt")
for r in best.predict(source="data/intel/test", imgsz=160, stream=True, verbose=False):
    top1 = r.probs.top1                     # int class index
    print(Path(r.path).stem, r.names[top1], float(r.probs.top1conf))
