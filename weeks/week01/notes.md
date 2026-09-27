# Week 1: The exam pipeline with Ultralytics

## 1. Mental model
images in class folders → YOLO-cls (CNN backbone + classification head) → probabilities over classes → top-1 → CSV row.

## 2. Folder layout Ultralytics expects
```
root/
  train/<class>/*.jpg
  val/<class>/*.jpg        # if val/ is missing it falls back to test/<class>/
```
- `data=` is the **root folder path**, not a YAML file.
- Class names are the folder names; the index order is alphabetical.
- The exam gives you `train/` plus a flat `test/`, so **you** must make `val/` (drill 02).

## 3. Models
`yolo11n-cls.pt` (sizes n < s < m < l < x), pretrained on ImageNet, default input 224. Smaller `imgsz` and a smaller model train faster, which matters on CPU.

## 4. Train
```python
from ultralytics import YOLO
model = YOLO("yolo11n-cls.pt")
model.train(data="work/intel_cls", epochs=5, imgsz=160, batch=64,
            device=0,          # "cpu", "mps" (Mac), 0 (first GPU)
            workers=4, seed=0)
```
- Output goes to `runs/classify/train*/`, which holds `weights/best.pt`, `weights/last.pt` and `results.csv`.
- `model.trainer.save_dir` is that folder.

## 5. Predict
```python
best = YOLO("runs/classify/train/weights/best.pt")
for r in best.predict(source="data/intel/test", imgsz=160, stream=True, verbose=False):
    r.path              # file path
    r.probs.top1        # int index of the best class
    r.probs.top1conf    # confidence (tensor), float(...) it
    r.probs.top5        # list of 5 indices
    r.probs.data        # tensor (num_classes,) of all probabilities
    r.names             # dict {index: class_name}
```
- `stream=True` returns results one at a time, which saves memory on big folders.

## 6. Validate
```python
m = best.val(data="work/intel_cls", imgsz=160)
m.top1, m.top5      # accuracy on val/
```

## 7. Inputs and outputs (shapes)
- **Input:** images resized to `imgsz` → tensor `(B, 3, H, W)`, values 0–1.
- **Output:** `(B, num_classes)` probabilities; softmax is already applied at predict time.

## 8. Offline help (allowed in the exam)
- `yolo cfg` prints every default training argument, which is your memory aid for arg names.
- `help(YOLO.predict)` and `dir(r.probs)`.

## 9. Exam checklist
1. Read the submission schema **twice** (columns, extension, name vs index).
2. Inspect the data: count per class; check the test folder is flat.
3. Make the val split.
4. Run a quick train (few epochs) to get a working submission early.
5. Predict, write the CSV, then sanity-check it: row count = number of test files, the header, and that the labels are valid names.
6. With time left: more epochs, larger `imgsz`, or a bigger model, then resubmit.
