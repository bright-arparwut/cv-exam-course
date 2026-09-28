"""Week 1 - Day 2 (Tue): predict, read the output, write and grade the submission CSV.

Needs Day 1 finished (work/last_run.txt must exist). Read notes.md sections 5-9 first.
Run:  python weeks/week01/practice/day2_predict.py <step number>

TODO list
  [ ] Step 1  Load best.pt from the Day 1 run
  [ ] Step 2  Predict 10 test images and print name / class / confidence
  [ ] Step 3  Inspect one result: probability vector, shape, top-5
  [ ] Step 4  Predict every test image and write submission.csv
  [ ] Step 5  Grade it (practice run, not logged)
  [ ] Step 6  (optional, slow) Time 1 epoch on CPU vs MPS
  [ ] Terminal: run `yolo cfg` and skim it - it's allowed on test day
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runner import run  # noqa: E402

DATA = Path("data/intel")
DEVICE = "mps"
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]


def step1_load(ctx):
    """Load best.pt from the Day 1 run"""
    # TODO: read the run folder from work/last_run.txt and load <run>/weights/best.pt with YOLO.
    # Tools:
    #   Path("work/last_run.txt").read_text().strip()
    #   Path(run) / "weights" / "best.pt"
    #   YOLO(path)
    #   model.names  -> {index: class_name}
    # Never type runs/classify/train/... by hand: a 2nd run is train-2, train-3...
    # Store the model in ctx["model"].
    raise NotImplementedError
    # --- check (don't edit) ---
    assert sorted(ctx["model"].names.values()) == CLASSES


def step2_predict_10(ctx):
    """Predict 10 test images and print name / class / confidence"""
    # TODO: take the first 10 files of DATA/"test" (sorted), predict them, and for each result print:
    #       <filename without extension>  <class name>  <confidence>
    # Tools:
    #   ctx["model"].predict(source=[str(p) for p in files], imgsz=160, device=DEVICE, verbose=False)
    #   r.path                   -> the image path
    #   Path(r.path).stem        -> "img_00042"
    #   r.probs.top1             -> int index;  r.names[...] -> class name
    #   float(r.probs.top1conf)  -> confidence as a plain number
    # Store the list of results in ctx["results"].
    raise NotImplementedError
    # --- check (don't edit) ---
    assert len(ctx["results"]) == 10


def step3_inspect(ctx):
    """Inspect one result: probability vector, shape, top-5"""
    # TODO: for ctx["results"][0] print:
    #   - r.probs.data and its .shape      (expect (6,) - one probability per class)
    #   - the sum of the probabilities     (expect ~1.0 - softmax is already applied)
    #   - the top-5 class names            (r.probs.top5 is a list of indices)
    # Tools: dir(r.probs) shows everything a Probs object has.
    # Store the shape as a tuple in ctx["shape"] and the sum as a float in ctx["total"].
    raise NotImplementedError
    # --- check (don't edit) ---
    assert ctx["shape"] == (6,) and abs(ctx["total"] - 1.0) < 1e-3


def step4_submission(ctx):
    """Predict every test image and write submission.csv"""
    # TODO: predict the whole DATA/"test" folder and write submission.csv with
    #       header exactly: image_id,label   (image_id WITHOUT extension, label = class name)
    # Tools:
    #   predict(source=str(DATA / "test"), ..., stream=True)   -> yields results one by one (saves memory)
    #   rows.append({"image_id": ..., "label": ...})
    #   pd.DataFrame(rows, columns=["image_id", "label"]).to_csv("submission.csv", index=False)
    #   index=False matters: otherwise pandas adds an extra unnamed column -> format error = score 0
    raise NotImplementedError
    # --- check (don't edit) ---
    import pandas as pd
    df = pd.read_csv("submission.csv", dtype=str)
    assert list(df.columns) == ["image_id", "label"], f"header is {list(df.columns)}"
    assert len(df) == 1200 and df["image_id"].is_unique
    assert not df["image_id"].str.endswith(".jpg").any(), "image_id must NOT have the extension"
    assert set(df["label"]) <= set(CLASSES)


def step5_grade(ctx):
    """Grade it (practice run, not logged)"""
    # TODO: call the grader from Python with --no-log (this is practice, not your mock).
    # Tools:
    #   from grader.score import main
    #   main(["--submission", "submission.csv",
    #         "--schema", "weeks/week01/mock/schema.json",
    #         "--labels", "data/_hidden/intel/test_labels.csv",
    #         "--minutes", "0", "--week", "1", "--dataset", "intel", "--no-log"])
    # Same thing in the terminal: python -m grader.score ... --no-log
    # Write your accuracy after 1 epoch in a comment here.
    raise NotImplementedError


def step6_cpu_vs_mps(ctx):
    """(optional, slow) Time 1 epoch on CPU vs MPS"""
    # TODO: train a fresh yolo11n-cls for 1 epoch on "work/intel_cls" twice - device="cpu" and
    #       device="mps" - and print how long each took. How slow would a GPU-less test machine be?
    # Tools:
    #   import time; t0 = time.perf_counter(); ...; time.perf_counter() - t0
    # Skip this step by simply not running it (it's last).
    raise NotImplementedError


STEPS = [step1_load, step2_predict_10, step3_inspect, step4_submission, step5_grade, step6_cpu_vs_mps]

if __name__ == "__main__":
    run(STEPS)
