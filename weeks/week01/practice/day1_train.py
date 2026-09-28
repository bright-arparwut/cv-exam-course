"""Week 1 - Day 1 (Mon): explore the data, make the val split, train for 1 epoch.

How to work:
  1. Read notes.md sections 1-4 first.
  2. Open this file. For each step: write code under "TODO", delete the
     `raise NotImplementedError` line, then run:
         python weeks/week01/practice/day1_train.py <step number>
  3. The "check" block under each step tells you if you got it right. Don't edit it.
  4. TYPE the code, don't copy-paste. Use help(...) / dir(...) before the notes.

TODO list
  [ ] Step 1  Check your environment (Python, torch, MPS)
  [ ] Step 2  Count images per class and test images
  [ ] Step 3  Make the train/val split in work/intel_cls
  [ ] Step 4  Train yolo11n-cls for 1 epoch
  [ ] Step 5  Explore what training produced (weights, results.csv)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runner import ROOT, run  # noqa: E402

DATA = Path("data/intel")
WORK = Path("work/intel_cls")
DEVICE = "mps"  # your M1 GPU. Try "cpu" too - the test machine may have no GPU.


def step1_env(ctx):
    """Check your environment (Python, torch, MPS)"""
    # TODO: print the Python version, the torch version, and whether MPS is available.
    # Tools:
    #   sys.version
    #   import torch; torch.__version__
    #   torch.backends.mps.is_available()    -> True on your M1
    # Store the MPS result in ctx["mps"].
    raise NotImplementedError
    # --- check (don't edit) ---
    assert "mps" in ctx, "store the MPS result in ctx['mps']"


def step2_explore(ctx):
    """Count images per class and test images"""
    # TODO: build a dict {class_name: number_of_images} for DATA / "train",
    #       and count the files in DATA / "test". Print both.
    # Tools:
    #   Path.iterdir()           -> children of a folder
    #   p.is_dir()               -> keep only class folders
    #   p.glob("*.jpg")          -> files matching a pattern; wrap in list(...) to count
    #   sorted(...)              -> class order = alphabetical (same as Ultralytics)
    # Store results in ctx["counts"] (dict) and ctx["n_test"] (int).
    # Question to answer in a comment: is the dataset balanced?
    raise NotImplementedError
    # --- check (don't edit) ---
    assert len(ctx["counts"]) == 6, f"expected 6 classes, got {list(ctx['counts'])}"
    assert sum(ctx["counts"].values()) == 4800 and ctx["n_test"] == 1200, "did you run python -m data_prep.prepare_intel?"


def step3_val_split(ctx):
    """Make the train/val split in work/intel_cls"""
    # TODO: copy ~10% of every class into WORK/val/<class>/ and the rest into WORK/train/<class>/.
    #       Ultralytics needs root/train/<class> and root/val/<class>.
    # Tools:
    #   shutil.rmtree(WORK, ignore_errors=True)   -> start clean (stale files leak into training!)
    #   random.Random(0).shuffle(list_of_files)   -> reproducible shuffle
    #   max(1, int(len(files) * 0.1))             -> at least 1 val image per class
    #   out.mkdir(parents=True, exist_ok=True)
    #   shutil.copy(src, dst)
    #   skip hidden files: f.name.startswith(".")
    raise NotImplementedError
    # --- check (don't edit) ---
    n_train = len(list((WORK / "train").rglob("*.jpg")))
    n_val = len(list((WORK / "val").rglob("*.jpg")))
    print(f"train={n_train} val={n_val}")
    assert n_train + n_val == 4800 and 400 <= n_val <= 560, "expected ~480 val images out of 4800"
    for cls in (WORK / "val").iterdir():
        overlap = {p.name for p in cls.iterdir()} & {p.name for p in (WORK / "train" / cls.name).iterdir()}
        assert not overlap, f"{cls.name}: the same file is in train and val"


def step4_train(ctx):
    """Train yolo11n-cls for 1 epoch"""
    # TODO: load the pretrained classifier and train it for 1 epoch.
    #   data = the ROOT FOLDER of the split (str(WORK)), not a yaml file
    #   epochs=1, imgsz=160, batch=64, device=DEVICE
    # Tools:
    #   from ultralytics import YOLO
    #   YOLO("yolo11n-cls.pt")          -> downloads weights the first time
    #   help(YOLO.train)                -> little info; the real arg list is:
    #   terminal: `yolo cfg`            -> every default training argument (allowed on test day)
    # Store the model in ctx["model"]. Watch the log: which device did it use? What is top1_acc?
    raise NotImplementedError
    # --- check (don't edit) ---
    save_dir = Path(ctx["model"].trainer.save_dir)
    (ROOT / "work").mkdir(exist_ok=True)
    (ROOT / "work" / "last_run.txt").write_text(str(save_dir))  # day 2 reads this
    assert (save_dir / "weights" / "best.pt").exists(), "no best.pt - did training finish?"
    print(f"run folder: {save_dir}")


def step5_outputs(ctx):
    """Explore what training produced (weights, results.csv)"""
    # TODO: open the run folder saved in work/last_run.txt and look around:
    #   - list every file in it
    #   - read results.csv with pandas and print its columns and the last row
    #   - write in a comment: which column is validation accuracy?
    # Tools:
    #   Path("work/last_run.txt").read_text()
    #   Path(...).iterdir()
    #   import pandas as pd; df = pd.read_csv(...); df.columns; df.tail(1)
    # Store the DataFrame in ctx["results"].
    raise NotImplementedError
    # --- check (don't edit) ---
    assert any("top1" in c for c in ctx["results"].columns), "results.csv should have a top1 accuracy column"


STEPS = [step1_env, step2_explore, step3_val_split, step4_train, step5_outputs]

if __name__ == "__main__":
    run(STEPS)
