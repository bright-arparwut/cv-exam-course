"""Week 1 - Days 3-4 (Wed-Thu): learn what each training setting does to accuracy and time.

On test day you'll have to pick epochs / imgsz / model size under a timer.
Today you find out, on your own machine, what each knob buys.
Run:  python weeks/week01/practice/day3_4_experiments.py <step number>

TODO list
  [ ] Step 1  Write run_experiment() - train, validate, time one config
  [ ] Step 2  Run the experiment grid (Wed: epochs & imgsz, Thu: model size & device)
  [ ] Step 3  Save a results table to weeks/week01/practice/experiments.md
  [ ] Step 4  Answer the questions at the bottom of experiments.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runner import run  # noqa: E402

WORK = Path("work/intel_cls")  # made on Day 1, step 3

# Edit freely. Keep each run short - you want many data points, not one perfect model.
GRID = [
    # Wed: epochs and image size
    {"model": "yolo11n-cls.pt", "epochs": 1, "imgsz": 160, "device": "mps"},
    {"model": "yolo11n-cls.pt", "epochs": 3, "imgsz": 160, "device": "mps"},
    {"model": "yolo11n-cls.pt", "epochs": 3, "imgsz": 224, "device": "mps"},
    # Thu: model size and device
    {"model": "yolo11s-cls.pt", "epochs": 3, "imgsz": 160, "device": "mps"},
    {"model": "yolo11n-cls.pt", "epochs": 1, "imgsz": 160, "device": "cpu"},
]


def run_experiment(cfg):
    """Train one config, validate it, return a dict with the numbers."""
    # TODO: 1) start a timer   2) YOLO(cfg["model"]).train(data=str(WORK), epochs=..., imgsz=..., device=...)
    #       3) stop the timer  4) validate: metrics = model.val(data=str(WORK), imgsz=..., device=...)
    #       5) return {**cfg, "top1": ..., "minutes": ...}
    # Tools:
    #   time.perf_counter()
    #   metrics.top1                  -> validation accuracy (0-1)
    #   seed=0 in train()             -> makes runs comparable
    #   verbose=False / plots=False   -> quieter, a bit faster (check `yolo cfg`)
    raise NotImplementedError


def step1_one_run(ctx):
    """Write run_experiment() - train, validate, time one config"""
    # Nothing to write here - this step just tests run_experiment() on GRID[0].
    ctx["rows"] = [run_experiment(GRID[0])]
    # --- check (don't edit) ---
    row = ctx["rows"][0]
    assert {"top1", "minutes"} <= set(row), "return a dict with top1 and minutes"
    assert 0 <= row["top1"] <= 1, "top1 should be a fraction between 0 and 1"
    print(row)


def step2_grid(ctx):
    """Run the experiment grid (Wed: epochs & imgsz, Thu: model size & device)"""
    # TODO: run run_experiment() on every config in GRID and collect the rows in ctx["rows"].
    #       Print each row as it finishes so you can watch progress.
    # Tip: this takes a while. Run it, then read notes.md or `yolo cfg` while it trains.
    raise NotImplementedError
    # --- check (don't edit) ---
    assert len(ctx["rows"]) == len(GRID)


def step3_table(ctx):
    """Save a results table to weeks/week01/practice/experiments.md"""
    # TODO: turn ctx["rows"] into a markdown table and write it to experiments.md.
    # Tools:
    #   pd.DataFrame(ctx["rows"])
    #   df.to_markdown(index=False)    -> needs `pip install tabulate`; or build the "| a | b |" lines yourself
    #   Path(...).write_text(...)
    # Add these questions under the table and answer them (step 4):
    #   1. Which knob raised accuracy the most per extra minute: epochs, imgsz or model size?
    #   2. How many times slower is CPU than MPS?
    #   3. With a 45-minute exam and no GPU, what settings would you choose?
    raise NotImplementedError
    # --- check (don't edit) ---
    out = Path("weeks/week01/practice/experiments.md")
    assert out.exists() and "top1" in out.read_text()


STEPS = [step1_one_run, step2_grid, step3_table]

if __name__ == "__main__":
    run(STEPS)
