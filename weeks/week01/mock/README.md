# Week 1 Mock: Scene classification

**Time limit: 90 minutes.** Start the timer when you open this file's "Task" section.

## One-time: set the pass target (before your first attempt)
`schema.json` ships with `"target_accuracy": null`, so the grader prints `NO TARGET` until you set it.
On a GPU Studio, after the setup below, run the reference solution once and grade it without logging:

    python weeks/week01/mock/reference.py
    python -m grader.score --submission submission.csv --schema weeks/week01/mock/schema.json \
      --labels data/_hidden/intel/test_labels.csv --minutes 0 --week 1 --dataset intel --no-log

Set `target_accuracy` to that accuracy minus 0.03, rounded down to 2 decimals (0.912 → `0.88`), then delete `submission.csv`.
Do this without reading `reference.py`.

## Before the timer (setup, not timed)
    python -m data_prep.prepare_intel
    python -c "from ultralytics import YOLO; YOLO('yolo11n-cls.pt')"   # pre-download weights

## Rules
- No browser, no LLM, no AI autocomplete, no Ultralytics or PyTorch docs.
- Allowed: `help()`, `dir()`, Jupyter `?`, `yolo cfg`, `--help`.
- Don't open `data/_hidden/` or `reference.py`.

## Task
Classify natural-scene photos into 6 classes: buildings, forest, glacier, mountain, sea, street.

Data:
    data/intel/train/<class>/*.jpg   # labelled training images
    data/intel/test/img_XXXXX.jpg    # unlabelled test images

Predict a class for **every** test image.

## Submission format
Write `submission.csv` with:
- The header exactly `image_id,label`.
- `image_id`: the test file name **WITHOUT** the extension, e.g. `img_00042`.
- `label`: the class **name**, e.g. `glacier`.
- One row per test image, no extra columns, no index column.

A submission with any format error scores 0.

## Grading (after you stop the timer)
    python -m grader.score --submission submission.csv \
      --schema weeks/week01/mock/schema.json \
      --labels data/_hidden/intel/test_labels.csv \
      --minutes <minutes you used> --week 1 --dataset intel \
      --blockers "<where you got stuck>"
