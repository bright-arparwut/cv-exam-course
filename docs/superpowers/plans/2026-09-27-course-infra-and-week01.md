# Course Infrastructure + Week 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reusable course tooling (the submission grader and the private dataset splitter) and all Week 1 material (notes, drills, a mock brief with a reference solution and a measured baseline) so the learner can start Week 1.

**Architecture:** There are two small Python packages. `grader/` checks a `submission.csv` against a per-mock `schema.json` and hidden labels, computes metrics, and appends a row to `logs/progress.md`. `data_prep/` downloads Kaggle datasets and makes a stratified private split (`train/` labelled, `test/` anonymised, `test_labels.csv` hidden). Weekly material lives in `weeks/weekNN/` as markdown plus small reference scripts. The pure-Python parts of those scripts are unit-tested; the parts that need a GPU are dry-run by hand.

**Tech Stack:** Python 3.10+, pandas, scikit-learn, pytest, Kaggle CLI, Ultralytics (YOLO classify), run on Lightning AI Studios. Tests only need pandas, scikit-learn and pytest.

**Spec:** `docs/superpowers/specs/2026-09-27-cv-exam-course-design.md`

## Global Constraints

- The private split seed is `42` (spec §5.5).
- A submission with any format error scores `0` (spec §5.4).
- The target accuracy is `baseline − 3 percentage points`, where the baseline is `yolo11n-cls` with default-ish settings run for about 10 minutes on a GPU (spec §5.4).
- The Week 1 mock time limit is `90` minutes (spec §5.2).
- Week 1 dataset: `puneet6060/intel-image-classification`, 6 classes: buildings, forest, glacier, mountain, sea, street (spec §7).
- Log columns: week, dataset, accuracy, macro-F1, minutes, pass/fail, blockers (spec §5.4), plus date and target.
- **Deviation from spec §9:** the spec's `datasets/` folder is named `data_prep/`. A local package called `datasets` would shadow Hugging Face's `datasets` library, which weeks 6–8 import.
- Raw data, splits and hidden labels live under `data/`, which is git-ignored. Hidden labels go in `data/_hidden/<name>/test_labels.csv`.
- The mock rules (no docs, no LLM) are on the honour system. Nothing technically stops the learner from opening `data/_hidden/`.

## Review Focus

1. **Id extension mismatch.** For example, the submission uses `img_00001.jpg` when the schema says ids have no extension, or the reverse. Expected: a format error that names the problem ("schema says ids have NO file extension"), not a confusing list of "unknown ids". This is tested in Task 1.
2. **CSV quirks from Excel or pandas.** A UTF-8 BOM in the header should still parse. Index labels written as `1.0` are a format error with a clear hint. This is tested in Task 1.
3. **An empty or unreadable submission file.** Expected: a format error and score 0, not a Python traceback. This is tested in Task 2.
4. **Mac junk files and tiny classes in a source folder.** `.DS_Store` and other non-image files are ignored. A class with fewer than 2 images raises a clear error that names the class. This is tested in Task 3.
5. **Re-running dataset preparation over an existing split.** Expected: it refuses unless `overwrite=True`. That way old test files can't mix with new hidden labels. This is tested in Task 3.

---

## File Structure

```
cv-exam-course/
  .gitignore
  pytest.ini
  requirements.txt
  README.md                               # setup + weekly workflow
  grader/
    __init__.py
    schema.py                             # Schema dataclass + load_schema()
    validate.py                           # read_submission(), validate_submission()
    metrics.py                            # compute_metrics()
    log.py                                # append_log()
    score.py                              # score(), verdict(), format_report(), main() CLI
  data_prep/
    __init__.py
    split.py                              # list_images(), collect_samples(), make_private_split()
    kaggle_io.py                          # download(), find_class_root()
    prepare_intel.py                      # Week 1 dataset CLI
  logs/progress.md                        # mock log (header only at start)
  weeks/week01/
    notes.md
    drills/README.md
    drills/01_ultralytics_train_predict/{prompt.md,reference.py}
    drills/02_val_split/{prompt.md,reference.py}
    drills/03_write_submission/{prompt.md,reference.py}
    mock/README.md                        # the brief
    mock/schema.json
    mock/reference.py                     # reference solution (for baseline)
  tests/
    test_schema_validate.py
    test_metrics.py
    test_score.py
    test_split.py
    test_kaggle_io.py
    test_week01_drills.py
```

---

### Task 1: Repo scaffold, submission schema and format validation

**Files:**
- Create: `.gitignore`, `pytest.ini`, `requirements.txt`, `grader/__init__.py`, `grader/schema.py`, `grader/validate.py`
- Test: `tests/test_schema_validate.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `grader.schema.Schema` is a frozen dataclass with fields `id_column: str`, `label_column: str`, `label_type: str` (`"name"` or `"index"`), `id_has_extension: bool`, `classes: tuple[str, ...]`, `target_accuracy: float | None = None`, `time_limit_minutes: int | None = None`.
  - `grader.schema.load_schema(path: str | Path) -> Schema`
  - `grader.validate.read_submission(path: str | Path) -> pd.DataFrame`: every cell is a `str`, and the BOM is stripped.
  - `grader.validate.validate_submission(df: pd.DataFrame, schema: Schema, expected_ids: set[str]) -> tuple[list[str], dict[str, str]]` returns `(errors, predictions)`. `predictions` maps an id with no extension to a class name, and is `{}` whenever `errors` is non-empty.

- [ ] **Step 1: Create the scaffold files**

`.gitignore`:
```gitignore
data/
runs/
work/
submission*.csv
*.pt
*.onnx
__pycache__/
.ipynb_checkpoints/
.DS_Store
.venv/
```

`pytest.ini`:
```ini
[pytest]
testpaths = tests
pythonpath = .
```

`requirements.txt`:
```text
# needed for tests
pandas>=2.0
scikit-learn>=1.3
pytest>=8.0
# needed for course work (install on Lightning AI)
kaggle>=1.6
ultralytics>=8.3
torch>=2.2
torchvision>=0.17
timm>=1.0
lightning>=2.2
transformers>=4.40
```

`grader/__init__.py`: an empty file.

Run: `pip install pandas scikit-learn pytest`

- [ ] **Step 2: Write the failing tests**

`tests/test_schema_validate.py`:
```python
import json

import pandas as pd
import pytest

from grader.schema import Schema, load_schema
from grader.validate import read_submission, validate_submission

CLASSES = ("cat", "dog", "fox")
IDS = {"img_00000", "img_00001", "img_00002"}


def make_schema(**overrides):
    base = dict(
        id_column="image_id",
        label_column="label",
        label_type="name",
        id_has_extension=False,
        classes=CLASSES,
    )
    base.update(overrides)
    return Schema(**base)


def frame(rows, cols=("image_id", "label")):
    return pd.DataFrame(rows, columns=list(cols), dtype=str)


GOOD_ROWS = [["img_00000", "cat"], ["img_00001", "dog"], ["img_00002", "fox"]]


def test_load_schema_roundtrip(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text(json.dumps({
        "id_column": "image_id", "label_column": "label", "label_type": "index",
        "id_has_extension": True, "classes": ["a", "b"],
        "target_accuracy": 0.8, "time_limit_minutes": 90,
    }))
    s = load_schema(path)
    assert s == Schema("image_id", "label", "index", True, ("a", "b"), 0.8, 90)


def test_load_schema_optional_fields_default_to_none(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text(json.dumps({
        "id_column": "id", "label_column": "y", "label_type": "name",
        "id_has_extension": False, "classes": ["a", "b"],
    }))
    s = load_schema(path)
    assert s.target_accuracy is None and s.time_limit_minutes is None


def test_load_schema_rejects_bad_label_type(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text(json.dumps({
        "id_column": "id", "label_column": "y", "label_type": "onehot",
        "id_has_extension": False, "classes": ["a", "b"],
    }))
    with pytest.raises(ValueError, match="label_type"):
        load_schema(path)


def test_load_schema_rejects_duplicate_classes(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text(json.dumps({
        "id_column": "id", "label_column": "y", "label_type": "name",
        "id_has_extension": False, "classes": ["a", "a"],
    }))
    with pytest.raises(ValueError, match="classes"):
        load_schema(path)


def test_valid_name_submission():
    errors, preds = validate_submission(frame(GOOD_ROWS), make_schema(), IDS)
    assert errors == []
    assert preds == {"img_00000": "cat", "img_00001": "dog", "img_00002": "fox"}


def test_column_order_does_not_matter():
    df = frame([[l, i] for i, l in GOOD_ROWS], cols=("label", "image_id"))
    errors, _ = validate_submission(df, make_schema(), IDS)
    assert errors == []


def test_wrong_columns():
    errors, preds = validate_submission(frame(GOOD_ROWS, cols=("id", "label")), make_schema(), IDS)
    assert len(errors) == 1 and "expected columns" in errors[0]
    assert preds == {}


def test_no_rows():
    errors, _ = validate_submission(frame([]), make_schema(), IDS)
    assert errors == ["submission has no rows"]


def test_missing_id():
    errors, _ = validate_submission(frame(GOOD_ROWS[:2]), make_schema(), IDS)
    assert any("missing" in e and "img_00002" in e for e in errors)


def test_unknown_id():
    rows = GOOD_ROWS + [["img_99999", "cat"]]
    errors, _ = validate_submission(frame(rows), make_schema(), IDS)
    assert any("unknown ids" in e and "img_99999" in e for e in errors)


def test_duplicate_id():
    rows = GOOD_ROWS + [["img_00000", "dog"]]
    errors, _ = validate_submission(frame(rows), make_schema(), IDS)
    assert any("duplicate" in e for e in errors)


def test_invalid_label_name():
    rows = [["img_00000", "Cat"], ["img_00001", "dog"], ["img_00002", "fox"]]
    errors, _ = validate_submission(frame(rows), make_schema(), IDS)
    assert any("unknown class names" in e and "'Cat'" in e for e in errors)


def test_empty_label():
    rows = [["img_00000", ""], ["img_00001", "dog"], ["img_00002", "fox"]]
    errors, _ = validate_submission(frame(rows), make_schema(), IDS)
    assert any("empty label" in e for e in errors)


def test_index_labels_valid_and_mapped_to_names():
    rows = [["img_00000", "0"], ["img_00001", "1"], ["img_00002", "2"]]
    errors, preds = validate_submission(frame(rows), make_schema(label_type="index"), IDS)
    assert errors == []
    assert preds == {"img_00000": "cat", "img_00001": "dog", "img_00002": "fox"}


def test_index_label_written_as_float_is_rejected():
    rows = [["img_00000", "0"], ["img_00001", "1.0"], ["img_00002", "2"]]
    errors, _ = validate_submission(frame(rows), make_schema(label_type="index"), IDS)
    assert any("integer class indices" in e and "'1.0'" in e for e in errors)


def test_index_out_of_range():
    rows = [["img_00000", "0"], ["img_00001", "1"], ["img_00002", "3"]]
    errors, _ = validate_submission(frame(rows), make_schema(label_type="index"), IDS)
    assert any("0..2" in e for e in errors)


def test_extension_present_when_schema_says_no():
    rows = [[i + ".jpg", l] for i, l in GOOD_ROWS]
    errors, _ = validate_submission(frame(rows), make_schema(), IDS)
    assert len(errors) == 1
    assert "NO file extension" in errors[0]


def test_extension_required_and_present():
    rows = [[i + ".jpg", l] for i, l in GOOD_ROWS]
    errors, preds = validate_submission(frame(rows), make_schema(id_has_extension=True), IDS)
    assert errors == []
    assert set(preds) == IDS


def test_extension_required_but_missing():
    errors, _ = validate_submission(frame(GOOD_ROWS), make_schema(id_has_extension=True), IDS)
    assert len(errors) == 1
    assert "MUST include the file extension" in errors[0]


def test_read_submission_strips_bom_and_keeps_strings(tmp_path):
    path = tmp_path / "sub.csv"
    path.write_text("﻿image_id,label\nimg_00000,0\nimg_00001,1\n", encoding="utf-8")
    df = read_submission(path)
    assert list(df.columns) == ["image_id", "label"]
    assert df["label"].tolist() == ["0", "1"]
```

- [ ] **Step 3: Run the tests and check they fail**

Run: `pytest tests/test_schema_validate.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'grader.schema'`

- [ ] **Step 4: Implement `grader/schema.py`**

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

VALID_LABEL_TYPES = ("name", "index")


@dataclass(frozen=True)
class Schema:
    id_column: str
    label_column: str
    label_type: str
    id_has_extension: bool
    classes: tuple[str, ...]
    target_accuracy: float | None = None
    time_limit_minutes: int | None = None


def load_schema(path: str | Path) -> Schema:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    label_type = raw["label_type"]
    if label_type not in VALID_LABEL_TYPES:
        raise ValueError(f"label_type must be one of {VALID_LABEL_TYPES}, got {label_type!r}")
    classes = tuple(raw["classes"])
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("classes must list at least 2 unique class names")
    return Schema(
        id_column=raw["id_column"],
        label_column=raw["label_column"],
        label_type=label_type,
        id_has_extension=bool(raw["id_has_extension"]),
        classes=classes,
        target_accuracy=raw.get("target_accuracy"),
        time_limit_minutes=raw.get("time_limit_minutes"),
    )
```

- [ ] **Step 5: Implement `grader/validate.py`**

```python
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from grader.schema import Schema

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_INT_RE = re.compile(r"^\d+$")


def read_submission(path: str | Path) -> pd.DataFrame:
    """Read a CSV with every cell as a string; utf-8-sig strips an Excel BOM."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def _examples(items, k: int = 3) -> str:
    return ", ".join(repr(x) for x in sorted(items)[:k])


def validate_submission(
    df: pd.DataFrame, schema: Schema, expected_ids: set[str]
) -> tuple[list[str], dict[str, str]]:
    expected_cols = [schema.id_column, schema.label_column]
    actual_cols = list(df.columns)
    if sorted(actual_cols) != sorted(expected_cols):
        return [f"expected columns {expected_cols}, got {actual_cols}"], {}
    if len(df) == 0:
        return ["submission has no rows"], {}

    errors: list[str] = []
    raw_ids = df[schema.id_column].tolist()
    labels = df[schema.label_column].tolist()

    # --- ids: extension rule ---
    if schema.id_has_extension:
        bad_ext = [i for i in raw_ids if os.path.splitext(i)[1] == ""]
        hint = "(schema says ids MUST include the file extension)"
        ids = [os.path.splitext(i)[0] for i in raw_ids]
    else:
        bad_ext = [i for i in raw_ids if i.lower().endswith(IMAGE_EXTS)]
        hint = "(schema says ids have NO file extension)"
        ids = list(raw_ids)
    if bad_ext:
        errors.append(f"{len(bad_ext)} ids break the extension rule, e.g. {_examples(bad_ext)} {hint}")

    # --- ids: duplicates, unknown, missing ---
    seen: set[str] = set()
    dups: set[str] = set()
    for i in ids:
        if i in seen:
            dups.add(i)
        seen.add(i)
    if dups:
        errors.append(f"{len(dups)} duplicate ids, e.g. {_examples(dups)}")
    if not bad_ext:  # otherwise every id would also show up as unknown/missing
        unknown = seen - expected_ids
        if unknown:
            errors.append(f"{len(unknown)} unknown ids, e.g. {_examples(unknown)}")
        missing = expected_ids - seen
        if missing:
            errors.append(f"{len(missing)} test ids missing, e.g. {_examples(missing)}")

    # --- labels ---
    empty = [i for i, lab in zip(ids, labels) if lab.strip() == ""]
    if empty:
        errors.append(f"{len(empty)} rows have an empty label, e.g. ids {_examples(empty)}")

    if schema.label_type == "name":
        bad = {lab for lab in labels if lab.strip() != "" and lab not in schema.classes}
        if bad:
            errors.append(
                f"{len(bad)} unknown class names, e.g. {_examples(bad)}; valid: {list(schema.classes)}"
            )
        names = labels
    else:
        n = len(schema.classes)
        bad_fmt = {lab for lab in labels if lab.strip() != "" and not _INT_RE.match(lab)}
        if bad_fmt:
            errors.append(f"labels must be integer class indices like 0, 1, 2; got e.g. {_examples(bad_fmt)}")
        out_of_range = {lab for lab in labels if _INT_RE.match(lab) and int(lab) >= n}
        if out_of_range:
            errors.append(f"class indices must be 0..{n - 1}; got e.g. {_examples(out_of_range)}")
        names = [schema.classes[int(lab)] if _INT_RE.match(lab) and int(lab) < n else "" for lab in labels]

    if errors:
        return errors, {}
    return [], dict(zip(ids, names))
```

- [ ] **Step 6: Run the tests and check they pass**

Run: `pytest tests/test_schema_validate.py -v`
Expected: 20 passed

- [ ] **Step 7: Commit**

```bash
git add .gitignore pytest.ini requirements.txt grader/ tests/test_schema_validate.py
git commit -m "feat(grader): submission schema and format validation"
```

---

### Task 2: Metrics, progress log and the `grader.score` CLI

**Files:**
- Create: `grader/metrics.py`, `grader/log.py`, `grader/score.py`, `logs/progress.md`
- Test: `tests/test_metrics.py`, `tests/test_score.py`

**Interfaces:**
- Consumes: `load_schema`, `Schema`, `read_submission`, `validate_submission` from Task 1.
- Produces:
  - `grader.metrics.compute_metrics(y_true: list[str], y_pred: list[str], classes: tuple[str, ...]) -> dict`, with keys `accuracy: float`, `macro_f1: float`, `per_class_recall: dict[str, float]`.
  - `grader.log.LOG_COLUMNS: list[str]` and `grader.log.append_log(log_path: str | Path, row: dict) -> None`.
  - `grader.score.load_labels(path) -> dict[str, str]`. It reads a CSV with columns `id,label`.
  - `grader.score.score(submission_path, schema_path, labels_path) -> dict`, with keys `format_ok: bool`, `errors: list[str]` and the metrics keys. On a format error, accuracy and macro-F1 are `0.0`.
  - `grader.score.verdict(result: dict, target: float | None, minutes: float, limit: int | None) -> str`. It returns one of `"PASS"`, `"FAIL (format)"`, `"FAIL (accuracy)"`, `"FAIL (time)"`, `"NO TARGET"`.
  - The CLI is `python -m grader.score --submission S --schema J --labels L --minutes M --week W --dataset D [--blockers TEXT] [--log PATH] [--no-log]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:
```python
import pytest

from grader.metrics import compute_metrics

CLASSES = ("cat", "dog", "fox")


def test_perfect_predictions():
    m = compute_metrics(["cat", "dog", "fox"], ["cat", "dog", "fox"], CLASSES)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["per_class_recall"] == {"cat": 1.0, "dog": 1.0, "fox": 1.0}


def test_one_wrong():
    m = compute_metrics(["cat", "cat", "dog", "fox"], ["cat", "dog", "dog", "fox"], CLASSES)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["per_class_recall"]["cat"] == pytest.approx(0.5)
    assert m["per_class_recall"]["dog"] == pytest.approx(1.0)


def test_class_absent_from_truth_scores_zero_without_crashing():
    m = compute_metrics(["cat", "cat"], ["cat", "cat"], CLASSES)
    assert m["per_class_recall"]["fox"] == 0.0
    assert isinstance(m["macro_f1"], float)
```

`tests/test_score.py`:
```python
import json

import pytest

from grader.log import LOG_COLUMNS, append_log
from grader.score import main, score, verdict

CLASSES = ["cat", "dog", "fox"]


@pytest.fixture
def mock_files(tmp_path):
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({
        "id_column": "image_id", "label_column": "label", "label_type": "name",
        "id_has_extension": False, "classes": CLASSES,
        "target_accuracy": 0.6, "time_limit_minutes": 90,
    }))
    labels = tmp_path / "test_labels.csv"
    labels.write_text("id,label\nimg_00000,cat\nimg_00001,dog\nimg_00002,fox\n")
    return tmp_path, schema, labels


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_score_perfect(mock_files):
    d, schema, labels = mock_files
    sub = write(d / "sub.csv", "image_id,label\nimg_00000,cat\nimg_00001,dog\nimg_00002,fox\n")
    r = score(sub, schema, labels)
    assert r["format_ok"] is True
    assert r["accuracy"] == 1.0


def test_score_format_error_scores_zero(mock_files):
    d, schema, labels = mock_files
    sub = write(d / "sub.csv", "id,label\nimg_00000,cat\n")
    r = score(sub, schema, labels)
    assert r["format_ok"] is False
    assert r["accuracy"] == 0.0 and r["macro_f1"] == 0.0
    assert r["errors"]


def test_score_empty_file_is_format_error(mock_files):
    d, schema, labels = mock_files
    sub = write(d / "sub.csv", "")
    r = score(sub, schema, labels)
    assert r["format_ok"] is False
    assert "could not read submission" in r["errors"][0]


def test_score_missing_file_is_format_error(mock_files):
    d, schema, labels = mock_files
    r = score(d / "nope.csv", schema, labels)
    assert r["format_ok"] is False


@pytest.mark.parametrize("result,target,minutes,limit,expected", [
    ({"format_ok": False, "accuracy": 0.0}, 0.8, 10, 90, "FAIL (format)"),
    ({"format_ok": True, "accuracy": 0.9}, None, 10, 90, "NO TARGET"),
    ({"format_ok": True, "accuracy": 0.7}, 0.8, 10, 90, "FAIL (accuracy)"),
    ({"format_ok": True, "accuracy": 0.9}, 0.8, 95, 90, "FAIL (time)"),
    ({"format_ok": True, "accuracy": 0.9}, 0.8, 95, None, "PASS"),
    ({"format_ok": True, "accuracy": 0.8}, 0.8, 90, 90, "PASS"),
])
def test_verdict(result, target, minutes, limit, expected):
    assert verdict(result, target, minutes, limit) == expected


def test_append_log_writes_header_once(tmp_path):
    log = tmp_path / "logs" / "progress.md"
    append_log(log, {"week": 1, "dataset": "intel", "result": "PASS"})
    append_log(log, {"week": 2, "dataset": "rice", "result": "FAIL (time)", "blockers": "a|b\nc"})
    text = log.read_text()
    assert text.count("| " + " | ".join(LOG_COLUMNS) + " |") == 1
    assert "| intel |" in text
    assert "a/b c" in text  # pipes and newlines are sanitised


def test_main_prints_report_and_logs(mock_files, capsys):
    d, schema, labels = mock_files
    sub = write(d / "sub.csv", "image_id,label\nimg_00000,cat\nimg_00001,dog\nimg_00002,cat\n")
    log = d / "progress.md"
    code = main([
        "--submission", str(sub), "--schema", str(schema), "--labels", str(labels),
        "--minutes", "42", "--week", "1", "--dataset", "toy", "--log", str(log),
        "--blockers", "forgot val split",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "Result: PASS" in out
    assert "Accuracy: 0.6667" in out
    assert "| toy |" in log.read_text()


def test_main_no_log(mock_files, capsys):
    d, schema, labels = mock_files
    sub = write(d / "sub.csv", "image_id,label\nimg_00000,cat\nimg_00001,dog\nimg_00002,fox\n")
    log = d / "progress.md"
    main([
        "--submission", str(sub), "--schema", str(schema), "--labels", str(labels),
        "--minutes", "5", "--week", "1", "--dataset", "toy", "--log", str(log), "--no-log",
    ])
    assert not log.exists()
```

- [ ] **Step 2: Run the tests and check they fail**

Run: `pytest tests/test_metrics.py tests/test_score.py -v`
Expected: collection errors, `ModuleNotFoundError: No module named 'grader.metrics'` and `'grader.log'`

- [ ] **Step 3: Implement `grader/metrics.py`**

```python
from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score, recall_score


def compute_metrics(y_true: list[str], y_pred: list[str], classes: tuple[str, ...]) -> dict:
    labels = list(classes)
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_recall": {c: float(r) for c, r in zip(labels, recalls)},
    }
```

- [ ] **Step 4: Implement `grader/log.py`**

```python
from __future__ import annotations

from pathlib import Path

LOG_COLUMNS = ["date", "week", "dataset", "accuracy", "macro_f1", "minutes", "target", "result", "blockers"]


def _header() -> str:
    return (
        "# Mock progress log\n\n"
        "| " + " | ".join(LOG_COLUMNS) + " |\n"
        "|" + "---|" * len(LOG_COLUMNS) + "\n"
    )


def append_log(log_path: str | Path, row: dict) -> None:
    path = Path(log_path)
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_header(), encoding="utf-8")
    cells = [str(row.get(c, "")).replace("|", "/").replace("\n", " ") for c in LOG_COLUMNS]
    with path.open("a", encoding="utf-8") as f:
        f.write("| " + " | ".join(cells) + " |\n")
```

- [ ] **Step 5: Implement `grader/score.py`**

```python
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from grader.log import append_log
from grader.metrics import compute_metrics
from grader.schema import Schema, load_schema
from grader.validate import read_submission, validate_submission


def load_labels(path: str | Path) -> dict[str, str]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return dict(zip(df["id"], df["label"]))


def _format_fail(errors: list[str]) -> dict:
    return {"format_ok": False, "errors": errors, "accuracy": 0.0, "macro_f1": 0.0, "per_class_recall": {}}


def score(submission_path, schema_path, labels_path) -> dict:
    schema = load_schema(schema_path)
    truth = load_labels(labels_path)
    try:
        df = read_submission(submission_path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        return _format_fail([f"could not read submission: {exc}"])
    errors, preds = validate_submission(df, schema, set(truth))
    if errors:
        return _format_fail(errors)
    ids = sorted(truth)
    metrics = compute_metrics([truth[i] for i in ids], [preds[i] for i in ids], schema.classes)
    return {"format_ok": True, "errors": [], **metrics}


def verdict(result: dict, target: float | None, minutes: float, limit: int | None) -> str:
    if not result["format_ok"]:
        return "FAIL (format)"
    if target is None:
        return "NO TARGET"
    if result["accuracy"] < target:
        return "FAIL (accuracy)"
    if limit is not None and minutes > limit:
        return "FAIL (time)"
    return "PASS"


def format_report(result: dict, v: str, minutes: float, schema: Schema) -> str:
    lines = [f"Result: {v}"]
    if not result["format_ok"]:
        lines.append("Format errors (score = 0):")
        lines += [f"  - {e}" for e in result["errors"]]
    else:
        lines.append(f"Accuracy: {result['accuracy']:.4f}")
        lines.append(f"Macro-F1: {result['macro_f1']:.4f}")
        lines.append("Per-class recall:")
        lines += [f"  {c}: {r:.3f}" for c, r in result["per_class_recall"].items()]
    limit = f" / limit {schema.time_limit_minutes} min" if schema.time_limit_minutes else ""
    lines.append(f"Time: {minutes:g} min{limit}")
    if schema.target_accuracy is not None:
        lines.append(f"Target accuracy: {schema.target_accuracy:.2f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score a mock-test submission.")
    p.add_argument("--submission", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--minutes", type=float, required=True, help="time you took, in minutes")
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--blockers", default="", help="where you got stuck")
    p.add_argument("--log", default="logs/progress.md")
    p.add_argument("--no-log", action="store_true", help="do not append to the progress log")
    a = p.parse_args(argv)

    schema = load_schema(a.schema)
    result = score(a.submission, a.schema, a.labels)
    v = verdict(result, schema.target_accuracy, a.minutes, schema.time_limit_minutes)
    print(format_report(result, v, a.minutes, schema))
    if not a.no_log:
        append_log(a.log, {
            "date": date.today().isoformat(),
            "week": a.week,
            "dataset": a.dataset,
            "accuracy": f"{result['accuracy']:.4f}",
            "macro_f1": f"{result['macro_f1']:.4f}",
            "minutes": f"{a.minutes:g}",
            "target": "" if schema.target_accuracy is None else f"{schema.target_accuracy:.2f}",
            "result": v,
            "blockers": a.blockers,
        })
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Create `logs/progress.md` with just the header**

```markdown
# Mock progress log

| date | week | dataset | accuracy | macro_f1 | minutes | target | result | blockers |
|---|---|---|---|---|---|---|---|---|
```

- [ ] **Step 7: Run all tests and check they pass**

Run: `pytest -v`
Expected: all tests in `test_schema_validate.py`, `test_metrics.py` and `test_score.py` pass (36 in total)

- [ ] **Step 8: Commit**

```bash
git add grader/ logs/progress.md tests/test_metrics.py tests/test_score.py
git commit -m "feat(grader): metrics, progress log and score CLI"
```

---

### Task 3: Private dataset splitter

**Files:**
- Create: `data_prep/__init__.py` (empty), `data_prep/split.py`
- Test: `tests/test_split.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `data_prep.split.IMAGE_EXTS: set[str]`
  - `data_prep.split.list_images(class_dir: Path) -> list[Path]`: sorted image files; hidden and non-image files are skipped.
  - `data_prep.split.collect_samples(src_root: Path, max_per_class: int | None = None, seed: int = 42) -> list[tuple[Path, str]]`
  - `data_prep.split.make_private_split(src_root, out_root, hidden_root, test_size=0.2, seed=42, max_per_class=None, train_layout="folders", overwrite=False) -> dict`, which returns `{"train": int, "test": int, "classes": list[str]}`.
  - Output layout:
    - `out_root/train/<class>/<original name>` when `train_layout="folders"`.
    - `out_root/train/images/train_NNNNN.<ext>` plus `out_root/train_labels.csv` (`filename,label`) when `train_layout="csv"`.
    - `out_root/test/img_NNNNN.<ext>` in both cases.
    - `hidden_root/test_labels.csv` with columns `id,label` (the id has no extension).

- [ ] **Step 1: Write the failing tests**

`tests/test_split.py`:
```python
import pandas as pd
import pytest

from data_prep.split import collect_samples, list_images, make_private_split


def make_tree(root, counts, ext=".jpg"):
    for cls, n in counts.items():
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"{cls}_{i}{ext}").write_bytes(b"fake-image")
    return root


def test_list_images_skips_junk(tmp_path):
    d = tmp_path / "cat"
    d.mkdir()
    for name in ["a.jpg", "b.PNG", ".DS_Store", "._a.jpg", "notes.txt"]:
        (d / name).write_bytes(b"x")
    (d / "sub").mkdir()
    assert [p.name for p in list_images(d)] == ["a.jpg", "b.PNG"]


def test_collect_samples_rejects_class_with_one_image(tmp_path):
    make_tree(tmp_path, {"cat": 5, "dog": 1})
    with pytest.raises(ValueError, match="'dog'"):
        collect_samples(tmp_path)


def test_collect_samples_needs_two_classes(tmp_path):
    make_tree(tmp_path, {"cat": 5})
    with pytest.raises(ValueError, match="at least 2 class folders"):
        collect_samples(tmp_path)


def test_collect_samples_caps_per_class_deterministically(tmp_path):
    make_tree(tmp_path, {"cat": 20, "dog": 20})
    a = collect_samples(tmp_path, max_per_class=5, seed=42)
    b = collect_samples(tmp_path, max_per_class=5, seed=42)
    assert a == b
    assert sum(1 for _, c in a if c == "cat") == 5


def test_split_folders_layout(tmp_path):
    src = make_tree(tmp_path / "src", {"cat": 10, "dog": 10, "fox": 10})
    out, hidden = tmp_path / "out", tmp_path / "hidden"
    info = make_private_split(src, out, hidden, test_size=0.2)
    assert info == {"train": 24, "test": 6, "classes": ["cat", "dog", "fox"]}
    assert sorted(p.name for p in (out / "train").iterdir()) == ["cat", "dog", "fox"]
    assert len(list((out / "train" / "cat").iterdir())) == 8
    test_files = sorted(p.name for p in (out / "test").iterdir())
    assert len(test_files) == 6
    assert all(f.startswith("img_") and f.endswith(".jpg") for f in test_files)
    labels = pd.read_csv(hidden / "test_labels.csv", dtype=str)
    assert list(labels.columns) == ["id", "label"]
    assert sorted(labels["id"] + ".jpg") == test_files
    assert labels["label"].value_counts().to_dict() == {"cat": 2, "dog": 2, "fox": 2}


def test_split_has_no_train_test_leak(tmp_path):
    src = make_tree(tmp_path / "src", {"cat": 10, "dog": 10})
    # make every file's content unique so we can trace it
    for p in src.rglob("*.jpg"):
        p.write_bytes(p.name.encode())
    out, hidden = tmp_path / "out", tmp_path / "hidden"
    make_private_split(src, out, hidden)
    train_bytes = {p.read_bytes() for p in (out / "train").rglob("*.jpg")}
    test_bytes = {p.read_bytes() for p in (out / "test").iterdir()}
    assert train_bytes.isdisjoint(test_bytes)
    assert len(train_bytes) + len(test_bytes) == 20


def test_split_is_deterministic(tmp_path):
    src = make_tree(tmp_path / "src", {"cat": 10, "dog": 10})
    make_private_split(src, tmp_path / "o1", tmp_path / "h1")
    make_private_split(src, tmp_path / "o2", tmp_path / "h2")
    assert (tmp_path / "h1" / "test_labels.csv").read_text() == (tmp_path / "h2" / "test_labels.csv").read_text()


def test_split_csv_layout(tmp_path):
    src = make_tree(tmp_path / "src", {"cat": 10, "dog": 10})
    out, hidden = tmp_path / "out", tmp_path / "hidden"
    make_private_split(src, out, hidden, train_layout="csv")
    df = pd.read_csv(out / "train_labels.csv", dtype=str)
    assert list(df.columns) == ["filename", "label"]
    assert len(df) == 16
    assert sorted(df["filename"]) == sorted(p.name for p in (out / "train" / "images").iterdir())
    assert all(not name.startswith(("cat", "dog")) for name in df["filename"])


def test_split_refuses_to_overwrite(tmp_path):
    src = make_tree(tmp_path / "src", {"cat": 10, "dog": 10})
    out, hidden = tmp_path / "out", tmp_path / "hidden"
    make_private_split(src, out, hidden)
    with pytest.raises(FileExistsError, match="overwrite"):
        make_private_split(src, out, hidden)
    make_private_split(src, out, hidden, overwrite=True)
    assert len(list((out / "test").iterdir())) == 4


def test_split_rejects_bad_layout(tmp_path):
    src = make_tree(tmp_path / "src", {"cat": 10, "dog": 10})
    with pytest.raises(ValueError, match="train_layout"):
        make_private_split(src, tmp_path / "o", tmp_path / "h", train_layout="parquet")
```

- [ ] **Step 2: Run the tests and check they fail**

Run: `pytest tests/test_split.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'data_prep'`

- [ ] **Step 3: Implement `data_prep/split.py`** (and create an empty `data_prep/__init__.py`)

```python
from __future__ import annotations

import random
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_LAYOUTS = ("folders", "csv")


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in Path(class_dir).iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTS
    )


def collect_samples(src_root: Path, max_per_class: int | None = None, seed: int = 42) -> list[tuple[Path, str]]:
    src_root = Path(src_root)
    class_dirs = sorted(d for d in src_root.iterdir() if d.is_dir() and not d.name.startswith("."))
    if len(class_dirs) < 2:
        raise ValueError(f"{src_root} must contain at least 2 class folders, found {len(class_dirs)}")
    rng = random.Random(seed)
    samples: list[tuple[Path, str]] = []
    for d in class_dirs:
        imgs = list_images(d)
        if len(imgs) < 2:
            raise ValueError(f"class {d.name!r} has {len(imgs)} images; need at least 2 to split")
        if max_per_class is not None and len(imgs) > max_per_class:
            imgs = sorted(rng.sample(imgs, max_per_class))
        samples += [(p, d.name) for p in imgs]
    return samples


def _prepare_dir(d: Path, overwrite: bool) -> None:
    if d.exists() and any(d.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{d} is not empty; pass overwrite=True to replace it")
        shutil.rmtree(d)


def make_private_split(
    src_root,
    out_root,
    hidden_root,
    test_size: float = 0.2,
    seed: int = 42,
    max_per_class: int | None = None,
    train_layout: str = "folders",
    overwrite: bool = False,
) -> dict:
    src_root, out_root, hidden_root = Path(src_root), Path(out_root), Path(hidden_root)
    if train_layout not in TRAIN_LAYOUTS:
        raise ValueError(f"train_layout must be one of {TRAIN_LAYOUTS}, got {train_layout!r}")
    samples = collect_samples(src_root, max_per_class=max_per_class, seed=seed)
    _prepare_dir(out_root, overwrite)
    _prepare_dir(hidden_root, overwrite)

    paths = [p for p, _ in samples]
    labels = [c for _, c in samples]
    tr_p, te_p, tr_y, te_y = train_test_split(
        paths, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    # test: anonymous ids in shuffled order, labels hidden
    test_dir = out_root / "test"
    test_dir.mkdir(parents=True)
    order = list(range(len(te_p)))
    random.Random(seed).shuffle(order)
    rows = []
    for n, i in enumerate(order):
        img_id = f"img_{n:05d}"
        shutil.copy2(te_p[i], test_dir / f"{img_id}{te_p[i].suffix.lower()}")
        rows.append((img_id, te_y[i]))
    hidden_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["id", "label"]).to_csv(hidden_root / "test_labels.csv", index=False)

    # train
    if train_layout == "folders":
        for p, y in zip(tr_p, tr_y):
            dst = out_root / "train" / y
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / p.name)
    else:
        img_dir = out_root / "train" / "images"
        img_dir.mkdir(parents=True)
        train_rows = []
        for n, (p, y) in enumerate(zip(tr_p, tr_y)):
            name = f"train_{n:05d}{p.suffix.lower()}"
            shutil.copy2(p, img_dir / name)
            train_rows.append((name, y))
        pd.DataFrame(train_rows, columns=["filename", "label"]).to_csv(out_root / "train_labels.csv", index=False)

    return {"train": len(tr_p), "test": len(te_p), "classes": sorted(set(labels))}
```

- [ ] **Step 4: Run the tests and check they pass**

Run: `pytest tests/test_split.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add data_prep/ tests/test_split.py
git commit -m "feat(data_prep): stratified private split with hidden labels"
```

---

### Task 4: Kaggle download helpers and the Intel dataset script

**Files:**
- Create: `data_prep/kaggle_io.py`, `data_prep/prepare_intel.py`
- Test: `tests/test_kaggle_io.py`

**Interfaces:**
- Consumes: `make_private_split` from Task 3.
- Produces:
  - `data_prep.kaggle_io.download(slug: str, dest: Path) -> Path`: runs the Kaggle CLI with `--unzip`, and skips the download if `dest` is already non-empty.
  - `data_prep.kaggle_io.find_class_root(root: Path, classes: Iterable[str]) -> Path`: returns the first directory under `root` whose sub-folders include every class. It prefers paths that contain `"train"`.
  - `data_prep.prepare_intel.SLUG`, `CLASSES` and `main(argv=None) -> int`.
  - The CLI is `python -m data_prep.prepare_intel [--raw data/raw/intel] [--out data/intel] [--hidden data/_hidden/intel] [--max-per-class 1000] [--overwrite]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_kaggle_io.py`:
```python
from pathlib import Path

import pytest

from data_prep import kaggle_io
from data_prep.kaggle_io import download, find_class_root

CLASSES = ("buildings", "forest")


def make_dirs(root: Path, rels):
    for r in rels:
        (root / r).mkdir(parents=True, exist_ok=True)


def test_find_class_root_prefers_train(tmp_path):
    make_dirs(tmp_path, [
        "seg_pred/seg_pred",
        "seg_test/seg_test/buildings", "seg_test/seg_test/forest",
        "seg_train/seg_train/buildings", "seg_train/seg_train/forest",
    ])
    assert find_class_root(tmp_path, CLASSES) == tmp_path / "seg_train" / "seg_train"


def test_find_class_root_accepts_root_itself(tmp_path):
    make_dirs(tmp_path, ["buildings", "forest"])
    assert find_class_root(tmp_path, CLASSES) == tmp_path


def test_find_class_root_raises_when_absent(tmp_path):
    make_dirs(tmp_path, ["a/buildings"])
    with pytest.raises(FileNotFoundError, match="forest"):
        find_class_root(tmp_path, CLASSES)


def test_download_skips_when_cached(tmp_path, monkeypatch):
    (tmp_path / "something.txt").write_text("x")
    calls = []
    monkeypatch.setattr(kaggle_io.subprocess, "run", lambda *a, **k: calls.append(a))
    assert download("owner/slug", tmp_path) == tmp_path
    assert calls == []


def test_download_calls_kaggle_cli(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(kaggle_io.subprocess, "run", lambda cmd, check: calls.append(cmd))
    dest = tmp_path / "raw"
    download("owner/slug", dest)
    assert calls == [["kaggle", "datasets", "download", "-d", "owner/slug", "-p", str(dest), "--unzip"]]
```

- [ ] **Step 2: Run the tests and check they fail**

Run: `pytest tests/test_kaggle_io.py -v`
Expected: collection error, `ImportError: cannot import name 'kaggle_io'`

- [ ] **Step 3: Implement `data_prep/kaggle_io.py`**

```python
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable


def download(slug: str, dest: Path) -> Path:
    """Download and unzip a Kaggle dataset into dest; skip if dest already has files."""
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        print(f"[kaggle] {dest} already populated, skipping download")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"], check=True)
    return dest


def find_class_root(root: Path, classes: Iterable[str]) -> Path:
    root = Path(root)
    wanted = set(classes)
    candidates = [root, *sorted(p for p in root.rglob("*") if p.is_dir())]
    matches = [d for d in candidates if wanted <= {c.name for c in d.iterdir() if c.is_dir()}]
    if not matches:
        raise FileNotFoundError(f"no folder under {root} contains all class folders {sorted(wanted)}")
    train_matches = [m for m in matches if "train" in str(m.relative_to(root)).lower()]
    return (train_matches or matches)[0]
```

- [ ] **Step 4: Implement `data_prep/prepare_intel.py`**

```python
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
```

- [ ] **Step 5: Run the tests and check they pass**

Run: `pytest -v`
Expected: everything passes, including the 5 new tests in `test_kaggle_io.py`

- [ ] **Step 6: Commit**

```bash
git add data_prep/kaggle_io.py data_prep/prepare_intel.py tests/test_kaggle_io.py
git commit -m "feat(data_prep): kaggle helpers and Intel dataset script"
```

---

### Task 5: Week 1 drills

**Files:**
- Create: `weeks/week01/drills/README.md`
- Create: `weeks/week01/drills/01_ultralytics_train_predict/prompt.md`, `reference.py`
- Create: `weeks/week01/drills/02_val_split/prompt.md`, `reference.py`
- Create: `weeks/week01/drills/03_write_submission/prompt.md`, `reference.py`
- Test: `tests/test_week01_drills.py`

**Interfaces:**
- Consumes: `Schema` and `validate_submission` from Task 1 (the tests use them to check drill output against the real grader).
- Produces:
  - `02_val_split/reference.py::make_val_split(src: Path, dst: Path, val_frac: float = 0.1, seed: int = 0) -> dict[str, int]`, which returns `{"train": n, "val": n}`.
  - `03_write_submission/reference.py::write_submission(preds: list[tuple[str, str]], out_path, id_col: str, label_col: str, classes: list[str], label_type: str = "name", keep_ext: bool = False) -> None`

- [ ] **Step 1: Write the failing tests**

`tests/test_week01_drills.py`:
```python
import importlib.util
from pathlib import Path

from grader.schema import Schema
from grader.validate import read_submission, validate_submission

DRILLS = Path(__file__).resolve().parent.parent / "weeks" / "week01" / "drills"


def load(rel: str):
    path = DRILLS / rel
    spec = importlib.util.spec_from_file_location(path.parent.name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_make_val_split(tmp_path):
    mod = load("02_val_split/reference.py")
    src = tmp_path / "train"
    for cls in ("cat", "dog"):
        (src / cls).mkdir(parents=True)
        for i in range(20):
            (src / cls / f"{i}.jpg").write_bytes(b"x")
    (src / "cat" / ".DS_Store").write_bytes(b"x")
    counts = mod.make_val_split(src, tmp_path / "work", val_frac=0.1, seed=0)
    assert counts == {"train": 36, "val": 4}
    assert len(list((tmp_path / "work" / "val" / "cat").iterdir())) == 2
    assert not (tmp_path / "work" / "train" / "cat" / ".DS_Store").exists()
    val = {p.name for p in (tmp_path / "work" / "val" / "dog").iterdir()}
    train = {p.name for p in (tmp_path / "work" / "train" / "dog").iterdir()}
    assert val.isdisjoint(train)


def test_make_val_split_tiny_class_keeps_one_val(tmp_path):
    mod = load("02_val_split/reference.py")
    src = tmp_path / "train"
    for cls, n in (("cat", 3), ("dog", 3)):
        (src / cls).mkdir(parents=True)
        for i in range(n):
            (src / cls / f"{i}.jpg").write_bytes(b"x")
    counts = mod.make_val_split(src, tmp_path / "work", val_frac=0.1)
    assert counts == {"train": 4, "val": 2}


def test_write_submission_names_passes_grader(tmp_path):
    mod = load("03_write_submission/reference.py")
    preds = [("data/intel/test/img_00000.jpg", "sea"), ("data/intel/test/img_00001.jpg", "forest")]
    out = tmp_path / "submission.csv"
    mod.write_submission(preds, out, "image_id", "label", ["forest", "sea"])
    schema = Schema("image_id", "label", "name", False, ("forest", "sea"))
    errors, got = validate_submission(read_submission(out), schema, {"img_00000", "img_00001"})
    assert errors == []
    assert got == {"img_00000": "sea", "img_00001": "forest"}


def test_write_submission_index_with_ext_passes_grader(tmp_path):
    mod = load("03_write_submission/reference.py")
    preds = [("x/img_00000.jpg", "sea"), ("x/img_00001.jpg", "forest")]
    out = tmp_path / "submission.csv"
    mod.write_submission(preds, out, "filename", "class", ["forest", "sea"], label_type="index", keep_ext=True)
    assert out.read_text().splitlines() == ["filename,class", "img_00000.jpg,1", "img_00001.jpg,0"]
    schema = Schema("filename", "class", "index", True, ("forest", "sea"))
    errors, _ = validate_submission(read_submission(out), schema, {"img_00000", "img_00001"})
    assert errors == []
```

- [ ] **Step 2: Run the tests and check they fail**

Run: `pytest tests/test_week01_drills.py -v`
Expected: FAIL with `FileNotFoundError` for `02_val_split/reference.py`

- [ ] **Step 3: Write `02_val_split/reference.py`**

```python
"""Drill 02 reference: split a folder-per-class train set into train/ and val/ for Ultralytics."""
import random
import shutil
from pathlib import Path


def make_val_split(src, dst, val_frac=0.1, seed=0):
    rng = random.Random(seed)
    counts = {"train": 0, "val": 0}
    for cls_dir in sorted(p for p in Path(src).iterdir() if p.is_dir()):
        files = sorted(f for f in cls_dir.iterdir() if f.is_file() and not f.name.startswith("."))
        rng.shuffle(files)
        n_val = max(1, int(len(files) * val_frac))
        for split, subset in (("val", files[:n_val]), ("train", files[n_val:])):
            out = Path(dst) / split / cls_dir.name
            out.mkdir(parents=True, exist_ok=True)
            for f in subset:
                shutil.copy(f, out / f.name)
            counts[split] += len(subset)
    return counts


if __name__ == "__main__":
    print(make_val_split("data/intel/train", "work/intel_cls"))
```

- [ ] **Step 4: Write `03_write_submission/reference.py`**

```python
"""Drill 03 reference: turn (image path, predicted class) pairs into the required CSV."""
from pathlib import Path

import pandas as pd


def write_submission(preds, out_path, id_col, label_col, classes, label_type="name", keep_ext=False):
    rows = []
    for path, cls in preds:
        p = Path(path)
        rows.append({
            id_col: p.name if keep_ext else p.stem,
            label_col: classes.index(cls) if label_type == "index" else cls,
        })
    pd.DataFrame(rows, columns=[id_col, label_col]).to_csv(out_path, index=False)


if __name__ == "__main__":
    demo = [("data/intel/test/img_00000.jpg", "sea")]
    write_submission(demo, "submission.csv", "image_id", "label", ["buildings", "forest", "glacier", "mountain", "sea", "street"])
```

- [ ] **Step 5: Write `01_ultralytics_train_predict/reference.py`** (needs a GPU and weights, so it's checked by hand in Task 6)

```python
"""Drill 01 reference: train YOLO classify on a train/val folder, predict a folder, get class names."""
from pathlib import Path

from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")
model.train(data="work/intel_cls", epochs=3, imgsz=160, batch=64, seed=0)

best = YOLO(Path(model.trainer.save_dir) / "weights" / "best.pt")
for r in best.predict(source="data/intel/test", imgsz=160, stream=True, verbose=False):
    top1 = r.probs.top1                     # int class index
    print(Path(r.path).stem, r.names[top1], float(r.probs.top1conf))
```

- [ ] **Step 6: Write the three `prompt.md` files and the drills README**

`weeks/week01/drills/README.md`:
```markdown
# Week 1 drills (Friday)

Rules: no docs, no LLM, no autocomplete. `help()`, `dir()` and `yolo cfg` are allowed.

For each drill:
1. Start a timer set to the time box.
2. Open `prompt.md` only (not `reference.py`). Write your answer in `work/drills/<drill>.py`.
3. Run it.
4. Diff against `reference.py` and note every difference in `logs/progress.md` under "Drill notes".
5. A drill passes when it runs correctly inside the time box. If it fails, repeat it next Friday until it passes twice in a row.

| Drill | Time box |
|---|---|
| 01 Ultralytics train + predict | 15 min |
| 02 Train/val split | 10 min |
| 03 Write the submission CSV | 10 min |
```

`01_ultralytics_train_predict/prompt.md`:
```markdown
# Drill 01: Ultralytics train + predict (15 min)

`work/intel_cls/` contains `train/<class>/` and `val/<class>/` (run drill 02 first if it doesn't exist).

From memory, write a script that:
1. Loads the pretrained `yolo11n-cls.pt` classification model.
2. Trains it for 3 epochs at image size 160 with batch size 64.
3. Loads the **best** weights from that run (not the last).
4. Predicts every image in `data/intel/test/`, printing one line per image: `<filename without extension> <class name> <confidence>`.
```

`02_val_split/prompt.md`:
```markdown
# Drill 02: Train/val split for Ultralytics (10 min)

`data/intel/train/<class>/*.jpg` holds all labelled images. Ultralytics classify needs `root/train/<class>/` and `root/val/<class>/`.

From memory, write `make_val_split(src, dst, val_frac=0.1, seed=0)` that:
- Copies ~10% of each class into `dst/val/<class>/` and the rest into `dst/train/<class>/`.
- Always puts at least 1 image per class in val.
- Skips hidden files such as `.DS_Store`.
- Returns `{"train": n_train, "val": n_val}`.
```

`03_write_submission/prompt.md`:
```markdown
# Drill 03: Write the submission CSV (10 min)

You have `preds`, a list of `(image_path, predicted_class_name)` tuples.

From memory, write `write_submission(preds, out_path, id_col, label_col, classes, label_type="name", keep_ext=False)` that writes a CSV:
- Columns are exactly `[id_col, label_col]`, in that order, with no index column.
- The id is the file name, with the extension only if `keep_ext=True`.
- The label is the class name, or its position in `classes` if `label_type == "index"`.

Then check your output by eye: row count, header, and a few rows.
```

- [ ] **Step 7: Run the tests and check they pass**

Run: `pytest -v`
Expected: everything passes, including the 4 new tests in `test_week01_drills.py`

- [ ] **Step 8: Commit**

```bash
git add weeks/week01/drills tests/test_week01_drills.py
git commit -m "feat(week01): drills with tested reference solutions"
```

---

### Task 6: Week 1 notes, mock brief, reference solution, baseline and README

**Files:**
- Create: `weeks/week01/notes.md`, `weeks/week01/mock/README.md`, `weeks/week01/mock/schema.json`, `weeks/week01/mock/reference.py`, `README.md`
- Test: `tests/test_week01_mock.py`

**Interfaces:**
- Consumes: `load_schema` (Task 1), the `grader.score` CLI (Task 2), the `data_prep.prepare_intel` CLI and `CLASSES` (Task 4).
- Produces: `weeks/week01/mock/schema.json`, which the learner passes to `grader.score`.

- [ ] **Step 1: Write the failing test**

`tests/test_week01_mock.py`:
```python
from pathlib import Path

from data_prep.prepare_intel import CLASSES
from grader.schema import load_schema

MOCK = Path(__file__).resolve().parent.parent / "weeks" / "week01" / "mock"


def test_week01_schema_matches_dataset_and_spec():
    s = load_schema(MOCK / "schema.json")
    assert s.classes == CLASSES
    assert (s.id_column, s.label_column, s.label_type, s.id_has_extension) == ("image_id", "label", "name", False)
    assert s.time_limit_minutes == 90


def test_week01_brief_states_the_schema():
    brief = (MOCK / "README.md").read_text()
    assert "image_id,label" in brief
    assert "WITHOUT" in brief and "90 minutes" in brief
```

- [ ] **Step 2: Run the test and check it fails**

Run: `pytest tests/test_week01_mock.py -v`
Expected: FAIL with `FileNotFoundError` for `schema.json`

- [ ] **Step 3: Write `weeks/week01/mock/schema.json`** (`target_accuracy` stays `null` until Step 7 measures the baseline)

```json
{
  "id_column": "image_id",
  "label_column": "label",
  "label_type": "name",
  "id_has_extension": false,
  "classes": ["buildings", "forest", "glacier", "mountain", "sea", "street"],
  "time_limit_minutes": 90,
  "target_accuracy": null
}
```

- [ ] **Step 4: Write `weeks/week01/mock/README.md`**

```markdown
# Week 1 Mock: Scene classification

**Time limit: 90 minutes.** Start the timer when you open this file's "Task" section.

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
```

- [ ] **Step 5: Write `weeks/week01/mock/reference.py`**

```python
"""Week 1 mock reference solution. Used to measure the baseline; learner opens it only after grading."""
import random
import shutil
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

DATA = Path("data/intel")
WORK = Path("work/intel_cls")

# 1. train/val split (Ultralytics needs root/train/<cls> and root/val/<cls>)
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
```

- [ ] **Step 6: Write `weeks/week01/notes.md`**

````markdown
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
````

- [ ] **Step 7: Measure the baseline on Lightning AI (GPU Studio, by hand)**

Run:
```bash
pip install -r requirements.txt
python -m data_prep.prepare_intel
python weeks/week01/mock/reference.py
python -m grader.score --submission submission.csv --schema weeks/week01/mock/schema.json \
  --labels data/_hidden/intel/test_labels.csv --minutes 0 --week 1 --dataset intel --no-log
```
Expected:
- `prepare_intel` prints `train=4800 test=1200`.
- `reference.py` prints `wrote 1200 rows (expected 1200)`.
- The grader prints `Result: NO TARGET` and `Accuracy: ≈0.88–0.93`.

Then set `target_accuracy` in `schema.json` to the measured accuracy minus 0.03, rounded down to 2 decimals. For example, a measured 0.912 gives `0.88`. Re-run the grader and check it prints `Result: PASS`.

- [ ] **Step 8: Write the top-level `README.md`**

````markdown
# CV Exam Course

A 12-week course to prepare for an image-classification coding test (no docs, no LLM, CSV submission) and a system design interview. Design: `docs/superpowers/specs/2026-09-27-cv-exam-course-design.md`.

## Setup (Lightning AI Studio)
```bash
git clone https://github.com/bright-arparwut/cv-exam-course.git && cd cv-exam-course
pip install -r requirements.txt
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json   # from kaggle.com → Settings → API
pytest                                                                             # checks the tooling
```

## Weekly rhythm
| Day | Block | Where |
|---|---|---|
| Mon–Tue | Concept | `weeks/weekNN/notes.md` |
| Wed–Thu | Build | `weeks/weekNN/build/` |
| Fri | Drills | `weeks/weekNN/drills/` |
| Sat | Mock | `weeks/weekNN/mock/README.md` → `python -m grader.score ...` |
| Sun | Review | `logs/progress.md` |

## Layout
- `grader/`: scores a `submission.csv` (format check, then accuracy, macro-F1 and per-class recall) and logs it.
- `data_prep/`: downloads Kaggle datasets and makes private splits (hidden labels go in `data/_hidden/`).
- `weeks/`: notes, drills and mocks.
- `logs/progress.md`: one row per mock.
````

- [ ] **Step 9: Run all tests and check they pass**

Run: `pytest -v`
Expected: all tests pass, including the 2 in `test_week01_mock.py`

- [ ] **Step 10: Commit and push**

```bash
git add README.md weeks/week01/notes.md weeks/week01/mock tests/test_week01_mock.py
git commit -m "feat(week01): notes, mock brief, reference solution and baseline target"
git push
```
