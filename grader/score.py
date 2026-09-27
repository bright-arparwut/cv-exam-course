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
