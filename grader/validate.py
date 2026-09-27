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
