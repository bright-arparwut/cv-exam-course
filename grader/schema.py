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
