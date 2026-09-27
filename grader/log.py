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
