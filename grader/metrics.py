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
