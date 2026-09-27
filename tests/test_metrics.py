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
