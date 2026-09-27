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
