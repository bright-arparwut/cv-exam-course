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
