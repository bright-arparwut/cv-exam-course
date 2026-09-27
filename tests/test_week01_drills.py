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
