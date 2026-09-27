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
