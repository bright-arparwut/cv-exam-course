import pandas as pd
import pytest

from data_prep.split import collect_samples, list_images, make_private_split


def make_tree(root, counts, ext=".jpg"):
    for cls, n in counts.items():
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"{cls}_{i}{ext}").write_bytes(b"fake-image")
    return root


def test_list_images_skips_junk(tmp_path):
    d = tmp_path / "cat"
    d.mkdir()
    for name in ["a.jpg", "b.PNG", ".DS_Store", "._a.jpg", "notes.txt"]:
        (d / name).write_bytes(b"x")
    (d / "sub").mkdir()
    assert [p.name for p in list_images(d)] == ["a.jpg", "b.PNG"]


def test_collect_samples_rejects_class_with_one_image(tmp_path):
    make_tree(tmp_path, {"cat": 5, "dog": 1})
    with pytest.raises(ValueError, match="'dog'"):
        collect_samples(tmp_path)


def test_collect_samples_needs_two_classes(tmp_path):
    make_tree(tmp_path, {"cat": 5})
    with pytest.raises(ValueError, match="at least 2 class folders"):
        collect_samples(tmp_path)


def test_collect_samples_caps_per_class_deterministically(tmp_path):
    make_tree(tmp_path, {"cat": 20, "dog": 20})
    a = collect_samples(tmp_path, max_per_class=5, seed=42)
    b = collect_samples(tmp_path, max_per_class=5, seed=42)
    assert a == b
    assert sum(1 for _, c in a if c == "cat") == 5


def test_split_folders_layout(tmp_path):
    src = make_tree(tmp_path / "src", {"cat": 10, "dog": 10, "fox": 10})
    out, hidden = tmp_path / "out", tmp_path / "hidden"
    info = make_private_split(src, out, hidden, test_size=0.2)
    assert info == {"train": 24, "test": 6, "classes": ["cat", "dog", "fox"]}
    assert sorted(p.name for p in (out / "train").iterdir()) == ["cat", "dog", "fox"]
    assert len(list((out / "train" / "cat").iterdir())) == 8
    test_files = sorted(p.name for p in (out / "test").iterdir())
    assert len(test_files) == 6
    assert all(f.startswith("img_") and f.endswith(".jpg") for f in test_files)
    labels = pd.read_csv(hidden / "test_labels.csv", dtype=str)
    assert list(labels.columns) == ["id", "label"]
    assert sorted(labels["id"] + ".jpg") == test_files
    assert labels["label"].value_counts().to_dict() == {"cat": 2, "dog": 2, "fox": 2}


def test_split_has_no_train_test_leak(tmp_path):
    src = make_tree(tmp_path / "src", {"cat": 10, "dog": 10})
    # make every file's content unique so we can trace it
    for p in src.rglob("*.jpg"):
        p.write_bytes(p.name.encode())
    out, hidden = tmp_path / "out", tmp_path / "hidden"
    make_private_split(src, out, hidden)
    train_bytes = {p.read_bytes() for p in (out / "train").rglob("*.jpg")}
    test_bytes = {p.read_bytes() for p in (out / "test").iterdir()}
    assert train_bytes.isdisjoint(test_bytes)
    assert len(train_bytes) + len(test_bytes) == 20


def test_split_is_deterministic(tmp_path):
    src = make_tree(tmp_path / "src", {"cat": 10, "dog": 10})
    make_private_split(src, tmp_path / "o1", tmp_path / "h1")
    make_private_split(src, tmp_path / "o2", tmp_path / "h2")
    assert (tmp_path / "h1" / "test_labels.csv").read_text() == (tmp_path / "h2" / "test_labels.csv").read_text()


def test_split_csv_layout(tmp_path):
    src = make_tree(tmp_path / "src", {"cat": 10, "dog": 10})
    out, hidden = tmp_path / "out", tmp_path / "hidden"
    make_private_split(src, out, hidden, train_layout="csv")
    df = pd.read_csv(out / "train_labels.csv", dtype=str)
    assert list(df.columns) == ["filename", "label"]
    assert len(df) == 16
    assert sorted(df["filename"]) == sorted(p.name for p in (out / "train" / "images").iterdir())
    assert all(not name.startswith(("cat", "dog")) for name in df["filename"])


def test_split_refuses_to_overwrite(tmp_path):
    src = make_tree(tmp_path / "src", {"cat": 10, "dog": 10})
    out, hidden = tmp_path / "out", tmp_path / "hidden"
    make_private_split(src, out, hidden)
    with pytest.raises(FileExistsError, match="overwrite"):
        make_private_split(src, out, hidden)
    make_private_split(src, out, hidden, overwrite=True)
    assert len(list((out / "test").iterdir())) == 4


def test_split_rejects_bad_layout(tmp_path):
    src = make_tree(tmp_path / "src", {"cat": 10, "dog": 10})
    with pytest.raises(ValueError, match="train_layout"):
        make_private_split(src, tmp_path / "o", tmp_path / "h", train_layout="parquet")
