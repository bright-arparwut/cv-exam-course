import pytest

from data_prep.yolo_layout import main, make_yolo_cls_dir


def make_tree(root, counts):
    for cls, n in counts.items():
        (root / cls).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (root / cls / f"{cls}_{i}.jpg").write_bytes(f"{cls}{i}".encode())
    (root / list(counts)[0] / ".DS_Store").write_bytes(b"x")
    return root


def names(d):
    return {p.name for p in d.iterdir()}


def test_split_mode_makes_symlinked_train_and_val(tmp_path):
    src = make_tree(tmp_path / "train", {"cat": 20, "dog": 10})
    info = make_yolo_cls_dir(src, tmp_path / "cls", val_frac=0.1)
    assert info == {"train": 27, "val": 3, "classes": ["cat", "dog"]}
    for cls in ("cat", "dog"):
        tr, va = tmp_path / "cls" / "train" / cls, tmp_path / "cls" / "val" / cls
        assert names(tr).isdisjoint(names(va))
        assert names(tr) | names(va) == {p.name for p in (src / cls).glob("*.jpg")}
        assert all(p.is_symlink() and p.resolve().parent == (src / cls).resolve() for p in tr.iterdir())
    assert not (tmp_path / "cls" / "train" / "cat" / ".DS_Store").exists()


def test_full_mode_uses_every_image_for_train_and_val(tmp_path):
    src = make_tree(tmp_path / "train", {"cat": 5, "dog": 5})
    info = make_yolo_cls_dir(src, tmp_path / "cls", full=True)
    assert info == {"train": 10, "val": 10, "classes": ["cat", "dog"]}
    assert names(tmp_path / "cls" / "val" / "cat") == names(tmp_path / "cls" / "train" / "cat")


def test_rebuild_replaces_old_folder(tmp_path):
    src = make_tree(tmp_path / "train", {"cat": 5, "dog": 5})
    stale = tmp_path / "cls" / "train" / "cat" / "stale.jpg"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"x")
    make_yolo_cls_dir(src, tmp_path / "cls")
    assert not stale.exists()


def test_refuses_to_write_inside_source(tmp_path):
    src = make_tree(tmp_path / "train", {"cat": 5, "dog": 5})
    with pytest.raises(ValueError, match="source"):
        make_yolo_cls_dir(src, src / "cls")
    with pytest.raises(ValueError, match="source"):
        make_yolo_cls_dir(src, src)


def test_cli(tmp_path, capsys):
    src = make_tree(tmp_path / "train", {"cat": 10, "dog": 10})
    assert main(["--src", str(src), "--out", str(tmp_path / "cls")]) == 0
    assert "train=18 val=2" in capsys.readouterr().out
