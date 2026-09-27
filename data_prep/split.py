from __future__ import annotations

import random
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_LAYOUTS = ("folders", "csv")


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in Path(class_dir).iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTS
    )


def collect_samples(src_root: Path, max_per_class: int | None = None, seed: int = 42) -> list[tuple[Path, str]]:
    src_root = Path(src_root)
    class_dirs = sorted(d for d in src_root.iterdir() if d.is_dir() and not d.name.startswith("."))
    if len(class_dirs) < 2:
        raise ValueError(f"{src_root} must contain at least 2 class folders, found {len(class_dirs)}")
    rng = random.Random(seed)
    samples: list[tuple[Path, str]] = []
    for d in class_dirs:
        imgs = list_images(d)
        if len(imgs) < 2:
            raise ValueError(f"class {d.name!r} has {len(imgs)} images; need at least 2 to split")
        if max_per_class is not None and len(imgs) > max_per_class:
            imgs = sorted(rng.sample(imgs, max_per_class))
        samples += [(p, d.name) for p in imgs]
    return samples


def _prepare_dir(d: Path, overwrite: bool) -> None:
    if d.exists() and any(d.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{d} is not empty; pass overwrite=True to replace it")
        shutil.rmtree(d)


def make_private_split(
    src_root,
    out_root,
    hidden_root,
    test_size: float = 0.2,
    seed: int = 42,
    max_per_class: int | None = None,
    train_layout: str = "folders",
    overwrite: bool = False,
) -> dict:
    src_root, out_root, hidden_root = Path(src_root), Path(out_root), Path(hidden_root)
    if train_layout not in TRAIN_LAYOUTS:
        raise ValueError(f"train_layout must be one of {TRAIN_LAYOUTS}, got {train_layout!r}")
    samples = collect_samples(src_root, max_per_class=max_per_class, seed=seed)
    _prepare_dir(out_root, overwrite)
    _prepare_dir(hidden_root, overwrite)

    paths = [p for p, _ in samples]
    labels = [c for _, c in samples]
    tr_p, te_p, tr_y, te_y = train_test_split(
        paths, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    # test: anonymous ids in shuffled order, labels hidden
    test_dir = out_root / "test"
    test_dir.mkdir(parents=True)
    order = list(range(len(te_p)))
    random.Random(seed).shuffle(order)
    rows = []
    for n, i in enumerate(order):
        img_id = f"img_{n:05d}"
        shutil.copy2(te_p[i], test_dir / f"{img_id}{te_p[i].suffix.lower()}")
        rows.append((img_id, te_y[i]))
    hidden_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["id", "label"]).to_csv(hidden_root / "test_labels.csv", index=False)

    # train
    if train_layout == "folders":
        for p, y in zip(tr_p, tr_y):
            dst = out_root / "train" / y
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / p.name)
    else:
        img_dir = out_root / "train" / "images"
        img_dir.mkdir(parents=True)
        train_rows = []
        for n, (p, y) in enumerate(zip(tr_p, tr_y)):
            name = f"train_{n:05d}{p.suffix.lower()}"
            shutil.copy2(p, img_dir / name)
            train_rows.append((name, y))
        pd.DataFrame(train_rows, columns=["filename", "label"]).to_csv(out_root / "train_labels.csv", index=False)

    return {"train": len(tr_p), "test": len(te_p), "classes": sorted(set(labels))}
