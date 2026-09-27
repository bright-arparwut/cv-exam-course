# Drill 02: Train/val split for Ultralytics (10 min)

`data/intel/train/<class>/*.jpg` holds all labelled images. Ultralytics classify needs `root/train/<class>/` and `root/val/<class>/`.

From memory, write `make_val_split(src, dst, val_frac=0.1, seed=0)` that:
- Copies ~10% of each class into `dst/val/<class>/` and the rest into `dst/train/<class>/`.
- Always puts at least 1 image per class in val.
- Skips hidden files such as `.DS_Store`.
- Returns `{"train": n_train, "val": n_val}`.
