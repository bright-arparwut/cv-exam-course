"""Drill 03 reference: turn (image path, predicted class) pairs into the required CSV."""
from pathlib import Path

import pandas as pd


def write_submission(preds, out_path, id_col, label_col, classes, label_type="name", keep_ext=False):
    rows = []
    for path, cls in preds:
        p = Path(path)
        rows.append({
            id_col: p.name if keep_ext else p.stem,
            label_col: classes.index(cls) if label_type == "index" else cls,
        })
    pd.DataFrame(rows, columns=[id_col, label_col]).to_csv(out_path, index=False)


if __name__ == "__main__":
    demo = [("data/intel/test/img_00000.jpg", "sea")]
    write_submission(demo, "submission.csv", "image_id", "label", ["buildings", "forest", "glacier", "mountain", "sea", "street"])
