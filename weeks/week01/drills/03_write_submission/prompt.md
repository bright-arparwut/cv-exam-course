# Drill 03: Write the submission CSV (10 min)

You have `preds`, a list of `(image_path, predicted_class_name)` tuples.

From memory, write `write_submission(preds, out_path, id_col, label_col, classes, label_type="name", keep_ext=False)` that writes a CSV:
- Columns are exactly `[id_col, label_col]`, in that order, with no index column.
- The id is the file name, with the extension only if `keep_ext=True`.
- The label is the class name, or its position in `classes` if `label_type == "index"`.

Then check your output by eye: row count, header, and a few rows.
