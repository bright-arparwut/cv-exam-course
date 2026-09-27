# Drill 01: Ultralytics train + predict (15 min)

`work/intel_cls/` contains `train/<class>/` and `val/<class>/` (run drill 02 first if it doesn't exist).

From memory, write a script that:
1. Loads the pretrained `yolo11n-cls.pt` classification model.
2. Trains it for 3 epochs at image size 160 with batch size 64.
3. Loads the **best** weights from that run (not the last).
4. Predicts every image in `data/intel/test/`, printing one line per image: `<filename without extension> <class name> <confidence>`.
