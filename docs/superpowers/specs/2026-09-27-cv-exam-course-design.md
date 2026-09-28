# CV Exam Course — Design Spec

**Date:** 2026-09-27
**Author:** Arparwut (with Claude)
**Status:** Draft — awaiting review

---

## 1. Goal and context

**Target:** pass the Axons (CPF) AI Engineer hiring process:
1. A **HackerRank image-classification coding test**. No docs, no LLM. The output is a **predictions CSV**.
2. A **system design interview**.

**Known facts**
- The candidate is a Python developer with CNN experience and an overview-level knowledge of ViT. They know Hugging Face and Ultralytics.
- An acquaintance at Axons passed using **Ultralytics YOLO**. Candidates who used plain PyTorch or boilerplate-heavy code did not pass.
- The submission is a predictions CSV.
- Unknown: time limit, whether a GPU is available, internet access, whether Ultralytics is pre-installed, the exact dataset layout, and whether training is required.

**Assumptions** (the course covers the hardest plausible version of each)
- The test is timed and follows a Kaggle-style flow: get images → train or fine-tune → predict → write a CSV, scored mainly on accuracy.
- There may be no GPU and no internet. So we practise a CPU-friendly setup and an offline-weights fallback.
- Dataset layouts vary: folder-per-class, labels in a CSV, or pre-split vs unsplit.

**Success criteria**
- By **week 10**, finish a mock in **≤45 min**, from a blank file to a correctly formatted CSV, with no docs, reaching the per-dataset target accuracy (§5.4).
- Be able to explain the input and output of **CNN, ViT and CLIP** (tensor shapes, preprocessing, logits vs embeddings) without notes.
- Have **4 one-page system design docs** plus a working capstone to talk about in the interview.

**Out of scope:** object detection and segmentation, research-level maths, TensorFlow/Keras, and writing a reusable framework.

## 2. Constraints

| Item | Value |
|---|---|
| Duration | 12 weeks |
| Weekly time | 15–20 hrs |
| Main compute | MacBook M1 Pro (MPS for training, CPU for some mocks) — changed 2026-09-28 from Lightning AI |
| Optional | Lightning AI GPU Studio when a model is too slow on the M1 (e.g. ViT fine-tuning) |
| Exam tool | Ultralytics YOLO classify (`yolo11n-cls` / `yolov8n-cls`) |
| Learning tools | PyTorch, PyTorch Lightning, Hugging Face (`transformers`, `datasets`, pipelines), timm |
| Data source | Kaggle, via the Kaggle API |

## 3. Approach: mock-exam ladder

There are two tracks, and both run every week.
- **Exam track:** Ultralytics classify, drilled until it's automatic. Every week ends with a timed mock that produces a CSV.
- **Understanding track:** one model family per week in PyTorch, Lightning or Hugging Face, focused on how it works at an overview level and its input and output.

A **minimal fallback** (a timm or torchvision pretrained model in about 30 lines) is drilled in case Ultralytics or pretrained weights aren't available in the test environment.

Rejected alternatives:
- **Concepts first, then cram:** exam muscle memory would only start in week 9.
- **One growing framework repo:** this trains the boilerplate habit that failed other candidates.

## 4. Weekly rhythm

| Day | Block | Hrs | Content |
|---|---|---|---|
| Mon–Tue | Concept | ~5 | Study the week's model family. Trace one forward pass and write down the shapes: input → intermediate → output. Output: `notes.md` |
| Wed–Thu | Build | ~5 | Implement a train or fine-tune loop in Lightning or HF on the week's dataset. Docs allowed. Output: `build/` |
| Fri | Drill | ~2 | 3–4 no-docs reps of 10–20 min each, from the drill bank (§6) |
| Sat | Mock | ~3 | Timed, no docs, no LLM. Brief → CSV → graded (§5) |
| Sun | Review | ~1–2 | Log where you got stuck in `logs/progress.md`; update the personal cheat sheet. From week 8, +45 min system design (§8) |

## 5. Mock tests and grading

### 5.1 Brief format
Each mock has a `README.md` written like a HackerRank task. It covers:
- The task.
- The folder layout.
- The class list.
- The **required CSV schema**.

The schema changes from week to week to force careful reading. Variations include:
- Column names: `image_id,label` vs `filename,class`.
- Labels as class names vs integer indices.
- Ids with or without file extensions.

### 5.2 Time limits
| Weeks | Limit |
|---|---|
| 1–4 | 90 min |
| 5–8 | 60 min |
| 9–12 | 45 min |

### 5.3 Rules
- No browser, no LLM, no AI autocomplete, and no Ultralytics or PyTorch docs.
- Allowed: `help()`, `dir()`, Jupyter `?`, and `--help` in the shell.
- Ultralytics `-cls` weights and one timm/torchvision backbone are pre-downloaded. From week 9, some mocks **block the internet and/or the GPU**.

### 5.4 Grading (`grader/score.py`)
Inputs are `submission.csv`, the hidden `test_labels.csv` and the brief's schema. It checks:
1. **Format:** correct columns, every id present, no duplicates, and labels in the valid set. **Any format error scores 0.**
2. **Metrics:** accuracy (primary), per-class recall, and macro-F1.
3. **Time:** entered manually (start and end).

It then appends a row to `logs/progress.md`: week, dataset, accuracy, macro-F1, minutes, pass/fail, blockers.

**Target accuracy per dataset:** the author (Claude, when building each week) runs a quick Ultralytics baseline (`yolo11n-cls`, default settings, ~10–20 min on the M1). The target is `baseline − 3 percentage points`, recorded in the brief. A pass means reaching the target within the time limit.

### 5.5 Private splits (`datasets/prepare_<name>.py`)
- Downloads with `kaggle datasets download`.
- Makes a stratified split into `train/` (labelled) and `test/` (no labels, files renamed to anonymous ids).
- Writes `test_labels.csv` to a hidden folder. Seed = 42.
- May deliberately reshape the layout (e.g. convert folders into a labels CSV) to match that week's brief.

## 6. Drill bank

Each drill = a prompt, a time box, and a reference solution. You write from memory, run it, then diff against the reference.

| Group | Drills |
|---|---|
| Ultralytics | train with `data/epochs/imgsz/batch/device`; predict on a folder and take `probs.top1` → class name through `names`; load `best.pt`; `val` metrics |
| Data wrangling | labels CSV → folder-per-class; stratified train/val split; match filenames to ids; write the CSV with pandas in the required schema |
| PyTorch / Lightning | `Dataset` class; transforms and ImageNet normalisation; plain training loop; `LightningModule` + `Trainer` |
| Fallback | timm/torchvision pretrained model → replace the head → fine-tune → predict → CSV, in about 30 lines |
| Understanding | write the input/output shapes from memory for a CNN, ViT and CLIP forward pass |

**Schedule:** each Friday = 1 Ultralytics drill + 1 wrangling drill + 1–2 drills from the week's topic. From week 4, one of those topic slots is always the fallback drill. Any failed drill comes back the following Friday until it passes twice in a row.

## 7. Twelve-week curriculum

In every mock, **Ultralytics is the primary submission**. If time remains, you may submit a second CSV using the week's model.

| Wk | Concept & build focus | Dataset (Kaggle slug) | Notes |
|---|---|---|---|
| 1 | Exam pipeline v1: Ultralytics classify, folder layout, train/val/predict, results → CSV | Intel Image Classification (`puneet6060/intel-image-classification`) | ~25k images, 150×150, 6 classes; comes pre-split |
| 2 | PyTorch basics: tensors, Dataset/DataLoader, transforms, normalisation, training loop, then the Lightning version | Rice Image Dataset (`muratkokludataset/rice-image-dataset`) | 75k images, 5 classes, CC0; subsample to ~1k per class |
| 3 | CNN internals: conv, pooling, BatchNorm, receptive field, feature-map shapes; small CNN in Lightning | Weather Image Recognition (`jehanbhathena/weather-dataset`) | 6,862 images, 11 classes, not pre-split, so the split is practised |
| 4 | Transfer learning: ResNet/EfficientNet, freeze vs fine-tune, torchvision and timm | Oxford 102 Flowers (a Kaggle mirror; slug picked at build time) | Few images per class |
| 5 | Training craft: augmentation, LR schedules, overfitting, confusion matrix, F1, class imbalance | Chest X-Ray Pneumonia (`paultimothymooney/chest-xray-pneumonia`) | Imbalanced; macro-F1 is shown next to accuracy |
| 6 | ViT: patches, CLS token, attention, position embeddings; HF `ViTForImageClassification` and the image processor | Food-101 (a Kaggle mirror), subset of 20 classes | Fine-grained classes |
| 7 | CLIP: image/text encoders, embeddings, zero-shot, linear probe | PlantVillage (`emmarex/plantdisease`), subset | Agriculture; compare zero-shot vs linear probe vs Ultralytics |
| 8 | HF ecosystem + comparison: pipelines, Trainer, `datasets`; CNN vs ViT vs CLIP on one dataset | A Large Scale Fish Dataset (`crowww/a-large-scale-fish-dataset`) | 9 classes × 1k images; ignore the ground-truth mask folders |
| 9 | Hard cases I: CPU-only, no internet (offline weights, fallback path), tiny data | Chicken disease fecal images (`allandclive/chicken-disease-1`) | Healthy / Coccidiosis / Salmonella / Newcastle; imbalanced |
| 10 | Hard cases II: speed under a timer, test-time augmentation, error analysis, unfamiliar layouts | A surprise dataset that Claude picks and hides until the mock starts | Simulates the unknown |
| 11–12 | Capstone + system design (§8), plus 2 full mocks per week | Meat Quality Assessment (`crowww/meat-quality-assessment-based-on-deep-learning`) + surprise datasets | Fresh vs spoiled red meat |

**Dataset fallback rule:** if a slug is unavailable or too large when a week is built, replace it with a Kaggle dataset that has a similar class count and difficulty, and note the swap in that week's `notes.md`.

## 8. Capstone and system design

### 8.1 Capstone (weeks 11–12): meat quality inspection
1. **Data:** quality audit and split; write down how labels would be collected at a plant.
2. **Model:** train with Ultralytics; compare against the best model from the understanding track (ViT or CLIP linear probe) on accuracy, CPU latency and model size.
3. **Export:** ONNX; benchmark CPU inference.
4. **Serve:** a FastAPI `POST /predict` endpoint (image → label + confidence) in Docker (run locally on the Mac; Lightning AI optional).
5. **Monitor:** log confidence per request, plus a simple drift check (brightness and colour statistics vs the training set).

### 8.2 System design practice (from week 8)
**Framework:** requirements → data (collection and labelling) → training pipeline → evaluation → serving (edge or cloud, batch or real-time) → monitoring and retraining → cost and trade-offs.

| Wk | Case |
|---|---|
| 8 | Meat or carcass defect detection on a production line (edge camera, latency) |
| 9 | Egg crack and dirt sorting at high throughput |
| 10 | Poultry health monitoring from farm cameras (few labels, seasonal drift) |
| 11–12 | The capstone, as a 30-minute mock interview. Claude plays the interviewer |

**Output:** a one-page design doc with a diagram for each case, saved in `system-design/`.

## 9. Repository layout

```
cv-exam-course/
  docs/superpowers/specs/      # this spec
  weeks/weekNN/
    notes.md                   # concept notes + shape traces
    build/                     # Lightning / HF code
    drills/                    # prompt.md + reference.py per drill
    mock/README.md             # the brief
    practice/                  # the learner's own code (committed)
  datasets/prepare_<name>.py   # download + private split
  grader/score.py              # format check + metrics + log
  logs/progress.md             # one row per mock
  system-design/               # one-page designs
  capstone/                    # weeks 11–12
```

## 10. Testing the course itself
- `grader/score.py` gets unit tests: a correct CSV, a missing id, a wrong column name, an invalid label, and a duplicate id.
- Each `prepare_*.py` is checked by confirming split counts and that no file appears in both train and test.
- Each mock brief is dry-run by Claude with the reference solution before release, which also produces the baseline for the target accuracy.

## 11. Open risks
| Risk | Mitigation |
|---|---|
| The real test forbids Ultralytics or has no pretrained weights | The fallback drill runs weekly from week 4, and week 9 is dedicated to it |
| The real test is not about training (e.g. inference only, or implementing a function) | Drills include predict-only and "implement a function" styles; the understanding track covers inputs and outputs |
| Kaggle slugs change | Dataset fallback rule (§7) |
| Burnout at 15–20 hrs/week | Sunday review can be skipped; week 10 has lighter build work |
