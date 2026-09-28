# CONTEXT

A 12-week self-study course to pass an image-classification coding test (HackerRank, no docs, no LLM, predictions-CSV submission) and a system design interview for an AI Engineer role. Main learning device: MacBook M1 Pro (MPS); Lightning AI GPU Studio is optional. Design: `docs/superpowers/specs/2026-09-27-cv-exam-course-design.md`.

## Glossary

| Term | Meaning |
|---|---|
| **Exam track** | Ultralytics YOLO classify, drilled until automatic. Every mock is solved with it first. |
| **Understanding track** | One model family per week (CNN, transfer learning, ViT, CLIP) in PyTorch / Lightning / Hugging Face, focused on inputs, outputs and shapes. Lives in `build/`. |
| **Notes** | `weeks/weekNN/notes.md` — the Mon–Tue concept reading. |
| **Build** | `weeks/weekNN/build/` — Wed–Thu exercises for the understanding track; docs allowed, no timer. Starts in week 2. |
| **Drill** | A 10–20 min no-docs coding rep (Fri). Prompt in `drills/NN_name/prompt.md`, answer key in `reference.py`. |
| **Mock** | The Sat timed exam: a **brief** (`mock/README.md`) → the learner's `submission.csv` → graded. |
| **Schema** | `mock/schema.json` — the CSV rules for one mock (column names, id extension, label as name or index, classes, time limit, target). |
| **Hidden labels** | `data/_hidden/<dataset>/test_labels.csv` (`id,label`) — ground truth the grader uses; never opened before grading. |
| **Target accuracy** | Pass bar in the schema = reference-solution accuracy − 0.03. `null` until the learner measures it. |
| **Format error** | Any schema violation in a submission; scores 0. |
| **Practice** | `weeks/weekNN/practice/` — the learner's own code (daily work, drill attempts, mock attempts). Committed. |
| **Reference solution** | `reference.py` files — answer keys; the learner opens them only after attempting. |
| **Fallback path** | A timm/torchvision pretrained model in ~30 lines, for when Ultralytics or its weights are unavailable. Drilled from week 4. |

## Layout

```
cv-exam-course/
  CONTEXT.md                 # this file
  AGENTS.md                  # agent config (issue tracker, labels, domain docs)
  README.md                  # Mac setup + weekly rhythm
  grader/                    # scores submission.csv: schema check → metrics → logs/progress.md
  data_prep/                 # Kaggle download + private split (train/, anonymised test/, hidden labels)
  weeks/weekNN/
    notes.md                 # Mon–Tue concept
    build/                   # Wed–Thu understanding-track exercises (from week 2)
    drills/NN_name/          # prompt.md + reference.py
    mock/                    # README.md (brief), schema.json, reference.py
    practice/                # the learner's code — committed
  logs/progress.md           # one row per graded mock
  tests/                     # pytest for grader, data_prep and weekly references
  docs/superpowers/          # specs/ (design) and plans/ (build plans)
  docs/agents/               # agent-skill conventions
  # git-ignored, generated:
  data/                      # raw downloads, splits, data/_hidden/ labels
  work/                      # image copies (e.g. train/val split for Ultralytics)
  runs/                      # Ultralytics training output
  *.pt, submission*.csv
```

## Conventions

- Run scripts from the repo root so `data/...` paths resolve and `runs/` lands in the ignored root folder.
- `data_prep/` is not called `datasets/`, which would shadow Hugging Face's `datasets` library.
- Code never hard-codes a `runs/.../trainN/` path; it uses `model.trainer.save_dir`.
- Ultralytics uses the M1 GPU only with `device="mps"`; practise with `device="cpu"` too, as the test machine may lack a GPU.
