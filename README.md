# CV Exam Course

A 12-week course to prepare for an image-classification coding test (no docs, no LLM, CSV submission) and a system design interview. Design: `docs/superpowers/specs/2026-09-27-cv-exam-course-design.md`.

## Setup (Mac, main learning device)
Needs Python 3.10+ (`python3 --version`; if it's older, `brew install python@3.12`).
```bash
git clone https://github.com/bright-arparwut/cv-exam-course.git && cd cv-exam-course
python3 -m venv .venv && source .venv/bin/activate      # run `source .venv/bin/activate` in every new terminal
pip install -r requirements.txt                          # torch on Apple Silicon includes MPS (M1 GPU) support
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json   # kaggle.com → Settings → API → Create New Token
pytest                                                   # checks the tooling
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"   # expect True
```

**Device:** Ultralytics runs on CPU on a Mac unless you pass `device="mps"`. The reference solutions pick MPS automatically. In your own code, pass `device="mps"` for speed, and practise with `device="cpu"` too, because the real test machine may have no GPU.

**Lightning AI (optional):** use a GPU Studio only when a later week's model is too slow on the M1 (e.g. ViT fine-tuning). The setup is the same, minus the venv.

## Weekly rhythm
| Day | Block | Where |
|---|---|---|
| Mon–Tue | Concept | `weeks/weekNN/notes.md` |
| Wed–Thu | Build | `weeks/weekNN/build/` |
| Fri | Drills | `weeks/weekNN/drills/` |
| Sat | Mock | `weeks/weekNN/mock/README.md` → `python -m grader.score ...` |
| Sun | Review | `logs/progress.md` |

## Layout
- `grader/`: scores a `submission.csv` (format check, then accuracy, macro-F1 and per-class recall) and logs it.
- `data_prep/`: downloads Kaggle datasets and makes private splits (hidden labels go in `data/_hidden/`).
- `weeks/`: notes, drills and mocks.
- `logs/progress.md`: one row per mock.
