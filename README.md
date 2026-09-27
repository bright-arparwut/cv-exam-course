# CV Exam Course

A 12-week course to prepare for an image-classification coding test (no docs, no LLM, CSV submission) and a system design interview. Design: `docs/superpowers/specs/2026-09-27-cv-exam-course-design.md`.

## Setup (Lightning AI Studio)
```bash
git clone https://github.com/bright-arparwut/cv-exam-course.git && cd cv-exam-course
pip install -r requirements.txt
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json   # from kaggle.com → Settings → API
pytest                                                                             # checks the tooling
```

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
