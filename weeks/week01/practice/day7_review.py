"""Week 1 - Day 7 (Sun): review. ~1 hour, no coding needed.

Run:  python weeks/week01/practice/day7_review.py   -> prints your mock log and this checklist

TODO list
  [ ] Look at your mock row in logs/progress.md: accuracy vs target, minutes vs 90
  [ ] List the 3 things that cost you the most time this week (write them below)
  [ ] Re-do any drill you failed on Friday (new tryN file) - no notes
  [ ] Update your "cheat sheet in your head": write from memory, in one screen, the full
      pipeline - split -> train -> load best -> predict -> CSV. Compare with the notes.
  [ ] Ask Claude for the week 2 plan when you're ready

What cost me the most time
  1.
  2.
  3.
"""
from pathlib import Path

if __name__ == "__main__":
    log = Path(__file__).resolve().parents[3] / "logs" / "progress.md"
    print(log.read_text() if log.exists() else "no logs/progress.md yet")
    print(__doc__)
