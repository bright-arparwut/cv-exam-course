import importlib.util
import py_compile
from pathlib import Path

PRACTICE = Path(__file__).resolve().parent.parent / "weeks" / "week01" / "practice"
DAY_FILES = ["day1_train.py", "day2_predict.py", "day3_4_experiments.py",
             "day5_drills.py", "day6_mock.py", "day7_review.py"]


def load(name):
    spec = importlib.util.spec_from_file_location(name[:-3], PRACTICE / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_day_file_exists_and_compiles():
    for name in DAY_FILES + ["_runner.py"]:
        py_compile.compile(str(PRACTICE / name), doraise=True)


def test_importing_a_day_file_runs_nothing(monkeypatch):
    monkeypatch.chdir(PRACTICE)
    mod = load("day1_train.py")  # would raise / train if steps ran on import
    assert len(mod.STEPS) >= 4


def test_runner_stops_at_first_todo_and_supports_resume(monkeypatch, capsys):
    monkeypatch.chdir(PRACTICE)
    runner = load("_runner.py")
    called = []

    def a(ctx):
        """first"""
        called.append("a")

    def b(ctx):
        """second"""
        raise NotImplementedError("step 2")

    def c(ctx):
        """third"""
        called.append("c")

    runner.run([a, b, c], argv=[])
    assert called == ["a"]
    assert "TODO" in capsys.readouterr().out
    called.clear()
    runner.run([a, b, c], argv=["3"])
    assert called == ["c"]
