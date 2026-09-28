from pathlib import Path

from data_prep.prepare_intel import CLASSES
from grader.schema import load_schema

MOCK = Path(__file__).resolve().parent.parent / "weeks" / "week01" / "mock"


def test_week01_schema_matches_dataset_and_spec():
    s = load_schema(MOCK / "schema.json")
    assert s.classes == CLASSES
    assert (s.id_column, s.label_column, s.label_type, s.id_has_extension) == ("image_id", "label", "name", False)
    assert s.time_limit_minutes == 90


def test_week01_brief_states_the_schema():
    brief = (MOCK / "README.md").read_text()
    assert "image_id,label" in brief
    assert "WITHOUT" in brief and "90 minutes" in brief


def test_week01_brief_explains_how_to_set_target():
    brief = (MOCK / "README.md").read_text()
    assert "target_accuracy" in brief and "0.03" in brief


def test_readme_and_brief_target_mac_setup():
    readme = (MOCK.parent.parent.parent / "README.md").read_text()
    brief = (MOCK / "README.md").read_text()
    assert "## Setup (Mac" in readme and "venv" in readme
    assert "GPU Studio" not in brief
