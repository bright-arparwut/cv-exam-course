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
