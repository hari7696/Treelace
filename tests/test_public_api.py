from __future__ import annotations

from pathlib import Path

import treelace


def test_root_api_is_small_and_app_focused() -> None:
    assert hasattr(treelace, "Treelace")
    assert hasattr(treelace, "Run")
    assert hasattr(treelace, "NeedsInput")
    assert hasattr(treelace, "tool")
    assert not hasattr(treelace, "Runner")
    assert not hasattr(treelace, "coding_tool")


def test_readme_references_existing_examples() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "examples/00_generic_math.py" in readme
    assert "examples/01_weather_lookups.py" in readme
    assert "examples/02_employee_questions.py" in readme
    assert "examples/03_weather_location_clarification.py" in readme
    assert "examples/02_advanced_sqlite_llm_tools.py" not in readme
    assert (root / "examples" / "00_generic_math.py").exists()
    assert (root / "examples" / "01_weather_lookups.py").exists()
    assert (root / "examples" / "02_employee_questions.py").exists()
    assert (root / "examples" / "03_weather_location_clarification.py").exists()


def test_docs_include_clean_room_provenance_note() -> None:
    root = Path(__file__).resolve().parents[1]
    note = root / "docs" / "clean-room-provenance.md"
    assert note.exists()
