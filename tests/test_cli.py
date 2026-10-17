from __future__ import annotations

import json
from pathlib import Path

from treelace.cli import main


def _workflow_payload() -> dict[str, object]:
    return {
        "id": "cli-workflow",
        "result_node_id": "ask",
        "root": {
            "kind": "clarification",
            "id": "ask",
            "prompt": "Need more detail",
            "options": ["ok"],
            "missing_fields": [],
            "depends_on": [],
        },
    }


def test_cli_validate_and_inspect_and_render(tmp_path: Path) -> None:
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(_workflow_payload()), encoding="utf-8")
    output_mermaid = tmp_path / "plan.mmd"

    assert main(["validate", str(path)]) == 0
    assert main(["inspect", str(path)]) == 0
    assert main(["render", str(path), "-o", str(output_mermaid)]) == 0
    assert output_mermaid.exists()


def test_cli_schema_targets() -> None:
    assert main(["schema", "--target", "workflow"]) == 0
    assert main(["schema", "--target", "plan_decision"]) == 0
