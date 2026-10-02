from __future__ import annotations

import pytest

from treelace.compiler.compile import compile_workflow
from treelace.ir.nodes import ToolCall, Workflow
from treelace.runtime.engine import Engine
from treelace.sessions.artifacts import InMemoryArtifactStore
from treelace.tools.context import ToolContext
from treelace.tools.registry import ToolRegistry
from treelace.tools.result import ToolResult


def build_dataframe(property_name: str, *, context: ToolContext) -> ToolResult:
    """Build a dataframe-like payload.

    Args:
        property_name: Property name.
    """

    rows = [{"property": property_name, "occupancy": 0.91}]
    summary = {"property": property_name, "rows": len(rows)}
    return ToolResult(
        model_context=summary,
        artifact_refs=[context.store_artifact(rows, kind="table")],
    )


@pytest.mark.asyncio
async def test_artifact_refs_are_stored_and_retrievable() -> None:
    tools = ToolRegistry()
    tools.register_callable(build_dataframe)
    artifacts = InMemoryArtifactStore()
    engine = Engine(tools=tools, artifact_store=artifacts)
    workflow = Workflow(
        id="artifact-flow",
        root=ToolCall(id="build", tool="build_dataframe", arguments={"property_name": "X"}),
    )
    compiled = compile_workflow(workflow, tools=tools)
    result = await engine.run(compiled, inputs={})
    assert result.status == "succeeded"
    assert isinstance(result.output, dict)
    refs = result.output["artifact_refs"]
    assert len(refs) == 1
    stored = artifacts.get(session_id=result.session_id, artifact_id=refs[0])
    assert stored == [{"property": "X", "occupancy": 0.91}]
