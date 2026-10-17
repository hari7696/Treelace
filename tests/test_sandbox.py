from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from treelace.app import Treelace
from treelace.compiler.compile import compile_workflow
from treelace.ir.nodes import Parallel, ToolCall, Workflow
from treelace.models.fake import FakeModel
from treelace.models.protocol import (
    ModelContext,
    ModelRequest,
    ModelResponse,
    StructuredOutput,
)
from treelace.runner import Runner
from treelace.runtime.engine import Engine
from treelace.sandbox import SandboxError, execute_python, run_python_sandbox
from treelace.tools.context import ToolContext
from treelace.tools.registry import ToolRegistry

_OCCUPANCY_CODE = """
def process(records):
    occupancy = []
    forecast = None
    property_name = None
    for record in records:
        if property_name is None and record.get("property") is not None:
            property_name = record.get("property")
        if "forecast_occupancy" in record:
            forecast = record.get("forecast_occupancy")
            continue
        value = record.get("occupancy")
        if value is None:
            value = record.get("avg_occupancy")
        occupancy.append({"period": record.get("period"), "value": value})
    occupancy = sorted(occupancy, key=lambda item: item.get("period") or "")
    last_year = occupancy[0]["value"]
    current = occupancy[-1]["value"]
    return {
        "property": property_name,
        "comparison_periods": [occupancy[0]["period"], occupancy[-1]["period"]],
        "current_occupancy": current,
        "last_year_occupancy": last_year,
        "delta": round(current - last_year, 4),
        "next_year_forecast": forecast,
    }
"""


class ScriptedModel:
    def __init__(self, codes: list[str]) -> None:
        self.codes = list(codes)
        self.requests: list[ModelRequest] = []
        self.contexts: list[ModelContext] = []

    async def complete(self, request: ModelRequest, *, context: ModelContext) -> ModelResponse:
        self.requests.append(request)
        self.contexts.append(context)
        code = self.codes.pop(0)
        return ModelResponse(output=StructuredOutput(data={"code": code}))


def test_execute_python_merges_occupancy_records() -> None:
    records = [
        {"property": "Aurora Suites", "period": "2026-01", "occupancy": 0.86},
        {"property": "Aurora Suites", "period": "2025-01", "occupancy": 0.81},
        {"property": "Aurora Suites", "period": "2027-01", "forecast_occupancy": 0.9},
    ]
    result = execute_python(_OCCUPANCY_CODE, records)
    assert result == {
        "property": "Aurora Suites",
        "comparison_periods": ["2025-01", "2026-01"],
        "current_occupancy": 0.86,
        "last_year_occupancy": 0.81,
        "delta": 0.05,
        "next_year_forecast": 0.9,
    }
    assert records[0]["occupancy"] == 0.86


def test_execute_python_does_not_mutate_input_records() -> None:
    records = [{"a": 1}]
    result = execute_python(
        "def process(records):\n    records[0]['a'] = 2\n    return {'a': records[0]['a']}\n",
        records,
    )
    assert result == {"a": 2}
    assert records == [{"a": 1}]


def test_execute_python_rejects_imports_and_unsafe_names() -> None:
    with pytest.raises(SandboxError, match="imports are not allowed"):
        execute_python("import os\n\ndef process(records):\n    return {}\n", [])
    with pytest.raises(SandboxError, match="open"):
        execute_python("def process(records):\n    return {'x': open('secret')}\n", [])
    with pytest.raises(SandboxError, match="__class__"):
        execute_python("def process(records):\n    return {'x': records.__class__}\n", [])


def test_execute_python_stops_runaway_loops() -> None:
    code = "def process(records):\n    while True:\n        records = list(records)\n    return {}\n"
    with pytest.raises(SandboxError, match="exceeded 20 steps"):
        execute_python(code, [], max_steps=20)


def test_execute_python_requires_a_dict_result() -> None:
    with pytest.raises(SandboxError, match="must return a dict"):
        execute_python("def process(records):\n    return len(records)\n", [])


def test_execute_python_accepts_fenced_code() -> None:
    code = "```python\ndef process(records):\n    return {'n': len(records)}\n```"
    assert execute_python(code, [{}, {}]) == {"n": 2}


@pytest.mark.asyncio
async def test_run_python_sandbox_calls_model_with_records_and_context() -> None:
    model = ScriptedModel(
        ["def process(records):\n    return {'n': len(records), 'label': records[0].get('name')}\n"]
    )
    result = await run_python_sandbox(
        [{"name": "Aurora"}],
        task="Count the records and keep the first name.",
        model=model,
        context=ModelContext(node_id="merge", metadata={"tool": "merge_summary"}),
    )
    assert result == {"n": 1, "label": "Aurora"}
    assert len(model.requests) == 1
    request = model.requests[0]
    assert [item.role for item in request.messages] == ["system", "user"]
    assert "Count the records" in request.messages[1].content
    assert "Aurora" in request.messages[1].content
    assert model.contexts[0].node_id == "merge"
    assert model.contexts[0].metadata["tool"] == "merge_summary"
    assert model.contexts[0].metadata["source"] == "python_sandbox"


@pytest.mark.asyncio
async def test_run_python_sandbox_retries_with_an_independent_call() -> None:
    model = ScriptedModel(
        [
            "def process(records):\n    return records[0]['missing']\n",
            "def process(records):\n    return {'count': len(records)}\n",
        ]
    )
    result = await run_python_sandbox([{"a": 1}, {"a": 2}], task="Count records.", model=model)
    assert result == {"count": 2}
    assert len(model.requests) == 2
    retry = model.requests[1].messages[1].content
    assert "missing" in retry
    assert "Count records." in retry
    assert model.requests[0].messages[1].content != retry
    assert model.contexts[1].metadata["source"] == "python_sandbox"


@pytest.mark.asyncio
async def test_run_python_sandbox_accepts_text_json() -> None:
    code = "def process(records):\n    return {'n': len(records)}\n"
    model = FakeModel.fixed_text(json.dumps({"code": code}))
    result = await run_python_sandbox([{"a": 1}], task="Count records.", model=model)
    assert result == {"n": 1}


@pytest.mark.asyncio
async def test_engine_passes_model_and_runs_sandbox_on_record_list() -> None:
    code = "def process(records):\n    return {'count': len(records), 'first': records[0].get('name')}\n"
    model = FakeModel.fixed_structured({"code": code})

    def lookup(name: str) -> dict[str, object]:
        """Return one named record.

        Args:
            name: Record name.
        """

        return {"name": name}

    async def merge_records(
        records: list[dict[str, object]],
        *,
        context: ToolContext,
    ) -> dict[str, object]:
        """Merge records by asking the python sandbox to write code.

        Args:
            records: Upstream records.
        """

        assert context.model is model
        assert context.question == "compare occupancy"
        assert context.model is not None
        return await run_python_sandbox(
            records,
            task="Count the records and keep the first name.",
            model=context.model,
        )

    tools = ToolRegistry()
    tools.register_callable(lookup)
    tools.register_callable(merge_records)
    workflow = Workflow(
        id="sandbox-merge",
        root=Parallel(
            id="root",
            branches=[
                ToolCall(id="left", tool="lookup", arguments={"name": "L"}),
                ToolCall(id="right", tool="lookup", arguments={"name": "R"}),
                ToolCall(
                    id="merge",
                    tool="merge_records",
                    arguments={
                        "records": [
                            {"kind": "output_ref", "node_id": "left", "pointer": "/model_context"},
                            {"kind": "output_ref", "node_id": "right", "pointer": "/model_context"},
                        ]
                    },
                    depends_on=["left", "right"],
                ),
            ],
        ),
        result_node_id="merge",
    )
    compiled = compile_workflow(workflow, tools=tools)
    result = await Engine(tools=tools, model=model).run(
        compiled,
        inputs={},
        question="compare occupancy",
    )
    assert result.status == "succeeded"
    assert result.output is not None
    assert result.output["model_context"]["count"] == 2
    assert result.output["model_context"]["first"] == "L"


def test_runner_shares_model_with_tools() -> None:
    model = FakeModel.fixed_text("ok")
    runner = Runner(model=model, tools=[])
    assert runner.engine.model is model

    custom = Engine(model=FakeModel.fixed_text("tool"))
    attached = Runner(model=model, tools=[], engine=custom)
    assert attached.engine.model is custom.model

    empty = Engine()
    filled = Runner(model=model, tools=[], engine=empty)
    assert filled.engine.model is model


def test_treelace_exposes_the_model_used_for_tool_completions() -> None:
    model = FakeModel.fixed_text("ok")
    app = Treelace(model=model, tools=[])
    assert app.model is model


def test_example_merge_tool_accepts_a_record_list() -> None:
    root = Path(__file__).resolve().parents[1]
    examples = root / "examples"
    sys.path.insert(0, str(examples))
    try:
        spec = importlib.util.spec_from_file_location(
            "weather_lookups_example",
            examples / "01_weather_lookups.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(examples))

    definition = module.merge_summary.__treelace_tool__
    assert definition.is_async
    assert set(definition.input_schema["properties"]) == {"records"}
