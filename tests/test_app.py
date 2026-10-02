from __future__ import annotations

import asyncio

from treelace import Treelace, tool
from treelace.models.fake import FakeModel
from treelace.models.protocol import ModelResponse, StructuredOutput, TextOutput
from treelace.tools.result import ClarificationRequest, ToolResult


@tool
def add(left: int, right: int) -> dict[str, int]:
    """Add two integers."""

    return {"value": left + right}


@tool
def ask_user() -> ToolResult:
    """Ask for approval before continuing."""

    return ToolResult(
        model_context={"status": "waiting"},
        clarification_request=ClarificationRequest(prompt="Continue?", options=["yes", "no"]),
    )


@tool
def finalize(decision: dict[str, object]) -> dict[str, str]:
    """Persist clarification decision."""

    return {"decision": str(decision["decision"])}


@tool
def always_fail() -> dict[str, str]:
    """Raise an execution error."""

    raise RuntimeError("boom")


@tool
def lucky_number() -> int:
    """Return a scalar payload to test automatic result wrapping."""

    return 7


def _execution_plan_payload() -> dict[str, object]:
    return {
        "kind": "execution",
        "question": "Add 2 and 3",
        "steps": [
            {"id": "sum", "tool": "add", "arguments": {"left": 2, "right": 3}},
        ],
        "result_step_id": "sum",
    }


def test_treelace_returns_answer_and_data() -> None:
    model = FakeModel(
        responses=[
            ModelResponse(output=StructuredOutput(data=_execution_plan_payload())),
            ModelResponse(output=TextOutput(text="The sum is 5.")),
        ]
    )
    app = Treelace(model=model, tools=[add], instructions="Answer in one sentence.")
    run = app.run("Add 2 and 3")
    assert run.status == "succeeded"
    assert run.answer == "The sum is 5."
    assert run.data == {"value": 5}
    assert run.plan_text is not None
    assert "Plan" in run.plan_text


def test_treelace_raw_mode_skips_response_synthesis() -> None:
    model = FakeModel(responses=[ModelResponse(output=StructuredOutput(data=_execution_plan_payload()))])
    app = Treelace(model=model, tools=[add])
    run = app.run("Add 2 and 3", respond=False)
    assert run.status == "succeeded"
    assert run.answer is None
    assert run.data == {"value": 5}


def test_treelace_resume_uses_waiting_run_object() -> None:
    planner_payload = {
        "kind": "execution",
        "question": "confirm",
        "steps": [
            {"id": "ask", "tool": "ask_user"},
            {
                "id": "final",
                "tool": "finalize",
                "deps": ["ask"],
                "arguments": {
                    "decision": {"kind": "output_ref", "node_id": "ask", "pointer": "/model_context"}
                },
            },
        ],
        "result_step_id": "final",
    }
    model = FakeModel(responses=[ModelResponse(output=StructuredOutput(data=planner_payload))])
    app = Treelace(model=model, tools=[ask_user, finalize])

    first = app.run("confirm action", respond=False)
    assert first.status == "waiting"
    assert first.needs_input is not None
    assert first.prompt == "Continue?"

    resumed = app.resume(first, "yes", respond=False)
    assert resumed.status == "succeeded"
    assert resumed.data == {"decision": "yes"}


def test_treelace_surfaces_failed_execution_without_synthesis() -> None:
    planner_payload = {
        "kind": "execution",
        "question": "run failing tool",
        "steps": [{"id": "boom", "tool": "always_fail"}],
        "result_step_id": "boom",
    }
    model = FakeModel(responses=[ModelResponse(output=StructuredOutput(data=planner_payload))])
    app = Treelace(model=model, tools=[always_fail])
    run = app.run("trigger failure")
    assert run.status == "failed"
    assert run.error is not None
    assert "boom" in run.error
    assert run.answer is None


def test_treelace_wraps_scalar_tool_returns() -> None:
    planner_payload = {
        "kind": "execution",
        "question": "what is your lucky number?",
        "steps": [{"id": "number", "tool": "lucky_number"}],
        "result_step_id": "number",
    }
    model = FakeModel(responses=[ModelResponse(output=StructuredOutput(data=planner_payload))])
    app = Treelace(model=model, tools=[lucky_number])
    run = app.run("what is your lucky number?", respond=False)
    assert run.status == "succeeded"
    assert run.data == {"result": 7}


def test_treelace_sync_and_async_paths_match_for_raw_runs() -> None:
    payload = _execution_plan_payload()
    sync_model = FakeModel(responses=[ModelResponse(output=StructuredOutput(data=payload))])
    async_model = FakeModel(responses=[ModelResponse(output=StructuredOutput(data=payload))])

    sync_app = Treelace(model=sync_model, tools=[add])
    sync_run = sync_app.run("Add 2 and 3", respond=False)

    async_app = Treelace(model=async_model, tools=[add])

    async def _execute() -> object:
        return await async_app.arun("Add 2 and 3", respond=False)

    async_run = asyncio.run(_execute())
    assert isinstance(async_run, type(sync_run))
    assert async_run.status == sync_run.status
    assert async_run.data == sync_run.data
    assert async_run.plan_text == sync_run.plan_text
