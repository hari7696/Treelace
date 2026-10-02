from __future__ import annotations

import asyncio
from time import monotonic

import pytest

from treelace.compiler.compile import compile_workflow
from treelace.ir.expressions import OutputRef
from treelace.ir.nodes import Parallel, Sequence, ToolCall, Workflow
from treelace.runtime.engine import Engine
from treelace.tools.registry import ToolRegistry
from treelace.tools.result import ClarificationRequest, ToolResult


async def delayed_value(name: str, delay: float) -> ToolResult:
    """Return a delayed value.

    Args:
        name: Value name.
        delay: Delay in seconds.
    """

    await asyncio.sleep(delay)
    return ToolResult(model_context={"name": name})


def merge_payloads(left: dict[str, object], right: dict[str, object]) -> ToolResult:
    """Merge two payloads.

    Args:
        left: Left payload.
        right: Right payload.
    """

    return ToolResult(model_context={"left": left["name"], "right": right["name"]})


def failing_tool() -> ToolResult:
    """Always fail."""

    raise RuntimeError("boom")


def needs_user_confirmation() -> ToolResult:
    """Pause and ask for clarification."""

    return ToolResult(
        model_context={"status": "waiting"},
        clarification_request=ClarificationRequest(
            prompt="Continue execution?",
            options=["yes", "no"],
            missing_fields=["decision"],
        ),
    )


def final_decision(step_result: dict[str, object]) -> ToolResult:
    """Finalize using clarification decision.

    Args:
        step_result: Prior step output.
    """

    return ToolResult(model_context={"decision": step_result["decision"]})


def ask_a() -> ToolResult:
    return ToolResult(
        model_context={"status": "waiting"},
        clarification_request=ClarificationRequest(prompt="answer a?", options=["yes", "no"]),
    )


def ask_b() -> ToolResult:
    return ToolResult(
        model_context={"status": "waiting"},
        clarification_request=ClarificationRequest(prompt="answer b?", options=["yes", "no"]),
    )


@pytest.mark.asyncio
async def test_dependency_ready_nodes_run_in_parallel() -> None:
    tools = ToolRegistry()
    tools.register_callable(delayed_value)
    tools.register_callable(merge_payloads)
    workflow = Workflow(
        id="parallel-runtime",
        root=Parallel(
            id="root",
            branches=[
                ToolCall(id="left", tool="delayed_value", arguments={"name": "L", "delay": 0.05}),
                ToolCall(id="right", tool="delayed_value", arguments={"name": "R", "delay": 0.05}, depends_on=[]),
                ToolCall(
                    id="join",
                    tool="merge_payloads",
                    arguments={
                        "left": OutputRef(node_id="left", pointer="/model_context"),
                        "right": OutputRef(node_id="right", pointer="/model_context"),
                    },
                    depends_on=["left", "right"],
                ),
            ],
        ),
    )
    compiled = compile_workflow(workflow, tools=tools)
    engine = Engine(tools=tools)
    started_at = monotonic()
    result = await engine.run(compiled, inputs={})
    elapsed = monotonic() - started_at
    assert result.status == "succeeded"
    assert elapsed < 0.10
    assert result.output["model_context"]["left"] == "L"
    assert result.output["model_context"]["right"] == "R"


@pytest.mark.asyncio
async def test_fail_fast_stops_downstream_nodes() -> None:
    tools = ToolRegistry()
    tools.register_callable(failing_tool)
    tools.register_callable(delayed_value)
    workflow = Workflow(
        id="fail-fast",
        root=Sequence(
            id="root",
            steps=[
                ToolCall(id="bad", tool="failing_tool"),
                ToolCall(id="never", tool="delayed_value", arguments={"name": "X", "delay": 0.01}),
            ],
        ),
    )
    compiled = compile_workflow(workflow, tools=tools)
    result = await Engine(tools=tools).run(compiled, inputs={})
    assert result.status == "failed"
    started = [event.node_id for event in result.events if event.kind == "node_started"]
    assert "bad" in started
    assert "never" not in started


@pytest.mark.asyncio
async def test_clarification_pause_and_resume() -> None:
    tools = ToolRegistry()
    tools.register_callable(needs_user_confirmation)
    tools.register_callable(final_decision)
    workflow = Workflow(
        id="clarify",
        root=Sequence(
            id="root",
            steps=[
                ToolCall(id="ask", tool="needs_user_confirmation"),
                ToolCall(
                    id="final",
                    tool="final_decision",
                    arguments={"step_result": OutputRef(node_id="ask", pointer="/model_context")},
                ),
            ],
        ),
    )
    compiled = compile_workflow(workflow, tools=tools)
    engine = Engine(tools=tools)
    first = await engine.run(compiled, inputs={})
    assert first.status == "waiting"
    assert first.pending_clarification is not None
    assert first.pending_clarification.source == "tool"
    resumed = await engine.resume(
        compiled,
        session_id=first.session_id,
        token=first.pending_clarification.token,
        decision="yes",
    )
    assert resumed.status == "succeeded"
    assert resumed.output["model_context"]["decision"] == "yes"


@pytest.mark.asyncio
async def test_multiple_clarifications_are_queued_deterministically() -> None:
    tools = ToolRegistry()
    tools.register_callable(ask_b)
    tools.register_callable(ask_a)
    workflow = Workflow(
        id="many-clarifications",
        root=Parallel(
            id="root",
            branches=[
                ToolCall(id="b", tool="ask_b"),
                ToolCall(id="a", tool="ask_a"),
            ],
        ),
        result_node_id="b",
    )
    compiled = compile_workflow(workflow, tools=tools)
    engine = Engine(tools=tools)
    first = await engine.run(compiled, inputs={})
    assert first.status == "waiting"
    assert first.pending_clarification is not None
    assert first.pending_clarification.node_id == "a"

    second = await engine.resume(
        compiled,
        session_id=first.session_id,
        token=first.pending_clarification.token,
        decision="yes",
    )
    assert second.status == "waiting"
    assert second.pending_clarification is not None
    assert second.pending_clarification.node_id == "b"
