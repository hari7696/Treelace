from __future__ import annotations

import asyncio

import pytest

from treelace.models.fake import FakeModel
from treelace.models.protocol import ModelResponse, StructuredOutput
from treelace.planner.models import ChatPlan, PlanSpec
from treelace.planner.prompts import ConversationTurn
from treelace.runner import Runner
from treelace.runtime.engine import Engine
from treelace.sessions.artifacts import InMemoryArtifactStore
from treelace.sessions.memory import InMemorySessionStore
from treelace.tools.context import ToolContext
from treelace.tools.result import ClarificationRequest, ToolResult


async def get_occupancy(
    property_name: str,
    period: str,
    *,
    context: ToolContext,
) -> ToolResult:
    """Get occupancy and store rows as artifacts.

    Args:
        property_name: Property identifier.
        period: Month period.
    """

    await asyncio.sleep(0.03)
    rows = [
        {"property": property_name, "period": period, "occupancy": 0.82},
        {"property": property_name, "period": period, "occupancy": 0.87},
    ]
    summary = {
        "property": property_name,
        "period": period,
        "avg_occupancy": sum(item["occupancy"] for item in rows) / len(rows),
        "micro_prompt": context.micro_prompt or "",
    }
    return ToolResult(
        model_context=summary,
        artifact_refs=[context.store_artifact(rows, kind="table")],
    )


async def get_forecast(
    property_name: str,
    period: str,
    *,
    context: ToolContext,
) -> ToolResult:
    """Get forecast rows.

    Args:
        property_name: Property identifier.
        period: Month period.
    """

    await asyncio.sleep(0.03)
    rows = [{"property": property_name, "period": period, "forecast_occupancy": 0.9}]
    return ToolResult(
        model_context={"property": property_name, "period": period, "forecast_occupancy": 0.9},
        artifact_refs=[context.store_artifact(rows, kind="table")],
    )


def merge_tool(
    current: dict[str, object],
    last: dict[str, object],
    forecast: dict[str, object],
) -> ToolResult:
    """Merge all summaries into one response.

    Args:
        current: Current period summary.
        last: Historical period summary.
        forecast: Forecast summary.
    """

    delta = float(current["avg_occupancy"]) - float(last["avg_occupancy"])
    return ToolResult(
        model_context={
            "comparison": {
                "current_avg": current["avg_occupancy"],
                "last_avg": last["avg_occupancy"],
                "delta": round(delta, 4),
                "forecast_next": forecast["forecast_occupancy"],
            }
        }
    )


def ask_then_continue() -> ToolResult:
    return ToolResult(
        model_context={"status": "waiting"},
        clarification_request=ClarificationRequest(prompt="Proceed?", options=["yes", "no"]),
    )


def finalize(step_result: dict[str, object]) -> ToolResult:
    return ToolResult(model_context={"decision": step_result["decision"]})


@pytest.mark.asyncio
async def test_runner_chat_plan_does_not_execute_tools() -> None:
    planner_payload = {
        "kind": "chat",
        "question": "hi",
        "response": "Hello there.",
    }
    r = Runner(model=FakeModel.fixed_structured(planner_payload), tools=[get_occupancy, get_forecast, merge_tool])
    result = await r.run(question="hi", inputs={})
    assert result.status == "succeeded"
    assert isinstance(result.plan, ChatPlan)
    assert result.output == {"message": "Hello there."}


@pytest.mark.asyncio
async def test_runner_executes_occupancy_flow_with_planner_generated_graph() -> None:
    planner_payload = {
        "kind": "execution",
        "question": "compare occupancy for property X and forecast next January",
        "intent_entities": {"property": "X", "month_current": "2026-01", "month_last": "2025-01"},
        "rationale": "Occupancy and forecast are independent until merge.",
        "steps": [
            {
                "id": "occ_this",
                "tool": "get_occupancy",
                "arguments": {"property_name": "X", "period": "2026-01"},
                "prompt_focus": "Current January occupancy.",
            },
            {
                "id": "occ_last",
                "tool": "get_occupancy",
                "arguments": {"property_name": "X", "period": "2025-01"},
                "prompt_focus": "Last year January occupancy.",
            },
            {
                "id": "forecast_next",
                "tool": "get_forecast",
                "arguments": {"property_name": "X", "period": "2027-01"},
                "prompt_focus": "Next January forecast.",
            },
            {
                "id": "merge",
                "tool": "merge_tool",
                "deps": ["occ_this", "occ_last", "forecast_next"],
                "arguments": {
                    "current": {"kind": "output_ref", "node_id": "occ_this", "pointer": "/model_context"},
                    "last": {"kind": "output_ref", "node_id": "occ_last", "pointer": "/model_context"},
                    "forecast": {"kind": "output_ref", "node_id": "forecast_next", "pointer": "/model_context"},
                },
                "prompt_focus": "Merge summaries only.",
            },
        ],
        "result_step_id": "merge",
    }
    r = Runner(model=FakeModel.fixed_structured(planner_payload), tools=[get_occupancy, get_forecast, merge_tool])
    history = [
        ConversationTurn(role="user", content=f"history-{index}")
        for index in range(14)
    ]
    result = await r.run(
        question="compare occupancy",
        inputs={},
        history=history,
        session_id="session-1",
        workflow_id="occupancy-plan",
    )
    assert isinstance(result.plan, PlanSpec)
    assert result.plan.steps[0].micro_prompt is not None
    assert "Tool: get_occupancy" in result.plan.steps[0].micro_prompt
    assert result.status == "succeeded"
    assert result.output is not None
    comparison = result.output["model_context"]["comparison"]
    assert comparison["forecast_next"] == 0.9
    # Ensure intermediate tools produced artifacts and references were tracked.
    session = await r.engine.session_store.get("session-1")
    assert session is not None
    assert "occ_this" in session.node_artifacts
    assert "occ_last" in session.node_artifacts


@pytest.mark.asyncio
async def test_runner_can_resume_planner_clarification_across_instances() -> None:
    model = FakeModel(
        responses=[
            # First planner decision requires clarification.
            ModelResponse(
                output=StructuredOutput(
                    data={
                        "kind": "clarification",
                        "question": "Need more detail",
                        "clarification": {
                            "prompt": "Which number should I echo?",
                            "missing_fields": ["value"],
                        },
                    }
                )
            ),
            # Second planner decision is executable.
            ModelResponse(
                output=StructuredOutput(
                    data={
                        "kind": "execution",
                        "question": "Need more detail",
                        "steps": [
                            {
                                "id": "echo",
                                "tool": "finalize",
                                "arguments": {"step_result": {"decision": "yes"}},
                            }
                        ],
                        "result_step_id": "echo",
                    }
                )
            ),
        ]
    )
    shared_engine = Engine(
        session_store=InMemorySessionStore(),
        artifact_store=InMemoryArtifactStore(),
    )
    first_runner = Runner(model=model, tools=[finalize], engine=shared_engine)
    first = await first_runner.run(question="help", inputs={}, session_id="planner-session")
    assert first.status == "waiting"
    assert first.pending_clarification is not None
    assert first.pending_clarification.source == "planner"

    second_runner = Runner(model=model, tools=[finalize], engine=shared_engine)
    resumed = await second_runner.resume(
        session_id=first.session_id,
        token=first.pending_clarification.token,
        decision="yes",
    )
    assert resumed.status == "succeeded"
    assert resumed.output is not None


@pytest.mark.asyncio
async def test_runner_can_resume_tool_clarification_across_instances() -> None:
    planner_payload = {
        "kind": "execution",
        "question": "confirm action",
        "steps": [
            {"id": "ask", "tool": "ask_then_continue"},
            {
                "id": "final",
                "tool": "finalize",
                "deps": ["ask"],
                "arguments": {"step_result": {"kind": "output_ref", "node_id": "ask", "pointer": "/model_context"}},
            },
        ],
        "result_step_id": "final",
    }
    model = FakeModel.fixed_structured(planner_payload)
    shared_engine = Engine(
        session_store=InMemorySessionStore(),
        artifact_store=InMemoryArtifactStore(),
    )
    first_runner = Runner(model=model, tools=[ask_then_continue, finalize], engine=shared_engine)
    first = await first_runner.run(question="confirm", inputs={}, session_id="tool-session")
    assert first.status == "waiting"
    assert first.pending_clarification is not None
    assert first.pending_clarification.source == "tool"

    second_runner = Runner(model=model, tools=[ask_then_continue, finalize], engine=shared_engine)
    resumed = await second_runner.resume(
        session_id=first.session_id,
        token=first.pending_clarification.token,
        decision="yes",
    )
    assert resumed.status == "succeeded"
    assert resumed.output is not None
    assert resumed.output["model_context"]["decision"] == "yes"
