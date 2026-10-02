from __future__ import annotations

import pytest

from treelace.ir.nodes import Parallel
from treelace.models.fake import FakeModel
from treelace.planner.inspect import execution_plan_text
from treelace.planner.models import ChatPlan, ClarificationPlan, PlanSpec
from treelace.planner.orchestrator import PlannerOrchestrator
from treelace.planner.prompts import ConversationTurn, MasterPromptManager
from treelace.tools.registry import ToolRegistry
from treelace.tools.result import ToolResult


def get_occupancy(property_name: str, period: str) -> ToolResult:
    """Get occupancy summary.

    Args:
        property_name: Property identifier.
        period: Month period.
    """

    return ToolResult(model_context={"property": property_name, "period": period})


def get_forecast(period: str) -> ToolResult:
    """Get forecast occupancy summary for a month."""

    return ToolResult(model_context={"period": period, "forecast_occupancy": 0.85})


def get_occupancy_with_parameters_style(period: str, property_name: str) -> ToolResult:
    """Get occupancy summary from Parameters section.

    Parameters:
    - period: str - Month in YYYY-MM format.
    - property_name: str - Property identifier.
    """

    return ToolResult(model_context={"period": period, "property": property_name})


def merge_tool(runs: dict[str, object]) -> ToolResult:
    """Merge step outputs."""

    return ToolResult(model_context={"runs": runs})


def test_tool_spec_uses_concise_docstring_summary() -> None:
    tools = ToolRegistry()
    definition = tools.register_callable(get_occupancy)

    assert definition.spec.description == "Get occupancy summary."


def test_tool_spec_parses_parameters_style_argument_descriptions() -> None:
    tools = ToolRegistry()
    definition = tools.register_callable(get_occupancy_with_parameters_style)
    properties = definition.spec.input_schema.get("properties")
    assert isinstance(properties, dict)
    period = properties.get("period")
    property_name = properties.get("property_name")
    assert isinstance(period, dict)
    assert isinstance(property_name, dict)
    assert period.get("description") == "Month in YYYY-MM format."
    assert property_name.get("description") == "Property identifier."


def test_execution_plan_text_uses_compact_terminal_friendly_layout() -> None:
    plan = PlanSpec.model_validate(
        {
            "kind": "execution",
            "question": "compare january forecast of 2025 and 2026",
            "steps": [
                {
                    "id": "forecast_2025_01",
                    "tool": "get_forecast",
                    "arguments": {"period": "2025-01"},
                },
                {
                    "id": "forecast_2026_01",
                    "tool": "get_forecast",
                    "arguments": {"period": "2026-01"},
                },
                {
                    "id": "merge_comparison",
                    "tool": "merge_tool",
                    "arguments": {},
                    "deps": ["forecast_2025_01", "forecast_2026_01"],
                },
            ],
            "result_step_id": "merge_comparison",
        }
    )

    rendered = execution_plan_text(plan)
    assert "Plan" in rendered
    assert "Wave 1 (parallel): forecast_2025_01, forecast_2026_01" in rendered
    assert "Wave 2: merge_comparison" in rendered
    assert "- merge_comparison(forecast_2025_01, forecast_2026_01)" in rendered
    assert "Result: merge_comparison" in rendered


def test_master_prompt_uses_tool_catalog_and_last_12_messages() -> None:
    tools = ToolRegistry()
    tools.register_callable(get_occupancy)
    history = [
        ConversationTurn(role="user", content=f"message-{index}")
        for index in range(15)
    ]
    manager = MasterPromptManager(master_prompt="Domain rule: property identifiers are case-sensitive.")
    prompt = manager.build_master_prompt(history=history, tools=tools)
    assert "Planner base policy:" in prompt
    assert "Application master prompt:" in prompt
    assert "Domain rule: property identifiers are case-sensitive." in prompt
    assert '"id": "get_occupancy"' in prompt
    assert "message-0" not in prompt
    assert "user: message-2" not in prompt
    assert "user: message-3" in prompt
    assert "message-14" in prompt


@pytest.mark.asyncio
async def test_planner_adds_micro_prompts() -> None:
    tools = ToolRegistry()
    tools.register_callable(get_occupancy)
    payload = {
        "kind": "execution",
        "question": "Compare occupancy",
        "steps": [
            {
                "id": "occ_this",
                "tool": "get_occupancy",
                "arguments": {"property_name": "X", "period": "2026-01"},
                "prompt_focus": "Current month occupancy",
            }
        ],
        "result_step_id": "occ_this",
    }
    planner = PlannerOrchestrator(model=FakeModel.fixed_structured(payload))
    plan = await planner.create_plan(question="Compare occupancy", history=[], tools=tools)
    assert isinstance(plan, PlanSpec)
    assert len(plan.steps) == 1
    assert plan.steps[0].micro_prompt is not None
    assert "Tool: get_occupancy" in plan.steps[0].micro_prompt


@pytest.mark.asyncio
async def test_planner_repairs_common_execution_aliases_once() -> None:
    tools = ToolRegistry()
    tools.register_callable(get_occupancy)
    payload = {
        "question": "Compare occupancy",
        "plan": [
            {
                "id": "occ_this",
                "tool": "get_occupancy",
                "arguments": {"property_name": "X", "period": "2026-01"},
                "depends_on": [],
            },
            {
                "id": "merge",
                "tool": "merge_tool",
                "args": {"runs": ["occ_this"]},
                "depends_on": ["occ_this"],
            },
        ],
        "final_step": "merge",
    }
    planner = PlannerOrchestrator(model=FakeModel.fixed_structured(payload))
    plan = await planner.create_plan(question="Compare occupancy", history=[], tools=tools)
    assert isinstance(plan, PlanSpec)
    assert [step.id for step in plan.steps] == ["occ_this", "merge"]
    assert plan.steps[1].deps == ["occ_this"]
    assert plan.result_step_id == "merge"


@pytest.mark.asyncio
async def test_planner_repairs_tool_calls_with_tool_name_as_id() -> None:
    tools = ToolRegistry()
    tools.register_callable(get_occupancy)
    payload = {
        "kind": "execution",
        "tool_calls": [
            {
                "id": "get_occupancy",
                "arguments": {"property_name": "X", "period": "2025-01"},
            }
        ],
    }
    planner = PlannerOrchestrator(model=FakeModel.fixed_structured(payload))
    plan = await planner.create_plan(
        question="Give me occupancy for Jan 2025",
        history=[],
        tools=tools,
    )
    assert isinstance(plan, PlanSpec)
    assert plan.question == "Give me occupancy for Jan 2025"
    assert plan.steps[0].id == "get_occupancy"
    assert plan.steps[0].tool == "get_occupancy"
    assert plan.result_step_id == "get_occupancy"


@pytest.mark.asyncio
async def test_planner_repairs_step_without_id() -> None:
    tools = ToolRegistry()
    tools.register_callable(get_occupancy)
    payload = {
        "kind": "execution",
        "steps": [
            {
                "tool": "get_occupancy",
                "arguments": {"property_name": "X", "period": "2025-01"},
            }
        ],
    }
    planner = PlannerOrchestrator(model=FakeModel.fixed_structured(payload))
    plan = await planner.create_plan(
        question="Give me occupancy for Jan 2025",
        history=[],
        tools=tools,
    )
    assert isinstance(plan, PlanSpec)
    assert plan.steps[0].id == "get_occupancy"
    assert plan.steps[0].tool == "get_occupancy"
    assert plan.result_step_id == "get_occupancy"


@pytest.mark.asyncio
async def test_planner_repairs_step_refs_into_output_refs_and_deps() -> None:
    tools = ToolRegistry()
    tools.register_many([get_forecast, merge_tool])
    payload = {
        "kind": "execution",
        "steps": [
            {
                "id": "forecast_2025_01",
                "tool": "get_forecast",
                "arguments": {"period": "2025-01"},
            },
            {
                "id": "forecast_2026_01",
                "tool": "get_forecast",
                "arguments": {"period": "2026-01"},
            },
            {
                "id": "compare_forecasts",
                "tool": "merge_tool",
                "arguments": {
                    "runs": {
                        "january_2025": {"$step": "forecast_2025_01"},
                        "january_2026": {"$step": "forecast_2026_01"},
                    }
                },
            },
        ],
    }
    planner = PlannerOrchestrator(model=FakeModel.fixed_structured(payload))
    plan = await planner.create_plan(
        question="compare january forecast of 2025 and 2026",
        history=[],
        tools=tools,
    )
    assert isinstance(plan, PlanSpec)
    assert plan.result_step_id == "compare_forecasts"
    merge_step = plan.steps[2]
    assert set(merge_step.deps) == {"forecast_2025_01", "forecast_2026_01"}
    jan_2025 = merge_step.arguments["runs"]["january_2025"]
    jan_2026 = merge_step.arguments["runs"]["january_2026"]
    assert isinstance(jan_2025, dict)
    assert isinstance(jan_2026, dict)
    assert jan_2025["kind"] == "output_ref"
    assert jan_2025["node_id"] == "forecast_2025_01"
    assert jan_2026["kind"] == "output_ref"
    assert jan_2026["node_id"] == "forecast_2026_01"


@pytest.mark.asyncio
async def test_planner_repairs_dollar_step_shorthand_into_output_refs_and_deps() -> None:
    tools = ToolRegistry()
    tools.register_many([get_forecast, merge_tool])
    payload = {
        "kind": "execution",
        "steps": [
            {
                "id": "forecast_2025_01",
                "tool": "get_forecast",
                "arguments": {"period": "2025-01"},
            },
            {
                "id": "forecast_2026_01",
                "tool": "get_forecast",
                "arguments": {"period": "2026-01"},
            },
            {
                "id": "compare_forecasts",
                "tool": "merge_tool",
                "arguments": {
                    "runs": {
                        "january_2025": "$forecast_2025_01",
                        "january_2026": "$forecast_2026_01",
                    }
                },
            },
        ],
    }
    planner = PlannerOrchestrator(model=FakeModel.fixed_structured(payload))
    plan = await planner.create_plan(
        question="compare january forecast of 2025 and 2026",
        history=[],
        tools=tools,
    )
    assert isinstance(plan, PlanSpec)
    assert plan.result_step_id == "compare_forecasts"
    merge_step = plan.steps[2]
    assert set(merge_step.deps) == {"forecast_2025_01", "forecast_2026_01"}
    jan_2025 = merge_step.arguments["runs"]["january_2025"]
    jan_2026 = merge_step.arguments["runs"]["january_2026"]
    assert isinstance(jan_2025, dict)
    assert isinstance(jan_2026, dict)
    assert jan_2025["kind"] == "output_ref"
    assert jan_2025["node_id"] == "forecast_2025_01"
    assert jan_2026["kind"] == "output_ref"
    assert jan_2026["node_id"] == "forecast_2026_01"


@pytest.mark.asyncio
async def test_planner_repairs_output_ref_alias_key_into_node_id() -> None:
    tools = ToolRegistry()
    tools.register_many([get_forecast, merge_tool])
    payload = {
        "kind": "execution",
        "steps": [
            {
                "id": "forecast_2025_01",
                "tool": "get_forecast",
                "arguments": {"period": "2025-01"},
            },
            {
                "id": "compare_forecasts",
                "tool": "merge_tool",
                "arguments": {
                    "runs": {
                        "january_2025": {
                            "kind": "output_ref",
                            "output_ref": "forecast_2025_01",
                        }
                    }
                },
            },
        ],
    }
    planner = PlannerOrchestrator(model=FakeModel.fixed_structured(payload))
    plan = await planner.create_plan(
        question="compare january forecast",
        history=[],
        tools=tools,
    )
    assert isinstance(plan, PlanSpec)
    merge_step = plan.steps[1]
    assert merge_step.deps == ["forecast_2025_01"]
    jan_2025 = merge_step.arguments["runs"]["january_2025"]
    assert isinstance(jan_2025, dict)
    assert jan_2025["kind"] == "output_ref"
    assert jan_2025["node_id"] == "forecast_2025_01"
    assert jan_2025["pointer"] == "/"


@pytest.mark.asyncio
async def test_planner_returns_explicit_clarification_plan() -> None:
    tools = ToolRegistry()
    tools.register_callable(get_occupancy)
    payload = {
        "kind": "clarification",
        "question": "hi",
        "clarification": {"prompt": "Please specify a period", "missing_fields": ["period"]},
    }
    planner = PlannerOrchestrator(model=FakeModel.fixed_structured(payload))
    plan = await planner.create_plan(question="hi", history=[], tools=tools)
    assert isinstance(plan, ClarificationPlan)
    assert plan.clarification.prompt == "Please specify a period"


@pytest.mark.asyncio
async def test_planner_returns_chat_plan_for_non_tool_conversation() -> None:
    tools = ToolRegistry()
    tools.register_callable(get_occupancy)
    payload = {
        "kind": "chat",
        "question": "hi",
        "response": "Hello! Tell me what occupancy question you want to run.",
    }
    planner = PlannerOrchestrator(model=FakeModel.fixed_structured(payload))
    plan = await planner.create_plan(question="hi", history=[], tools=tools)
    assert isinstance(plan, ChatPlan)
    assert "Hello!" in plan.response


@pytest.mark.asyncio
async def test_planner_repairs_chat_message_alias() -> None:
    tools = ToolRegistry()
    tools.register_callable(get_occupancy)
    payload = {
        "kind": "chat",
        "message": "Hi! How can I help?",
    }
    planner = PlannerOrchestrator(model=FakeModel.fixed_structured(payload))
    plan = await planner.create_plan(question="hi", history=[], tools=tools)
    assert isinstance(plan, ChatPlan)
    assert plan.question == "hi"
    assert plan.response == "Hi! How can I help?"


@pytest.mark.asyncio
async def test_planner_repairs_clarification_without_nested_object_to_chat() -> None:
    tools = ToolRegistry()
    tools.register_callable(get_occupancy)
    payload = {
        "kind": "clarification",
        "question": "Hi! What occupancy information would you like?",
    }
    planner = PlannerOrchestrator(model=FakeModel.fixed_structured(payload))
    plan = await planner.create_plan(question="hi", history=[], tools=tools)
    assert isinstance(plan, ChatPlan)
    assert "occupancy information" in plan.response


@pytest.mark.asyncio
async def test_plan_to_workflow_sets_explicit_result_node() -> None:
    tools = ToolRegistry()
    tools.register_callable(get_occupancy)
    payload = {
        "kind": "execution",
        "question": "Compare occupancy",
        "steps": [
            {
                "id": "occ_this",
                "tool": "get_occupancy",
                "arguments": {"property_name": "X", "period": "2026-01"},
            }
        ],
        "result_step_id": "occ_this",
    }
    planner = PlannerOrchestrator(model=FakeModel.fixed_structured(payload))
    plan = await planner.create_plan(question="Compare occupancy", history=[], tools=tools)
    assert isinstance(plan, PlanSpec)
    workflow = planner.plan_to_workflow(plan, workflow_id="planned")
    assert workflow.result_node_id == "occ_this"
    assert isinstance(workflow.root, Parallel)
