"""Treelace example: parallel weather lookups merged by generated Python."""

from __future__ import annotations

import asyncio
import re

from _model_setup import resolve_model_spec

from treelace import ToolContext, Treelace, execution_plan_text, tool
from treelace.compiler import compile_workflow, dependency_text
from treelace.models.protocol import ModelContext
from treelace.planner import PlannerOrchestrator
from treelace.planner.models import PlanSpec
from treelace.runtime.engine import Engine
from treelace.sandbox import run_python_sandbox

HISTORICAL_TEMPERATURES: dict[tuple[str, str], float] = {
    ("Seattle", "2025-01"): 6.2,
    ("Seattle", "2026-01"): 7.1,
}

TEMPERATURE_FORECASTS: dict[tuple[str, str], float] = {
    ("Seattle", "2027-01"): 8.4,
}

_PERIOD_PATTERN = re.compile(r"(20\d{2})[_-](\d{2})")


def _period_from_node_id(node_id: str) -> str | None:
    match = _PERIOD_PATTERN.search(node_id)
    if match is None:
        return None
    return f"{match.group(1)}-{match.group(2)}"


@tool
def get_weather_observation(
    city: str = "Seattle",
    period: str | None = None,
    *,
    context: ToolContext,
) -> dict[str, object]:
    """Return average observed temperature for one city and period."""

    resolved_period = period or _period_from_node_id(context.node_id) or "2026-01"
    if (city, resolved_period) not in HISTORICAL_TEMPERATURES:
        resolved_period = "2026-01"
    avg_temp_c = HISTORICAL_TEMPERATURES[(city, resolved_period)]
    return {
        "city": city,
        "period": resolved_period,
        "avg_temp_c": avg_temp_c,
    }


@tool
def get_weather_forecast(
    city: str = "Seattle",
    period: str | None = None,
    *,
    context: ToolContext,
) -> dict[str, object]:
    """Return forecasted average temperature for one city and period."""

    resolved_period = period or _period_from_node_id(context.node_id) or "2027-01"
    if (city, resolved_period) not in TEMPERATURE_FORECASTS:
        resolved_period = "2027-01"
    forecast_temp_c = TEMPERATURE_FORECASTS[(city, resolved_period)]
    return {
        "city": city,
        "period": resolved_period,
        "forecast_temp_c": forecast_temp_c,
    }


def _sandbox_task(context: ToolContext) -> str:
    lines = [
        "Merge the observed and forecast weather records into one JSON object.",
        "Include these fields:",
        "- city: city name from the records",
        "- comparison_periods: the two observed periods, earlier period first",
        "- current_avg_temp_c: observed temperature for the later period",
        "- last_year_avg_temp_c: observed temperature for the earlier period",
        "- delta_c: current_avg_temp_c minus last_year_avg_temp_c, rounded to 2 decimal places",
        "- next_year_forecast_temp_c: forecast temperature",
        "Observed values use the key avg_temp_c or observed_temp_c.",
        "Forecast values use the key forecast_temp_c.",
    ]
    if context.question:
        lines.append(f"User question: {context.question}")
    if context.micro_prompt:
        lines.append("Step context:")
        lines.append(context.micro_prompt)
    return "\n".join(lines)


@tool
async def merge_summary(
    records: list[dict[str, object]],
    *,
    context: ToolContext,
) -> dict[str, object]:
    """Merge observed and forecast weather records into one summary.

    Args:
        records: Retrieval outputs. Pass a list of output_ref objects, one per
            upstream step, with pointer /model_context.
    """

    if context.model is None:
        raise RuntimeError("merge_summary requires a model so the python sandbox can write code")
    normalized_records = [
        record.get("model_context")
        if isinstance(record.get("model_context"), dict)
        else record
        for record in records
    ]
    return await run_python_sandbox(
        normalized_records,
        task=_sandbox_task(context),
        model=context.model,
        context=ModelContext(
            run_id=context.run_id,
            workflow_id=context.workflow_id,
            node_id=context.node_id,
            session_id=context.session_id,
            metadata={"tool": "merge_summary"},
        ),
    )


async def amain() -> None:
    model_spec = resolve_model_spec()
    app = Treelace(
        model=model_spec,
        tools=[get_weather_observation, get_weather_forecast, merge_summary],
        instructions=(
            "Use only these tools: get_weather_observation, get_weather_forecast, merge_summary. "
            "Call get_weather_observation once per observed period and get_weather_forecast "
            "for the forecast period. "
            "merge_summary takes one argument, records: a list of output_ref objects, one per "
            "retrieval step. Each output_ref uses kind output_ref, the step id as node_id, and "
            "pointer /model_context. Do not pass template strings. "
            "Set merge_summary deps to every retrieval step. "
            "The merge step prompt_focus should ask for one summary with city, comparison "
            "periods, current_avg_temp_c, last_year_avg_temp_c, delta_c, and "
            "next_year_forecast_temp_c."
        ),
    )
    question = (
        "How did Seattle's average temperature change from 2025-01 to 2026-01, "
        "and what is the forecast for 2027-01?"
    )
    plan = await app.aplan(question=question)

    print(f"Model: {model_spec}")
    print("\nQuestion:")
    print(question)
    print("\nPlanner decision:")
    print(execution_plan_text(plan))

    if not isinstance(plan, PlanSpec):
        print("\nPlanner did not return an execution plan.")
        return

    workflow = PlannerOrchestrator.plan_to_workflow(plan, workflow_id="weather_forecast_merge")
    compiled = compile_workflow(workflow, tools=app.tools)
    print("\nCompiled dependency graph (text):")
    print(dependency_text(compiled))

    engine = Engine(tools=app.tools, model=app.model)
    result = await engine.run(compiled, inputs={}, question=question)
    print("\nRun status:")
    print(result.status)
    if result.error:
        print("\nRun error:")
        print(result.error)
    final_data: object = result.output
    if isinstance(final_data, dict) and "model_context" in final_data:
        final_data = final_data["model_context"]
    print("\nFinal structured data:")
    print(final_data)


if __name__ == "__main__":
    asyncio.run(amain())
