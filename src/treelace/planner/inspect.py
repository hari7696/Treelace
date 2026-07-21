"""Planner inspection and human-readable rendering helpers."""

from __future__ import annotations

from treelace.planner.models import ChatPlan, ClarificationPlan, PlanDecision, PlanSpec


def _execution_waves(plan: PlanSpec) -> list[list[str]]:
    steps_by_id = {step.id: step for step in plan.steps}
    ordered = [step.id for step in plan.steps]
    remaining = set(ordered)
    completed: set[str] = set()
    waves: list[list[str]] = []

    while remaining:
        ready = [
            step_id
            for step_id in ordered
            if step_id in remaining and all(dep in completed for dep in steps_by_id[step_id].deps)
        ]
        if not ready:
            # Keep this helper robust even if an invalid plan slips through.
            break
        waves.append(ready)
        for step_id in ready:
            remaining.remove(step_id)
            completed.add(step_id)

    return waves


def execution_plan_text(plan: PlanDecision) -> str:
    """Render a compact plan summary suitable for terminal output."""

    if isinstance(plan, ChatPlan):
        return "\n".join(
            [
                "Plan",
                f"  Chat: {plan.response}",
            ]
        )
    if isinstance(plan, ClarificationPlan):
        return "\n".join(
            [
                "Plan",
                f"  NeedsInput: {plan.clarification.prompt}",
            ]
        )

    waves = _execution_waves(plan)
    lines = ["Plan"]
    for index, wave in enumerate(waves, start=1):
        rendered = ", ".join(wave)
        suffix = " (parallel)" if len(wave) > 1 else ""
        lines.append(f"  Wave {index}{suffix}: {rendered}")

    lines.append("  Calls:")

    for step in plan.steps:
        deps = ", ".join(step.deps)
        lines.append(f"    - {step.id}({deps})" if deps else f"    - {step.id}()")

    lines.append(f"  Result: {plan.result_step_id}")
    return "\n".join(lines)
