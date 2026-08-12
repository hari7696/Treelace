"""Dependency graph inspection helpers."""

from __future__ import annotations

import json
from typing import Any

from treelace.compiler.compiled import CompiledWorkflow


def _step_ids(plan: CompiledWorkflow) -> list[str]:
    # Keep display order aligned with the authored workflow for readability.
    return list(plan.operations.keys())


def _execution_waves(plan: CompiledWorkflow) -> list[list[str]]:
    remaining = set(plan.operations.keys())
    completed: set[str] = set()
    waves: list[list[str]] = []
    ordered = _step_ids(plan)

    while remaining:
        ready = [step_id for step_id in ordered if step_id in remaining and all(dep in completed for dep in plan.operations[step_id].deps)]
        if not ready:
            # The compiler already enforces acyclic workflows; this keeps inspect robust.
            break
        waves.append(ready)
        for step_id in ready:
            remaining.remove(step_id)
            completed.add(step_id)

    return waves


def _render_pointer(*, node_id: str, pointer: str | None) -> str:
    if not pointer:
        return node_id
    if pointer.startswith("/"):
        parts = [part for part in pointer.split("/") if part]
        if parts:
            return f"{node_id}." + ".".join(parts)
    return f"{node_id}{pointer}"


def _render_value(value: Any) -> str:
    if isinstance(value, dict):
        if value.get("kind") == "output_ref":
            node_id = value.get("node_id")
            pointer = value.get("pointer")
            if isinstance(node_id, str):
                return _render_pointer(node_id=node_id, pointer=pointer if isinstance(pointer, str) else None)
        return "{...}"
    if isinstance(value, list):
        if not value:
            return "[]"
        return "[" + ", ".join(_render_value(item) for item in value) + "]"
    return repr(value)


def _render_arguments(arguments: dict[str, Any]) -> str:
    if not arguments:
        return ""
    rendered = ", ".join(f"{key}={_render_value(value)}" for key, value in arguments.items())
    return rendered


def _render_operation(plan: CompiledWorkflow, step_id: str) -> str:
    op = plan.operations[step_id]
    if op.kind == "tool_call":
        tool = op.payload.get("tool")
        arguments = op.payload.get("arguments")
        if isinstance(tool, str):
            if isinstance(arguments, dict):
                args_text = _render_arguments(arguments)
            else:
                args_text = ""
            return f"{tool}({args_text})" if args_text else f"{tool}()"
    if op.kind == "clarification":
        prompt = op.payload.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return f"clarification(prompt={prompt!r})"
        return "clarification()"
    return op.kind


def dependency_text(plan: CompiledWorkflow) -> str:
    lines = [f"Execution plan: {plan.workflow_id}"]
    if plan.final_node_id is not None:
        lines.append(f"Final step: {plan.final_node_id}")
    lines.append(f"Digest: {plan.digest[:12]}...")
    lines.append("")
    lines.append("Steps:")
    for index, step_id in enumerate(_step_ids(plan), start=1):
        op = plan.operations[step_id]
        deps = ", ".join(op.deps) if op.deps else "none"
        lines.append(f"  {index}. {step_id}: {_render_operation(plan, step_id)}")
        lines.append(f"     waits for: {deps}")

    waves = _execution_waves(plan)
    if waves:
        lines.append("")
        lines.append("Execution waves:")
        for index, wave in enumerate(waves, start=1):
            suffix = " (parallel)" if len(wave) > 1 else ""
            lines.append(f"  Wave {index}{suffix}: {', '.join(wave)}")

    return "\n".join(lines)


def dependency_json(plan: CompiledWorkflow) -> str:
    payload = {
        "workflow_id": plan.workflow_id,
        "digest": plan.digest,
        "order": plan.order,
        "operations": {
            op_id: {"kind": op.kind, "deps": op.deps}
            for op_id, op in plan.operations.items()
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def dependency_mermaid(plan: CompiledWorkflow) -> str:
    lines = ["flowchart LR"]
    for op_id in plan.order:
        lines.append(f"    {op_id}[{op_id}]")
    for op_id in plan.order:
        op = plan.operations[op_id]
        if not op.deps:
            continue
        for dep in op.deps:
            lines.append(f"    {dep} --> {op_id}")
    return "\n".join(lines)
