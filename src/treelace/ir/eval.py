"""Expression evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from treelace.ir.expressions import (
    CompareExpr,
    ConstExpr,
    Expression,
    FormatExpr,
    InputRef,
    OutputRef,
)


class MissingValueError(ValueError):
    """Raised when an expression references missing data."""


@dataclass(slots=True)
class EvalContext:
    inputs: Any
    outputs: dict[str, Any]


def _resolve_pointer(value: Any, pointer: str) -> Any:
    if pointer == "/" or pointer == "":
        return value
    if not pointer.startswith("/"):
        raise MissingValueError(f"pointer must start with '/': {pointer}")
    current = value
    for part in pointer.split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if part not in current:
                raise MissingValueError(f"missing key '{part}' in pointer {pointer}")
            current = current[part]
            continue
        if isinstance(current, list):
            try:
                index = int(part)
            except ValueError as exc:
                raise MissingValueError(f"list index is not int in pointer {pointer}") from exc
            if index < 0 or index >= len(current):
                raise MissingValueError(f"index out of range in pointer {pointer}")
            current = current[index]
            continue
        raise MissingValueError(f"cannot resolve pointer {pointer} against scalar value")
    return current


def eval_expression(expr: Expression | Any, ctx: EvalContext) -> Any:
    if isinstance(expr, ConstExpr):
        return expr.value
    if isinstance(expr, InputRef):
        return _resolve_pointer(ctx.inputs, expr.pointer)
    if isinstance(expr, OutputRef):
        if expr.node_id not in ctx.outputs:
            raise MissingValueError(f"output for node '{expr.node_id}' is missing")
        return _resolve_pointer(ctx.outputs[expr.node_id], expr.pointer)
    if isinstance(expr, CompareExpr):
        left = eval_expression(expr.left, ctx)
        right = eval_expression(expr.right, ctx)
        if expr.op == "eq":
            return left == right
        if expr.op == "ne":
            return left != right
        if expr.op == "gt":
            return left > right
        if expr.op == "gte":
            return left >= right
        if expr.op == "lt":
            return left < right
        if expr.op == "lte":
            return left <= right
        if expr.op == "contains":
            return right in left
        raise MissingValueError(f"unknown compare op: {expr.op}")
    if isinstance(expr, FormatExpr):
        payload = {key: eval_expression(value, ctx) for key, value in expr.values.items()}
        return expr.template.format(**payload)
    if isinstance(expr, dict):
        kind = expr.get("kind")
        if kind == "const":
            return eval_expression(ConstExpr.model_validate(expr), ctx)
        if kind == "input_ref":
            return eval_expression(InputRef.model_validate(expr), ctx)
        if kind == "output_ref":
            return eval_expression(OutputRef.model_validate(expr), ctx)
        if kind == "compare":
            return eval_expression(CompareExpr.model_validate(expr), ctx)
        if kind == "format":
            return eval_expression(FormatExpr.model_validate(expr), ctx)
        return {key: eval_expression(value, ctx) for key, value in expr.items()}
    if isinstance(expr, list):
        return [eval_expression(item, ctx) for item in expr]
    return expr
