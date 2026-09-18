"""Restricted Python runner for model-written record processing.

This is not an operating-system sandbox. Registered tools still run with the
host process permissions. The helpers here only accept a single ``process``
function, reject unsafe syntax, and stop runaway loops with a step limit.
"""

from __future__ import annotations

import ast
import builtins
import json
import math
import sys
from types import FrameType
from typing import Any

from treelace.models.protocol import (
    Message,
    Model,
    ModelContext,
    ModelRequest,
    ModelResponse,
    StructuredOutput,
    TextOutput,
)

_MAX_ATTEMPTS = 2
_MAX_STEPS = 2_000
_MAX_INT = 100_000
_MAX_STRING = 5_000

_SYSTEM_PROMPT = (
    "You write one Python function for a restricted sandbox. "
    "Return JSON with a single string field named code. "
    "The code must be exactly one function:\n"
    "def process(records):\n"
    "    ...\n"
    "    return {...}\n"
    "records is a list of dictionaries and is the only input. "
    "Read every value from records. Do not hardcode numbers or names copied from the sample. "
    "Do not import modules. Do not use files, network, eval, or exec. "
    "Return a JSON object. "
    "You may use arithmetic, comparisons, comprehensions, if, for, and these calls: "
    "abs, all, any, bool, dict, enumerate, float, int, isinstance, len, list, max, min, "
    "range, reversed, round, set, sorted, str, sum, tuple, zip. "
    "You may call these methods: append, endswith, get, items, join, keys, lower, replace, "
    "split, startswith, strip, upper, values."
)

_CODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"code": {"type": "string"}},
    "required": ["code"],
    "additionalProperties": False,
}

_SAFE_BUILTINS: dict[str, Any] = {
    "abs": builtins.abs,
    "all": builtins.all,
    "any": builtins.any,
    "bool": builtins.bool,
    "dict": builtins.dict,
    "enumerate": builtins.enumerate,
    "float": builtins.float,
    "int": builtins.int,
    "isinstance": builtins.isinstance,
    "len": builtins.len,
    "list": builtins.list,
    "max": builtins.max,
    "min": builtins.min,
    "range": builtins.range,
    "reversed": builtins.reversed,
    "round": builtins.round,
    "set": builtins.set,
    "sorted": builtins.sorted,
    "str": builtins.str,
    "sum": builtins.sum,
    "tuple": builtins.tuple,
    "zip": builtins.zip,
}

_DENIED_NAMES = frozenset(
    {
        "__import__",
        "breakpoint",
        "classmethod",
        "compile",
        "copyright",
        "credits",
        "delattr",
        "dir",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "hasattr",
        "help",
        "id",
        "input",
        "license",
        "locals",
        "memoryview",
        "object",
        "open",
        "print",
        "property",
        "quit",
        "setattr",
        "staticmethod",
        "super",
        "type",
        "vars",
    }
)

_ALLOWED_METHODS = frozenset(
    {
        "append",
        "endswith",
        "get",
        "items",
        "join",
        "keys",
        "lower",
        "replace",
        "split",
        "startswith",
        "strip",
        "upper",
        "values",
    }
)

_ALLOWED_NODES = frozenset(
    {
        ast.Add,
        ast.And,
        ast.AnnAssign,
        ast.Assign,
        ast.Attribute,
        ast.AugAssign,
        ast.BinOp,
        ast.BoolOp,
        ast.Break,
        ast.Call,
        ast.Compare,
        ast.Constant,
        ast.Continue,
        ast.Dict,
        ast.DictComp,
        ast.Div,
        ast.Eq,
        ast.Expr,
        ast.FloorDiv,
        ast.For,
        ast.FormattedValue,
        ast.FunctionDef,
        ast.GeneratorExp,
        ast.Gt,
        ast.GtE,
        ast.If,
        ast.IfExp,
        ast.In,
        ast.Is,
        ast.IsNot,
        ast.JoinedStr,
        ast.Lambda,
        ast.List,
        ast.ListComp,
        ast.Load,
        ast.Lt,
        ast.LtE,
        ast.Mod,
        ast.Module,
        ast.Mult,
        ast.Name,
        ast.Not,
        ast.NotEq,
        ast.NotIn,
        ast.Or,
        ast.Pass,
        ast.Return,
        ast.Set,
        ast.SetComp,
        ast.Slice,
        ast.Store,
        ast.Sub,
        ast.Subscript,
        ast.Tuple,
        ast.UAdd,
        ast.USub,
        ast.UnaryOp,
        ast.While,
        ast.arg,
        ast.arguments,
        ast.comprehension,
        ast.keyword,
    }
)


class SandboxError(ValueError):
    """Raised when generated Python cannot be validated or executed."""


def execute_python(
    code: str,
    records: list[dict[str, Any]],
    *,
    max_steps: int = _MAX_STEPS,
) -> dict[str, Any]:
    """Validate and run a ``process(records)`` function."""

    if max_steps < 1:
        raise SandboxError("max_steps must be at least 1")
    tree = _parse_code(code)
    _validate(tree)
    safe_records = _copy_records(records)
    namespace: dict[str, Any] = {"__builtins__": dict(_SAFE_BUILTINS)}
    steps = 0

    def tracer(frame: FrameType, event: str, arg: object) -> Any:
        nonlocal steps
        if event == "line" and frame.f_code.co_filename == "<python_sandbox>":
            steps += 1
            if steps > max_steps:
                raise SandboxError(f"python sandbox exceeded {max_steps} steps")
        return tracer

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        exec(compile(tree, "<python_sandbox>", "exec"), namespace, namespace)
        process = namespace.get("process")
        if not callable(process):
            raise SandboxError("process was not defined")
        try:
            result = process(safe_records)
        except SandboxError:
            raise
        except Exception as exc:
            raise SandboxError(f"{type(exc).__name__}: {exc}") from exc
    finally:
        sys.settrace(previous)
    return _as_object_dict(result)


async def run_python_sandbox(
    records: list[dict[str, Any]],
    *,
    task: str,
    model: Model,
    context: ModelContext | None = None,
    max_attempts: int = _MAX_ATTEMPTS,
    max_steps: int = _MAX_STEPS,
) -> dict[str, Any]:
    """Ask the model to write ``process(records)``, then run it.

    Each attempt is a new completion. The records, task, and run context are
    sent with that call. A failed program is sent back once so the model can
    write a replacement.
    """

    if max_attempts < 1:
        raise SandboxError("max_attempts must be at least 1")
    cleaned_task = task.strip()
    if not cleaned_task:
        raise SandboxError("task must be non-empty")
    safe_records = _copy_records(records)
    previous_code: str | None = None
    error_text: str | None = None
    last_error: SandboxError | None = None
    for _attempt in range(max_attempts):
        code: str | None = None
        try:
            code = await _request_code(
                model,
                task=cleaned_task,
                records=safe_records,
                context=context,
                previous_code=previous_code,
                error=error_text,
            )
            return execute_python(code, safe_records, max_steps=max_steps)
        except SandboxError as exc:
            last_error = exc
            error_text = str(exc)
            if code is not None:
                previous_code = code
    if last_error is None:
        raise SandboxError("python sandbox did not run")
    raise last_error


def _parse_code(code: str) -> ast.Module:
    text = _strip_fences(code)
    try:
        return _module_from_source(text)
    except SyntaxError as first_error:
        marker = "def process"
        index = text.find(marker)
        if index <= 0:
            raise SandboxError(f"sandbox code is not valid Python: {first_error.msg}") from first_error
        try:
            return _module_from_source(text[index:])
        except SyntaxError as exc:
            raise SandboxError(f"sandbox code is not valid Python: {exc.msg}") from exc


def _module_from_source(text: str) -> ast.Module:
    tree = ast.parse(text, mode="exec")
    if not isinstance(tree, ast.Module):
        raise SyntaxError("expected a module")
    return tree


def _strip_fences(code: str) -> str:
    text = code.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _validate(tree: ast.Module) -> None:
    if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)):
        raise SandboxError("imports are not allowed")
    statements = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]
    if len(statements) != 1 or not isinstance(statements[0], ast.FunctionDef):
        raise SandboxError("sandbox code must contain only def process(records)")
    if statements[0].name != "process":
        raise SandboxError("sandbox code must define process")
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            raise SandboxError(f"syntax is not allowed: {type(node).__name__}")
        _validate_node(node)


def _validate_node(node: ast.AST) -> None:
    if isinstance(node, ast.FunctionDef):
        _validate_function(node)
        return
    if isinstance(node, ast.Lambda):
        _reject_variadic(node.args)
        return
    if isinstance(node, ast.Name) and (node.id.startswith("_") or node.id in _DENIED_NAMES):
        raise SandboxError(f"name {node.id} is not allowed")
    if isinstance(node, ast.Attribute) and (
        node.attr.startswith("_") or node.attr not in _ALLOWED_METHODS
    ):
        raise SandboxError(f"attribute {node.attr} is not allowed")
    if isinstance(node, ast.Call) and not isinstance(node.func, (ast.Name, ast.Attribute)):
        raise SandboxError("only function and method calls are allowed")
    if isinstance(node, ast.Constant):
        _validate_constant(node.value)
        return
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        _validate_target(_assignment_target(node))
        return
    if isinstance(node, ast.For):
        _validate_target(node.target)
        return
    if isinstance(node, ast.comprehension):
        _validate_target(node.target)


def _validate_function(node: ast.FunctionDef) -> None:
    if node.name.startswith("_"):
        raise SandboxError(f"function {node.name} is not allowed")
    if node.decorator_list or getattr(node, "type_params", None):
        raise SandboxError("decorators and type parameters are not allowed")
    _reject_variadic(node.args)
    if node.name != "process":
        return
    args = node.args.args
    if len(args) != 1 or args[0].arg != "records" or node.args.defaults:
        raise SandboxError("process must take exactly one argument named records")


def _reject_variadic(args: ast.arguments) -> None:
    if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs:
        raise SandboxError("variadic arguments are not allowed")


def _assignment_target(node: ast.Assign | ast.AnnAssign | ast.AugAssign) -> ast.AST:
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1:
            raise SandboxError("only one assignment target is allowed")
        return node.targets[0]
    return node.target


def _validate_target(target: ast.AST) -> None:
    if isinstance(target, ast.Name):
        if target.id.startswith("_") or target.id in _DENIED_NAMES:
            raise SandboxError(f"assignment to {target.id} is not allowed")
        return
    if isinstance(target, ast.Subscript):
        return
    if isinstance(target, (ast.Tuple, ast.List)):
        for item in target.elts:
            _validate_target(item)
        return
    raise SandboxError("only simple assignment targets are allowed")


def _validate_constant(value: object) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, int):
        if abs(value) > _MAX_INT:
            raise SandboxError("integer constant is too large")
        return
    if isinstance(value, float):
        if not math.isfinite(value) or abs(value) > 1_000_000_000:
            raise SandboxError("float constant is not allowed")
        return
    if isinstance(value, str):
        if len(value) > _MAX_STRING:
            raise SandboxError("string constant is too long")
        return
    raise SandboxError("constant type is not allowed")


def _copy_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        rendered = json.dumps(records)
    except TypeError as exc:
        raise SandboxError("records must be JSON-serializable") from exc
    parsed = json.loads(rendered)
    if not isinstance(parsed, list):
        raise SandboxError("records must be a list of objects")
    copied: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise SandboxError("records must be a list of objects")
        copied.append({str(key): value for key, value in item.items()})
    return copied


def _as_object_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SandboxError("process(records) must return a dict")
    try:
        rendered = json.dumps(value)
    except TypeError as exc:
        raise SandboxError("process(records) must return JSON-serializable data") from exc
    parsed = json.loads(rendered)
    if not isinstance(parsed, dict):
        raise SandboxError("process(records) must return a dict")
    return {str(key): item for key, item in parsed.items()}


async def _request_code(
    model: Model,
    *,
    task: str,
    records: list[dict[str, Any]],
    context: ModelContext | None,
    previous_code: str | None,
    error: str | None,
) -> str:
    response = await model.complete(
        ModelRequest(
            messages=[
                Message(role="system", content=_SYSTEM_PROMPT),
                Message(
                    role="user",
                    content=_user_prompt(
                        task=task,
                        records=records,
                        previous_code=previous_code,
                        error=error,
                    ),
                ),
            ],
            output_schema=_CODE_SCHEMA,
        ),
        context=_call_context(context),
    )
    return _code_from_response(response)


def _call_context(context: ModelContext | None) -> ModelContext:
    base = context or ModelContext()
    metadata = dict(base.metadata)
    metadata["source"] = "python_sandbox"
    return base.model_copy(update={"metadata": metadata})


def _user_prompt(
    *,
    task: str,
    records: list[dict[str, Any]],
    previous_code: str | None,
    error: str | None,
) -> str:
    parts = [
        "Task:",
        task,
        "",
        "Records JSON:",
        _records_preview(records),
        "",
        "Write process(records). Compute only from the records argument.",
    ]
    if previous_code is not None or error is not None:
        parts.extend(
            [
                "",
                "The previous function failed. Write a corrected function.",
                "Previous code:",
                previous_code or "",
                "Error:",
                error or "",
            ]
        )
    return "\n".join(parts)


def _records_preview(records: list[dict[str, Any]]) -> str:
    rendered = json.dumps(records, indent=2, sort_keys=True)
    if len(rendered) <= 8_000:
        return rendered
    keys = sorted({key for item in records for key in item})
    preview = json.dumps(records[:3], indent=2, sort_keys=True)
    return f"record_count: {len(records)}\nkeys: {keys}\npreview:\n{preview[:4_000]}"


def _code_from_response(response: ModelResponse) -> str:
    output = response.output
    if isinstance(output, StructuredOutput):
        raw = output.data.get("code")
        if not isinstance(raw, str) or not raw.strip():
            raise SandboxError("model response did not include code")
        return raw
    if isinstance(output, TextOutput):
        text = output.text.strip()
        if not text:
            raise SandboxError("model response did not include code")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(payload, dict) and isinstance(payload.get("code"), str):
            return str(payload["code"])
        return text
    raise SandboxError("model response did not include code")
