"""Infer tool metadata from Python callables."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

from pydantic import ConfigDict, Field, TypeAdapter, create_model

from treelace.tools.context import ToolContext
from treelace.tools.definition import ToolDefinition


@dataclass(slots=True)
class ParsedDocstring:
    description: str
    args: dict[str, str]


_ARG_HEADERS = {"Args:", "Arguments:", "Parameters:"}


def _looks_like_type_hint(value: str) -> bool:
    normalized = value.strip()
    if not normalized:
        return False
    simple = {
        "str",
        "int",
        "float",
        "bool",
        "dict",
        "list",
        "tuple",
        "set",
        "any",
        "object",
    }
    if normalized.lower() in simple:
        return True
    return "[" in normalized and normalized.endswith("]")


def _parse_google_docstring(doc: str | None) -> ParsedDocstring:
    if not doc:
        return ParsedDocstring(description="", args={})
    lines = [line.rstrip() for line in doc.strip().splitlines()]
    summary_parts: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            if summary_parts:
                break
            continue
        if stripped in _ARG_HEADERS:
            break
        summary_parts.append(stripped)
    summary = " ".join(summary_parts).strip()

    args: dict[str, str] = {}
    in_args = False
    active_name: str | None = None
    active_chunks: list[str] = []

    def flush_active() -> None:
        nonlocal active_name, active_chunks
        if active_name is None:
            return
        merged = " ".join(chunk.strip() for chunk in active_chunks if chunk.strip()).strip()
        if merged:
            args[active_name] = merged
        active_name = None
        active_chunks = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in _ARG_HEADERS:
            flush_active()
            in_args = True
            continue
        if in_args:
            raw = stripped.lstrip("-* ").strip()
            if raw.endswith(":"):
                flush_active()
                in_args = False
                continue
            if ":" in raw:
                candidate_name, desc = raw.split(":", 1)
                candidate = candidate_name.strip()
                if candidate:
                    flush_active()
                    active_name = candidate
                    detail = desc.strip()
                    if " - " in detail:
                        maybe_type, maybe_desc = detail.split(" - ", 1)
                        if _looks_like_type_hint(maybe_type):
                            detail = maybe_desc.strip()
                    active_chunks = [detail] if detail else []
                    continue
            if active_name is not None:
                active_chunks.append(raw)
    flush_active()
    description = summary or (lines[0].strip() if lines else "")
    return ParsedDocstring(description=description, args=args)


def infer_tool(func: Callable[..., Any], *, name: str | None = None) -> ToolDefinition:
    signature = inspect.signature(func)
    hints = get_type_hints(func)
    parsed_doc = _parse_google_docstring(inspect.getdoc(func))
    fields: dict[str, tuple[Any, Any]] = {}
    context_param: str | None = None

    for param_name, parameter in signature.parameters.items():
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(
                f"tool {func.__name__} uses unsupported parameter kind for {param_name}"
            )
        if param_name not in hints:
            raise TypeError(f"tool {func.__name__} requires type annotation for {param_name}")
        annotation = hints[param_name]
        if annotation is ToolContext:
            if parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
                raise TypeError("ToolContext parameter must be keyword-only")
            if context_param is not None:
                raise TypeError("only one ToolContext parameter is allowed")
            context_param = param_name
            continue
        default: Any = ... if parameter.default is inspect._empty else parameter.default
        field = Field(default=default, description=parsed_doc.args.get(param_name))
        fields[param_name] = (annotation, field)

    output_annotation = hints.get("return")
    if output_annotation is None:
        raise TypeError(f"tool {func.__name__} requires a return annotation")

    input_model = create_model(
        f"{func.__name__.title().replace('_', '')}Input",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )  # type: ignore[call-overload]
    output_adapter = TypeAdapter(output_annotation)
    return ToolDefinition(
        name=name or func.__name__,
        description=parsed_doc.description or f"Tool {func.__name__}",
        input_model=input_model,
        output_adapter=output_adapter,
        func=func,
        is_async=inspect.iscoroutinefunction(func),
        context_parameter=context_param,
    )


def tool(func: Callable[..., Any] | None = None, *, name: str | None = None) -> Callable[..., Any]:
    """Decorator that attaches a ToolDefinition as ``__treelace_tool__``."""

    def _decorate(inner: Callable[..., Any]) -> Callable[..., Any]:
        definition = infer_tool(inner, name=name)
        setattr(inner, "__treelace_tool__", definition)
        return inner

    if func is None:
        return _decorate
    return _decorate(func)
