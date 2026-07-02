"""Tool definition and invocation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, TypeAdapter

from treelace.tools.context import ToolContext
from treelace.tools.result import ToolOutput, ToolResult
from treelace.tools.spec import ToolSpec


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    input_model: type[BaseModel]
    output_adapter: TypeAdapter[Any]
    func: Callable[..., Any] | Callable[..., Awaitable[Any]]
    is_async: bool
    context_parameter: str | None = None

    @property
    def input_schema(self) -> dict[str, Any]:
        return self.input_model.model_json_schema()

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(id=self.name, description=self.description, input_schema=self.input_schema)

    def _coerce_tool_result(self, value: Any) -> ToolResult:
        if isinstance(value, ToolOutput):
            return ToolResult.model_validate(value.model_dump(mode="python"))
        if isinstance(value, dict) and ("model_context" in value or "data" in value):
            return ToolResult.model_validate(value)
        if isinstance(value, dict):
            return ToolResult.model_validate({"model_context": value})
        return ToolResult.model_validate({"model_context": {"result": value}})

    async def invoke(self, args: dict[str, Any], *, context: ToolContext) -> ToolResult:
        payload = self.input_model.model_validate(args)
        kwargs = payload.model_dump(mode="python")
        if self.context_parameter is not None:
            kwargs[self.context_parameter] = context
        if self.is_async:
            raw = await self.func(**kwargs)
        else:
            raw = await asyncio.to_thread(self.func, **kwargs)
        validated = self.output_adapter.validate_python(raw)
        return self._coerce_tool_result(validated)
