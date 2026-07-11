"""Tool registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from treelace.tools.context import ToolContext
from treelace.tools.definition import ToolDefinition
from treelace.tools.inference import infer_tool
from treelace.tools.result import ToolResult
from treelace.tools.spec import ToolSpec


@dataclass(slots=True)
class ToolRegistry:
    _items: dict[str, ToolDefinition] = field(default_factory=dict)

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._items:
            raise ValueError(f"tool already registered: {definition.name}")
        self._items[definition.name] = definition

    def register_callable(self, func: Any, *, name: str | None = None) -> ToolDefinition:
        definition = infer_tool(func, name=name)
        self.register(definition)
        return definition

    def register_many(self, items: Iterable[Any]) -> list[ToolDefinition]:
        definitions: list[ToolDefinition] = []
        for item in items:
            if isinstance(item, ToolDefinition):
                self.register(item)
                definitions.append(item)
                continue
            if getattr(item, "__treelace_tool__", None) is not None:
                definitions.append(self.register_decorated(item))
                continue
            definitions.append(self.register_callable(item))
        return definitions

    def register_decorated(self, func: Any) -> ToolDefinition:
        definition = getattr(func, "__treelace_tool__", None)
        if definition is None:
            definition = infer_tool(func)
        self.register(definition)
        return definition

    def get(self, name: str) -> ToolDefinition | None:
        return self._items.get(name)

    def require(self, name: str) -> ToolDefinition:
        definition = self.get(name)
        if definition is None:
            raise KeyError(f"tool not found: {name}")
        return definition

    @property
    def items(self) -> Mapping[str, ToolDefinition]:
        return dict(self._items)

    def catalog_lines(self) -> list[str]:
        lines: list[str] = []
        for spec in self.list_specs():
            properties_raw = spec.input_schema.get("properties", {})
            properties = properties_raw if isinstance(properties_raw, dict) else {}
            args = ", ".join(properties.keys()) if properties else "-"
            lines.append(f"- {spec.id}: {spec.description} | args: {args}")
        return lines

    def list_specs(self) -> list[ToolSpec]:
        return [self._items[name].spec for name in sorted(self._items.keys())]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "ToolRegistry":
        registry = cls()
        if not mapping:
            return registry
        for name, value in mapping.items():
            if isinstance(value, ToolDefinition):
                registry.register(value)
            else:
                registry.register_callable(value, name=name)
        return registry

    async def invoke(self, name: str, args: dict[str, Any], *, context: ToolContext) -> ToolResult:
        definition = self.require(name)
        return await definition.invoke(args, context=context)
