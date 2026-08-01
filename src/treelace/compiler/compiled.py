"""Compiled workflow contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from treelace.compiler.diagnostics import Diagnostic
from treelace.json_types import JsonValue


class CompiledOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    kind: Literal["tool_call", "clarification"]
    payload: dict[str, JsonValue]
    deps: list[str] = Field(default_factory=list)


class CompiledWorkflow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    workflow_id: str
    workflow: dict[str, JsonValue]
    operations: dict[str, CompiledOperation]
    order: list[str]
    final_node_id: str | None = None
    canonical_json: str
    digest: str
    diagnostics: list[Diagnostic] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_references(self) -> "CompiledWorkflow":
        op_ids = set(self.operations.keys())
        if set(self.order) != op_ids:
            raise ValueError("order must contain each operation id exactly once")
        for op_id, op in self.operations.items():
            for dep in op.deps:
                if dep not in op_ids:
                    raise ValueError(f"operation {op_id} depends on unknown operation {dep}")
        if self.final_node_id is not None and self.final_node_id not in op_ids:
            raise ValueError("final_node_id must reference an operation id")
        return self
