"""Workflow node IR."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from treelace.ir.expressions import Expression
from treelace.json_types import JsonValue


class BaseNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("id must be non-empty")
        return value.strip()


class Sequence(BaseNode):
    kind: Literal["sequence"] = "sequence"
    steps: list["Node"]

    @field_validator("steps")
    @classmethod
    def _steps_non_empty(cls, steps: list["Node"]) -> list["Node"]:
        if not steps:
            raise ValueError("sequence requires at least one step")
        return steps


class Parallel(BaseNode):
    kind: Literal["parallel"] = "parallel"
    branches: list["Node"]
    max_concurrency: int | None = None

    @field_validator("branches")
    @classmethod
    def _branches_non_empty(cls, branches: list["Node"]) -> list["Node"]:
        if not branches:
            raise ValueError("parallel requires at least one branch")
        return branches


class ToolCall(BaseNode):
    kind: Literal["tool_call"] = "tool_call"
    tool: str
    arguments: dict[str, JsonValue | Expression] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    micro_prompt: str | None = None


class Clarification(BaseNode):
    kind: Literal["clarification"] = "clarification"
    prompt: str | Expression
    options: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


Node = Annotated[
    Sequence | Parallel | ToolCall | Clarification,
    Field(discriminator="kind"),
]


class Workflow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ir_version: Literal["1.0"] = "1.0"
    id: str
    root: Node
    result_node_id: str | None = None
    description: str | None = None

    @field_validator("id")
    @classmethod
    def _workflow_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("workflow id must be non-empty")
        return value.strip()

    @model_validator(mode="after")
    def _validate_result_node(self) -> "Workflow":
        if self.result_node_id is None:
            return self
        selected = self.result_node_id.strip()
        if not selected:
            raise ValueError("result_node_id must be non-empty when provided")
        node_ids = {node.id for node in walk_nodes(self.root)}
        if selected not in node_ids:
            raise ValueError("result_node_id must reference an existing node id")
        return self


def walk_nodes(node: Node) -> list[Node]:
    nodes: list[Node] = [node]
    if isinstance(node, Sequence):
        for child in node.steps:
            nodes.extend(walk_nodes(child))
    elif isinstance(node, Parallel):
        for child in node.branches:
            nodes.extend(walk_nodes(child))
    return nodes


Sequence.model_rebuild()
Parallel.model_rebuild()
ToolCall.model_rebuild()
Clarification.model_rebuild()
Workflow.model_rebuild()
