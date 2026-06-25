"""Model protocol and DTOs."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from treelace.json_types import JsonObject


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    role: Literal["system", "user", "assistant"]
    content: str


class TextOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["text"] = "text"
    text: str


class StructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["structured"] = "structured"
    data: JsonObject


class ModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    messages: list[Message]
    output_schema: JsonObject | None = None


class ModelContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str | None = None
    workflow_id: str | None = None
    node_id: str | None = None
    session_id: str | None = None
    metadata: JsonObject = Field(default_factory=dict)


class ModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    output: TextOutput | StructuredOutput


class Model(Protocol):
    async def complete(self, request: ModelRequest, *, context: ModelContext) -> ModelResponse:
        ...
