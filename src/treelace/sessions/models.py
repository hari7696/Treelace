"""Session persistence models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from treelace.json_types import JsonValue


class PendingClarification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: Literal["planner", "tool"]
    token: str
    prompt: str
    node_id: str | None = None
    options: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)

    @field_validator("token", "prompt")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()


class RuntimeState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    completed_status: dict[str, Literal["succeeded", "failed"]] = Field(default_factory=dict)
    outputs: dict[str, JsonValue] = Field(default_factory=dict)
    pending: list[str] = Field(default_factory=list)


class CompiledWorkflowSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    canonical_json: str
    digest: str


class Session(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    workflow_id: str
    workflow_digest: str | None = None
    status: Literal["running", "waiting", "succeeded", "failed"]
    question: str | None = None
    inputs: JsonValue
    output: JsonValue | None = None
    runtime_state: RuntimeState = Field(default_factory=RuntimeState)
    pending_clarifications: list[PendingClarification] = Field(default_factory=list)
    compiled_workflow: CompiledWorkflowSnapshot | None = None
    # Last conversational turns, used by prompt rewriting.
    history: list[dict[str, str]] = Field(default_factory=list)
    # Node id -> produced artifact references.
    node_artifacts: dict[str, list[str]] = Field(default_factory=dict)
    revision: str = "0"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("id", "workflow_id")
    @classmethod
    def _non_empty_ids(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()

    @field_validator("history")
    @classmethod
    def _history_shape(cls, turns: list[dict[str, str]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for turn in turns:
            role = str(turn.get("role", "")).strip()
            content = str(turn.get("content", "")).strip()
            if not role or not content:
                continue
            normalized.append({"role": role, "content": content})
        return normalized

    @model_validator(mode="after")
    def _status_consistency(self) -> "Session":
        if self.status == "waiting" and not self.pending_clarifications:
            raise ValueError("waiting session requires pending clarifications")
        if self.status != "waiting" and self.pending_clarifications:
            raise ValueError("pending clarifications require waiting session status")
        return self

    @property
    def pending_clarification(self) -> PendingClarification | None:
        return self.pending_clarifications[0] if self.pending_clarifications else None
