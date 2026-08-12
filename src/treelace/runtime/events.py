"""Runtime event contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from treelace.json_types import JsonValue


class _EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sequence: int
    run_id: str
    session_id: str
    workflow_id: str
    node_id: str | None = None


class RunStartedEvent(_EventBase):
    kind: Literal["run_started"] = "run_started"


class NodeStartedEvent(_EventBase):
    kind: Literal["node_started"] = "node_started"
    node_id: str


class NodeSucceededEvent(_EventBase):
    kind: Literal["node_succeeded"] = "node_succeeded"
    node_id: str
    output: dict[str, JsonValue]


class NodeFailedEvent(_EventBase):
    kind: Literal["node_failed"] = "node_failed"
    node_id: str
    error: str


class ClarificationRequestedEvent(_EventBase):
    kind: Literal["clarification_requested"] = "clarification_requested"
    node_id: str
    source: Literal["planner", "tool"]
    token: str
    prompt: str
    options: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)


class RunPausedEvent(_EventBase):
    kind: Literal["run_paused"] = "run_paused"
    node_id: str | None = None
    source: Literal["planner", "tool"]


class RunFailedEvent(_EventBase):
    kind: Literal["run_failed"] = "run_failed"
    error: str


class RunSucceededEvent(_EventBase):
    kind: Literal["run_succeeded"] = "run_succeeded"
    output_node: str | None = None


Event = Annotated[
    RunStartedEvent
    | NodeStartedEvent
    | NodeSucceededEvent
    | NodeFailedEvent
    | ClarificationRequestedEvent
    | RunPausedEvent
    | RunFailedEvent
    | RunSucceededEvent,
    Field(discriminator="kind"),
]
EVENT_ADAPTER = TypeAdapter(Event)
