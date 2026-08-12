"""Run result contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from treelace.json_types import JsonValue
from treelace.planner.models import PlanDecision
from treelace.runtime.events import Event
from treelace.sessions.models import PendingClarification


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["succeeded", "failed", "waiting"]
    run_id: str
    session_id: str
    output: JsonValue | None = None
    error: str | None = None
    pending_clarification: PendingClarification | None = None
    events: list[Event] = Field(default_factory=list)
    plan: PlanDecision | None = None

    @model_validator(mode="after")
    def _validate_state(self) -> "RunResult":
        if self.status == "waiting" and self.pending_clarification is None:
            raise ValueError("waiting result requires pending_clarification")
        if self.status == "failed" and not self.error:
            raise ValueError("failed result requires error")
        if self.status == "succeeded" and self.error is not None:
            raise ValueError("succeeded result cannot include error")
        return self
