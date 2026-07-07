"""Standardized tool result envelopes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from treelace.json_types import JsonValue


class ClarificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    prompt: str
    options: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)


class ToolOutput(BaseModel):
    """Preferred tool return envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
    data: dict[str, JsonValue] = Field(alias="model_context")
    artifact_refs: list[str] = Field(default_factory=list)
    clarification_request: ClarificationRequest | None = None

    @property
    def model_context(self) -> dict[str, JsonValue]:
        """Backward-compatible alias used by the runtime payload contract."""

        return self.data


class ToolResult(ToolOutput):
    """Backward-compatible alias for ToolOutput."""
