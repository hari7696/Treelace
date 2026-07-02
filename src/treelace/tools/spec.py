"""Machine-readable tool discovery models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from treelace.json_types import JsonObject


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    description: str
    input_schema: JsonObject
