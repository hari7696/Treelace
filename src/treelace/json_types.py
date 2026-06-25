"""JSON-compatible type aliases used across Treelace contracts."""

from __future__ import annotations

from typing import TypeAlias

from pydantic import JsonValue as PydanticJsonValue

JsonValue: TypeAlias = PydanticJsonValue
JsonObject: TypeAlias = dict[str, JsonValue]
