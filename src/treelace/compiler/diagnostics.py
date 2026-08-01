"""Compiler diagnostics."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class Diagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    severity: Literal["error", "warning"]
    code: str
    message: str
    node_id: str | None = None


def error(code: str, message: str, *, node_id: str | None = None) -> Diagnostic:
    return Diagnostic(severity="error", code=code, message=message, node_id=node_id)


def warning(code: str, message: str, *, node_id: str | None = None) -> Diagnostic:
    return Diagnostic(severity="warning", code=code, message=message, node_id=node_id)
