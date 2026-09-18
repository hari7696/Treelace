"""Treelace exception hierarchy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class TreelaceError(Exception):
    """Base error for Treelace."""


@dataclass(slots=True)
class CompileFailure(TreelaceError):
    """Raised when workflow compilation fails."""

    message: str
    diagnostics: list[Any] = field(default_factory=list)

    def __str__(self) -> str:
        return self.message


class ExecutionError(TreelaceError):
    """Raised when execution fails."""


class SessionConflictError(TreelaceError):
    """Raised when optimistic concurrency checks fail."""


class SessionNotFoundError(TreelaceError):
    """Raised when a session does not exist."""


class ResumeTokenError(TreelaceError):
    """Raised when resume token validation fails."""


class NodeExecutionError(ExecutionError):
    """Raised when a specific node fails."""

    def __init__(self, node_id: str, reason: str) -> None:
        super().__init__(f"{node_id}: {reason}")
        self.node_id = node_id
        self.reason = reason
