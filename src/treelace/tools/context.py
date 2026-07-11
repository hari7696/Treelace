"""Tool execution context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from treelace.models.protocol import Model
from treelace.sessions.protocols import ArtifactStoreProtocol


@dataclass(slots=True)
class ToolContext:
    """Per-invocation data passed into a tool.

    ``model`` is set when the engine has a model, so a tool can make its own
    completion. ``question`` is the user question for the current run.
    """

    run_id: str
    session_id: str
    workflow_id: str
    node_id: str
    micro_prompt: str | None
    artifact_store: ArtifactStoreProtocol
    inputs: Any
    model: Model | None = None
    question: str | None = None

    def store_artifact(self, value: Any, *, kind: str = "generic") -> str:
        return self.artifact_store.put(session_id=self.session_id, value=value, kind=kind)

    def load_artifact(self, artifact_id: str) -> Any:
        return self.artifact_store.get(session_id=self.session_id, artifact_id=artifact_id)
