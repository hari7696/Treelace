"""In-memory artifact registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class ArtifactRecord:
    id: str
    session_id: str
    kind: str
    value: Any
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class InMemoryArtifactStore:
    def __init__(self, *, max_artifacts_per_session: int = 1000) -> None:
        self._lock = Lock()
        self._items: dict[str, dict[str, ArtifactRecord]] = {}
        self.max_artifacts_per_session = max_artifacts_per_session

    def put(self, *, session_id: str, value: Any, kind: str = "generic") -> str:
        with self._lock:
            session_items = self._items.setdefault(session_id, {})
            if len(session_items) >= self.max_artifacts_per_session:
                raise ValueError(f"artifact limit exceeded for session {session_id}")
            artifact_id = str(uuid4())
            session_items[artifact_id] = ArtifactRecord(
                id=artifact_id,
                session_id=session_id,
                kind=kind,
                value=value,
            )
            return artifact_id

    def get(self, *, session_id: str, artifact_id: str) -> Any:
        with self._lock:
            session_items = self._items.get(session_id, {})
            if artifact_id not in session_items:
                raise KeyError(f"artifact not found: {artifact_id}")
            return session_items[artifact_id].value

    def list_refs(self, *, session_id: str) -> list[str]:
        with self._lock:
            return sorted(self._items.get(session_id, {}).keys())

    def delete_session(self, *, session_id: str) -> None:
        with self._lock:
            self._items.pop(session_id, None)
