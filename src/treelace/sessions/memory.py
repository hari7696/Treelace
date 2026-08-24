"""In-memory session persistence."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from treelace.errors import SessionConflictError, SessionNotFoundError
from treelace.sessions.models import Session


class InMemorySessionStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions: dict[str, Session] = {}

    async def create(self, session: Session) -> Session:
        async with self._lock:
            if session.id in self._sessions:
                raise SessionConflictError(f"session already exists: {session.id}")
            clone = session.model_copy(deep=True)
            self._sessions[session.id] = clone
            return clone.model_copy(deep=True)

    async def get(self, session_id: str) -> Session | None:
        async with self._lock:
            item = self._sessions.get(session_id)
            return None if item is None else item.model_copy(deep=True)

    async def save(self, session: Session, *, expected_revision: str) -> Session:
        async with self._lock:
            current = self._sessions.get(session.id)
            if current is None:
                raise SessionNotFoundError(session.id)
            if current.revision != expected_revision:
                raise SessionConflictError(
                    f"revision mismatch for {session.id}: {expected_revision} != {current.revision}"
                )
            updated = session.model_copy(deep=True)
            updated.revision = str(int(current.revision) + 1)
            updated.updated_at = datetime.now(UTC)
            self._sessions[session.id] = updated
            return updated.model_copy(deep=True)

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)
