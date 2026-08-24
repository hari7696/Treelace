"""Protocols for session and artifact persistence backends."""

from __future__ import annotations

from typing import Any, Protocol

from treelace.sessions.models import Session


class SessionStoreProtocol(Protocol):
    async def create(self, session: Session) -> Session:
        ...

    async def get(self, session_id: str) -> Session | None:
        ...

    async def save(self, session: Session, *, expected_revision: str) -> Session:
        ...

    async def delete(self, session_id: str) -> None:
        ...


class ArtifactStoreProtocol(Protocol):
    def put(self, *, session_id: str, value: Any, kind: str = "generic") -> str:
        ...

    def get(self, *, session_id: str, artifact_id: str) -> Any:
        ...

    def delete_session(self, *, session_id: str) -> None:
        ...
