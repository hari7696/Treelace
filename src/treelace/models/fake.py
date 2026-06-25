"""Deterministic fake model for tests and offline examples."""

from __future__ import annotations

import json
from collections import deque
from typing import Any

from treelace.models.protocol import (
    Model,
    ModelContext,
    ModelRequest,
    ModelResponse,
    StructuredOutput,
    TextOutput,
)


class FakeModel(Model):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = deque(responses)

    @classmethod
    def fixed_text(cls, text: str) -> "FakeModel":
        return cls([ModelResponse(output=TextOutput(text=text))])

    @classmethod
    def fixed_structured(cls, data: dict[str, Any]) -> "FakeModel":
        return cls([ModelResponse(output=StructuredOutput(data=data))])

    async def complete(self, request: ModelRequest, *, context: ModelContext) -> ModelResponse:
        _ = (request, context)
        if self._responses:
            return self._responses.popleft()
        if request.output_schema is not None:
            return ModelResponse(output=StructuredOutput(data={}))
        return ModelResponse(output=TextOutput(text=json.dumps({})))
