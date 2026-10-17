from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from treelace.models.openai_adapter import OpenAIChatModel
from treelace.models.protocol import Message, ModelContext, ModelRequest, StructuredOutput


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list[_Choice]


class _FakeCompletions:
    def __init__(self, *, fail_schema_once: bool) -> None:
        self.calls: list[dict[str, Any]] = []
        self._fail_schema_once = fail_schema_once

    async def create(self, **payload: Any) -> _Response:
        self.calls.append(payload)
        if self._fail_schema_once and len(self.calls) == 1:
            raise RuntimeError(
                "Invalid schema for response_format 'treelace_response': "
                "schema must be a JSON Schema of 'type: \"object\"', got 'type: \"None\"'."
            )
        return _Response(choices=[_Choice(message=_Message(content='{"ok": true}'))])


def test_normalize_output_schema_adds_top_level_object_type() -> None:
    input_schema = {"oneOf": [{"type": "object", "properties": {"x": {"type": "string"}}}]}
    normalized = OpenAIChatModel._normalize_output_schema(input_schema)
    assert normalized["type"] == "object"
    assert "oneOf" in normalized


@pytest.mark.asyncio
async def test_schema_error_falls_back_to_json_object_mode() -> None:
    model = OpenAIChatModel(model="fake-model", api_key="test-key", base_url="http://localhost")
    fake = _FakeCompletions(fail_schema_once=True)
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))

    request = ModelRequest(
        messages=[Message(role="user", content="hello")],
        output_schema={"oneOf": [{"type": "object"}]},
    )
    result = await model.complete(request, context=ModelContext())
    assert isinstance(result.output, StructuredOutput)
    assert result.output.data == {"ok": True}
    assert len(fake.calls) == 2
    assert fake.calls[0]["response_format"]["type"] == "json_schema"
    assert fake.calls[1]["response_format"]["type"] == "json_object"
