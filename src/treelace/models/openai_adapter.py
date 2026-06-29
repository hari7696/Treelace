"""OpenAI-compatible model adapter."""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from treelace.models.protocol import (
    Message,
    Model,
    ModelContext,
    ModelRequest,
    ModelResponse,
    StructuredOutput,
    TextOutput,
)


class OpenAIChatModel(Model):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)

    @staticmethod
    def _to_openai_messages(messages: list[Message]) -> list[dict[str, str]]:
        return [{"role": item.role, "content": item.content} for item in messages]

    @staticmethod
    def _normalize_output_schema(schema: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(schema)
        if normalized.get("type") is None and any(
            key in normalized for key in ("oneOf", "anyOf", "allOf")
        ):
            normalized["type"] = "object"
        return normalized

    @staticmethod
    def _is_schema_response_format_error(exc: Exception) -> bool:
        text = str(exc)
        return "Invalid schema for response_format" in text

    async def complete(self, request: ModelRequest, *, context: ModelContext) -> ModelResponse:
        _ = context
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_openai_messages(request.messages),
        }
        if request.output_schema is not None:
            schema = self._normalize_output_schema(request.output_schema)
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "treelace_response",
                    "schema": schema,
                },
            }
        try:
            response = await self.client.chat.completions.create(**payload)
        except Exception as exc:
            if request.output_schema is None or not self._is_schema_response_format_error(exc):
                raise
            fallback_payload = dict(payload)
            fallback_payload["response_format"] = {"type": "json_object"}
            response = await self.client.chat.completions.create(**fallback_payload)
        content = response.choices[0].message.content or ""
        if request.output_schema is None:
            return ModelResponse(output=TextOutput(text=content))
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("model returned invalid JSON content") from exc
        if not isinstance(data, dict):
            raise ValueError("model returned non-object JSON for structured output")
        return ModelResponse(output=StructuredOutput(data=data))
