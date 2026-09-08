"""High-level Treelace application facade."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Coroutine, Iterable, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from treelace.json_types import JsonValue
from treelace.models import OpenAIChatModel
from treelace.models.protocol import Message, Model, ModelContext, ModelRequest, StructuredOutput
from treelace.planner.inspect import execution_plan_text
from treelace.planner.models import ChatPlan, PlanDecision
from treelace.planner.prompts import ConversationTurn, MasterPromptManager
from treelace.runner import Runner
from treelace.runtime.engine import Engine
from treelace.runtime.events import Event
from treelace.runtime.result import RunResult
from treelace.tools.definition import ToolDefinition
from treelace.tools.registry import ToolRegistry


class NeedsInput(BaseModel):
    """Public waiting-state payload used by applications."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    prompt: str
    options: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)


class Run(BaseModel):
    """Public result envelope returned by Treelace runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["succeeded", "failed", "waiting"]
    question: str
    answer: str | None = None
    data: JsonValue | None = None
    error: str | None = None
    needs_input: NeedsInput | None = None
    plan: PlanDecision | None = None
    plan_text: str | None = None
    events: list[Event] = Field(default_factory=list)
    session_id: str = Field(repr=False)
    clarification_token: str | None = Field(default=None, repr=False)
    raw_output: JsonValue | None = Field(default=None, repr=False)

    @property
    def prompt(self) -> str | None:
        return self.needs_input.prompt if self.needs_input is not None else None

    @model_validator(mode="after")
    def _validate_state(self) -> "Run":
        if self.status == "waiting":
            if self.needs_input is None:
                raise ValueError("waiting run requires needs_input")
            if not self.clarification_token:
                raise ValueError("waiting run requires clarification token")
        if self.status == "failed" and not self.error:
            raise ValueError("failed run requires error")
        if self.status == "succeeded" and self.error is not None:
            raise ValueError("succeeded run cannot include error")
        return self


def _resolve_openai_model(spec: str) -> Model:
    provider, separator, model_name = spec.partition(":")
    if not separator:
        provider = "openai"
        model_name = spec
    provider = provider.strip().lower()
    if provider not in {"openai", "azure-openai", "azure"}:
        raise ValueError(f"unsupported model provider: {provider}")

    model = model_name.strip() or (
        os.getenv("OPENAI_MODEL")
        or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_MODEL_NAME")
    )
    if not model:
        raise ValueError("missing model name (for example: 'openai:gpt-5-mini')")

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("missing OPENAI_API_KEY or AZURE_OPENAI_API_KEY")

    base_url = os.getenv("OPENAI_BASE_URL")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not base_url and endpoint:
        base_url = endpoint.rstrip("/") + "/openai/v1/"

    return OpenAIChatModel(model=model, api_key=api_key, base_url=base_url)


def _resolve_model(model: Model | str) -> Model:
    if isinstance(model, str):
        return _resolve_openai_model(model)
    return model


def _extract_data(output: JsonValue | None) -> JsonValue | None:
    if not isinstance(output, dict):
        return output
    model_context = output.get("model_context")
    if model_context is not None:
        return model_context
    trimmed = {
        key: value
        for key, value in output.items()
        if key not in {"artifact_refs", "clarification_request"}
    }
    return trimmed or None


def _extract_artifact_refs(output: JsonValue | None) -> list[str]:
    if not isinstance(output, dict):
        return []
    refs = output.get("artifact_refs")
    if not isinstance(refs, list):
        return []
    return [item for item in refs if isinstance(item, str)]


T = TypeVar("T")


class Treelace:
    """Simple facade for running small LLM-driven tool applications."""

    def __init__(
        self,
        *,
        model: Model | str | None = None,
        tools: ToolRegistry | Iterable[Any] | None = None,
        instructions: str | None = None,
        planner_model: Model | str | None = None,
        responder_model: Model | str | None = None,
        prompt_manager: MasterPromptManager | None = None,
        engine: Engine | None = None,
    ) -> None:
        planner_source = planner_model if planner_model is not None else model
        if planner_source is None:
            raise ValueError("model or planner_model is required")
        responder_source = responder_model if responder_model is not None else planner_source

        if prompt_manager is not None and instructions:
            raise ValueError("provide either prompt_manager or instructions, not both")

        manager = prompt_manager or MasterPromptManager(master_prompt=instructions)
        self._instructions = instructions.strip() if isinstance(instructions, str) else ""
        self._runner = Runner(
            model=_resolve_model(planner_source),
            tools=tools,
            prompt_manager=manager,
            engine=engine,
        )
        self._responder_model = _resolve_model(responder_source)

    @property
    def tools(self) -> ToolRegistry:
        return self._runner.tools

    @property
    def model(self) -> Model:
        """Model used for planning and for tool-side completions."""

        return self._runner.planner.model

    def register_tool(self, tool_like: Any) -> ToolDefinition:
        return self._runner.register_tool(tool_like)

    async def aplan(
        self,
        *,
        question: str,
        history: list[ConversationTurn] | None = None,
        session_id: str | None = None,
        workflow_id: str = "planned_workflow",
    ) -> PlanDecision:
        plan, _ = await self._runner.plan(
            question=question,
            history=history,
            session_id=session_id,
            workflow_id=workflow_id,
        )
        return plan

    def plan(
        self,
        *,
        question: str,
        history: list[ConversationTurn] | None = None,
        session_id: str | None = None,
        workflow_id: str = "planned_workflow",
    ) -> PlanDecision:
        return _run_sync(
            self.aplan(
                question=question,
                history=history,
                session_id=session_id,
                workflow_id=workflow_id,
            )
        )

    async def arun(
        self,
        question: str,
        *,
        history: list[ConversationTurn] | None = None,
        inputs: JsonValue | None = None,
        respond: bool = True,
        session_id: str | None = None,
        workflow_id: str = "planned_workflow",
    ) -> Run:
        result = await self._runner.run(
            question=question,
            inputs={} if inputs is None else inputs,
            history=history,
            session_id=session_id,
            workflow_id=workflow_id,
        )
        return await self._to_public_run(question=question, result=result, respond=respond)

    def run(
        self,
        question: str,
        *,
        history: list[ConversationTurn] | None = None,
        inputs: JsonValue | None = None,
        respond: bool = True,
        session_id: str | None = None,
        workflow_id: str = "planned_workflow",
    ) -> Run:
        return _run_sync(
            self.arun(
                question,
                history=history,
                inputs=inputs,
                respond=respond,
                session_id=session_id,
                workflow_id=workflow_id,
            )
        )

    run_sync = run

    async def aresume(self, run: Run, answer: str, *, respond: bool = True) -> Run:
        if run.status != "waiting":
            raise ValueError("resume requires a waiting run")
        if run.clarification_token is None:
            raise ValueError("run does not have a clarification token")
        resumed = await self._runner.resume(
            session_id=run.session_id,
            token=run.clarification_token,
            decision=answer,
        )
        return await self._to_public_run(question=run.question, result=resumed, respond=respond)

    def resume(self, run: Run, answer: str, *, respond: bool = True) -> Run:
        return _run_sync(self.aresume(run, answer, respond=respond))

    resume_sync = resume

    async def _to_public_run(
        self,
        *,
        question: str,
        result: RunResult,
        respond: bool,
    ) -> Run:
        plan_text = execution_plan_text(result.plan) if result.plan is not None else None
        pending = result.pending_clarification
        needs_input = (
            NeedsInput(
                prompt=pending.prompt,
                options=list(pending.options),
                missing_fields=list(pending.missing_fields),
            )
            if pending is not None
            else None
        )

        data = _extract_data(result.output)
        answer: str | None = None
        if isinstance(result.plan, ChatPlan):
            answer = result.plan.response
            data = {"message": result.plan.response}
        elif result.status == "succeeded" and respond:
            answer = await self._synthesize_answer(
                question=question,
                plan_text=plan_text or "(no plan)",
                data=data,
                artifact_refs=_extract_artifact_refs(result.output),
                session_id=result.session_id,
            )

        return Run(
            status=result.status,
            question=question,
            answer=answer,
            data=data,
            error=result.error,
            needs_input=needs_input,
            plan=result.plan,
            plan_text=plan_text,
            events=list(result.events),
            session_id=result.session_id,
            clarification_token=pending.token if pending is not None else None,
            raw_output=result.output,
        )

    async def _synthesize_answer(
        self,
        *,
        question: str,
        plan_text: str,
        data: JsonValue | None,
        artifact_refs: list[str],
        session_id: str,
    ) -> str:
        if data is None:
            return "No result data was produced."

        system_lines = [
            "You are Treelace's response synthesizer.",
            "Answer strictly from the provided execution data.",
            "Do not invent tools, values, or files.",
            "If data is missing for part of the question, say so clearly.",
            "Keep the answer concise unless the question asks for detail.",
        ]
        if self._instructions:
            system_lines.append(f"Application instructions:\n{self._instructions}")

        prompt_payload = {
            "question": question,
            "plan": plan_text,
            "data": data,
            "artifact_refs": artifact_refs,
        }
        response = await self._responder_model.complete(
            ModelRequest(
                messages=[
                    Message(role="system", content="\n".join(system_lines)),
                    Message(
                        role="user",
                        content=json.dumps(prompt_payload, indent=2, ensure_ascii=True, sort_keys=True),
                    ),
                ]
            ),
            context=ModelContext(session_id=session_id, metadata={"stage": "response_synthesis"}),
        )
        if isinstance(response.output, StructuredOutput):
            answer = response.output.data.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer.strip()
            return json.dumps(response.output.data, ensure_ascii=True, sort_keys=True)
        text = response.output.text.strip()
        if text:
            return text
        return json.dumps(data, ensure_ascii=True, sort_keys=True)


def _run_sync(coro: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("synchronous Treelace methods cannot run inside an active event loop")
