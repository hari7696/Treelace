"""High-level planner + runtime orchestration."""

from __future__ import annotations

import json
from typing import Any, Iterable
from uuid import uuid4

from treelace.compiler.compile import compile_workflow
from treelace.compiler.compiled import CompiledWorkflow
from treelace.errors import ResumeTokenError, SessionNotFoundError
from treelace.ir.nodes import Workflow
from treelace.json_types import JsonValue
from treelace.models.protocol import Model
from treelace.planner.models import ChatPlan, ClarificationPlan, PlanDecision
from treelace.planner.orchestrator import PlannerOrchestrator
from treelace.planner.prompts import ConversationTurn, MasterPromptManager
from treelace.runtime.engine import Engine
from treelace.runtime.events import ClarificationRequestedEvent, Event, RunPausedEvent
from treelace.runtime.result import RunResult
from treelace.sessions.models import PendingClarification, RuntimeState, Session
from treelace.tools.definition import ToolDefinition
from treelace.tools.registry import ToolRegistry


class Runner:
    """Convenience facade around planner, compiler, and engine."""

    def __init__(
        self,
        *,
        model: Model,
        tools: ToolRegistry | Iterable[Any] | None = None,
        prompt_manager: MasterPromptManager | None = None,
        engine: Engine | None = None,
    ) -> None:
        if isinstance(tools, ToolRegistry):
            self.tools = tools
        else:
            self.tools = ToolRegistry()
            if tools is not None:
                self.tools.register_many(tools)
        self.prompt_manager = prompt_manager or MasterPromptManager()
        self.planner = PlannerOrchestrator(model=model, prompt_manager=self.prompt_manager)
        self.engine = engine or Engine(tools=self.tools, model=model)
        self.engine.tools = self.tools
        if self.engine.model is None:
            self.engine.model = model

    def register_tool(self, tool_like: Any) -> ToolDefinition:
        return self.tools.register_decorated(tool_like)

    async def plan(
        self,
        *,
        question: str,
        history: list[ConversationTurn] | None = None,
        session_id: str | None = None,
        workflow_id: str = "planned_workflow",
    ) -> tuple[PlanDecision, CompiledWorkflow | None]:
        plan = await self.planner.create_plan(
            question=question,
            history=history or [],
            tools=self.tools,
            session_id=session_id,
        )
        if isinstance(plan, (ClarificationPlan, ChatPlan)):
            return plan, None
        workflow = self.planner.plan_to_workflow(plan, workflow_id=workflow_id)
        compiled = compile_workflow(workflow, tools=self.tools)
        return plan, compiled

    async def run(
        self,
        *,
        question: str,
        inputs: JsonValue,
        history: list[ConversationTurn] | None = None,
        session_id: str | None = None,
        workflow_id: str = "planned_workflow",
    ) -> RunResult:
        plan, compiled = await self.plan(
            question=question,
            history=history,
            session_id=session_id,
            workflow_id=workflow_id,
        )
        if isinstance(plan, ChatPlan):
            result = self._chat_result(plan=plan, session_id=session_id)
            return result.model_copy(update={"plan": plan})
        if isinstance(plan, ClarificationPlan):
            result = await self._planner_wait_result(
                plan=plan,
                question=question,
                inputs=inputs,
                history=history or [],
                session_id=session_id,
                workflow_id=workflow_id,
            )
            return result.model_copy(update={"plan": plan})
        if compiled is None:
            raise ValueError("compiled workflow missing for execution plan")
        result = await self.engine.run(
            compiled,
            inputs=inputs,
            session_id=session_id,
            history=[item.model_dump(mode="python") for item in (history or [])],
            question=question,
        )
        return result.model_copy(update={"plan": plan})

    async def resume(
        self,
        *,
        session_id: str,
        token: str,
        decision: str,
    ) -> RunResult:
        session = await self.engine.session_store.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        if not session.pending_clarifications:
            raise ResumeTokenError("session is not waiting for clarification")
        pending = next((item for item in session.pending_clarifications if item.token == token), None)
        if pending is None:
            raise ResumeTokenError("clarification token mismatch")
        if pending.options and decision not in pending.options:
            raise ResumeTokenError(f"invalid clarification decision: {decision}")

        if pending.source == "planner":
            if not session.question:
                raise ResumeTokenError("planner clarification cannot resume without original question")
            history_turns = [ConversationTurn.model_validate(item) for item in session.history]
            history_turns.append(ConversationTurn(role="user", content=decision))
            await self.engine.session_store.delete(session_id)
            return await self.run(
                question=session.question,
                inputs=session.inputs,
                history=history_turns,
                session_id=session_id,
                workflow_id=session.workflow_id,
            )

        compiled = self._compiled_from_session(session)
        return await self.engine.resume(
            compiled,
            session_id=session_id,
            token=token,
            decision=decision,
        )

    def _compiled_from_session(self, session: Session) -> CompiledWorkflow:
        snapshot = session.compiled_workflow
        if snapshot is None:
            raise ResumeTokenError("compiled workflow snapshot missing from session")
        workflow_data = json.loads(snapshot.canonical_json)
        workflow = Workflow.model_validate(workflow_data)
        compiled = compile_workflow(workflow, tools=self.tools)
        if compiled.digest != snapshot.digest:
            raise ResumeTokenError("compiled workflow digest mismatch")
        return compiled

    async def _planner_wait_result(
        self,
        *,
        plan: ClarificationPlan,
        question: str,
        inputs: JsonValue,
        history: list[ConversationTurn],
        session_id: str | None,
        workflow_id: str,
    ) -> RunResult:
        sid = session_id or str(uuid4())
        run_id = str(uuid4())
        pending = PendingClarification(
            source="planner",
            token=str(uuid4()),
            prompt=plan.clarification.prompt,
            options=list(plan.clarification.options),
            missing_fields=list(plan.clarification.missing_fields),
        )
        session = Session(
            id=sid,
            workflow_id=workflow_id,
            status="waiting",
            question=question,
            inputs=inputs,
            history=[item.model_dump(mode="python") for item in history][-12:],
            runtime_state=RuntimeState(pending=[]),
            pending_clarifications=[pending],
        )
        session = await self.engine.session_store.create(session)
        events: list[Event] = [
            ClarificationRequestedEvent(
                sequence=1,
                run_id=run_id,
                session_id=session.id,
                workflow_id=session.workflow_id,
                node_id="planner_clarification",
                source="planner",
                token=pending.token,
                prompt=pending.prompt,
                options=list(pending.options),
                missing_fields=list(pending.missing_fields),
            ),
            RunPausedEvent(
                sequence=2,
                run_id=run_id,
                session_id=session.id,
                workflow_id=session.workflow_id,
                node_id="planner_clarification",
                source="planner",
            ),
        ]
        return RunResult(
            status="waiting",
            run_id=run_id,
            session_id=session.id,
            pending_clarification=pending,
            events=events,
        )

    @staticmethod
    def _chat_result(*, plan: ChatPlan, session_id: str | None) -> RunResult:
        return RunResult(
            status="succeeded",
            run_id=str(uuid4()),
            session_id=session_id or str(uuid4()),
            output={"message": plan.response},
            events=[],
        )
