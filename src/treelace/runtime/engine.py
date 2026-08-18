"""Dependency scheduler runtime engine."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from treelace.compiler.compiled import CompiledOperation, CompiledWorkflow
from treelace.errors import NodeExecutionError, ResumeTokenError, SessionNotFoundError
from treelace.ir.eval import EvalContext, MissingValueError, eval_expression
from treelace.ir.nodes import Clarification, ToolCall
from treelace.json_types import JsonValue
from treelace.models.protocol import Model
from treelace.runtime.events import (
    ClarificationRequestedEvent,
    Event,
    NodeFailedEvent,
    NodeStartedEvent,
    NodeSucceededEvent,
    RunFailedEvent,
    RunPausedEvent,
    RunStartedEvent,
    RunSucceededEvent,
)
from treelace.runtime.result import RunResult
from treelace.sessions.artifacts import InMemoryArtifactStore
from treelace.sessions.memory import InMemorySessionStore
from treelace.sessions.models import (
    CompiledWorkflowSnapshot,
    PendingClarification,
    RuntimeState,
    Session,
)
from treelace.sessions.protocols import ArtifactStoreProtocol, SessionStoreProtocol
from treelace.tools.context import ToolContext
from treelace.tools.registry import ToolRegistry


@dataclass(slots=True)
class _PauseSignal:
    node_id: str
    prompt: str
    options: list[str]
    missing_fields: list[str]


class Engine:
    """Executes compiled workflows with dependency-aware parallelism."""

    def __init__(
        self,
        *,
        tools: ToolRegistry | None = None,
        session_store: SessionStoreProtocol | None = None,
        artifact_store: ArtifactStoreProtocol | None = None,
        max_concurrency: int = 8,
        fail_fast: bool = True,
        cleanup_artifacts_on_terminal: bool = False,
        model: Model | None = None,
    ) -> None:
        self.tools = tools or ToolRegistry()
        self.session_store = session_store or InMemorySessionStore()
        self.artifact_store = artifact_store or InMemoryArtifactStore()
        self.max_concurrency = max_concurrency
        self.fail_fast = fail_fast
        self.cleanup_artifacts_on_terminal = cleanup_artifacts_on_terminal
        self.model = model

    async def run(
        self,
        plan: CompiledWorkflow,
        *,
        inputs: JsonValue,
        session_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        question: str | None = None,
    ) -> RunResult:
        sid = session_id or str(uuid4())
        session = Session(
            id=sid,
            workflow_id=plan.workflow_id,
            workflow_digest=plan.digest,
            status="running",
            question=question,
            inputs=inputs,
            history=list(history or [])[-12:],
            compiled_workflow=CompiledWorkflowSnapshot(
                canonical_json=plan.canonical_json,
                digest=plan.digest,
            ),
            runtime_state=RuntimeState(pending=list(plan.order)),
        )
        session = await self.session_store.create(session)
        return await self._continue(plan=plan, session=session, run_id=str(uuid4()))

    async def resume(
        self,
        plan: CompiledWorkflow,
        *,
        session_id: str,
        token: str,
        decision: str,
    ) -> RunResult:
        session = await self.session_store.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        if session.workflow_digest != plan.digest:
            raise ResumeTokenError("workflow digest mismatch")
        if not session.pending_clarifications:
            raise ResumeTokenError("session is not waiting for clarification")
        pending_index = next(
            (index for index, item in enumerate(session.pending_clarifications) if item.token == token),
            None,
        )
        if pending_index is None:
            raise ResumeTokenError("clarification token mismatch")
        pending = session.pending_clarifications[pending_index]
        if pending.source != "tool":
            raise ResumeTokenError("planner clarification must be resumed through Runner.resume")
        if pending.options and decision not in pending.options:
            raise ResumeTokenError(f"invalid clarification decision: {decision}")
        if pending.node_id is None:
            raise ResumeTokenError("clarification node id is missing")

        state = session.runtime_state
        waiting_node = pending.node_id
        state.completed_status[waiting_node] = "succeeded"
        state.outputs[waiting_node] = {
            "model_context": {"decision": decision},
            "artifact_refs": [],
        }
        if waiting_node in state.pending:
            state.pending.remove(waiting_node)
        remaining = [
            item
            for index, item in enumerate(session.pending_clarifications)
            if index != pending_index
        ]
        if remaining:
            session.pending_clarifications = remaining
            session.status = "waiting"
            session.runtime_state = RuntimeState(
                completed_status=state.completed_status,
                outputs=state.outputs,
                pending=state.pending,
            )
            session.updated_at = datetime.now(UTC)
            session = await self.session_store.save(session, expected_revision=session.revision)
            run_id = str(uuid4())
            events: list[Event] = [
                ClarificationRequestedEvent(
                    sequence=1,
                    run_id=run_id,
                    session_id=session.id,
                    workflow_id=session.workflow_id,
                    node_id=remaining[0].node_id or "clarification",
                    source="tool",
                    token=remaining[0].token,
                    prompt=remaining[0].prompt,
                    options=list(remaining[0].options),
                    missing_fields=list(remaining[0].missing_fields),
                ),
                RunPausedEvent(
                    sequence=2,
                    run_id=run_id,
                    session_id=session.id,
                    workflow_id=session.workflow_id,
                    source="tool",
                    node_id=remaining[0].node_id,
                ),
            ]
            return RunResult(
                status="waiting",
                run_id=run_id,
                session_id=session.id,
                pending_clarification=remaining[0],
                events=events,
            )
        session.pending_clarifications = []
        session.status = "running"
        session.updated_at = datetime.now(UTC)
        session = await self.session_store.save(session, expected_revision=session.revision)
        return await self._continue(plan=plan, session=session, run_id=str(uuid4()))

    async def _continue(self, *, plan: CompiledWorkflow, session: Session, run_id: str) -> RunResult:
        events: list[Event] = []
        sequence = 0

        def emit(
            kind: str,
            *,
            node_id: str | None = None,
            output: dict[str, JsonValue] | None = None,
            error: str | None = None,
            output_node: str | None = None,
            clarification: PendingClarification | None = None,
            source: str | None = None,
        ) -> None:
            nonlocal sequence
            sequence += 1
            if kind == "run_started":
                events.append(
                    RunStartedEvent(
                        sequence=sequence,
                        run_id=run_id,
                        session_id=session.id,
                        workflow_id=session.workflow_id,
                    )
                )
                return
            if kind == "node_started":
                if node_id is None:
                    raise ValueError("node_started requires node_id")
                events.append(
                    NodeStartedEvent(
                        sequence=sequence,
                        run_id=run_id,
                        session_id=session.id,
                        workflow_id=session.workflow_id,
                        node_id=node_id,
                    )
                )
                return
            if kind == "node_succeeded":
                if node_id is None or output is None:
                    raise ValueError("node_succeeded requires node_id and output")
                events.append(
                    NodeSucceededEvent(
                        sequence=sequence,
                        run_id=run_id,
                        session_id=session.id,
                        workflow_id=session.workflow_id,
                        node_id=node_id,
                        output=output,
                    )
                )
                return
            if kind == "node_failed":
                if node_id is None or error is None:
                    raise ValueError("node_failed requires node_id and error")
                events.append(
                    NodeFailedEvent(
                        sequence=sequence,
                        run_id=run_id,
                        session_id=session.id,
                        workflow_id=session.workflow_id,
                        node_id=node_id,
                        error=error,
                    )
                )
                return
            if kind == "clarification_requested":
                if clarification is None or source is None:
                    raise ValueError("clarification_requested requires clarification and source")
                events.append(
                    ClarificationRequestedEvent(
                        sequence=sequence,
                        run_id=run_id,
                        session_id=session.id,
                        workflow_id=session.workflow_id,
                        node_id=clarification.node_id or "clarification",
                        source="tool" if source == "tool" else "planner",
                        token=clarification.token,
                        prompt=clarification.prompt,
                        options=list(clarification.options),
                        missing_fields=list(clarification.missing_fields),
                    )
                )
                return
            if kind == "run_paused":
                if source is None:
                    raise ValueError("run_paused requires source")
                events.append(
                    RunPausedEvent(
                        sequence=sequence,
                        run_id=run_id,
                        session_id=session.id,
                        workflow_id=session.workflow_id,
                        node_id=node_id,
                        source="tool" if source == "tool" else "planner",
                    )
                )
                return
            if kind == "run_failed":
                if error is None:
                    raise ValueError("run_failed requires error")
                events.append(
                    RunFailedEvent(
                        sequence=sequence,
                        run_id=run_id,
                        session_id=session.id,
                        workflow_id=session.workflow_id,
                        error=error,
                    )
                )
                return
            if kind == "run_succeeded":
                events.append(
                    RunSucceededEvent(
                        sequence=sequence,
                        run_id=run_id,
                        session_id=session.id,
                        workflow_id=session.workflow_id,
                        output_node=output_node,
                    )
                )
                return
            raise ValueError(f"unsupported event kind: {kind}")

        if not session.runtime_state.completed_status:
            emit("run_started")

        completed = dict(session.runtime_state.completed_status)
        outputs = dict(session.runtime_state.outputs)
        pending = list(session.runtime_state.pending)
        node_artifacts = dict(session.node_artifacts)

        while pending:
            ready = [
                op_id
                for op_id in pending
                if all(completed.get(dep) == "succeeded" for dep in plan.operations[op_id].deps)
            ]
            if not ready:
                error_text = "no ready operations; dependency deadlock or upstream failure"
                emit("run_failed", error=error_text)
                session.status = "failed"
                session.runtime_state = RuntimeState(
                    completed_status=completed,
                    outputs=outputs,
                    pending=pending,
                )
                session.node_artifacts = node_artifacts
                session.updated_at = datetime.now(UTC)
                session = await self.session_store.save(session, expected_revision=session.revision)
                if self.cleanup_artifacts_on_terminal:
                    self.artifact_store.delete_session(session_id=session.id)
                return RunResult(
                    status="failed",
                    run_id=run_id,
                    session_id=session.id,
                    error=error_text,
                    events=events,
                )

            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def execute(op_id: str) -> tuple[str, dict[str, JsonValue] | _PauseSignal]:
                op = plan.operations[op_id]
                emit("node_started", node_id=op_id)
                async with semaphore:
                    result = await self._execute_operation(op, session=session, run_id=run_id, outputs=outputs)
                return op_id, result

            wave_results = await asyncio.gather(*[execute(op_id) for op_id in ready], return_exceptions=True)

            pause_signals: list[_PauseSignal] = []
            for result in wave_results:
                if isinstance(result, BaseException):
                    reason = str(result)
                    node_id = result.node_id if isinstance(result, NodeExecutionError) else "unknown"
                    emit("node_failed", node_id=node_id, error=reason)
                    completed[node_id] = "failed"
                    if self.fail_fast:
                        emit("run_failed", error=reason)
                        session.status = "failed"
                        session.runtime_state = RuntimeState(
                            completed_status=completed,
                            outputs=outputs,
                            pending=pending,
                        )
                        session.node_artifacts = node_artifacts
                        session.updated_at = datetime.now(UTC)
                        session = await self.session_store.save(session, expected_revision=session.revision)
                        if self.cleanup_artifacts_on_terminal:
                            self.artifact_store.delete_session(session_id=session.id)
                        return RunResult(
                            status="failed",
                            run_id=run_id,
                            session_id=session.id,
                            error=reason,
                            events=events,
                        )
                    continue

                op_id, payload = result
                if isinstance(payload, _PauseSignal):
                    pause_signals.append(payload)
                    continue

                if op_id in pending:
                    pending.remove(op_id)
                completed[op_id] = "succeeded"
                outputs[op_id] = payload
                refs = payload.get("artifact_refs")
                if isinstance(refs, list):
                    node_artifacts[op_id] = [str(item) for item in refs]
                emit("node_succeeded", node_id=op_id, output=payload)

            if pause_signals:
                ordered_pauses = sorted(pause_signals, key=lambda item: item.node_id)
                pending_clarifications = [
                    PendingClarification(
                        source="tool",
                        node_id=signal.node_id,
                        token=str(uuid4()),
                        prompt=signal.prompt,
                        options=list(signal.options),
                        missing_fields=list(signal.missing_fields),
                    )
                    for signal in ordered_pauses
                ]
                session.pending_clarifications = pending_clarifications
                session.status = "waiting"
                session.runtime_state = RuntimeState(
                    completed_status=completed,
                    outputs=outputs,
                    pending=pending,
                )
                session.node_artifacts = node_artifacts
                session.updated_at = datetime.now(UTC)
                session = await self.session_store.save(session, expected_revision=session.revision)
                for clarification in pending_clarifications:
                    emit(
                        "clarification_requested",
                        node_id=clarification.node_id,
                        clarification=clarification,
                        source="tool",
                    )
                first_pending = pending_clarifications[0]
                emit("run_paused", node_id=first_pending.node_id, source="tool")
                return RunResult(
                    status="waiting",
                    run_id=run_id,
                    session_id=session.id,
                    pending_clarification=first_pending,
                    events=events,
                )

            session.runtime_state = RuntimeState(
                completed_status=completed,
                outputs=outputs,
                pending=pending,
            )
            session.node_artifacts = node_artifacts
            session.updated_at = datetime.now(UTC)
            session = await self.session_store.save(session, expected_revision=session.revision)

        final_node = plan.final_node_id or (plan.order[-1] if plan.order else None)
        final_output = outputs.get(final_node) if final_node is not None else None
        session.status = "succeeded"
        session.output = final_output
        session.runtime_state = RuntimeState(
            completed_status=completed,
            outputs=outputs,
            pending=[],
        )
        session.node_artifacts = node_artifacts
        session.updated_at = datetime.now(UTC)
        session = await self.session_store.save(session, expected_revision=session.revision)
        emit("run_succeeded", output_node=final_node)
        if self.cleanup_artifacts_on_terminal:
            self.artifact_store.delete_session(session_id=session.id)
        return RunResult(
            status="succeeded",
            run_id=run_id,
            session_id=session.id,
            output=final_output,
            events=events,
        )

    async def _execute_operation(
        self,
        op: CompiledOperation,
        *,
        session: Session,
        run_id: str,
        outputs: dict[str, JsonValue],
    ) -> dict[str, JsonValue] | _PauseSignal:
        ctx = EvalContext(inputs=session.inputs, outputs=outputs)
        try:
            if op.kind == "tool_call":
                tool_node = ToolCall.model_validate(op.payload)
                args = {
                    key: eval_expression(value, ctx)
                    for key, value in tool_node.arguments.items()
                }
                tool_context = ToolContext(
                    run_id=run_id,
                    session_id=session.id,
                    workflow_id=session.workflow_id,
                    node_id=tool_node.id,
                    micro_prompt=tool_node.micro_prompt,
                    artifact_store=self.artifact_store,
                    inputs=session.inputs,
                    model=self.model,
                    question=session.question,
                )
                result = await self.tools.invoke(tool_node.tool, args, context=tool_context)
                if result.clarification_request is not None:
                    request = result.clarification_request
                    return _PauseSignal(
                        node_id=tool_node.id,
                        prompt=request.prompt,
                        options=list(request.options),
                        missing_fields=list(request.missing_fields),
                    )
                return result.model_dump(mode="json", by_alias=True)
            if op.kind == "clarification":
                clarification_node = Clarification.model_validate(op.payload)
                prompt = (
                    eval_expression(clarification_node.prompt, ctx)
                    if not isinstance(clarification_node.prompt, str)
                    else clarification_node.prompt
                )
                return _PauseSignal(
                    node_id=clarification_node.id,
                    prompt=str(prompt),
                    options=list(clarification_node.options),
                    missing_fields=list(clarification_node.missing_fields),
                )
        except MissingValueError as exc:
            raise NodeExecutionError(op.id, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise NodeExecutionError(op.id, str(exc)) from exc
        raise NodeExecutionError(op.id, f"unsupported operation kind: {op.kind}")
