"""Planner orchestration and lowering into workflow IR."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from treelace.ir.nodes import Clarification, Node, Parallel, ToolCall, Workflow
from treelace.models.protocol import (
    Message,
    Model,
    ModelContext,
    ModelRequest,
    StructuredOutput,
    TextOutput,
)
from treelace.planner.models import (
    PLAN_DECISION_ADAPTER,
    ChatPlan,
    ClarificationPlan,
    PlanDecision,
    PlanSpec,
    PlanStep,
)
from treelace.planner.prompts import ConversationTurn, MasterPromptManager
from treelace.tools.registry import ToolRegistry

_STEP_REF_SHORTHAND = re.compile(r"^\$([A-Za-z_][A-Za-z0-9_-]*)$")


def _coerce_expression_or_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _coerce_expression_or_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_coerce_expression_or_value(item) for item in value]
    return value


def _normalize_argument_aliases(value: Any) -> Any:
    if isinstance(value, str):
        # Accept planner shorthand like "$forecast_2026_01" as a step output reference.
        match = _STEP_REF_SHORTHAND.fullmatch(value.strip())
        if match is not None:
            return {
                "kind": "output_ref",
                "node_id": match.group(1),
                "pointer": "/",
            }
        return value
    if isinstance(value, dict):
        if value.get("kind") == "output_ref":
            node_id = value.get("node_id")
            if not isinstance(node_id, str):
                for alias in ("step_id", "output_ref", "step", "ref"):
                    alias_value = value.get(alias)
                    if isinstance(alias_value, str):
                        node_id = alias_value
                        break
            if isinstance(node_id, str):
                pointer = value.get("pointer")
                return {
                    "kind": "output_ref",
                    "node_id": node_id,
                    "pointer": pointer if isinstance(pointer, str) else "/",
                }
        step_ref = value.get("$step")
        if isinstance(step_ref, str):
            pointer = value.get("pointer")
            return {
                "kind": "output_ref",
                "node_id": step_ref,
                "pointer": pointer if isinstance(pointer, str) else "/",
            }
        return {key: _normalize_argument_aliases(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_argument_aliases(item) for item in value]
    return value


def _collect_output_ref_ids(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        if value.get("kind") == "output_ref" and isinstance(value.get("node_id"), str):
            refs.add(value["node_id"])
        for item in value.values():
            refs |= _collect_output_ref_ids(item)
        return refs
    if isinstance(value, list):
        for item in value:
            refs |= _collect_output_ref_ids(item)
    return refs


def _normalize_step_payload(raw_step: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if isinstance(raw_step.get("id"), str):
        normalized["id"] = raw_step["id"]
    elif isinstance(raw_step.get("step_id"), str):
        normalized["id"] = raw_step["step_id"]
    elif isinstance(raw_step.get("tool"), str):
        # A single tool call often omits a separate step id.
        normalized["id"] = raw_step["tool"]
    elif isinstance(raw_step.get("name"), str):
        normalized["id"] = raw_step["name"]

    if isinstance(raw_step.get("tool"), str):
        normalized["tool"] = raw_step["tool"]
    elif isinstance(raw_step.get("name"), str):
        normalized["tool"] = raw_step["name"]
    elif isinstance(raw_step.get("id"), str):
        # Some models use the tool name as the tool-call id and omit `tool`.
        normalized["tool"] = raw_step["id"]

    arguments = raw_step.get("arguments")
    if not isinstance(arguments, dict):
        arguments = raw_step.get("args")
    if not isinstance(arguments, dict):
        arguments = raw_step.get("inputs")
    normalized_arguments = _normalize_argument_aliases(arguments if isinstance(arguments, dict) else {})
    normalized["arguments"] = normalized_arguments

    deps = raw_step.get("deps")
    if not isinstance(deps, list):
        deps = raw_step.get("depends_on")
    if not isinstance(deps, list):
        deps = raw_step.get("depends")
    normalized_deps = [item for item in deps if isinstance(item, str)] if isinstance(deps, list) else []
    inferred_deps = sorted(_collect_output_ref_ids(normalized_arguments))
    step_id = normalized.get("id")
    for dep in inferred_deps:
        if dep == step_id:
            continue
        if dep not in normalized_deps:
            normalized_deps.append(dep)
    normalized["deps"] = normalized_deps

    rationale = raw_step.get("rationale")
    if isinstance(rationale, str):
        normalized["rationale"] = rationale

    prompt_focus = raw_step.get("prompt_focus")
    if not isinstance(prompt_focus, str):
        prompt_focus = raw_step.get("focus")
    if isinstance(prompt_focus, str):
        normalized["prompt_focus"] = prompt_focus

    micro_prompt = raw_step.get("micro_prompt")
    if isinstance(micro_prompt, str):
        normalized["micro_prompt"] = micro_prompt
    return normalized


def _repair_plan_payload(payload: dict[str, Any], *, fallback_question: str) -> dict[str, Any]:
    repaired = dict(payload)
    if "question" not in repaired or not isinstance(repaired["question"], str):
        repaired["question"] = fallback_question

    if "kind" not in repaired:
        response_like = repaired.get("response")
        if isinstance(response_like, str) and response_like.strip():
            repaired["kind"] = "chat"
        if "clarification" in repaired and not repaired.get("steps") and not repaired.get("plan"):
            repaired["kind"] = "clarification"
        elif "kind" not in repaired:
            repaired["kind"] = "execution"

    if repaired.get("kind") == "execution":
        raw_steps = repaired.get("steps")
        if not isinstance(raw_steps, list):
            raw_steps = repaired.get("plan")
        if not isinstance(raw_steps, list):
            raw_steps = repaired.get("tool_calls")
        if isinstance(raw_steps, list):
            repaired["steps"] = [_normalize_step_payload(item) for item in raw_steps if isinstance(item, dict)]
        repaired.pop("plan", None)
        repaired.pop("tool_calls", None)
        if "result_step_id" not in repaired:
            result_step_id = repaired.get("final_step_id")
            if not isinstance(result_step_id, str):
                result_step_id = repaired.get("final_step")
            if not isinstance(result_step_id, str) and isinstance(repaired.get("steps"), list) and repaired["steps"]:
                last = repaired["steps"][-1]
                if isinstance(last, dict):
                    candidate = last.get("id")
                    if isinstance(candidate, str):
                        result_step_id = candidate
            if isinstance(result_step_id, str):
                repaired["result_step_id"] = result_step_id
        repaired.pop("final_step_id", None)
        repaired.pop("final_step", None)
        repaired.pop("clarification", None)
        repaired.pop("response", None)
    elif repaired.get("kind") == "clarification":
        clarification = repaired.get("clarification")
        if isinstance(clarification, str):
            repaired["clarification"] = {"prompt": clarification}
        elif clarification is None and isinstance(repaired.get("question"), str):
            # Some models return conversational/clarifying text in `question` without
            # the nested clarification object. Treat this as a direct chat response.
            repaired["kind"] = "chat"
            repaired["response"] = repaired["question"]
            repaired["question"] = fallback_question
        repaired.pop("steps", None)
        repaired.pop("plan", None)
        repaired.pop("result_step_id", None)
        repaired.pop("final_step_id", None)
        repaired.pop("final_step", None)
    elif repaired.get("kind") == "chat":
        response = repaired.get("response")
        if not isinstance(response, str):
            message = repaired.get("message")
            if isinstance(message, str):
                repaired["response"] = message
            elif isinstance(repaired.get("question"), str):
                repaired["response"] = repaired["question"]
                repaired["question"] = fallback_question
        repaired.pop("message", None)
        repaired.pop("steps", None)
        repaired.pop("plan", None)
        repaired.pop("result_step_id", None)
        repaired.pop("final_step_id", None)
        repaired.pop("final_step", None)
        repaired.pop("clarification", None)
    return repaired


class PlannerOrchestrator:
    def __init__(
        self,
        *,
        model: Model,
        prompt_manager: MasterPromptManager | None = None,
    ) -> None:
        self.model = model
        self.prompt_manager = prompt_manager or MasterPromptManager()

    async def create_plan(
        self,
        *,
        question: str,
        history: list[ConversationTurn],
        tools: ToolRegistry,
        session_id: str | None = None,
    ) -> PlanDecision:
        master_prompt = self.prompt_manager.build_master_prompt(history=history, tools=tools)
        request = ModelRequest(
            messages=[
                Message(role="system", content=master_prompt),
                Message(role="user", content=question),
            ],
            output_schema=PLAN_DECISION_ADAPTER.json_schema(),
        )
        response = await self.model.complete(
            request,
            context=ModelContext(session_id=session_id),
        )
        plan = self._coerce_plan(response=response, fallback_question=question)
        if isinstance(plan, (ClarificationPlan, ChatPlan)):
            return plan
        steps: list[PlanStep] = []
        for step in plan.steps:
            if step.micro_prompt is None:
                rendered = self.prompt_manager.render_tool_micro_prompt(
                    master_prompt=master_prompt,
                    step=step,
                )
                steps.append(step.model_copy(update={"micro_prompt": rendered}))
            else:
                steps.append(step)
        return plan.model_copy(update={"steps": steps})

    @staticmethod
    def _coerce_plan(*, response: Any, fallback_question: str) -> PlanDecision:
        if isinstance(response.output, StructuredOutput):
            payload = response.output.data
        elif isinstance(response.output, TextOutput):
            payload = json.loads(response.output.text)
        else:
            raise ValueError("unsupported model response output")
        if not isinstance(payload, dict):
            raise ValueError("plan payload must be a JSON object")

        try:
            return PLAN_DECISION_ADAPTER.validate_python(payload)
        except ValidationError as first_error:
            repaired = _repair_plan_payload(payload, fallback_question=fallback_question)
            try:
                return PLAN_DECISION_ADAPTER.validate_python(repaired)
            except ValidationError as second_error:
                raise ValueError(
                    "planner output failed schema validation after one repair attempt: "
                    f"first={first_error.errors()} second={second_error.errors()}"
                ) from second_error

    @staticmethod
    def plan_to_workflow(plan: PlanSpec, *, workflow_id: str = "planned_workflow") -> Workflow:
        tool_nodes: list[Node] = []
        for step in plan.steps:
            arguments = {
                key: _coerce_expression_or_value(value)
                for key, value in step.arguments.items()
            }
            tool_nodes.append(
                ToolCall(
                    id=step.id,
                    tool=step.tool,
                    arguments=arguments,
                    depends_on=list(step.deps),
                    micro_prompt=step.micro_prompt,
                )
            )

        root = Parallel(id="planned_parallel", branches=tool_nodes)
        return Workflow(
            id=workflow_id,
            description=plan.rationale,
            root=root,
            result_node_id=plan.result_step_id,
        )

    @staticmethod
    def clarification_to_workflow(plan: ClarificationPlan, *, workflow_id: str = "planned_workflow") -> Workflow:
        return Workflow(
            id=workflow_id,
            description=plan.rationale,
            root=Clarification(
                id="planner_clarification",
                prompt=plan.clarification.prompt,
                options=list(plan.clarification.options),
                missing_fields=list(plan.clarification.missing_fields),
            ),
            result_node_id="planner_clarification",
        )
