"""Prompt management for planning and tool micro-prompts."""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict

from treelace.planner.models import PlanStep
from treelace.tools.registry import ToolRegistry


class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    role: str
    content: str


DEFAULT_PLANNER_BASE_POLICY = (
    "You are the Treelace planner. Return strictly valid JSON matching the schema. "
    "Use registered tools only. Keep arguments compact and explicit. "
    "If the user intent is conversational and no tool execution is required, return kind='chat'. "
    "When information is missing for tool execution, return kind='clarification' with a clear prompt. "
    "Return kind='execution' only when tool planning is actually needed. "
    "For execution, use 'steps'; every step must include a unique 'id' and 'tool', and set "
    "'result_step_id'. Do not use 'tool_calls'. For chat, use 'question' and 'response', not 'message'."
)


class MasterPromptManager:
    def __init__(
        self,
        *,
        master_prompt: str | None = None,
        history_limit: int = 12,
        planner_base_policy: str | None = None,
    ) -> None:
        self.master_prompt = master_prompt.strip() if isinstance(master_prompt, str) and master_prompt.strip() else None
        self.planner_base_policy = planner_base_policy or DEFAULT_PLANNER_BASE_POLICY
        self.history_limit = history_limit

    def _history_block(self, history: list[ConversationTurn]) -> str:
        lines: list[str] = []
        for turn in history[-self.history_limit :]:
            role = turn.role.strip().lower()
            lines.append(f"{role}: {turn.content.strip()}")
        if not lines:
            return "(no prior history)"
        return "\n".join(lines)

    def build_master_prompt(self, *, history: list[ConversationTurn], tools: ToolRegistry) -> str:
        tool_specs = [item.model_dump(mode="json") for item in tools.list_specs()]
        tools_block = json.dumps(tool_specs, indent=2, sort_keys=True) if tool_specs else "[]"
        history_block = self._history_block(history)
        app_prompt_block = self.master_prompt or "(none provided)"
        return (
            "Planner base policy:\n"
            f"{self.planner_base_policy}\n\n"
            "Application master prompt:\n"
            f"{app_prompt_block}\n\n"
            "Registered tools (JSON array of tool specs):\n"
            f"{tools_block}\n\n"
            "Recent conversation context (last 12 messages max):\n"
            f"{history_block}\n\n"
            "Return only JSON that validates against the requested schema."
        )

    def render_tool_micro_prompt(self, *, master_prompt: str, step: PlanStep) -> str:
        focus = step.prompt_focus or step.rationale or "Execute this step safely and minimally."
        app_context = self.master_prompt or "(none provided)"
        return (
            f"Tool step: {step.id}\n"
            f"Tool: {step.tool}\n"
            f"Application context: {app_context}\n"
            f"Focus: {focus}\n"
            "Return compact model_context. Use artifact_refs only when explicit artifacts are produced."
        )
