"""Planner contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from treelace.json_types import JsonValue


class PlanClarification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    prompt: str
    options: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)

    @field_validator("prompt")
    @classmethod
    def _prompt_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must be non-empty")
        return value.strip()


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    tool: str
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    deps: list[str] = Field(default_factory=list)
    rationale: str | None = None
    prompt_focus: str | None = None
    micro_prompt: str | None = None

    @field_validator("id", "tool")
    @classmethod
    def _validate_names(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()

    @field_validator("deps")
    @classmethod
    def _dedupe_deps(cls, deps: list[str]) -> list[str]:
        return list(dict.fromkeys([item.strip() for item in deps if item.strip()]))


class PlanSpec(BaseModel):
    """Execution plan decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["execution"] = "execution"
    version: str = "1.0"
    question: str
    intent_entities: dict[str, JsonValue] = Field(default_factory=dict)
    rationale: str | None = None
    steps: list[PlanStep] = Field(default_factory=list)
    result_step_id: str

    @field_validator("question", "result_step_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()

    @field_validator("steps")
    @classmethod
    def _steps_unique_ids(cls, steps: list[PlanStep]) -> list[PlanStep]:
        ids = [item.id for item in steps]
        if len(set(ids)) != len(ids):
            raise ValueError("step ids must be unique")
        return steps

    @model_validator(mode="after")
    def _validate_graph(self) -> "PlanSpec":
        if not self.steps:
            raise ValueError("execution plan requires at least one step")
        known = {step.id for step in self.steps}
        if self.result_step_id not in known:
            raise ValueError("result_step_id must reference a step id")
        for step in self.steps:
            if step.id in step.deps:
                raise ValueError(f"step {step.id} cannot depend on itself")
            for dep in step.deps:
                if dep not in known:
                    raise ValueError(f"step {step.id} depends on unknown step {dep}")
        return self


class ClarificationPlan(BaseModel):
    """Planner clarification decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["clarification"] = "clarification"
    version: str = "1.0"
    question: str
    rationale: str | None = None
    clarification: PlanClarification

    @field_validator("question")
    @classmethod
    def _question_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must be non-empty")
        return value.strip()


class ChatPlan(BaseModel):
    """Direct conversational response decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["chat"] = "chat"
    version: str = "1.0"
    question: str
    response: str

    @field_validator("question", "response")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()


PlanDecision = Annotated[PlanSpec | ClarificationPlan | ChatPlan, Field(discriminator="kind")]
PLAN_DECISION_ADAPTER = TypeAdapter(PlanDecision)
