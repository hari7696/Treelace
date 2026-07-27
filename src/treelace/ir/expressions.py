"""Restricted expression IR."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from treelace.json_types import JsonValue


class ConstExpr(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["const"] = "const"
    value: JsonValue = None


class InputRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["input_ref"] = "input_ref"
    pointer: str = "/"


class OutputRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["output_ref"] = "output_ref"
    node_id: str
    pointer: str = "/"


class CompareExpr(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["compare"] = "compare"
    op: Literal["eq", "ne", "gt", "gte", "lt", "lte", "contains"]
    left: "Expression"
    right: "Expression"


class FormatExpr(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["format"] = "format"
    template: str
    values: dict[str, "Expression"] = Field(default_factory=dict)


Expression = Annotated[
    ConstExpr | InputRef | OutputRef | CompareExpr | FormatExpr,
    Field(discriminator="kind"),
]

ConstExpr.model_rebuild()
InputRef.model_rebuild()
OutputRef.model_rebuild()
CompareExpr.model_rebuild()
FormatExpr.model_rebuild()
