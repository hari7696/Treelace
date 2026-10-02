from __future__ import annotations

import pytest

from treelace.compiler.compile import compile_workflow
from treelace.errors import CompileFailure
from treelace.ir.expressions import OutputRef
from treelace.ir.nodes import Parallel, ToolCall, Workflow
from treelace.tools.registry import ToolRegistry
from treelace.tools.result import ToolResult


def echo(value: str) -> ToolResult:
    """Echo a value.

    Args:
        value: Arbitrary value.
    """

    return ToolResult(model_context={"value": value})


def merge(left: dict[str, object], right: dict[str, object]) -> ToolResult:
    """Merge two payloads.

    Args:
        left: Left payload.
        right: Right payload.
    """

    return ToolResult(model_context={"left": left, "right": right})


def test_compiler_infers_reference_dependencies() -> None:
    tools = ToolRegistry()
    tools.register_callable(echo)
    tools.register_callable(merge)
    workflow = Workflow(
        id="deps",
        root=Parallel(
            id="root",
            branches=[
                ToolCall(id="left", tool="echo", arguments={"value": "L"}),
                ToolCall(id="right", tool="echo", arguments={"value": "R"}),
                ToolCall(
                    id="join",
                    tool="merge",
                    arguments={
                        "left": OutputRef(node_id="left", pointer="/model_context"),
                        "right": OutputRef(node_id="right", pointer="/model_context"),
                    },
                ),
            ],
        ),
    )
    compiled = compile_workflow(workflow, tools=tools)
    assert set(compiled.operations["join"].deps) == {"left", "right"}
    assert compiled.order.index("left") < compiled.order.index("join")
    assert compiled.order.index("right") < compiled.order.index("join")
    assert compiled.final_node_id == "join"


def test_compiler_rejects_cycles() -> None:
    tools = ToolRegistry()
    tools.register_callable(echo)
    workflow = Workflow(
        id="cycle",
        root=Parallel(
            id="root",
            branches=[
                ToolCall(id="a", tool="echo", arguments={"value": "A"}, depends_on=["b"]),
                ToolCall(id="b", tool="echo", arguments={"value": "B"}, depends_on=["a"]),
            ],
        ),
    )
    with pytest.raises(CompileFailure):
        compile_workflow(workflow, tools=tools)


def test_compiler_rejects_ambiguous_terminal_nodes_without_explicit_result() -> None:
    tools = ToolRegistry()
    tools.register_callable(echo)
    workflow = Workflow(
        id="ambiguous",
        root=Parallel(
            id="root",
            branches=[
                ToolCall(id="left", tool="echo", arguments={"value": "L"}),
                ToolCall(id="right", tool="echo", arguments={"value": "R"}),
            ],
        ),
    )
    with pytest.raises(CompileFailure):
        compile_workflow(workflow, tools=tools)


def test_compiler_honors_explicit_result_node() -> None:
    tools = ToolRegistry()
    tools.register_callable(echo)
    workflow = Workflow(
        id="explicit-final",
        root=Parallel(
            id="root",
            branches=[
                ToolCall(id="left", tool="echo", arguments={"value": "L"}),
                ToolCall(id="right", tool="echo", arguments={"value": "R"}),
            ],
        ),
        result_node_id="right",
    )
    compiled = compile_workflow(workflow, tools=tools)
    assert compiled.final_node_id == "right"
