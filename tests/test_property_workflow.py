from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from treelace.compiler.compile import compile_workflow
from treelace.ir.nodes import Sequence, ToolCall, Workflow
from treelace.tools.registry import ToolRegistry
from treelace.tools.result import ToolResult


def echo(value: str) -> ToolResult:
    return ToolResult(model_context={"value": value})


@given(length=st.integers(min_value=1, max_value=8))
def test_compiled_order_stays_topological_for_sequences(length: int) -> None:
    tools = ToolRegistry()
    tools.register_callable(echo)
    steps = [
        ToolCall(id=f"n{index}", tool="echo", arguments={"value": f"v{index}"})
        for index in range(length)
    ]
    workflow = Workflow(
        id="prop-sequence",
        root=Sequence(id="root", steps=steps),
        result_node_id=f"n{length - 1}",
    )
    compiled = compile_workflow(workflow, tools=tools)

    assert len(compiled.order) == length
    assert compiled.order[0] == "n0"
    assert compiled.order[-1] == f"n{length - 1}"
    for index in range(1, length):
        current = f"n{index}"
        assert compiled.order.index(current) > compiled.order.index(f"n{index - 1}")
