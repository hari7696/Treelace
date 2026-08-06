"""Workflow compiler and DAG lowering."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping

from treelace.compiler.compiled import CompiledOperation, CompiledWorkflow
from treelace.compiler.diagnostics import Diagnostic, error
from treelace.errors import CompileFailure
from treelace.ir.expressions import OutputRef
from treelace.ir.nodes import Clarification, Node, Parallel, Sequence, ToolCall, Workflow
from treelace.tools.registry import ToolRegistry


@dataclass(slots=True)
class _MutableOp:
    id: str
    kind: str
    payload: dict[str, Any]
    deps: set[str] = field(default_factory=set)


class _Builder:
    def __init__(
        self,
        *,
        tools: ToolRegistry | Mapping[str, Any] | None,
        diagnostics: list[Diagnostic],
    ) -> None:
        self._tools = tools if isinstance(tools, ToolRegistry) else ToolRegistry.from_mapping(tools or {})
        self._diagnostics = diagnostics
        self.operations: dict[str, _MutableOp] = {}
        self.insertion_order: list[str] = []

    def _add(self, op: _MutableOp) -> None:
        if op.id in self.operations:
            self._diagnostics.append(error("E001", f"duplicate node id: {op.id}", node_id=op.id))
            return
        self.operations[op.id] = op
        self.insertion_order.append(op.id)

    def lower(self, node: Node, deps: set[str]) -> set[str]:
        if isinstance(node, Sequence):
            current = set(deps)
            for step in node.steps:
                current = self.lower(step, current)
            return current
        if isinstance(node, Parallel):
            terminals: set[str] = set()
            for child in node.branches:
                terminals |= self.lower(child, set(deps))
            return terminals
        if isinstance(node, ToolCall):
            if self._tools.get(node.tool) is None:
                self._diagnostics.append(error("E201", f"unknown tool registration: {node.tool}", node_id=node.id))
            op = _MutableOp(
                id=node.id,
                kind="tool_call",
                payload=node.model_dump(mode="json"),
                deps=set(deps) | set(node.depends_on),
            )
            self._add(op)
            return {node.id}
        if isinstance(node, Clarification):
            op = _MutableOp(
                id=node.id,
                kind="clarification",
                payload=node.model_dump(mode="json"),
                deps=set(deps) | set(node.depends_on),
            )
            self._add(op)
            return {node.id}
        raise AssertionError(f"unsupported node type: {type(node).__name__}")

    def add_reference_dependencies(self) -> None:
        for op in self.operations.values():
            refs = _collect_output_refs(op.payload)
            for ref in refs:
                if ref not in self.operations:
                    self._diagnostics.append(
                        error("E301", f"output reference points to unknown node: {ref}", node_id=op.id)
                    )
                    continue
                op.deps.add(ref)


def _collect_output_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, OutputRef):
        refs.add(value.node_id)
        return refs
    if hasattr(value, "model_dump"):
        refs |= _collect_output_refs(value.model_dump(mode="python"))
        return refs
    if isinstance(value, dict):
        if value.get("kind") == "output_ref" and isinstance(value.get("node_id"), str):
            refs.add(value["node_id"])
        for item in value.values():
            refs |= _collect_output_refs(item)
        return refs
    if isinstance(value, list):
        for item in value:
            refs |= _collect_output_refs(item)
    return refs


def _topological_order(ops: dict[str, _MutableOp], diagnostics: list[Diagnostic]) -> list[str]:
    indegree = {op_id: 0 for op_id in ops}
    outgoing: dict[str, list[str]] = {op_id: [] for op_id in ops}
    for op in ops.values():
        for dep in op.deps:
            if dep not in ops:
                diagnostics.append(error("E302", f"dependency {dep} does not exist", node_id=op.id))
                continue
            indegree[op.id] += 1
            outgoing[dep].append(op.id)
    ready = deque(sorted([op_id for op_id, degree in indegree.items() if degree == 0]))
    order: list[str] = []
    while ready:
        current = ready.popleft()
        order.append(current)
        for nxt in sorted(outgoing[current]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    if len(order) != len(ops):
        diagnostics.append(error("E400", "workflow dependencies contain a cycle"))
    return order


def compile_workflow(
    workflow: Workflow | dict[str, Any],
    *,
    tools: ToolRegistry | Mapping[str, Any] | None = None,
) -> CompiledWorkflow:
    wf = workflow if isinstance(workflow, Workflow) else Workflow.model_validate(workflow)
    diagnostics: list[Diagnostic] = []
    builder = _Builder(tools=tools, diagnostics=diagnostics)
    builder.lower(wf.root, set())
    builder.add_reference_dependencies()
    order = _topological_order(builder.operations, diagnostics)

    final_node_id: str | None = None
    if wf.result_node_id is not None:
        if wf.result_node_id not in builder.operations:
            diagnostics.append(
                error("E500", f"result node {wf.result_node_id} does not exist in compiled operations")
            )
        else:
            final_node_id = wf.result_node_id
    else:
        outgoing_count = {op_id: 0 for op_id in builder.operations}
        for op in builder.operations.values():
            for dep in op.deps:
                if dep in outgoing_count:
                    outgoing_count[dep] += 1
        terminals = sorted([op_id for op_id, count in outgoing_count.items() if count == 0])
        if len(terminals) == 1:
            final_node_id = terminals[0]
        elif len(terminals) > 1:
            diagnostics.append(
                error(
                    "E401",
                    f"workflow has multiple terminal nodes; set workflow.result_node_id explicitly ({', '.join(terminals)})",
                )
            )

    canonical_data = wf.model_dump(mode="json")
    canonical_json = json.dumps(canonical_data, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    if any(item.severity == "error" for item in diagnostics):
        raise CompileFailure("workflow compilation failed", diagnostics=diagnostics)

    operations = {
        op_id: CompiledOperation(
            id=mutable.id,
            kind=mutable.kind,  # type: ignore[arg-type]
            payload=mutable.payload,
            deps=sorted(mutable.deps),
        )
        for op_id, mutable in builder.operations.items()
    }

    return CompiledWorkflow(
        workflow_id=wf.id,
        workflow=canonical_data,
        operations=operations,
        order=order,
        final_node_id=final_node_id,
        canonical_json=canonical_json,
        digest=digest,
        diagnostics=diagnostics,
    )
