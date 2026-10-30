# Concepts

This page explains the Treelace execution model in the same order the framework
actually runs.

## Core idea

Treelace is a one-shot planner/executor for typed Python tools:

- A question is planned once.
- The plan is lowered into workflow IR.
- The workflow is compiled into a dependency DAG.
- Ready nodes run in parallel waves until a final result exists.

There are no autonomous loops or hidden background agents.

## Run lifecycle

High-level flow:

1. `Treelace.run(...)` delegates to `Runner.run(...)`.
2. `PlannerOrchestrator.create_plan(...)` returns a `PlanDecision`.
3. If the plan is executable, `compile_workflow(...)` builds a
   `CompiledWorkflow`.
4. `Engine.run(...)` executes dependency-safe waves.
5. A low-level `RunResult` is converted into public `Run`.

`PlanDecision` can be one of:

- `execution` (`PlanSpec`): normal tool workflow.
- `clarification` (`ClarificationPlan`): model needs user input first.
- `chat` (`ChatPlan`): direct response without tool execution.

## Graph and dependency model

Treelace supports explicit dependency edges in two ways:

- `deps` declared on steps/nodes.
- `output_ref` references inside arguments (auto-added as dependencies).

When all dependencies for a node are complete, the node becomes ready. Nodes in
the same wave run concurrently (subject to runtime concurrency limits).

## IR layers

Treelace separates planning from execution with typed contracts:

- Planner contract: `treelace.planner.models.PlanSpec`
- Workflow IR: `treelace.ir.nodes.Workflow`
- Compiled DAG: `treelace.compiler.compiled.CompiledWorkflow`
- Runtime result: `treelace.runtime.result.RunResult`
- Public app result: `treelace.app.Run`

This split keeps validation and inspection explicit at each stage.

## Tool output contract

Runtime payloads use this envelope:

```json
{
  "model_context": {"key": "value"},
  "artifact_refs": [],
  "clarification_request": null
}
```

Most tools can simply return dictionaries. Treelace coerces them to
`model_context` automatically.

## Waiting and resume

A run can pause in two places:

- **Planner clarification**: missing intent/inputs before execution.
- **Tool clarification**: a running tool explicitly asks for user input.

When `run.status == "waiting"`, the app receives `run.needs_input`. Resume with:

```python
run = app.resume(run, "user decision")
```

Treelace keeps the underlying clarification token and session bookkeeping in
the `Run` object.

## Sessions and artifacts

By default:

- Session state uses `InMemorySessionStore`.
- Artifacts use `InMemoryArtifactStore`.

You can swap both with custom implementations via protocols in
`treelace.sessions.protocols`.

## Safety and boundaries

- Tool functions execute in your Python process with your permissions.
- `sandbox.py` is a restricted code runner, not an OS sandbox.
- Register only trusted tools and use least-privilege credentials.
