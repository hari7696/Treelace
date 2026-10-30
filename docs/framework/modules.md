# Modules Reference

This is the "where is what" map for Treelace modules.

## Public entrypoint (`treelace`)

For most applications, start with:

```python
from treelace import Treelace, tool, ToolContext
```

`treelace.__init__` exposes:

- `Treelace`, `Run`, `NeedsInput`
- `tool`, `ToolContext`
- `ToolOutput`, `ToolResult`, `ClarificationRequest`
- `ConversationTurn`, `execution_plan_text`

## Package layout

```text
treelace/
  app.py
  runner.py
  planner/
  compiler/
  runtime/
  tools/
  models/
  ir/
  sessions/
  sandbox.py
  cli.py
```

## `treelace.app`

High-level app facade used by end users.

- `Treelace`: main API (`plan`, `run`, `resume`, async variants)
- `Run`: public run result envelope
- `NeedsInput`: waiting-state payload

Use this module for normal app integration.

## `treelace.runner`

Composes planning + compile + execution into one orchestrator.

- `Runner.plan(...)`
- `Runner.run(...)`
- `Runner.resume(...)`

Use this if you need lower-level control than `Treelace` but still want
end-to-end orchestration.

## `treelace.planner`

Planning contracts and orchestration.

- `PlannerOrchestrator` (`planner.orchestrator`)
- Plan models (`planner.models`):
  - `PlanSpec`, `PlanStep`
  - `ClarificationPlan`, `ChatPlan`
  - `PlanDecision`
- Prompt manager (`planner.prompts`):
  - `MasterPromptManager`, `ConversationTurn`
- Render helper (`planner.inspect`):
  - `execution_plan_text(...)`

Use this package when customizing planning behavior or inspecting plan objects.

## `treelace.compiler`

Workflow-to-DAG compilation and dependency inspection.

- `compile_workflow(...)`
- `CompiledWorkflow`, `CompiledOperation`
- `Diagnostic`
- Graph renderers:
  - `dependency_text(...)`
  - `dependency_json(...)`
  - `dependency_mermaid(...)`

Use this package for validation, graph diagnostics, and visualization.

## `treelace.runtime`

Dependency scheduler and runtime event/result contracts.

- `Engine` (`runtime.engine`)
- `RunResult` (`runtime.result`)
- `Event` union and event models (`runtime.events`)

Use this package when embedding Treelace runtime directly.

## `treelace.tools`

Tool registration, inference, invocation, and result contracts.

- Decorator + inference:
  - `tool`, `infer_tool`
- Registry/definitions:
  - `ToolRegistry`, `ToolDefinition`, `ToolSpec`
- Context and result contracts:
  - `ToolContext`
  - `ToolOutput`, `ToolResult`, `ClarificationRequest`

Use this package for typed tool authoring and custom tool registration flows.

## `treelace.models`

LLM protocol and model adapters.

- Protocol contracts (`models.protocol`):
  - `Model`, `ModelRequest`, `ModelResponse`
  - `Message`, `ModelContext`
  - `TextOutput`, `StructuredOutput`
- Implementations:
  - `OpenAIChatModel`
  - `FakeModel`

Use this package for adapter selection and custom model integrations.

## `treelace.ir`

Typed workflow and expression IR.

- Node IR (`ir.nodes`):
  - `Workflow`, `Sequence`, `Parallel`, `ToolCall`, `Clarification`
- Expression IR (`ir.expressions`):
  - `OutputRef`, `InputRef`, `ConstExpr`, `CompareExpr`, `FormatExpr`
- Eval helpers (`ir.eval`):
  - `EvalContext`, `eval_expression(...)`, `MissingValueError`

Use this package for direct workflow authoring or custom compilers/transforms.

## `treelace.sessions`

Session and artifact persistence contracts plus in-memory defaults.

- Protocols:
  - `SessionStoreProtocol`, `ArtifactStoreProtocol`
- In-memory implementations:
  - `InMemorySessionStore`, `InMemoryArtifactStore`
- Data models:
  - `Session`, `RuntimeState`, `PendingClarification`,
    `CompiledWorkflowSnapshot`

Use this package when persisting runs beyond process memory.

## Utility modules

- `treelace.sandbox`:
  restricted Python execution helper for model-generated record processing.
- `treelace.cli`:
  CLI commands for validate/inspect/render/schema on workflow JSON.
- `treelace.errors`:
  framework exception hierarchy.
- `treelace.json_types`:
  shared JSON type aliases.

## Stable vs internal guidance

- Prefer `from treelace import ...` for application code.
- Use subpackages (`planner`, `compiler`, `runtime`, `tools`) when building
  advanced integrations.
- Treat deeply internal implementation details as subject to change unless they
  are explicitly exported in package `__init__.py` files.
