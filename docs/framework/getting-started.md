# Getting Started

Treelace is built for small, predictable, one-shot tool workflows. A run
produces one plan, compiles dependencies into a DAG, and executes ready steps
in parallel.

## 1) Install

```bash
pip install treelace
```

If you want the built-in OpenAI-compatible adapter:

```bash
pip install "treelace[openai]"
```

## 2) Configure model environment

Set one of these environment configurations:

- OpenAI-compatible:
  - `OPENAI_API_KEY`
  - `OPENAI_MODEL` (or include the model in `model="openai:..."`)
- Azure OpenAI:
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_CHAT_DEPLOYMENT` (or `AZURE_OPENAI_MODEL_NAME`)

## 3) Define typed tools

Register normal Python functions with `@tool`.

Tool rules:

- Every parameter has a type annotation.
- The function has a return type annotation.
- Docstrings should describe arguments clearly.

```python
from treelace import tool


@tool
def add(left: float, right: float) -> dict[str, float]:
    """Add two numbers.

    Args:
        left: First addend.
        right: Second addend.
    """
    return {"value": left + right}


@tool
def multiply(left: float, right: float) -> dict[str, float]:
    """Multiply two numbers.

    Args:
        left: First factor.
        right: Second factor.
    """
    return {"value": left * right}
```

When a tool needs run metadata or model access, add a keyword-only
`context: ToolContext` parameter.

## 4) Build an app

```python
from treelace import Treelace

app = Treelace(
    model="openai:gpt-5-mini",
    tools=[add, multiply],
    instructions=(
        "Use tools for arithmetic. "
        "Prefer separate steps for independent subexpressions. "
        "Use output_ref when one step depends on another."
    ),
)
```

## 5) Run a question

```python
run = app.run("What is (12 + 8) * 3?", respond=True)

print(run.status)     # succeeded | failed | waiting
print(run.answer)     # natural-language response (unless respond=False)
print(run.data)       # structured output
print(run.plan_text)  # execution waves and dependencies
```

If you only want structured data:

```python
run = app.run("What is (12 + 8) * 3?", respond=False)
print(run.answer)  # None
print(run.data)
```

## 6) Understand run states

- `succeeded`: final step completed.
- `failed`: one or more nodes failed and execution stopped.
- `waiting`: planner or tool requested clarification.

When `waiting`, check `run.needs_input`:

- `run.needs_input.prompt`
- `run.needs_input.options`
- `run.needs_input.missing_fields`

## 7) Resume after clarification

```python
run = app.run("Proceed with default region?", respond=False)
if run.status == "waiting":
    print(run.needs_input.prompt)
    run = app.resume(run, "yes", respond=False)
```

Treelace stores the clarification token in `Run`, so the caller only needs the
`Run` object and the chosen answer.

## 8) Output references in dependent steps

Treelace runtime payloads use this envelope:

```json
{
  "model_context": {"value": 20.0},
  "artifact_refs": []
}
```

When writing plans manually, reference prior outputs with `output_ref`:

```json
{"kind": "output_ref", "node_id": "sum", "pointer": "/model_context/value"}
```

Use `pointer: "/"` only when the downstream step needs the full envelope.

## 9) Async APIs

In async applications, use async methods:

- `await app.aplan(...)`
- `await app.arun(...)`
- `await app.aresume(...)`

Use sync methods (`plan`, `run`, `resume`) only when not inside an active event
loop.

## 10) Pre-launch checks

- Validate at least one real run with `respond=False` and inspect `run.data`.
- Verify `run.plan_text` matches expected tool sequencing and dependencies.
- Confirm all tools are deterministic enough for your use case.
- Add explicit handling for `failed` and `waiting` states in your app code.
