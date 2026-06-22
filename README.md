# Treelace

Treelace is a lite Python framework for predictable multi-tool LLM workflows.

It converts a question into an AST-style execution plan, compiles dependencies
into a DAG, runs independent steps in parallel, and returns structured output.

No agents. No autonomous loops. No hidden control flow.

## Why Treelace

Treelace is designed for one-shot execution that stays inspectable:

- Decompose complex requests into small typed tool calls.
- Encode data flow with explicit `output_ref` edges.
- Parallelize independent subexpressions automatically.
- Execute the compiled graph deterministically.

This AST-style decomposition makes one-shot runs simpler to reason about and
more predictable to operate.

## How It Works

1. Register normal Python functions as tools.
2. Planner produces one execution plan with steps and dependencies.
3. Compiler builds a dependency DAG.
4. Runtime executes ready nodes in waves until the result node completes.

If user input is needed, the run pauses with a typed clarification payload and
can be resumed with `app.resume(...)`.

## How The LLM Knows Tool Details

Treelace builds tool metadata from your Python functions and passes it to the
planner/model:

- Function name and signature
- Type hints on parameters (and return shape)
- Docstring summary and parameter descriptions

Docstrings are strongly recommended because they make planning more accurate and
more predictable, especially when multiple tools are similar.

Use concise docstrings with `Args:` or `Parameters:` sections so each argument
has a clear meaning.

## Generic Math Example

`examples/00_generic_math.py` registers:

- `add`
- `subtract`
- `multiply`
- `divide`

For `((18 + 27) * (44 - 19) / 10)`, a typical plan is:

```text
Plan
  Wave 1 (parallel): sum, difference
  Wave 2: product
  Wave 3: quotient
```

This is AST decomposition in practice: evaluate leaves first, then combine them
toward the root result.

## Installation

```bash
pip install treelace
```

OpenAI-compatible adapter:

```bash
pip install "treelace[openai]"
```

## Quickstart

```python
from treelace import Treelace, tool


@tool
def add(left: float, right: float) -> dict[str, float]:
    """Add two numbers.

    Args:
        left: First addend.
        right: Second addend.
    """
    return {"value": left + right}


@tool
def subtract(left: float, right: float) -> dict[str, float]:
    """Subtract right from left.

    Args:
        left: Minuend.
        right: Subtrahend.
    """
    return {"value": left - right}


app = Treelace(
    model="openai:gpt-5-mini",
    tools=[add, subtract],
    instructions=(
        "Decompose arithmetic into tool calls. "
        "Use output_ref when one step depends on another."
    ),
)

run = app.run("What is (12 + 8) - 5?")
print(run.status)
print(run.answer)
print(run.data)
print(run.plan_text)
```

## Output Ref Shape

When one step depends on another, pass the canonical shape:

```json
{"kind": "output_ref", "node_id": "sum", "pointer": "/"}
```

Keep `node_id` at the top level of the ref object.

## Run Output

Each run returns:

- `run.plan_text`: execution waves and dependencies
- `run.data`: structured output from the final step
- `run.answer`: grounded natural-language response (default behavior)
- `run.status`: `succeeded`, `waiting`, or `failed`
- `run.needs_input`: clarification payload when `status == "waiting"`

Use `respond=False` when you want structured data only.

## Examples

- `examples/00_generic_math.py`
- `examples/01_weather_lookups.py`
- `examples/02_employee_questions.py`
- `examples/03_weather_location_clarification.py`

Examples are configured for real model adapters. Set either:

- `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_CHAT_DEPLOYMENT`
- `OPENAI_API_KEY` + `OPENAI_MODEL`

## Status

Treelace is alpha and intentionally scoped for simple, finite tool workflows.
See:

- `docs/framework/README.md`
- `docs/framework/getting-started.md`
- `docs/framework/concepts.md`
- `docs/framework/models.md`
- `docs/framework/modules.md`
- `docs/adr/`

