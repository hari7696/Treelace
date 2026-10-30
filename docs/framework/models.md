# Models

Treelace accepts either:

- A model string that uses the built-in OpenAI-compatible adapter, or
- A custom object implementing `treelace.models.protocol.Model`.

## Model input forms

Treelace accepts these string formats:

- `openai:<model_name>`
- `azure-openai:<deployment_name>`
- `azure:<deployment_name>` (alias of `azure-openai`)
- `<model_name>` (shorthand for `openai:<model_name>`)

Examples:

```python
from treelace import Treelace

app = Treelace(model="openai:gpt-5-mini", tools=[...])
app = Treelace(model="azure-openai:my-chat-deployment", tools=[...])
app = Treelace(model="gpt-5-mini", tools=[...])  # same as openai:gpt-5-mini
```

## Environment variables and resolution rules

Install the adapter first:

```bash
pip install "treelace[openai]"
```

### API key resolution

Treelace uses:

1. `OPENAI_API_KEY`, else
2. `AZURE_OPENAI_API_KEY`

At least one is required when using model strings.

### Base URL resolution

Treelace uses:

1. `OPENAI_BASE_URL`, else
2. Derived from `AZURE_OPENAI_ENDPOINT` as `<endpoint>/openai/v1/`

### Model name fallback

If the string does not include a model name (for example `openai:`), Treelace
falls back to:

1. `OPENAI_MODEL`
2. `AZURE_OPENAI_CHAT_DEPLOYMENT`
3. `AZURE_OPENAI_MODEL_NAME`

If no model can be resolved, Treelace raises a `ValueError`.

## Single-model and split-model setups

If you pass only `model=...`, the same model is used for both:

- planning/tool orchestration
- response synthesis (`run.answer`)

To split them:

```python
app = Treelace(
    planner_model="openai:gpt-5-mini",
    responder_model="openai:gpt-5-mini",
    tools=[...],
)
```

You can also pass model objects instead of strings.

## Custom model implementation

Any object implementing `Model` can be used.

```python
from treelace.models.protocol import (
    Model,
    ModelContext,
    ModelRequest,
    ModelResponse,
    TextOutput,
)


class MyModel(Model):
    async def complete(
        self,
        request: ModelRequest,
        *,
        context: ModelContext,
    ) -> ModelResponse:
        return ModelResponse(output=TextOutput(text="ok"))
```

Then pass it directly:

```python
app = Treelace(model=MyModel(), tools=[...])
```

## Testing and offline development

Use `FakeModel` for deterministic tests without external API calls:

```python
from treelace.models.fake import FakeModel

app = Treelace(model=FakeModel(...), tools=[...])
```

`FakeModel` is useful for unit tests that should not depend on network
availability or vendor model behavior.

## Common setup errors

- `missing OPENAI_API_KEY or AZURE_OPENAI_API_KEY`
  - Set one API key env var before creating `Treelace(...)`.
- `missing model name`
  - Use `openai:<name>` or set fallback env vars like `OPENAI_MODEL`.
- `unsupported model provider`
  - Use `openai`, `azure-openai`, or `azure`.
