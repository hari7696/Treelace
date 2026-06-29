"""Model exports."""

from treelace.models.fake import FakeModel
from treelace.models.protocol import (
    Message,
    Model,
    ModelContext,
    ModelRequest,
    ModelResponse,
    StructuredOutput,
    TextOutput,
)

try:
    from treelace.models.openai_adapter import OpenAIChatModel
except ModuleNotFoundError as import_error:
    _OPENAI_IMPORT_ERROR = import_error

    class OpenAIChatModel:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            _ = (args, kwargs)
            raise ImportError(
                "OpenAI adapter is not installed. Install treelace with the 'openai' extra."
            ) from _OPENAI_IMPORT_ERROR

__all__ = [
    "FakeModel",
    "Message",
    "Model",
    "ModelContext",
    "ModelRequest",
    "ModelResponse",
    "OpenAIChatModel",
    "StructuredOutput",
    "TextOutput",
]
