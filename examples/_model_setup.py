"""Shared runtime setup for real-model examples.

This module keeps example scripts dependency-free by loading a local .env file
directly when present and resolving a Treelace model string from environment
variables.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_local_env(*, dotenv_path: Path | None = None) -> None:
    """Load key-value pairs from .env into os.environ when not already set."""

    path = dotenv_path or Path(__file__).resolve().parents[1] / ".env"
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        cleaned = value.strip().strip('"').strip("'")
        os.environ[key] = cleaned


def resolve_model_spec() -> str:
    """Resolve an OpenAI-compatible model string for Treelace examples."""

    load_local_env()

    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or os.getenv("AZURE_OPENAI_MODEL_NAME")
    if azure_key and azure_endpoint and azure_model:
        return f"azure-openai:{azure_model}"

    openai_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL")
    if openai_key and openai_model:
        return f"openai:{openai_model}"

    raise RuntimeError(
        "No valid model environment found. Configure either:\n"
        "  1) AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT + "
        "AZURE_OPENAI_CHAT_DEPLOYMENT (or AZURE_OPENAI_MODEL_NAME), or\n"
        "  2) OPENAI_API_KEY + OPENAI_MODEL.\n"
        "You can place these in a local .env file at the project root."
    )
