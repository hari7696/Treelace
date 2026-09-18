"""Small local coding utility for dataframe-like payloads."""

from __future__ import annotations

from typing import Any


def coding_tool(
    dataframes: list[Any],
    dataframe_schemas: list[dict[str, Any]],
    system_message: str = "",
) -> tuple[dict[str, Any], list[Any]]:
    """Return compact metadata and pass-through artifacts.

    This intentionally avoids model calls and is safe for local use as a helper
    inside registered tools.
    """

    _ = dataframe_schemas
    summary = {
        "dataframe_count": len(dataframes),
        "system_message_preview": system_message[:200],
    }
    return summary, dataframes
