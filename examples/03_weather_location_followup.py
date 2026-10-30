"""Treelace example: weather location clarification with multi-turn history.

This flow demonstrates:
1) Attribute clarification when a misspelled location is ambiguous.
2) Resuming a paused run with app.resume(...).
3) A second user turn that includes conversation history.
"""

from __future__ import annotations

import json
from difflib import get_close_matches

from treelace import ClarificationRequest, ConversationTurn, Run, Treelace, ToolResult, tool

from _model_setup import resolve_model_spec

SPRINGFIELD_IL = "Springfield, IL, US"
SPRINGFIELD_MA = "Springfield, MA, US"
SEATTLE_WA = "Seattle, WA, US"
SUPPORTED_LOCATIONS = [SPRINGFIELD_IL, SPRINGFIELD_MA, SEATTLE_WA]

WEATHER_SNAPSHOTS: dict[str, dict[str, dict[str, object]]] = {
    SPRINGFIELD_IL: {
        "today": {"condition": "partly cloudy", "avg_temp_c": 18.3},
        "tomorrow": {"condition": "rain showers", "avg_temp_c": 16.8},
    },
    SPRINGFIELD_MA: {
        "today": {"condition": "clear", "avg_temp_c": 14.1},
        "tomorrow": {"condition": "light rain", "avg_temp_c": 12.9},
    },
    SEATTLE_WA: {
        "today": {"condition": "drizzle", "avg_temp_c": 11.7},
        "tomorrow": {"condition": "overcast", "avg_temp_c": 10.8},
    },
}


def _normalize(text: str) -> str:
    cleaned = "".join(char if char.isalnum() else " " for char in text.lower())
    return " ".join(cleaned.split())


def _candidate_locations(location_hint: str) -> list[str]:
    normalized = _normalize(location_hint)
    if not normalized:
        return []

    tokens = set(normalized.split())
    springfield_tokens = {"springfield", "springfeld", "springfeild", "sprngfield"}

    if {"seattle", "seatle"} & tokens or "seattle" in normalized or "seatle" in normalized:
        return [SEATTLE_WA]

    if springfield_tokens & tokens or any(token.startswith("springf") for token in tokens):
        if {"il", "illinois"} & tokens:
            return [SPRINGFIELD_IL]
        if {"ma", "massachusetts"} & tokens:
            return [SPRINGFIELD_MA]
        return [SPRINGFIELD_IL, SPRINGFIELD_MA]

    phrase_hints = {
        "springfield illinois": [SPRINGFIELD_IL],
        "springfield ma": [SPRINGFIELD_MA],
        "springfield massachusetts": [SPRINGFIELD_MA],
        "seatle wa": [SEATTLE_WA],
    }
    for phrase, locations in phrase_hints.items():
        if phrase in normalized:
            return locations

    close = get_close_matches(normalized, ["springfield", "seattle"], n=1, cutoff=0.84)
    if not close:
        return []
    if close[0] == "seattle":
        return [SEATTLE_WA]
    return [SPRINGFIELD_IL, SPRINGFIELD_MA]


def _normalize_day(value: str) -> str:
    lowered = value.strip().lower()
    if "tomorrow" in lowered:
        return "tomorrow"
    return "today"


def _selected_location(selection: dict[str, object]) -> str:
    for key in ("location", "decision"):
        value = selection.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise RuntimeError("selection must include either 'location' or 'decision'")


def _clarification_source(run: Run) -> str:
    for event in run.events:
        if getattr(event, "kind", "") == "clarification_requested":
            source = getattr(event, "source", None)
            if isinstance(source, str):
                return source
    return "unknown"


def _choose_answer(options: list[str]) -> str:
    if not options:
        return SPRINGFIELD_MA
    for option in options:
        if "MA" in option:
            return option
    return options[0]


def _assistant_content(run: Run) -> str:
    if run.answer:
        return run.answer
    if run.data is not None:
        return json.dumps(run.data, ensure_ascii=True, sort_keys=True)
    if run.error:
        return run.error
    return run.status


@tool
def resolve_weather_location(location_hint: str) -> ToolResult:
    """Resolve a possibly misspelled location into one supported weather target.

    Args:
        location_hint: Raw location text from the user question.
    """

    candidates = _candidate_locations(location_hint)
    if len(candidates) == 1:
        return ToolResult(
            model_context={
                "location": candidates[0],
                "location_hint": location_hint,
                "resolution": "resolved",
            }
        )

    options = candidates or SUPPORTED_LOCATIONS
    if candidates:
        prompt = f'I found multiple matches for "{location_hint}". Which location should I use?'
    else:
        prompt = f'I could not resolve "{location_hint}" to one location. Choose from the supported locations.'
    return ToolResult(
        model_context={
            "location_hint": location_hint,
            "candidates": options,
            "resolution": "ambiguous",
        },
        clarification_request=ClarificationRequest(
            prompt=prompt,
            options=options,
            missing_fields=["location"],
        ),
    )


@tool
def get_weather_for_selection(
    selection: dict[str, object],
    day: str = "today",
) -> ToolResult:
    """Return weather for a resolved location selection.

    Args:
        selection: Output from resolve_weather_location via output_ref pointer /model_context.
        day: Day indicator such as today or tomorrow.
    """

    location = _selected_location(selection)
    day_key = _normalize_day(day)
    daily = WEATHER_SNAPSHOTS.get(location)
    if daily is None:
        raise RuntimeError(f"Unsupported location selected: {location!r}")
    weather = daily.get(day_key, daily["today"])
    return ToolResult(
        model_context={
            "location": location,
            "day": day_key,
            "condition": weather["condition"],
            "avg_temp_c": weather["avg_temp_c"],
        }
    )


def run_turn(
    app: Treelace,
    *,
    question: str,
    history: list[ConversationTurn],
) -> list[ConversationTurn]:
    print("\nUser:")
    print(question)
    turns = [*history, ConversationTurn(role="user", content=question)]
    run = app.run(question, history=history, respond=False)
    print("\nRun status:")
    print(run.status)
    if run.plan_text:
        print("\nPlan text:")
        print(run.plan_text)

    while run.status == "waiting":
        if run.needs_input is None:
            raise RuntimeError("waiting status requires needs_input")
        source = _clarification_source(run)
        prompt = run.needs_input.prompt
        options = run.needs_input.options
        print("\nClarification requested:")
        print(f"source={source}")
        print(prompt)
        if options:
            print("options:", options)
        answer = _choose_answer(options)
        print(f"auto-answer: {answer}")
        turns.append(ConversationTurn(role="assistant", content=prompt))
        turns.append(ConversationTurn(role="user", content=answer))
        run = app.resume(run, answer, respond=False)
        print("\nResumed status:")
        print(run.status)
        if run.plan_text:
            print("\nResumed plan text:")
            print(run.plan_text)

    if run.status == "failed":
        print("\nError:")
        print(run.error)
    else:
        print("\nStructured data:")
        print(run.data)
    turns.append(ConversationTurn(role="assistant", content=_assistant_content(run)))
    return turns[-12:]


def main() -> None:
    model_spec = resolve_model_spec()
    app = Treelace(
        model=model_spec,
        tools=[resolve_weather_location, get_weather_for_selection],
        instructions=(
            "You are a weather assistant for these supported locations only: "
            "Springfield, IL, US; Springfield, MA, US; Seattle, WA, US. "
            "For each weather request, call resolve_weather_location once with the location text. "
            "If clarification is required, do not guess. Wait for the user's selection. "
            "Then call get_weather_for_selection once, passing resolve_weather_location output using "
            "an output_ref with pointer /model_context. "
            "Pass day as today or tomorrow when present. Keep tool arguments explicit."
        ),
    )

    print(f"Model: {model_spec}")

    history: list[ConversationTurn] = []
    history = run_turn(
        app,
        question="What's the weather in Springfeld today?",
        history=history,
    )
    history = run_turn(
        app,
        question="And what about Seatle tomorrow?",
        history=history,
    )


if __name__ == "__main__":
    main()
