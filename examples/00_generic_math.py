"""Baseline Treelace example: decompose arithmetic into primitive tools."""

from __future__ import annotations

from treelace import Treelace, tool

from _model_setup import resolve_model_spec

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
    """Subtract two numbers.

    Args:
        left: Minuend.
        right: Subtrahend.
    """

    return {"value": left - right}


@tool
def multiply(left: float, right: float) -> dict[str, float]:
    """Multiply two numbers.

    Args:
        left: First factor.
        right: Second factor.
    """

    return {"value": left * right}


@tool
def divide(left: float, right: float) -> dict[str, float]:
    """Divide two numbers.

    Args:
        left: Dividend.
        right: Divisor.
    """

    return {"value": left / right}


def main() -> None:
    model_spec = resolve_model_spec()
    app = Treelace(
        model=model_spec,
        tools=[add, subtract, multiply, divide],
        instructions=(
            "Decompose every arithmetic expression into primitive tools: "
            "add, subtract, multiply, divide. Do not compute the expression yourself. "
            "Independent subexpressions should be separate steps so they can run in parallel. "
            "When a later step needs a previous result, pass an output_ref "
            "with pointer /model_context/value (or use $step_id shorthand), "
            "not a guessed number."
        ),
    )
    question = "Evaluate this expression: ((18 + 27) * (44 - 19)/10)."
    run = app.run(question, respond=True)

    print(f"Model: {model_spec}")
    print("\nQuestion:")
    print(question)
    print("\nStatus:")
    print(run.status)
    if run.error:
        print("\nError:")
        print(run.error)
    print("\nAnswer:")
    print(run.answer)
    print("\nStructured data:")
    print(run.data)
    print("\nPlan text:")
    print(run.plan_text)


if __name__ == "__main__":
    main()
