"""Quote two weekend trips in parallel, then pick the cheaper one."""

from __future__ import annotations

from _model_setup import resolve_model_spec

from treelace import Treelace, tool

FLIGHTS = {
    ("Seattle", "Portland"): {"airline": "Alaska", "price": 89},
    ("Seattle", "San Francisco"): {"airline": "Alaska", "price": 142},
}

HOTELS = {
    "Portland": {"name": "Ace Hotel", "nightly": 189},
    "San Francisco": {"name": "Hotel Zoe", "nightly": 241},
}

CARS = {
    "Portland": {"kind": "compact", "daily": 41},
    "San Francisco": {"kind": "suv", "daily": 79},
}


@tool
def lookup_flight(origin: str, destination: str) -> dict[str, object]:
    """Look up a one-way flight.

    Args:
        origin: Departure city.
        destination: Arrival city.
    """

    flight = FLIGHTS[(origin, destination)]
    return {
        "origin": origin,
        "destination": destination,
        "airline": flight["airline"],
        "price": flight["price"],
    }


@tool
def lookup_hotel(city: str, nights: int) -> dict[str, object]:
    """Look up a hotel and return the stay total.

    Args:
        city: Destination city.
        nights: Number of nights.
    """

    hotel = HOTELS[city]
    return {
        "city": city,
        "name": hotel["name"],
        "nights": nights,
        "total": hotel["nightly"] * nights,
    }


@tool
def lookup_car(city: str, days: int) -> dict[str, object]:
    """Look up a rental car and return the rental total.

    Args:
        city: Destination city.
        days: Number of rental days.
    """

    car = CARS[city]
    return {
        "city": city,
        "kind": car["kind"],
        "days": days,
        "total": car["daily"] * days,
    }


@tool
def quote_trip(
    flight: dict[str, object],
    hotel: dict[str, object],
    car: dict[str, object],
) -> dict[str, object]:
    """Add up one trip's flight, hotel, and car.

    Args:
        flight: Output from lookup_flight.
        hotel: Output from lookup_hotel.
        car: Output from lookup_car.
    """

    return {
        "destination": hotel["city"],
        "flight": flight,
        "hotel": hotel,
        "car": car,
        "total": flight["price"] + hotel["total"] + car["total"],
    }


@tool
def cheaper_trip(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    """Pick the cheaper of two trip quotes.

    Args:
        left: First quote_trip output.
        right: Second quote_trip output.
    """

    winner = left if left["total"] <= right["total"] else right
    return {
        "quotes": [left, right],
        "cheaper": winner["destination"],
        "total": winner["total"],
        "savings": abs(left["total"] - right["total"]),
    }


def main() -> None:
    model_spec = resolve_model_spec()
    app = Treelace(
        model=model_spec,
        tools=[lookup_flight, lookup_hotel, lookup_car, quote_trip, cheaper_trip],
        instructions=(
            "Use city names Seattle, Portland, and San Francisco. "
            "Look up each flight, hotel, and car as its own step so independent lookups can run in parallel. "
            "Quote each destination with quote_trip, then call cheaper_trip on the two quotes. "
            "When a later step needs a previous result, pass an output_ref with pointer /model_context."
        ),
    )
    question = (
        "Compare 3-night weekends from Seattle to Portland vs Seattle to San Francisco. "
        "Include a flight, hotel, and a rental car for 3 days, then say which trip is cheaper."
    )
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
