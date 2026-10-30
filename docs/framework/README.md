# Treelace Framework Documentation

This folder contains practical docs for building, debugging, and shipping
Treelace applications.

## Recommended reading order

1. `getting-started.md` - Build your first app, run it, and handle waiting
   states.
2. `concepts.md` - Understand how planning, compilation, and execution fit
   together.
3. `models.md` - Configure model providers and split planner/responder models.
4. `modules.md` - Find where each module/class lives and when to use it.

## What these docs focus on

- One-shot, typed tool workflows with explicit dependencies.
- Predictable execution behavior over autonomous loops.
- Run output you can inspect (`status`, `plan_text`, `data`, `events`).
- Production basics: model config, resume flow, and common failure modes.

## Go-live checklist

- Every tool argument and return type is annotated.
- Tool docstrings clearly describe intent and parameters.
- Instructions constrain planner behavior to your allowed tools.
- You inspect `run.plan_text` and `run.data` in test runs.
- Waiting states are handled with `app.resume(...)`.
- Environment variables are configured for your selected model provider.

## Documentation map

- `getting-started.md` - Build and run your first Treelace app.
- `concepts.md` - Core runtime/planner/compiler concepts and lifecycle.
- `models.md` - Provider strings, env setup, and custom model protocol.
- `modules.md` - Package-by-package module and symbol reference.

## Reference examples

- `examples/00_generic_math.py` - Arithmetic decomposition with parallel waves.
- `examples/01_weather_lookups.py` - Parallel weather lookups merged with a
  sandboxed Python step.
- `examples/02_employee_questions.py` - Generated Python against an in-memory
  employee database.
- `examples/03_weather_location_clarification.py` - Ambiguous location
  clarification, resume flow, and multi-turn weather follow-ups.

For broader project docs, see `docs/guides.md`.
