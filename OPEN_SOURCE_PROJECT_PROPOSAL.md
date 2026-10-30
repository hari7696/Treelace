# Treelace Open-Source Direction (0.2)

## Positioning

Treelace is a lightweight framework for simple LLM-driven applications.

It is intentionally scoped to:

- register ordinary typed Python functions as tools,
- produce one finite, inspectable execution plan per request,
- execute independent steps concurrently from explicit dependencies,
- return grounded user-facing text plus structured tool data.

Treelace is not trying to be a full graph-runtime platform, agent team
orchestration layer, or hosted control plane.

## Product boundary

The default user path should stay small:

```python
from treelace import Treelace, tool
```

No mandatory graph authoring. No custom scheduler setup. No infrastructure
requirements for the first run.

## Architecture strategy

Treelace keeps deterministic internals:

- typed planner contracts,
- compile-first dependency validation,
- bounded parallel runtime execution,
- explicit clarification and resume lifecycle.

Treelace adds a simple facade:

- `Treelace` app class,
- `Run`/`NeedsInput` result contracts,
- response synthesis stage,
- raw mode for structured-data-first applications.

## Competitive stance

Treelace complements larger frameworks by optimizing for small apps where users
need clarity and speed over feature breadth.

- Use Treelace for a few coordinated tools and quick iteration.
- Use LangGraph or similar frameworks for long-lived state machines, loops,
  rich graph authoring, and complex orchestration patterns.

## Clean-room provenance policy

Treelace development follows a clean-room approach for any inspirations from
prior private/company work:

- copy no private source code,
- copy no private prompts or domain glossaries,
- copy no proprietary infrastructure wiring,
- reimplement only general patterns from first principles.

All examples and documentation in this repository use synthetic, public-safe
content.

## Open-source impact goals

This project is maintained as a public open-source contribution with the goals
of:

- clear technical design and testable contracts,
- reproducible examples runnable by new users quickly,
- meaningful external adoption and community contributions,
- sustained maintenance quality suitable for professional portfolio evidence.
