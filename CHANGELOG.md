# Changelog

All notable changes to this project are documented in this file.

## [0.1.0a1] - 2026-10-30

### Added

- New high-level `Treelace` facade with sync/async run methods and model string
  resolution (`openai:...`).
- Public `Run` and `NeedsInput` contracts focused on answer/data/plan output.
- Response synthesis stage that converts tool output into grounded user-facing
  text by default.
- `respond=False` raw mode for structured-data-first use cases.
- Resume UX via `app.resume(run, answer)` without exposing token plumbing.
- Framework-owned plan rendering attached to run output (`plan_text`).
- Support for tool docstrings that use either `Args:` or `Parameters:` formats.
- Real-model examples showing simple AST-style decomposition, parallel merge
  execution, and dependency inspection output.

### Changed

- Root package exports are now intentionally minimal for small-app usage.
- Tool output ergonomics now treat plain scalar/dict return values as first-class.
- Examples were rewritten around the new facade and compact plan readability.

### Removed

- `coding_tool` is no longer part of the top-level public API.
- Outdated quickstart style based on manually scripted planner payloads.

### Notes

- This release is an intentional alpha API reset toward a simpler default path.
- Advanced compiler/runtime modules remain available via subpackage imports.

## [0.1.0] - 2026-10-20

- Initial Treelace framework implementation:
  - Typed workflow AST and compiler
  - Dependency DAG scheduler with parallel execution of ready operations
  - Provider-neutral model protocol and OpenAI-compatible adapter
  - Tool registration from signatures and docstrings
  - In-memory session stores
  - CLI, examples, and test suite

