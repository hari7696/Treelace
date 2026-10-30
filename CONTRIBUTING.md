# Contributing to Treelace

Thanks for your interest in contributing.

## Development setup

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Run checks

```bash
ruff check .
mypy src
pytest -q
```

## Pull requests

- Keep changes focused and include tests.
- Document public API changes.
- Update `CHANGELOG.md` for user-facing behavior changes.

## Commit style

Suggested prefixes:

- `feat:`
- `fix:`
- `docs:`
- `test:`
- `refactor:`

## Security and responsible disclosure

Do not open public issues for vulnerabilities. Follow `SECURITY.md`.

