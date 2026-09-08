"""Treelace CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from treelace.compiler.compile import compile_workflow
from treelace.compiler.inspect import dependency_mermaid, dependency_text
from treelace.ir.nodes import Workflow


def _load_workflow(path: str) -> Workflow:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return Workflow.model_validate(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="treelace")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_cmd = subparsers.add_parser("validate")
    validate_cmd.add_argument("workflow")

    inspect_cmd = subparsers.add_parser("inspect")
    inspect_cmd.add_argument("workflow")

    render_cmd = subparsers.add_parser("render")
    render_cmd.add_argument("workflow")
    render_cmd.add_argument("-o", "--output", required=True)

    schema_cmd = subparsers.add_parser("schema")
    schema_cmd.add_argument(
        "--target",
        choices=["workflow", "plan_decision"],
        default="workflow",
    )

    args = parser.parse_args(argv)

    if args.command == "validate":
        workflow = _load_workflow(args.workflow)
        compile_workflow(workflow)
        print("ok")
        return 0

    if args.command == "inspect":
        workflow = _load_workflow(args.workflow)
        compiled = compile_workflow(workflow)
        print(dependency_text(compiled))
        return 0

    if args.command == "render":
        workflow = _load_workflow(args.workflow)
        compiled = compile_workflow(workflow)
        mermaid = dependency_mermaid(compiled)
        Path(args.output).write_text(mermaid, encoding="utf-8")
        print(args.output)
        return 0

    if args.command == "schema":
        if args.target == "workflow":
            schema = Workflow.model_json_schema()
        else:
            from treelace.planner.models import PLAN_DECISION_ADAPTER

            schema = PLAN_DECISION_ADAPTER.json_schema()
        print(json.dumps(schema, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
