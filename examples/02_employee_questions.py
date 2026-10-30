"""Write Python against a small employee database."""

from __future__ import annotations

import os
import sqlite3

from treelace import Treelace, tool
from treelace.models import Message, Model, ModelContext, ModelRequest, OpenAIChatModel, TextOutput

from _model_setup import resolve_model_spec


SEED = """
CREATE TABLE department (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE employee (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department_id INTEGER NOT NULL REFERENCES department(id),
    title TEXT NOT NULL,
    salary INTEGER NOT NULL
);
CREATE TABLE project (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    lead_id INTEGER NOT NULL REFERENCES employee(id)
);
INSERT INTO department VALUES (1, 'Engineering'), (2, 'Sales');
INSERT INTO employee VALUES
    (1, 'Lee Park', 1, 'Engineer', 128000),
    (2, 'Sam Brooks', 1, 'Manager', 151000),
    (3, 'Jo Hale', 2, 'Rep', 97000);
INSERT INTO project VALUES (1, 'Ledger', 1), (2, 'Renewals', 3);
"""


def open_db() -> tuple[sqlite3.Connection, str]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SEED)
    rows = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    return conn, "\n".join(str(row[0]) for row in rows)


def chat_model(spec: str) -> OpenAIChatModel:
    _, _, name = spec.partition(":")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url and endpoint:
        base_url = endpoint.rstrip("/") + "/openai/v1/"
    return OpenAIChatModel(
        model=name,
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"),
        base_url=base_url,
    )


def as_data(value: object) -> object:
    if isinstance(value, sqlite3.Row):
        return dict(value)
    if isinstance(value, (list, tuple)):
        return [as_data(item) for item in value]
    return value


def run_code(conn: sqlite3.Connection, code: str) -> object:
    scope: dict[str, object] = {"conn": conn}
    exec(code, scope)
    if "result" not in scope:
        raise RuntimeError(f"code did not set result\n{code}")
    return as_data(scope["result"])


def code_only(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return text.strip()


def query_tool(model: Model, conn: sqlite3.Connection, schema: str):
    system = (
        "Write Python that answers the question. conn is an open sqlite3 connection. "
        "Do not import anything. Set result to a dict, list, string, or number. "
        "Return only the code.\n\n"
        f"Schema:\n{schema}"
    )

    @tool
    async def query_database(question: str) -> dict[str, object]:
        """Write Python for the employee database and return what it produces.

        Args:
            question: Question about employees, departments, or projects.
        """

        reply = await model.complete(
            ModelRequest(
                messages=[
                    Message(role="system", content=system),
                    Message(role="user", content=question),
                ]
            ),
            context=ModelContext(),
        )
        if not isinstance(reply.output, TextOutput):
            raise RuntimeError("model did not return code")
        code = code_only(reply.output.text)
        return {"code": code, "result": run_code(conn, code)}

    return query_database


def main() -> None:
    spec = resolve_model_spec()
    model = chat_model(spec)
    conn, schema = open_db()
    question = "Who leads the Ledger project, and what is their department and salary?"
    app = Treelace(
        model=model,
        tools=[query_tool(model, conn, schema)],
        instructions="Call query_database once and pass the user's question through.",
    )
    run = app.run(question)

    print(f"Model: {spec}")
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
