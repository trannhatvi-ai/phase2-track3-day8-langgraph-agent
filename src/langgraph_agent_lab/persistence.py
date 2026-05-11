"""Checkpointer adapter."""

from __future__ import annotations

import sqlite3
from importlib import import_module
from pathlib import Path
from typing import Any


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer.

    Memory is the default for local tests. SQLite is used for the persistence extension and
    returns a ready-to-use saver rather than the context manager returned by from_conn_string().
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "SQLite checkpointer requires: pip install langgraph-checkpoint-sqlite"
            ) from exc
        db_path = Path(database_url or "checkpoints.db")
        if db_path.parent != Path("."):
            db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        saver = SqliteSaver(conn=conn)
        saver.setup()
        return saver
    if kind == "postgres":
        try:
            postgres_module = import_module("langgraph.checkpoint.postgres")
        except ImportError as exc:
            raise RuntimeError(
                "Postgres checkpointer requires: pip install langgraph-checkpoint-postgres"
            ) from exc
        postgres_saver = postgres_module.PostgresSaver
        return postgres_saver.from_conn_string(database_url or "")
    raise ValueError(f"Unknown checkpointer kind: {kind}")
