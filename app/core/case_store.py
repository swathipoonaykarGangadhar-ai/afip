"""
Case persistence layer.

Uses Postgres (via DATABASE_URL) when available -- cases and their
approval state survive restarts, which matters because Render's free
tier spins the container down after inactivity and loses anything
in-process. Falls back to an in-memory dict for local dev without a
database, so nothing breaks if DATABASE_URL isn't set -- it just won't
persist across restarts.
"""
import os
import json
from datetime import datetime
from typing import Optional

_DATABASE_URL = os.environ.get("DATABASE_URL")
_pool = None
_memory_store: dict = {}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    data JSONB NOT NULL,
    status TEXT NOT NULL,
    created_date TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
"""


def _get_pool():
    global _pool
    if _pool is None:
        from psycopg_pool import ConnectionPool
        _pool = ConnectionPool(conninfo=_DATABASE_URL, min_size=1, max_size=5, open=True)
        with _pool.connection() as conn:
            conn.execute(CREATE_TABLE_SQL)
            conn.commit()
    return _pool


def is_persistent() -> bool:
    return bool(_DATABASE_URL)


def save_case(case: dict) -> None:
    if _DATABASE_URL:
        pool = _get_pool()
        with pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO cases (case_id, data, status)
                VALUES (%s, %s, %s)
                ON CONFLICT (case_id) DO UPDATE
                SET data = EXCLUDED.data, status = EXCLUDED.status
                """,
                (case["case_id"], json.dumps(case), case["status"]),
            )
            conn.commit()
    else:
        _memory_store[case["case_id"]] = case


def get_case(case_id: str) -> Optional[dict]:
    if _DATABASE_URL:
        pool = _get_pool()
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT data FROM cases WHERE case_id = %s", (case_id,)
            ).fetchone()
            return row[0] if row else None
    return _memory_store.get(case_id)


def list_cases(status: Optional[str] = None) -> list:
    if _DATABASE_URL:
        pool = _get_pool()
        with pool.connection() as conn:
            if status:
                rows = conn.execute(
                    "SELECT data FROM cases WHERE status = %s ORDER BY created_date DESC", (status,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT data FROM cases ORDER BY created_date DESC"
                ).fetchall()
            return [r[0] for r in rows]
    cases = list(_memory_store.values())
    if status:
        cases = [c for c in cases if c["status"] == status]
    return cases
