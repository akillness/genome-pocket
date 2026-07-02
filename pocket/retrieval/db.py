import sqlite3
from pathlib import Path
import sqlite_vec
import pocket.config as config
from .base import _FTS_TABLE

def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _fts_available(conn: sqlite3.Connection) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (_FTS_TABLE,),
    )
    return cur.fetchone() is not None


def _graph_available(conn: sqlite3.Connection) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='entities'"
        ).fetchone()
        is not None
    )


def _has_status_column(conn: sqlite3.Connection, table: str) -> bool:
    """Whether `table` carries the POCKET-302 HITL `status` column."""
    try:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    except sqlite3.Error:
        return False
    return "status" in cols


def _status_clause(conn: sqlite3.Connection, table: str, alias: str = "") -> str:
    """SQL predicate restricting graph reads to approved (committed) facts.

    Pending facts staged by the HITL gate (POCKET-302) stay out of retrieval
    until ``pocket graph review`` accepts them. Legacy graphs built before the
    status column existed get an always-true predicate so reads still work.
    """
    col = f"{alias}.status" if alias else "status"
    return f"{col} = 'approved'" if _has_status_column(conn, table) else "1=1"
