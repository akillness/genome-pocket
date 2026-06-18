"""Administrative (lifecycle) operations for the Pocket target state.

This is the write-side companion to ``pocket.retrieval`` (which only reads).
It backs the ``pocket drop`` command: resetting the materialized target so a
subsequent ``pocket update`` rebuilds from scratch, or evicting a single source
file's chunks and lineage without touching the rest of the index.

Everything here speaks directly to the SQLite target the pocketindex engine
writes, so it stays consistent with the connector's table-naming convention
(``embeddings`` plus ``_pocket_{lineage,memo,fts}_embeddings`` companions).
"""
import sqlite3
from pathlib import Path
from typing import Dict, Optional

import pocket.config as config

# The target table the pipeline materializes (see pocket/pipeline.py) and its
# lineage/memo/FTS companions created by the sqlite connector.
_TARGET_TABLE = "embeddings"
_COMPANION_TABLES = (
    f"_pocket_lineage_{_TARGET_TABLE}",
    f"_pocket_memo_{_TARGET_TABLE}",
    f"_pocket_fts_{_TARGET_TABLE}",
)

# Knowledge-graph target tables (POCKET-404) and their lineage/memo/FTS
# companions. `drop_target` removes these too so a reset clears the whole index
# including any extracted graph.
_GRAPH_TABLES = ("entities", "relations")
_GRAPH_COMPANION_TABLES = tuple(
    f"_pocket_{kind}_{table}"
    for table in _GRAPH_TABLES
    for kind in ("lineage", "memo", "fts")
)


def drop_target(db_path: Optional[Path] = None) -> Dict:
    """Reset all materialized target state for the knowledge base.

    Drops the embeddings table together with its lineage, memo, and FTS
    companion tables. After this the database is empty and the next
    ``pocket update`` is treated as a full first build (no memo fast-path).

    Returns a summary dict: ``{"existed", "sources", "chunks", "dropped"}``.
    """
    db_path = Path(db_path or config.POCKET_SQLITE_DB)
    if not db_path.exists():
        return {"existed": False, "sources": 0, "chunks": 0, "dropped": []}

    conn = sqlite3.connect(str(db_path))
    try:
        sources, chunks = _count(conn)
        dropped = []
        for table in (
            _TARGET_TABLE,
            *_COMPANION_TABLES,
            *_GRAPH_TABLES,
            *_GRAPH_COMPANION_TABLES,
        ):
            if _table_exists(conn, table):
                conn.execute(f"DROP TABLE {table}")
                dropped.append(table)
        conn.commit()
    finally:
        conn.close()
    return {
        "existed": True,
        "sources": sources,
        "chunks": chunks,
        "dropped": dropped,
    }


def drop_source(file_path: str, db_path: Optional[Path] = None) -> Dict:
    """Evict a single source file's chunks and lineage from the target.

    Mirrors the engine's deletion path for one file: removes its rows from the
    main table, its FTS mirror, and its lineage/memo state so a later update
    re-adds it cleanly. Returns ``{"removed": <chunk count>}``.
    """
    db_path = Path(db_path or config.POCKET_SQLITE_DB)
    if not db_path.exists():
        return {"removed": 0}

    conn = sqlite3.connect(str(db_path))
    try:
        if not _table_exists(conn, _TARGET_TABLE):
            return {"removed": 0}
        ids = [
            r[0]
            for r in conn.execute(
                f"SELECT id FROM {_TARGET_TABLE} WHERE file_path = ?",
                (file_path,),
            ).fetchall()
        ]
        if not ids:
            return {"removed": 0}

        conn.executemany(
            f"DELETE FROM {_TARGET_TABLE} WHERE id = ?", [(i,) for i in ids]
        )
        fts = f"_pocket_fts_{_TARGET_TABLE}"
        if _table_exists(conn, fts):
            conn.executemany(
                f"DELETE FROM {fts} WHERE row_id = ?", [(i,) for i in ids]
            )
        # Lineage/memo are keyed by the engine's source_key (the relative path
        # walk_dir emits), which differs from the absolute file_path stored on
        # each row. Resolve the affected source_keys from the lineage table via
        # the row ids we just removed instead of assuming the two keys match.
        lineage = f"_pocket_lineage_{_TARGET_TABLE}"
        memo = f"_pocket_memo_{_TARGET_TABLE}"
        if _table_exists(conn, lineage):
            source_keys = {
                r[0]
                for r in conn.execute(
                    f"SELECT DISTINCT source_key FROM {lineage} "
                    f"WHERE row_id IN ({', '.join('?' * len(ids))})",
                    ids,
                ).fetchall()
            }
            conn.executemany(
                f"DELETE FROM {lineage} WHERE row_id = ?", [(i,) for i in ids]
            )
            # Only forget the memo fingerprint for sources that no longer have
            # any lineage rows left, so partial removals don't strand state.
            if _table_exists(conn, memo):
                for sk in source_keys:
                    remaining = conn.execute(
                        f"SELECT 1 FROM {lineage} WHERE source_key = ? LIMIT 1",
                        (sk,),
                    ).fetchone()
                    if remaining is None:
                        conn.execute(
                            f"DELETE FROM {memo} WHERE source_key = ?", (sk,)
                        )
        conn.commit()
        conn.commit()
    finally:
        conn.close()
    return {"removed": len(ids)}


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
            "AND name = ?",
            (name,),
        ).fetchone()
        is not None
    )


def _count(conn: sqlite3.Connection) -> tuple:
    if not _table_exists(conn, _TARGET_TABLE):
        return 0, 0
    chunks = conn.execute(f"SELECT COUNT(*) FROM {_TARGET_TABLE}").fetchone()[0]
    sources = conn.execute(
        f"SELECT COUNT(DISTINCT file_path) FROM {_TARGET_TABLE}"
    ).fetchone()[0]
    return sources, chunks
