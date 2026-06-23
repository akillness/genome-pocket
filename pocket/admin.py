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
        # source_key (engine relative path) ≠ file_path (absolute); resolve via lineage table
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


# POCKET-302 HITL graph review — pending facts (below POCKET_GRAPH_MIN_CONFIDENCE) held until `pocket graph review`

_ENTITIES = "entities"
_RELATIONS = "relations"


def _has_status_column(conn: sqlite3.Connection, table: str) -> bool:
    if not _table_exists(conn, table):
        return False
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    return "status" in cols


def list_pending(db_path: Optional[Path] = None) -> Dict:
    """Return the graph facts staged for review.

    ``{"entities": [...], "relations": [...]}`` — each entity carries
    ``id/name/type/confidence/source_file``; each relation additionally resolves
    its subject/object names (joining nodes of any status). Empty lists when no
    graph, no status column, or nothing pending.
    """
    db_path = Path(db_path or config.POCKET_SQLITE_DB)
    empty = {"entities": [], "relations": []}
    if not db_path.exists():
        return empty
    conn = sqlite3.connect(str(db_path))
    try:
        entities = []
        if _has_status_column(conn, _ENTITIES):
            entities = [
                {
                    "id": r[0],
                    "name": r[1],
                    "type": r[2],
                    "confidence": r[3],
                    "source_file": r[4],
                }
                for r in conn.execute(
                    "SELECT id, name, type, confidence, source_file "
                    "FROM entities WHERE status = 'pending' "
                    "ORDER BY confidence ASC"
                ).fetchall()
            ]
        relations = []
        if _has_status_column(conn, _RELATIONS):
            relations = [
                {
                    "id": r[0],
                    "predicate": r[1],
                    "subject": r[2],
                    "object": r[3],
                    "confidence": r[4],
                    "source_file": r[5],
                }
                for r in conn.execute(
                    "SELECT r.id, r.predicate, so.name, ob.name, r.confidence, "
                    "       r.source_file "
                    "FROM relations r "
                    "LEFT JOIN entities so ON so.id = r.subject_id "
                    "LEFT JOIN entities ob ON ob.id = r.object_id "
                    "WHERE r.status = 'pending' "
                    "ORDER BY r.confidence ASC"
                ).fetchall()
            ]
    finally:
        conn.close()
    return {"entities": entities, "relations": relations}


def _pending_ids(conn: sqlite3.Connection, table: str, ids: Optional[list]) -> list:
    """Pending row ids in `table`, optionally narrowed to `ids`."""
    if not _has_status_column(conn, table):
        return []
    if ids is None:
        rows = conn.execute(
            f"SELECT id FROM {table} WHERE status = 'pending'"
        ).fetchall()
    else:
        if not ids:
            return []
        ph = ", ".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT id FROM {table} WHERE status = 'pending' AND id IN ({ph})",
            list(ids),
        ).fetchall()
    return [r[0] for r in rows]


def approve_pending(ids: Optional[list] = None, db_path: Optional[Path] = None) -> Dict:
    """Commit staged facts: flip ``status`` from pending to approved.

    ``ids=None`` approves every pending fact; otherwise only matching ids (which
    may name entities and/or relations). Returns ``{"entities", "relations"}``
    counts actually approved.
    """
    db_path = Path(db_path or config.POCKET_SQLITE_DB)
    if not db_path.exists():
        return {"entities": 0, "relations": 0}
    conn = sqlite3.connect(str(db_path))
    try:
        counts = {}
        for key, table in (("entities", _ENTITIES), ("relations", _RELATIONS)):
            target = _pending_ids(conn, table, ids)
            if target:
                ph = ", ".join("?" * len(target))
                conn.execute(
                    f"UPDATE {table} SET status = 'approved' WHERE id IN ({ph})",
                    target,
                )
            counts[key] = len(target)
        conn.commit()
    finally:
        conn.close()
    return counts


def reject_pending(ids: Optional[list] = None, db_path: Optional[Path] = None) -> Dict:
    """Discard staged facts: delete pending rows (and entity FTS mirrors).

    ``ids=None`` rejects every pending fact. Returns counts actually removed.
    """
    db_path = Path(db_path or config.POCKET_SQLITE_DB)
    if not db_path.exists():
        return {"entities": 0, "relations": 0}
    conn = sqlite3.connect(str(db_path))
    try:
        counts = {}
        for key, table in (("entities", _ENTITIES), ("relations", _RELATIONS)):
            target = _pending_ids(conn, table, ids)
            if target:
                ph = ", ".join("?" * len(target))
                conn.execute(
                    f"DELETE FROM {table} WHERE id IN ({ph})", target
                )
                fts = f"_pocket_fts_{table}"
                if _table_exists(conn, fts):
                    conn.executemany(
                        f"DELETE FROM {fts} WHERE row_id = ?",
                        [(i,) for i in target],
                    )
            counts[key] = len(target)
        conn.commit()
    finally:
        conn.close()
    return counts