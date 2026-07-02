from pathlib import Path
import sqlite3
from typing import Dict, List, Optional
import pocket.config as config
from .base import RetrievalHit, _FTS_TABLE
from .db import _connect, _fts_available, _graph_available
from .router import _resolve_mode, _MODE_STRATEGIES


def get_lineage(file_path: str, db_path: Optional[Path] = None) -> List[Dict]:
    """Return the ordered chunk lineage for a source file."""
    db_path = db_path or config.POCKET_SQLITE_DB
    if not Path(db_path).exists():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            """
            SELECT id, start_offset, end_offset, text
            FROM embeddings
            WHERE file_path = ?
            ORDER BY start_offset ASC
            """,
            (file_path,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    return [
        {
            "chunk_id": cid,
            "start_offset": start,
            "end_offset": end,
            "snippet": text[:100].strip(),
        }
        for cid, start, end, text in rows
    ]


def list_sources(db_path: Optional[Path] = None) -> List[Dict]:
    """Return one row per indexed source file with its chunk count and offsets.

    This powers ``pocket ls`` — an inventory of the stable source paths the
    target currently materializes, derived from the same lineage the engine
    uses for incremental reconciliation.
    """
    db_path = db_path or config.POCKET_SQLITE_DB
    if not Path(db_path).exists():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            """
            SELECT file_path,
                   COUNT(*)        AS chunks,
                   MIN(start_offset) AS first_offset,
                   MAX(end_offset)   AS last_offset
            FROM embeddings
            GROUP BY file_path
            ORDER BY file_path ASC
            """
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    return [
        {
            "file_path": fp,
            "chunks": chunks,
            "first_offset": first,
            "last_offset": last,
        }
        for fp, chunks, first, last in rows
    ]


def target_stats(db_path: Optional[Path] = None) -> Dict:
    """Return aggregate counts describing the materialized target state.

    Used by ``pocket show`` to report what the database holds without running
    the pipeline: number of source files, total chunks, and whether the
    lexical (FTS5) companion index is present.
    """
    db_path = db_path or config.POCKET_SQLITE_DB
    if not Path(db_path).exists():
        return {"exists": False, "sources": 0, "chunks": 0, "fts_enabled": False}
    conn = sqlite3.connect(str(db_path))
    try:
        chunks = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        sources = conn.execute(
            "SELECT COUNT(DISTINCT file_path) FROM embeddings"
        ).fetchone()[0]
        fts_enabled = (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (_FTS_TABLE,),
            ).fetchone()
            is not None
        )
    finally:
        conn.close()
    return {
        "exists": True,
        "sources": sources,
        "chunks": chunks,
        "fts_enabled": fts_enabled,
    }


def routing_trace(
    query: str,
    limit: int = 5,
    db_path: Optional[Path] = None,
    model_name: Optional[str] = None,
    mode: str = "hybrid",
) -> Dict:
    """Explain how a query is routed and which sources answer it (POCKET-301).

    Returns a JSON-able trace the local tracing UI visualizes:

      * ``strategies`` — for ``vector``/``lexical``/``graph``: whether the
        chosen ``mode`` *activates* it, whether it is *available* on this target
        (FTS / graph tables present), and how many candidates it produced.
      * ``results`` — the fused hits, each annotated with the ``contributors``
        (the strategies whose ranked list surfaced that chunk).

    It reuses :func:`_gather` and :func:`_fuse`, so the trace can never diverge
    from a real :func:`search` for the same query/mode.
    """
    from .search import _gather, _fuse

    db_path = db_path or config.POCKET_SQLITE_DB
    model_name = model_name or config.EMBEDDING_MODEL
    active = set(_MODE_STRATEGIES.get(mode, ()))

    def _strategies(available: Dict[str, bool], candidates: Dict[str, int]):
        return [
            {
                "name": name,
                "active": name in active,
                "available": available[name],
                "candidates": candidates[name],
            }
            for name in ("vector", "lexical", "graph")
        ]

    zero = {"vector": 0, "lexical": 0, "graph": 0}
    if not Path(db_path).exists():
        absent = {"vector": False, "lexical": False, "graph": False}
        return {
            "query": query,
            "mode": mode,
            "limit": limit,
            "strategies": _strategies(absent, zero),
            "results": [],
        }

    fetch_n = max(limit * 4, limit)
    conn = _connect(Path(db_path))
    try:
        available = {
            "vector": True,
            "lexical": _fts_available(conn),
            "graph": _graph_available(conn),
        }
        # Apply the semantic router (POCKET-504) so the trace mirrors what a real
        # search would do, then recompute which strategies the routed mode
        # activates. ``active`` is read by the _strategies() closure below.
        mode = _resolve_mode(query, mode, conn)
        active = set(_MODE_STRATEGIES.get(mode, ()))
        vector_rows, lexical_rows, graph_rows = _gather(
            conn, query, mode, fetch_n, model_name
        )

    finally:
        conn.close()

    candidates = {
        "vector": len(vector_rows),
        "lexical": len(lexical_rows),
        "graph": len(graph_rows),
    }
    hits = _fuse(vector_rows, lexical_rows, limit, graph_rows)

    results = []
    for hit in hits:
        row = hit.to_dict()
        contributors = []
        if hit.vector_rank is not None:
            contributors.append("vector")
        if hit.lexical_rank is not None:
            contributors.append("lexical")
        if hit.graph_rank is not None:
            contributors.append("graph")
        row["contributors"] = contributors
        results.append(row)

    return {
        "query": query,
        "mode": mode,
        "limit": limit,
        "strategies": _strategies(available, candidates),
        "results": results,
    }


def format_hits(hits: List[RetrievalHit]) -> str:
    """Render hits as the human/agent-readable text used by CLI and MCP."""
    if not hits:
        return "No results found."
    parts = []
    for idx, hit in enumerate(hits, 1):
        parts.append(
            f"[{idx}] File: {hit.file_path} "
            f"(chars {hit.start_offset}-{hit.end_offset}) "
            f"[Score: {hit.score:.4f}]\n"
            f"Content:\n{hit.text.strip()}\n"
            f"{'=' * 40}"
        )
    return "\n\n".join(parts)
