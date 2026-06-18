"""Hybrid retrieval layer for Pocket.

This module is the single source of truth for *reading* the knowledge base.
Both the CLI (`pocket search`), the MCP server, and the REST API server call
into it, so query behavior can never drift between interfaces.

It implements the retrieval design documented in
``docs/architecture/retrieval-layer.md``:

  * **Vector search** over sqlite-vec embeddings (semantic similarity).
  * **Lexical search** over the FTS5 BM25 index (exact keywords / symbols).
  * **Reciprocal Rank Fusion (RRF)** to merge the two ranked lists.

Every returned hit carries full lineage (source file + character offsets) so an
agent can cite the exact source bytes, matching pocketindex's end-to-end lineage
guarantee.
"""
import sqlite3
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import sqlite_vec
from sentence_transformers import SentenceTransformer

import pocket.config as config

# Constant used by Reciprocal Rank Fusion; 60 is the value from the original
# RRF paper and the project's retrieval-layer design doc.
RRF_K = 60

_FTS_TABLE = "_pocket_fts_embeddings"


@dataclass
class RetrievalHit:
    """A single retrieved chunk with its lineage and fusion score."""

    file_path: str
    text: str
    start_offset: int
    end_offset: int
    score: float
    vector_rank: Optional[int] = None
    lexical_rank: Optional[int] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@lru_cache(maxsize=4)
def _get_model(model_name: str) -> SentenceTransformer:
    """Cache the embedding model so repeated queries don't reload weights."""
    return SentenceTransformer(model_name)


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


def _vector_search(conn: sqlite3.Connection, query_vector, limit: int) -> List[tuple]:
    cur = conn.execute(
        """
        SELECT id, file_path, text, start_offset, end_offset,
               vec_distance_cosine(embedding, ?) AS distance
        FROM embeddings
        ORDER BY distance ASC
        LIMIT ?
        """,
        (query_vector, limit),
    )
    return cur.fetchall()


def _fts_escape(query: str) -> str:
    """Turn a free-text query into a safe FTS5 MATCH expression.

    We quote each whitespace token as a phrase so punctuation and FTS operators
    in user input can't break the query, then OR them together for recall.
    """
    tokens = [t for t in query.replace('"', " ").split() if t]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


def _lexical_search(conn: sqlite3.Connection, query: str, limit: int) -> List[tuple]:
    match_expr = _fts_escape(query)
    if not match_expr:
        return []
    try:
        cur = conn.execute(
            f"""
            SELECT f.row_id, e.file_path, e.text, e.start_offset, e.end_offset,
                   bm25({_FTS_TABLE}) AS rank
            FROM {_FTS_TABLE} f
            JOIN embeddings e ON e.id = f.row_id
            WHERE {_FTS_TABLE} MATCH ?
            ORDER BY rank ASC
            LIMIT ?
            """,
            (match_expr, limit),
        )
        return cur.fetchall()
    except sqlite3.OperationalError:
        return []


def search(
    query: str,
    limit: int = 5,
    db_path: Optional[Path] = None,
    model_name: Optional[str] = None,
    mode: str = "hybrid",
) -> List[RetrievalHit]:
    """Run retrieval and return ranked, lineage-tagged hits.

    ``mode`` is one of ``"hybrid"`` (vector + lexical via RRF), ``"vector"``
    (semantic only), or ``"lexical"`` (keyword/BM25 only).
    """
    db_path = db_path or config.POCKET_SQLITE_DB
    model_name = model_name or config.EMBEDDING_MODEL
    if not Path(db_path).exists():
        return []

    # Over-fetch from each strategy so fusion has enough candidates to reorder.
    fetch_n = max(limit * 4, limit)

    conn = _connect(Path(db_path))
    try:
        vector_rows: List[tuple] = []
        lexical_rows: List[tuple] = []

        if mode in ("hybrid", "vector"):
            model = _get_model(model_name)
            query_embedding = model.encode(query, normalize_embeddings=True)
            query_vector = sqlite_vec.serialize_float32(query_embedding)
            vector_rows = _vector_search(conn, query_vector, fetch_n)

        if mode in ("hybrid", "lexical") and _fts_available(conn):
            lexical_rows = _lexical_search(conn, query, fetch_n)
    finally:
        conn.close()

    return _fuse(vector_rows, lexical_rows, limit)


def _fuse(
    vector_rows: List[tuple],
    lexical_rows: List[tuple],
    limit: int,
) -> List[RetrievalHit]:
    """Combine vector and lexical results with Reciprocal Rank Fusion.

    Rows are keyed by chunk id so the same chunk found by both strategies has
    its reciprocal-rank contributions summed.
    """
    accum: Dict[int, RetrievalHit] = {}

    for rank, row in enumerate(vector_rows, start=1):
        chunk_id, file_path, text, start, end, _distance = row
        hit = accum.get(chunk_id)
        if hit is None:
            hit = RetrievalHit(file_path, text, start, end, score=0.0)
            accum[chunk_id] = hit
        hit.vector_rank = rank
        hit.score += 1.0 / (RRF_K + rank)

    for rank, row in enumerate(lexical_rows, start=1):
        chunk_id, file_path, text, start, end, _bm25 = row
        hit = accum.get(chunk_id)
        if hit is None:
            hit = RetrievalHit(file_path, text, start, end, score=0.0)
            accum[chunk_id] = hit
        hit.lexical_rank = rank
        hit.score += 1.0 / (RRF_K + rank)

    ranked = sorted(accum.values(), key=lambda h: h.score, reverse=True)
    return ranked[:limit]


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
