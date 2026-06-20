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
import json
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
    graph_rank: Optional[int] = None

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

    ``mode`` is one of ``"hybrid"`` (vector + lexical + graph via RRF),
    ``"vector"`` (semantic only), ``"lexical"`` (keyword/BM25 only), or
    ``"graph"`` (GraphRAG: entity-anchored multi-hop traversal). The graph
    strategy only participates when an ``entities`` table is present, so
    ``"hybrid"`` stays backward compatible on graph-less databases.
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
        graph_rows: List[tuple] = []

        # Both vector search and graph entity-anchoring need the query embedding.
        query_vector = None
        if mode in ("hybrid", "vector", "graph"):
            model = _get_model(model_name)
            query_embedding = model.encode(query, normalize_embeddings=True)
            query_vector = sqlite_vec.serialize_float32(query_embedding)

        if mode in ("hybrid", "vector"):
            vector_rows = _vector_search(conn, query_vector, fetch_n)

        if mode in ("hybrid", "lexical") and _fts_available(conn):
            lexical_rows = _lexical_search(conn, query, fetch_n)

        if mode in ("hybrid", "graph") and _graph_available(conn):
            graph_rows = _graph_search(conn, query_vector, fetch_n)
    finally:
        conn.close()

    return _fuse(vector_rows, lexical_rows, limit, graph_rows)


def _fuse(
    vector_rows: List[tuple],
    lexical_rows: List[tuple],
    limit: int,
    graph_rows: Optional[List[tuple]] = None,
) -> List[RetrievalHit]:
    """Combine vector, lexical, and graph results with Reciprocal Rank Fusion.

    Rows are keyed by chunk id so the same chunk found by several strategies has
    its reciprocal-rank contributions summed. ``graph_rows`` is the optional
    third (GraphRAG) list; passing it preserves the original two-list signature.
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

    for rank, row in enumerate(graph_rows or [], start=1):
        chunk_id, file_path, text, start, end, _g = row
        hit = accum.get(chunk_id)
        if hit is None:
            hit = RetrievalHit(file_path, text, start, end, score=0.0)
            accum[chunk_id] = hit
        hit.graph_rank = rank
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


def _load_chunk_ids(raw: Optional[str]) -> List[int]:
    """Parse an entity's ``source_chunk_ids`` JSON column into a list of ints."""
    if not raw:
        return []
    try:
        ids = json.loads(raw)
    except (ValueError, TypeError):
        return []
    out: List[int] = []
    for cid in ids if isinstance(ids, list) else []:
        try:
            out.append(int(cid))
        except (ValueError, TypeError):
            continue
    return out


def _graph_search(
    conn: sqlite3.Connection,
    query_vector,
    limit: int,
) -> List[tuple]:
    """GraphRAG retriever: anchor the query to entities, traverse one hop, and
    surface the chunks that produced the touched nodes and edges (POCKET-404d).

    Returns rows shaped like the vector/lexical retrievers
    ``(chunk_id, file_path, text, start, end, _score)`` in graph-relevance order
    so :func:`_fuse` can blend them as a third ranked list. Every row resolves
    back to a real ``embeddings`` chunk, so the citation/lineage guarantee holds.
    Returns ``[]`` when there is no graph or no anchorable seed.
    """
    if not _graph_available(conn):
        return []

    # 1. Entity anchoring: nearest seed nodes by name embedding.
    seed_n = max(3, limit)
    seeds = conn.execute(
        "SELECT id, source_chunk_ids, vec_distance_cosine(embedding, ?) AS d "
        f"FROM entities WHERE {_status_clause(conn, 'entities')} "
        "ORDER BY d ASC LIMIT ?",
        (query_vector, seed_n),
    ).fetchall()
    if not seeds:
        return []

    ordered_ids: List[int] = []
    seen: set = set()

    def _push(cid: Optional[int]) -> None:
        if cid is not None and cid not in seen:
            seen.add(cid)
            ordered_ids.append(cid)

    for node_id, chunk_ids_json, _d in seeds:
        # 2. The seed's own mention chunks rank first.
        for cid in _load_chunk_ids(chunk_ids_json):
            _push(cid)
        # 3. One-hop traversal: the edge's source chunk plus the neighbor's
        #    mention chunks, ordered by edge confidence.
        edges = conn.execute(
            f"""
            SELECT r.source_chunk_id, r.subject_id, r.object_id,
                   sj.source_chunk_ids AS subj_chunks,
                   ob.source_chunk_ids AS obj_chunks
            FROM relations r
            LEFT JOIN entities sj ON sj.id = r.subject_id
            LEFT JOIN entities ob ON ob.id = r.object_id
            WHERE (r.subject_id = ? OR r.object_id = ?)
              AND {_status_clause(conn, 'relations', 'r')}
            ORDER BY r.confidence DESC
            """,
            (node_id, node_id),
        ).fetchall()
        for edge_chunk, subj_id, _obj_id, subj_chunks, obj_chunks in edges:
            _push(edge_chunk)
            neighbor_chunks = obj_chunks if subj_id == node_id else subj_chunks
            for cid in _load_chunk_ids(neighbor_chunks):
                _push(cid)

    ordered_ids = ordered_ids[:limit]
    if not ordered_ids:
        return []

    placeholders = ",".join("?" * len(ordered_ids))
    rows = conn.execute(
        f"SELECT id, file_path, text, start_offset, end_offset "
        f"FROM embeddings WHERE id IN ({placeholders})",
        ordered_ids,
    ).fetchall()
    by_id = {r[0]: r for r in rows}
    # Preserve graph-relevance order; some chunk ids may no longer exist.
    return [(*by_id[cid], 0.0) for cid in ordered_ids if cid in by_id]

def graph_neighborhood(
    entity: str,
    limit: int = 10,
    db_path: Optional[Path] = None,
    model_name: Optional[str] = None,
) -> Dict:
    """Return a knowledge-graph node's 1-hop neighborhood (POCKET-404d).

    Resolves ``entity`` to the nearest node by name (exact/alias match first,
    then vector similarity over entity-name embeddings) and returns it with the
    relations it participates in, each resolved to the neighbor's name. Every
    edge keeps its evidence span and source file so a caller can cite it.
    """
    db_path = db_path or config.POCKET_SQLITE_DB
    model_name = model_name or config.EMBEDDING_MODEL
    if not Path(db_path).exists():
        return {}
    conn = _connect(Path(db_path))
    try:
        if not _graph_available(conn):
            return {}
        # 1. Exact / alias match.
        row = conn.execute(
            "SELECT id, name, type, aliases, confidence, source_file "
            f"FROM entities WHERE lower(name) = lower(?) "
            f"AND {_status_clause(conn, 'entities')} LIMIT 1",
            (entity,),
        ).fetchone()
        # 2. Fall back to nearest by name embedding.
        if row is None:
            model = _get_model(model_name)
            qv = sqlite_vec.serialize_float32(
                model.encode(entity, normalize_embeddings=True)
            )
            row = conn.execute(
                "SELECT id, name, type, aliases, confidence, source_file, "
                "vec_distance_cosine(embedding, ?) AS d "
                f"FROM entities WHERE {_status_clause(conn, 'entities')} "
                "ORDER BY d ASC LIMIT 1",
                (qv,),
            ).fetchone()
        if row is None:
            return {}
        node_id, name, etype, aliases, confidence, source_file = row[:6]
        edges = conn.execute(
            f"""
            SELECT r.predicate, r.object_id, r.subject_id, r.evidence,
                   r.confidence, r.source_file,
                   so.name AS subject_name, ob.name AS object_name
            FROM relations r
            LEFT JOIN entities so ON so.id = r.subject_id
            LEFT JOIN entities ob ON ob.id = r.object_id
            WHERE (r.subject_id = ? OR r.object_id = ?)
              AND {_status_clause(conn, 'relations', 'r')}
            ORDER BY r.confidence DESC
            LIMIT ?
            """,
            (node_id, node_id, limit),
        ).fetchall()
    finally:
        conn.close()

    neighbors = []
    for (
        predicate,
        object_id,
        subject_id,
        evidence,
        edge_conf,
        edge_src,
        subject_name,
        object_name,
    ) in edges:
        outgoing = subject_id == node_id
        neighbors.append(
            {
                "direction": "out" if outgoing else "in",
                "predicate": predicate,
                "neighbor": object_name if outgoing else subject_name,
                "evidence": evidence,
                "confidence": edge_conf,
                "source_file": edge_src,
            }
        )
    return {
        "id": node_id,
        "name": name,
        "type": etype,
        "aliases": aliases,
        "confidence": confidence,
        "source_file": source_file,
        "neighbors": neighbors,
    }


def format_neighborhood(node: Dict) -> str:
    """Render a graph neighborhood for the CLI / MCP."""
    if not node:
        return "No matching entity found in the graph."
    import json as _json

    aliases = node.get("aliases")
    try:
        alias_list = _json.loads(aliases) if aliases else []
    except (ValueError, TypeError):
        alias_list = []
    lines = [
        f"Entity: {node['name']} ({node['type']}) "
        f"[confidence {node.get('confidence', 0):.2f}]",
        f"Source: {node.get('source_file', '?')}",
    ]
    if alias_list:
        lines.append(f"Aliases: {', '.join(alias_list)}")
    neighbors = node.get("neighbors", [])
    if not neighbors:
        lines.append("(no relations)")
    else:
        lines.append(f"Relations ({len(neighbors)}):")
        for n in neighbors:
            arrow = "->" if n["direction"] == "out" else "<-"
            lines.append(
                f"  {arrow} {n['predicate']} {arrow} {n['neighbor']} "
                f"[{n.get('confidence', 0):.2f}]"
            )
    return "\n".join(lines)


def list_graph_concepts(
    concept: str | None = None,
    limit: int = 20,
    db_path: Optional[Path] = None,
) -> List[Dict]:
    """Return top entities (and their top relation) from the knowledge graph.

    Requires a graph built with ``pocket update --graph`` (``POCKET_GRAPH=1``).
    When *concept* is given, filters by case-insensitive prefix match on the
    entity name; otherwise returns the highest-confidence entities up to *limit*.

    Each dict has keys: name, type, confidence, source_file, top_relation.
    Returns an empty list when the graph tables do not exist.
    """
    db_path = db_path or config.POCKET_SQLITE_DB
    if not Path(db_path).exists():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        if not _graph_available(conn):
            return []
        if concept:
            rows = conn.execute(
                f"""
                SELECT id, name, type, confidence, source_file
                FROM entities
                WHERE lower(name) LIKE lower(?)
                  AND {_status_clause(conn, 'entities')}
                ORDER BY confidence DESC
                LIMIT ?
                """,
                (concept.lower() + "%", limit),
            ).fetchall()
        else:
            rows = conn.execute(
                f"""
                SELECT id, name, type, confidence, source_file
                FROM entities
                WHERE {_status_clause(conn, 'entities')}
                ORDER BY confidence DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        results = []
        for eid, name, etype, conf, src in rows:
            # Fetch the single most-confident relation for context.
            rel = conn.execute(
                f"""
                SELECT r.predicate, ob.name AS obj_name, so.name AS sub_name,
                       r.subject_id
                FROM relations r
                LEFT JOIN entities ob ON ob.id = r.object_id
                LEFT JOIN entities so ON so.id = r.subject_id
                WHERE (r.subject_id = ? OR r.object_id = ?)
                  AND {_status_clause(conn, 'relations', 'r')}
                ORDER BY r.confidence DESC
                LIMIT 1
                """,
                (eid, eid),
            ).fetchone()
            top_relation = None
            if rel:
                pred, obj_name, sub_name, subj_id = rel
                if subj_id == eid:
                    top_relation = f"{name} -{pred}-> {obj_name}"
                else:
                    top_relation = f"{sub_name} -{pred}-> {name}"
            results.append(
                {
                    "name": name,
                    "type": etype,
                    "confidence": conf,
                    "source_file": src,
                    "top_relation": top_relation,
                }
            )
        return results
    finally:
        conn.close()
