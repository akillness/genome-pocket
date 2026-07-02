import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
import sqlite_vec
import pocket.config as config
from .db import _connect, _graph_available, _status_clause
from .encode import _get_model, _encode_query


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
            from .base import _resolve_callable
            model = _resolve_callable("_get_model", _get_model)(model_name)
            qv = sqlite_vec.serialize_float32(
                _encode_query(model, entity)
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
