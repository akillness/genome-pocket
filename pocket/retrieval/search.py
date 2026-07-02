import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
import sqlite_vec
import pocket.config as config
from .base import RetrievalHit, _FTS_TABLE
from .db import _connect, _fts_available, _graph_available
from .encode import _get_model, _encode_query
from .router import _resolve_mode, _MODE_STRATEGIES
from .fusion import _fuse_ranked, _fuse
from .rerank import _fetch_embeddings, _mmr_rerank, _hyde_expand, _rerank
from .graph import _graph_search


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


_QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _expand_query(query: str, synonyms: Dict[str, List[str]]) -> str:
    """Append synonym / acronym expansion terms to a query (POCKET-503).

    For each lowercased query token present in ``synonyms``, append every word of
    its expansion phrase(s) that is not already in the query. Order is preserved
    and tokens are de-duplicated, so the result is deterministic. Returns the
    query unchanged when ``synonyms`` is empty or nothing matches — making the
    default (expansion off / empty map) a strict no-op.

    Why append rather than replace: the original tokens still carry signal (BM25
    rank, vector mass), so expansion only *adds* recall. Only the bare-word
    expansion of acronyms is added, never paraphrase soup, keeping the change
    small and predictable.
    """
    if not synonyms:
        return query
    present = {t.lower() for t in _QUERY_TOKEN_RE.findall(query)}
    extra: List[str] = []
    for tok in _QUERY_TOKEN_RE.findall(query.lower()):
        for phrase in synonyms.get(tok, ()):  # type: ignore[arg-type]
            for word in _QUERY_TOKEN_RE.findall(phrase.lower()):
                if word not in present:
                    present.add(word)
                    extra.append(word)
    if not extra:
        return query
    return f"{query} {' '.join(extra)}"


def _gather(
    conn: sqlite3.Connection,
    query: str,
    mode: str,
    fetch_n: int,
    model_name: str,
    hyde_query: Optional[str] = None,
) -> tuple:
    """Run each enabled retrieval strategy and return their raw ranked rows.

    Shared by :func:`search` and :func:`routing_trace` so the routing decision —
    which strategies run, given the requested ``mode`` and which target tables
    exist — lives in exactly one place and can never drift between a real search
    and the trace the UI renders for the same query.
    """
    vector_rows: List[tuple] = []
    lexical_rows: List[tuple] = []
    graph_rows: List[tuple] = []

    # HyDE: embed hypothetical doc (hyde_query) instead of raw query; lexical always uses raw query
    query_vector = None
    if mode in ("hybrid", "vector", "graph"):
        from .base import _resolve_callable
        model = _resolve_callable("_get_model", _get_model)(model_name)
        vector_text = hyde_query if hyde_query else query
        query_embedding = _encode_query(model, vector_text)
        query_vector = sqlite_vec.serialize_float32(query_embedding)

    if mode in ("hybrid", "vector"):
        vector_rows = _vector_search(conn, query_vector, fetch_n)

    if mode in ("hybrid", "lexical") and _fts_available(conn):
        lexical_rows = _lexical_search(conn, query, fetch_n)

    if mode in ("hybrid", "graph") and _graph_available(conn):
        graph_rows = _graph_search(conn, query_vector, fetch_n)

    return vector_rows, lexical_rows, graph_rows


def search(
    query: str,
    limit: int = 5,
    db_path: Optional[Path] = None,
    model_name: Optional[str] = None,
    mode: str = "hybrid",
    use_mmr: Optional[bool] = None,
    mmr_lambda: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    use_reranker: Optional[bool] = None,
    use_hyde: Optional[bool] = None,
    use_expansion: Optional[bool] = None,
) -> List[RetrievalHit]:
    """Run retrieval and return ranked, lineage-tagged hits.

    ``mode`` is one of ``"hybrid"`` (vector + lexical + graph via RRF),
    ``"vector"`` (semantic only), ``"lexical"`` (keyword/BM25 only),
    ``"graph"`` (GraphRAG: entity-anchored multi-hop traversal), or ``"auto"``
    (POCKET-504: the semantic router picks a concrete mode from the query's
    shape — code-like queries route to lexical, relationship questions to graph,
    otherwise hybrid). A plain ``"hybrid"`` is auto-routed too when
    ``config.POCKET_QUERY_ROUTER`` is enabled. The graph strategy only
    participates when an ``entities`` table is present, so ``"hybrid"`` stays
    backward compatible on graph-less databases.

    ``weights`` scales each strategy's RRF contribution (POCKET-502); ``None``
    uses ``config.POCKET_RRF_WEIGHTS`` (1.0 each == plain unweighted RRF).

    When ``use_mmr`` is true (defaults to ``config.POCKET_MMR``) the fused
    candidates are re-ranked with Maximal Marginal Relevance so near-duplicate
    chunks don't crowd the top-k; ``mmr_lambda`` (defaults to
    ``config.POCKET_MMR_LAMBDA``) trades relevance (1.0) against diversity (0.0).

    When ``use_hyde`` is true (defaults to ``config.POCKET_HYDE``) the query is
    first expanded by a local Ollama model into a hypothetical answer passage
    which is used for vector/graph embedding instead of the raw query string.
    Lexical (BM25) search is always run on the original query.

    When ``use_expansion`` is true (defaults to ``config.POCKET_QUERY_EXPANSION``)
    the query is first augmented with deterministic synonym/acronym expansion
    terms from ``config.POCKET_QUERY_EXPANSION_MAP`` (POCKET-503), helping both the
    lexical index and the embedding match documents that only spell out the long
    form of an abbreviation. This runs before HyDE; when HyDE is also active its
    generated passage still takes precedence for the vector/graph embedding.

    When ``use_reranker`` is true (defaults to ``config.POCKET_RERANKER``) the
    top ``POCKET_RERANKER_TOP_N`` fused candidates are re-scored by a
    cross-encoder model, raising precision before the final top-``limit`` cut.
    """
    db_path = db_path or config.POCKET_SQLITE_DB
    model_name = model_name or config.EMBEDDING_MODEL
    if use_mmr is None:
        use_mmr = config.POCKET_MMR
    if mmr_lambda is None:
        mmr_lambda = config.POCKET_MMR_LAMBDA
    if use_reranker is None:
        use_reranker = config.POCKET_RERANKER
    if use_hyde is None:
        use_hyde = config.POCKET_HYDE
    if use_expansion is None:
        use_expansion = config.POCKET_QUERY_EXPANSION
    if not Path(db_path).exists():
        return []

    # Query expansion (POCKET-503): augment the query with synonym/acronym terms
    # so vector + lexical both see the long form of an abbreviation. The original
    # query is kept for HyDE generation and the reranker (raw user intent).
    gather_query = query
    if use_expansion:
        gather_query = _expand_query(query, config.POCKET_QUERY_EXPANSION_MAP)

    # HyDE: expand the query into a hypothetical passage for vector/graph encoding.
    # Lexical (BM25) always uses the original query keywords.
    hyde_query: Optional[str] = None
    if use_hyde:
        hyde_query = _hyde_expand(
            query,
            ollama_model=config.POCKET_HYDE_OLLAMA_MODEL,
            ollama_host=config.POCKET_HYDE_OLLAMA_HOST,
        )

    # When the reranker is active, gather a larger candidate pool first so the
    # cross-encoder has enough material to reorder before the final top-k cut.
    pre_limit = config.POCKET_RERANKER_TOP_N if use_reranker else limit
    fetch_n = max(pre_limit * 4, pre_limit)

    conn = _connect(Path(db_path))
    try:
        # POCKET-504: resolve "auto"/opt-in hybrid to a concrete mode; after connect so graph can fall back
        mode = _resolve_mode(query, mode, conn)

        vector_rows, lexical_rows, graph_rows = _gather(
            conn, gather_query, mode, fetch_n, model_name, hyde_query=hyde_query
        )

        if use_mmr:
            # MMR needs embeddings fetched while the connection is open.
            candidates = _fuse_ranked(vector_rows, lexical_rows, graph_rows, weights)
            embeddings = _fetch_embeddings(conn, [cid for cid, _ in candidates])
        else:
            pre_hits = _fuse(vector_rows, lexical_rows, pre_limit, graph_rows, weights)
    finally:
        conn.close()

    if use_mmr:
        paired = [(hit, embeddings.get(cid)) for cid, hit in candidates]
        from .base import _resolve_callable
        pre_hits = _resolve_callable("_mmr_rerank", _mmr_rerank)(paired, mmr_lambda, pre_limit)

    if use_reranker:
        pre_hits = _rerank(query, pre_hits, config.POCKET_RERANKER_MODEL)

    return pre_hits[:limit]
