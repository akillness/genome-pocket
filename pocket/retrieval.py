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
import re
import sqlite3
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import sqlite_vec
import numpy as np

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
    # Set by _rerank() when the cross-encoder reranker is active (POCKET_RERANKER=1).
    reranker_rank: Optional[int] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@lru_cache(maxsize=4)
def _get_model(model_name: str):
    """Cache the embedding model so repeated queries don't reload weights.

    Delegates model-type selection to the shared embedder registry
    (:func:`pocketindex.ops.sentence_transformers.resolve_backend`) so the query
    side can never drift from the ingestion side. Returns a multimodal
    :class:`SiglipEmbedder` for siglip2 ids (so a text query is encoded into the
    shared image/text space), or a plain SentenceTransformer for text models.
    """
    from pocketindex.ops.sentence_transformers import resolve_backend

    return resolve_backend(model_name).query_model(model_name)


def _encode_query(model, text: str):
    """Encode query-side text into the index's vector space.

    - Multimodal SigLIP2 (``encode_query``): text -> shared image/text space so the
      query can match stored image embeddings.
    - Instruction-aware text models (e.g. Qwen3-Embedding) define a ``query`` prompt
      that must wrap the query for the asymmetric retrieval recipe.
    - Symmetric models such as all-MiniLM expose no prompts and are encoded plainly.
    """
    if hasattr(model, "encode_query"):
        return model.encode_query(text)
    kwargs = {"normalize_embeddings": True}
    if "query" in (getattr(model, "prompts", None) or {}):
        kwargs["prompt_name"] = "query"
    return model.encode(text, **kwargs)


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



# Which strategies each retrieval mode activates. Single source of truth for the
# router so the CLI, REST API, and tracing UI agree on what "hybrid" means.
_MODE_STRATEGIES = {
    "hybrid": ("vector", "lexical", "graph"),
    "vector": ("vector",),
    "lexical": ("lexical",),
    "graph": ("graph",),
}


# POCKET-504 semantic query router — deterministic, offline; see config.POCKET_QUERY_ROUTER

# Conservative: only unambiguous code shapes — a false "lexical" route drops vector strategy
_CODE_SHAPE_RE = re.compile(
    r"""
      \b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]*\b   # snake_case identifier (parse_payload)
    | \b[a-z]+[A-Z][A-Za-z0-9]*\b              # camelCase identifier (parsePayload)
    | \b[A-Za-z_][A-Za-z0-9_]*\s*\(            # function/method call: foo(
    | ::[A-Za-z_]                              # C++/Rust scope: ns::sym
    | \b[A-Za-z_][A-Za-z0-9_]*\.(py|js|ts|tsx|jsx|go|rs|rb|java|c|cpp|h|hpp|sql|sh|md|json|yaml|yml|toml)\b  # filename.ext
    | [{}\[\];]                                # code punctuation
    | `[^`]+`                                  # an explicit `code span`
    """,
    re.VERBOSE,
)

# Concept/relationship phrasings → graph multi-hop; flat vector/lexical answers them poorly
_CONCEPT_PHRASES = (
    "related to",
    "relationship between",
    "relation between",
    "connection between",
    "connected to",
    "linked to",
    "links between",
    "associated with",
    "depends on",
    "depend on",
    "dependency between",
    "how does",
    "how do",
    "impact of",
    "interact with",
    "interaction between",
    "difference between",
)


def _route_query(query: str) -> str:
    """Classify a query's shape into a concrete retrieval mode (POCKET-504).

    Returns one of ``"lexical"``, ``"graph"``, or ``"hybrid"``. Pure and
    deterministic (regex + keyword shape only, no I/O) so it is unit-testable and
    its routing decision is reproducible.

    Priority: relationship/concept phrasing → ``graph`` first, because a question
    like *"how does write_ahead_log relate to recovery"* is a relationship query
    even though it embeds a code token; then unambiguous code shape → ``lexical``;
    otherwise the ``hybrid`` blend, which is the safe default for prose.
    """
    text = query.strip()
    lowered = text.lower()
    if any(phrase in lowered for phrase in _CONCEPT_PHRASES):
        return "graph"
    if _CODE_SHAPE_RE.search(text):
        return "lexical"
    return "hybrid"


def _resolve_mode(query: str, mode: str, conn: sqlite3.Connection) -> str:
    """Resolve ``mode`` to a concrete strategy, applying the router (POCKET-504).

    ``"auto"`` always routes; a plain ``"hybrid"`` routes only when
    ``config.POCKET_QUERY_ROUTER`` is enabled (opt-in upgrade for existing
    hybrid callers). Any other mode is returned unchanged. A routed ``"graph"``
    falls back to ``"hybrid"`` when the target has no graph tables, so routing
    can never silently return zero results on a graph-less database.
    """
    if mode == "auto" or (mode == "hybrid" and config.POCKET_QUERY_ROUTER):
        routed = _route_query(query)
        if routed == "graph" and not _graph_available(conn):
            return "hybrid"
        return routed
    return mode


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
        model = _get_model(model_name)
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
        pre_hits = _mmr_rerank(paired, mmr_lambda, pre_limit)

    if use_reranker:
        pre_hits = _rerank(query, pre_hits, config.POCKET_RERANKER_MODEL)

    return pre_hits[:limit]



def _resolve_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Merge caller weights over the configured defaults (POCKET-502).

    Returns a full ``{vector, lexical, graph}`` map: any strategy the caller
    omits falls back to ``config.POCKET_RRF_WEIGHTS`` (itself defaulting to 1.0
    each == plain unweighted RRF). Negative weights are clamped to 0 so a
    strategy can be disabled but never invert a chunk's score.
    """
    resolved = dict(config.POCKET_RRF_WEIGHTS)
    if weights:
        for name in ("vector", "lexical", "graph"):
            if name in weights:
                resolved[name] = max(float(weights[name]), 0.0)
    return resolved


def _fold_ranked(
    accum: Dict[int, RetrievalHit],
    rows: List[tuple],
    rank_attr: str,
    weight: float = 1.0,
) -> None:
    """Fold one strategy's ranked rows into the shared RRF accumulator.

    Each row contributes ``weight * 1/(RRF_K + rank)`` to its chunk's fused
    score (weighted Reciprocal Rank Fusion, POCKET-502) and records its 1-based
    position in ``rank_attr`` (``"vector_rank"`` / ``"lexical_rank"`` /
    ``"graph_rank"``). A chunk surfaced by several strategies is keyed by its
    chunk id, so the contributions land on the same :class:`RetrievalHit` and
    sum. ``weight`` defaults to 1.0, reproducing plain RRF.
    """
    for rank, row in enumerate(rows, start=1):
        chunk_id, file_path, text, start, end, _score = row
        hit = accum.get(chunk_id)
        if hit is None:
            hit = RetrievalHit(file_path, text, start, end, score=0.0)
            accum[chunk_id] = hit
        setattr(hit, rank_attr, rank)
        hit.score += weight * (1.0 / (RRF_K + rank))


def _fuse_ranked(
    vector_rows: List[tuple],
    lexical_rows: List[tuple],
    graph_rows: Optional[List[tuple]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> List[tuple]:
    """Fuse the strategies with weighted RRF and return ``(chunk_id, hit)`` pairs.

    Sorted by fused score (descending) but *not* truncated, so callers that need
    the full candidate pool keyed by chunk id (e.g. MMR re-ranking, which must
    join each candidate back to its embedding) get everything fusion saw.
    ``weights`` scales each strategy's contribution (POCKET-502); ``None`` uses
    the configured defaults (1.0 each == plain RRF).
    """
    w = _resolve_weights(weights)
    accum: Dict[int, RetrievalHit] = {}
    _fold_ranked(accum, vector_rows, "vector_rank", w["vector"])
    _fold_ranked(accum, lexical_rows, "lexical_rank", w["lexical"])
    _fold_ranked(accum, graph_rows or [], "graph_rank", w["graph"])
    return sorted(accum.items(), key=lambda kv: kv[1].score, reverse=True)


def _fuse(
    vector_rows: List[tuple],
    lexical_rows: List[tuple],
    limit: int,
    graph_rows: Optional[List[tuple]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> List[RetrievalHit]:
    """Combine vector, lexical, and graph results with weighted Reciprocal Rank
    Fusion.

    Rows are keyed by chunk id so the same chunk found by several strategies has
    its reciprocal-rank contributions summed. ``graph_rows`` is the optional
    third (GraphRAG) list; passing it preserves the original two-list signature.
    ``weights`` (POCKET-502) scales each strategy; ``None`` keeps plain RRF.
    """
    ranked = _fuse_ranked(vector_rows, lexical_rows, graph_rows, weights)
    return [hit for _cid, hit in ranked[:limit]]



def _fetch_embeddings(
    conn: sqlite3.Connection, chunk_ids: List[int]
) -> Dict[int, "np.ndarray"]:
    """Load the stored float32 embedding for each chunk id (for MMR).

    Embeddings are persisted as sqlite-vec ``serialize_float32`` blobs (raw
    little-endian float32), so ``np.frombuffer`` recovers the vector without a
    re-encode. Missing/NULL rows are simply omitted.
    """
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" * len(chunk_ids))
    cur = conn.execute(
        f"SELECT id, embedding FROM embeddings WHERE id IN ({placeholders})",
        tuple(chunk_ids),
    )
    out: Dict[int, "np.ndarray"] = {}
    for cid, blob in cur.fetchall():
        if blob is None:
            continue
        out[cid] = np.frombuffer(blob, dtype=np.float32)
    return out


def _cosine(a: Optional["np.ndarray"], b: Optional["np.ndarray"]) -> float:
    """Cosine similarity, defined as 0 when either vector is missing or zero.

    Returning 0 for degenerate inputs means MMR treats unmeasurable pairs as
    non-redundant and falls back toward the relevance order rather than erroring.
    """
    if a is None or b is None:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _mmr_rerank(
    candidates: List[tuple],
    mmr_lambda: float,
    limit: int,
) -> List[RetrievalHit]:
    """Re-rank fused candidates with Maximal Marginal Relevance.

    ``candidates`` is ``[(RetrievalHit, embedding_or_None), ...]`` pre-sorted by
    fused (RRF) relevance. Each pick maximises
    ``λ·rel(d) − (1−λ)·max_{s∈selected} cos(d, s)``: relevance is the fused score
    (normalised to the top candidate), redundancy is the highest cosine to an
    already-selected chunk. ``λ=1`` reproduces the plain relevance order; lower λ
    pushes diverse chunks up. Stable: ties keep the incoming (relevance) order.
    """
    mmr_lambda = min(max(mmr_lambda, 0.0), 1.0)
    if not candidates:
        return []
    max_score = max((hit.score for hit, _ in candidates), default=0.0) or 1.0

    remaining = list(candidates)
    selected: List[tuple] = []
    while remaining and len(selected) < limit:
        best_i = 0
        best_val = None
        for i, (hit, emb) in enumerate(remaining):
            rel = hit.score / max_score
            redundancy = max(
                (_cosine(emb, semb) for _, semb in selected), default=0.0
            )
            val = mmr_lambda * rel - (1.0 - mmr_lambda) * redundancy
            if best_val is None or val > best_val:
                best_val = val
                best_i = i
        selected.append(remaining.pop(best_i))
    return [hit for hit, _ in selected]


# ---------------------------------------------------------------------------
# HyDE — Hypothetical Document Embeddings (arXiv:2212.10496)
# ---------------------------------------------------------------------------

def _hyde_expand(query: str, *, ollama_model: str, ollama_host: str) -> str:
    """Generate a hypothetical passage for vector-query encoding (HyDE).

    Sends the query to a local Ollama model, asks it to write a short passage
    that would directly answer the question, then returns that passage.  The
    caller embeds the passage instead of the bare query so the query vector
    lands in the same region of the index as real document chunks — bridging
    the asymmetric short-query / long-document semantic gap.

    Falls back to the original ``query`` string when Ollama is unavailable so
    the search path degrades silently with no exception.
    """
    import urllib.error
    import urllib.request

    prompt = (
        "Write a short, dense passage (2–4 sentences) that directly answers "
        "the following question. Write only the passage, no preamble:\n\n"
        f"{query}"
    )
    payload = json.dumps(
        {"model": ollama_model, "prompt": prompt, "stream": False}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{ollama_host.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        generated = body.get("response", "").strip()
        return generated if generated else query
    except Exception as exc:
        print(f"[pocket.retrieval] HyDE expansion failed, using original query: {exc}")
        return query


# ---------------------------------------------------------------------------
# Cross-encoder reranker (precision pass after RRF / MMR)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2)
def _get_reranker(model_name: str):
    """Load and cache a sentence_transformers CrossEncoder.

    Returns ``None`` when the class or model is unavailable (e.g. model not
    yet downloaded) so the caller can degrade gracefully to RRF order.
    Cached per model-name so swapping ``POCKET_RERANKER_MODEL`` at runtime
    does not reload an already-warm model on the next call.
    """
    try:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415
        return CrossEncoder(model_name)
    except Exception as exc:
        print(f"[pocket.retrieval] Reranker model {model_name!r} failed to load: {exc}")
        return None


def _rerank(
    query: str,
    hits: List[RetrievalHit],
    model_name: str,
) -> List[RetrievalHit]:
    """Re-score *hits* with a cross-encoder and return them sorted by that score.

    Cross-encoders attend to (query, passage) jointly so they weigh lexical
    overlap and semantic entailment together — precision typically rises 5–15 %
    over single-vector dot-product scoring at the cost of one forward pass per
    candidate.  The :attr:`RetrievalHit.reranker_rank` field is set on every
    returned hit so callers and the tracing UI can see where the reranker moved
    each chunk relative to the original RRF position.

    Falls back silently to the incoming RRF order when the model is unavailable.
    """
    if not hits:
        return hits
    model = _get_reranker(model_name)
    if model is None:
        return hits
    try:
        pairs = [(query, h.text) for h in hits]
        scores = model.predict(pairs)
        ranked = sorted(
            zip(scores, hits), key=lambda x: float(x[0]), reverse=True
        )
        result: List[RetrievalHit] = []
        for rank, (score, hit) in enumerate(ranked, start=1):
            hit.reranker_rank = rank
            hit.score = float(score)
            result.append(hit)
        return result
    except Exception as exc:
        print(f"[pocket.retrieval] Reranker scoring failed, using RRF order: {exc}")
        return hits


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
