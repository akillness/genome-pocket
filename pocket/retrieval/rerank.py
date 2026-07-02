import json
import sqlite3
from functools import lru_cache
from typing import Dict, List, Optional
import urllib.request
import urllib.error
import numpy as np
import pocket.config as config
from .base import RetrievalHit


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
    from .base import _resolve_callable
    model = _resolve_callable("_get_reranker", _get_reranker)(model_name)
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
