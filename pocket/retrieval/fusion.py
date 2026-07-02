from typing import Dict, List, Optional
import pocket.config as config
from .base import RetrievalHit, RRF_K


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
