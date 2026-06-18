"""Entity-resolution op for PocketIndex.

Deduplicates extracted entities following the cost-effective
*blocking → cheap filters → (optional) LLM adjudication → label propagation*
pipeline from the spec (``docs/architecture/graph-target.md`` §4.2, grounded in
arXiv:2605.25814). Where the upstream ``cocoindex.ops.entity_resolution`` uses a
faiss index, we block with embedding cosine similarity (the same vectors the
sqlite-vec target stores) so there is no new dependency.

The core of this module is pure and offline-testable: given a set of entities
with optional embeddings, it produces merge clusters with no network or LLM call.
LLM adjudication is an optional injected callable, so the resolver itself never
depends on a model being available.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from pocketindex.ops.extract import ExtractedEntity


def normalize(name: str) -> str:
    """Canonical comparison key: lowercased, punctuation-stripped, despaced."""
    cleaned = re.sub(r"[^\w\s]", " ", name.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors; 0.0 if either is empty/degenerate."""
    if a is None or b is None or len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class ResolvedEntity:
    """A merged cluster of surface forms referring to one real entity."""

    name: str                      # chosen canonical surface form
    type: str
    aliases: List[str] = field(default_factory=list)
    confidence: float = 0.0        # max confidence across members
    members: List[ExtractedEntity] = field(default_factory=list)


# A merge-decision callable: given two entities, return True to merge. Used to
# inject optional LLM adjudication without coupling the resolver to a model.
MergeAdjudicator = Callable[[ExtractedEntity, ExtractedEntity], bool]


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def resolve_entities(
    entities: Sequence[ExtractedEntity],
    embeddings: Optional[Sequence[Optional[Sequence[float]]]] = None,
    *,
    block_threshold: float = 0.60,
    auto_merge_threshold: float = 0.92,
    adjudicator: Optional[MergeAdjudicator] = None,
    top_k: int = 10,
) -> List[ResolvedEntity]:
    """Resolve duplicate entities into canonical clusters.

    Pipeline (per the spec):

    1. **Blocking** — only consider candidate pairs whose name embeddings are at
       least ``block_threshold`` similar (top-K neighbours), avoiding O(n²) work on
       large graphs. With no embeddings, falls back to blocking on a shared
       normalized-token prefix so the path is still offline-testable.
    2. **Cheap filters** — exact normalized-name match merges immediately; an
       embedding similarity above ``auto_merge_threshold`` (with compatible types)
       merges with no LLM call.
    3. **LLM adjudication (optional)** — ambiguous pairs (similar but not certain)
       are passed to ``adjudicator`` *only if one is supplied*; otherwise they are
       left unmerged (the conservative fallback).
    4. **Label propagation** — transitively merge via union-find, pick a canonical
       name (longest / highest-confidence surface form), fold the rest into aliases.
    """
    n = len(entities)
    if n == 0:
        return []

    norms = [normalize(e.name) for e in entities]
    if embeddings is None:
        embeddings = [None] * n

    uf = _UnionFind(n)

    # Build candidate pairs via blocking.
    candidate_pairs = _candidate_pairs(norms, embeddings, block_threshold, top_k)

    for i, j in candidate_pairs:
        # (2) Cheap filters first.
        if norms[i] == norms[j]:
            uf.union(i, j)
            continue
        sim = cosine(embeddings[i], embeddings[j]) if embeddings[i] is not None else 0.0
        types_compatible = (
            entities[i].type == entities[j].type
            or entities[i].type in ("Concept", "")
            or entities[j].type in ("Concept", "")
        )
        if sim >= auto_merge_threshold and types_compatible:
            uf.union(i, j)
            continue
        # (3) Ambiguous: adjudicate only if a decider was injected.
        if sim >= block_threshold and types_compatible and adjudicator is not None:
            try:
                if adjudicator(entities[i], entities[j]):
                    uf.union(i, j)
            except Exception as exc:  # noqa: BLE001 - degrade, don't crash
                print(f"[pocketindex.entity_resolution] adjudicator failed: {exc}")

    # (4) Label propagation: gather clusters and choose canonical forms.
    clusters: Dict[int, List[int]] = {}
    for idx in range(n):
        clusters.setdefault(uf.find(idx), []).append(idx)

    resolved: List[ResolvedEntity] = []
    for members_idx in clusters.values():
        members = [entities[i] for i in members_idx]
        canonical = _choose_canonical(members)
        aliases = sorted(
            {m.name for m in members if m.name != canonical.name}
        )
        resolved.append(
            ResolvedEntity(
                name=canonical.name,
                type=canonical.type,
                aliases=aliases,
                confidence=max(m.confidence for m in members),
                members=members,
            )
        )
    # Deterministic ordering for stable downstream ids/tests.
    resolved.sort(key=lambda r: r.name.lower())
    return resolved


def _candidate_pairs(
    norms: Sequence[str],
    embeddings: Sequence[Optional[Sequence[float]]],
    block_threshold: float,
    top_k: int,
) -> List[tuple]:
    n = len(norms)
    pairs: List[tuple] = []
    have_embeddings = any(e is not None for e in embeddings)

    if have_embeddings:
        for i in range(n):
            if embeddings[i] is None:
                continue
            sims = []
            for j in range(n):
                if i == j or embeddings[j] is None:
                    continue
                s = cosine(embeddings[i], embeddings[j])
                if s >= block_threshold:
                    sims.append((s, j))
            sims.sort(reverse=True)
            for _s, j in sims[:top_k]:
                pairs.append((min(i, j), max(i, j)))
    # Always supplement embedding blocks (or fully substitute, when there are no
    # embeddings) with exact normalized-name groups and a shared-leading-token
    # block, so the resolver works even with no embeddings at all.
    by_norm: Dict[str, List[int]] = {}
    by_norm: Dict[str, List[int]] = {}
    by_prefix: Dict[str, List[int]] = {}
    for idx, norm in enumerate(norms):
        by_norm.setdefault(norm, []).append(idx)
        token = norm.split(" ", 1)[0] if norm else ""
        if token:
            by_prefix.setdefault(token, []).append(idx)
    for group in by_norm.values():
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                pairs.append((group[a], group[b]))
    if not have_embeddings:
        for group in by_prefix.values():
            for a in range(len(group)):
                for b in range(a + 1, len(group)):
                    pairs.append((group[a], group[b]))

    return sorted(set(pairs))


def _choose_canonical(members: Sequence[ExtractedEntity]) -> ExtractedEntity:
    # Highest confidence wins; ties broken by longest, then lexical for stability.
    return sorted(
        members,
        key=lambda m: (m.confidence, len(m.name), m.name),
        reverse=True,
    )[0]
