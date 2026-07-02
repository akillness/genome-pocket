from dataclasses import dataclass, asdict
from typing import Dict, Optional

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


def _resolve_callable(name: str, default):
    """Dynamic lookup to respect tests patching symbols on the package facade."""
    import sys
    pkg = sys.modules.get("pocket.retrieval")
    if pkg is not None:
        return getattr(pkg, name, default)
    return default
