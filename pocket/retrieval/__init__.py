import sys
import importlib
import pocket.config as config

# 1. Support reload(retrieval) in tests: reload submodules first if loaded.
# We lookup in sys.modules directly rather than importing relatives, because
# relative imports look up package attributes first, which can be shadowed by
# functions on second run (reload).
_submodules = [
    "pocket.retrieval.base",
    "pocket.retrieval.db",
    "pocket.retrieval.encode",
    "pocket.retrieval.router",
    "pocket.retrieval.fusion",
    "pocket.retrieval.rerank",
    "pocket.retrieval.graph",
    "pocket.retrieval.inspect",
    "pocket.retrieval.search",
]

for _name in _submodules:
    if _name in sys.modules:
        importlib.reload(sys.modules[_name])

# 2. Export public API and private names needed by tests/internal code
from .base import RetrievalHit, RRF_K, _FTS_TABLE
from .db import (
    _connect,
    _fts_available,
    _graph_available,
    _has_status_column,
    _status_clause,
)
from .encode import _get_model, _encode_query
from .router import (
    _CODE_SHAPE_RE,
    _CONCEPT_PHRASES,
    _route_query,
    _resolve_mode,
    _MODE_STRATEGIES,
)
from .fusion import _resolve_weights, _fold_ranked, _fuse_ranked, _fuse
from .rerank import (
    _cosine,
    _fetch_embeddings,
    _mmr_rerank,
    _get_reranker,
    _rerank,
    _hyde_expand,
)
from .graph import (
    _graph_search,
    _load_chunk_ids,
    graph_neighborhood,
    format_neighborhood,
    list_graph_concepts,
)
from .inspect import (
    get_lineage,
    list_sources,
    target_stats,
    routing_trace,
    format_hits,
)
from .search import (
    _vector_search,
    _fts_escape,
    _lexical_search,
    _gather,
    search,
    _expand_query,
)

__all__ = [
    # Public API
    "RetrievalHit",
    "RRF_K",
    "search",
    "get_lineage",
    "list_sources",
    "target_stats",
    "routing_trace",
    "format_hits",
    "graph_neighborhood",
    "format_neighborhood",
    "list_graph_concepts",
    # Internal names (for tests/LSP compatibility)
    "_connect",
    "_fts_available",
    "_graph_available",
    "_has_status_column",
    "_status_clause",
    "_get_model",
    "_encode_query",
    "_CODE_SHAPE_RE",
    "_CONCEPT_PHRASES",
    "_route_query",
    "_resolve_mode",
    "_MODE_STRATEGIES",
    "_resolve_weights",
    "_fold_ranked",
    "_fuse_ranked",
    "_fuse",
    "_cosine",
    "_fetch_embeddings",
    "_mmr_rerank",
    "_get_reranker",
    "_rerank",
    "_hyde_expand",
    "_graph_search",
    "_load_chunk_ids",
    "_expand_query",
    "_fts_escape",
    "_vector_search",
    "_lexical_search",
    "_gather",
    "_FTS_TABLE",
    "config",
]
