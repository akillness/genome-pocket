import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

POCKET_SOURCE_DIR = Path(os.getenv("POCKET_SOURCE_DIR", str(BASE_DIR / "notes")))
POCKET_SQLITE_DB  = Path(os.getenv("POCKET_SQLITE_DB",  str(BASE_DIR / ".pocket" / "pocket_data.db")))
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
os.environ["POCKET_EMBED_SIG"] = EMBEDDING_MODEL  # propagate to pocketindex (env-only layer); invalidates memos on model change


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


# POCKET-404 GraphRAG — opt-in; off = pure vector/lexical, zero extra cost
POCKET_GRAPH                = _truthy(os.getenv("POCKET_GRAPH", ""))
POCKET_LLM_PROVIDER         = os.getenv("POCKET_LLM_PROVIDER", "deterministic")  # deterministic | ollama | airllm
POCKET_LLM_MODEL            = os.getenv("POCKET_LLM_MODEL")
POCKET_GRAPH_MIN_CONFIDENCE = float(os.getenv("POCKET_GRAPH_MIN_CONFIDENCE", "0.0"))

# POCKET-501 MMR diversity — opt-in; lambda in [0,1]: 1=pure relevance, 0=pure diversity
POCKET_MMR        = _truthy(os.getenv("POCKET_MMR", ""))
POCKET_MMR_LAMBDA = min(max(float(os.getenv("POCKET_MMR_LAMBDA", "0.5")), 0.0), 1.0)

# Cross-encoder reranker (sentence_transformers) — opt-in; degrades gracefully if model absent
POCKET_RERANKER       = _truthy(os.getenv("POCKET_RERANKER", ""))
POCKET_RERANKER_MODEL = os.getenv("POCKET_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
POCKET_RERANKER_TOP_N = int(os.getenv("POCKET_RERANKER_TOP_N", "20"))

# HyDE arXiv:2212.10496 — opt-in; requires Ollama; falls back to raw query when daemon absent
POCKET_HYDE              = _truthy(os.getenv("POCKET_HYDE", ""))
POCKET_HYDE_OLLAMA_MODEL = os.getenv("POCKET_HYDE_OLLAMA_MODEL", "qwen3:0.6b")
POCKET_HYDE_OLLAMA_HOST  = os.getenv("POCKET_HYDE_OLLAMA_HOST") or os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434"

# Semantic splitter — opt-in; code files always use RecursiveSplitter regardless
POCKET_SEMANTIC_SPLIT           = _truthy(os.getenv("POCKET_SEMANTIC_SPLIT", ""))
POCKET_SEMANTIC_SPLIT_THRESHOLD = float(os.getenv("POCKET_SEMANTIC_SPLIT_THRESHOLD", "0.7"))

# POCKET-502 weighted RRF — default 1.0 each = plain unweighted RRF; `pocket eval --tune` writes overrides
_RRF_STRATEGIES = ("vector", "lexical", "graph")


def _clamp_weight(val) -> float:
    try:
        return max(float(val), 0.0)
    except (TypeError, ValueError):
        return 1.0


def _load_weight_overrides() -> dict:
    """Read POCKET_RRF_WEIGHTS_FILE JSON override written by `pocket eval --tune`."""
    path = os.getenv("POCKET_RRF_WEIGHTS_FILE")
    if not path or not Path(path).exists():
        return {}
    try:
        import json
        data = json.loads(Path(path).read_text())
    except (ValueError, OSError):
        return {}
    return {s: _clamp_weight(data[s]) for s in _RRF_STRATEGIES if isinstance(data, dict) and s in data}


def _resolved_rrf_weights() -> dict:
    weights = {s: _clamp_weight(os.getenv(f"POCKET_RRF_{s.upper()}_WEIGHT", "1.0")) for s in _RRF_STRATEGIES}
    weights.update(_load_weight_overrides())
    return weights


POCKET_RRF_WEIGHTS = _resolved_rrf_weights()

# POCKET-503 query expansion — opt-in; project-specific synonyms go in POCKET_QUERY_EXPANSION_FILE
POCKET_QUERY_EXPANSION = _truthy(os.getenv("POCKET_QUERY_EXPANSION", ""))

_DEFAULT_QUERY_EXPANSIONS = {
    "wal":  ["write ahead log"],
    "lru":  ["least recently used"],
    "mru":  ["most recently used"],
    "ttl":  ["time to live"],
    "bfs":  ["breadth first search"],
    "dfs":  ["depth first search"],
    "db":   ["database"],
    "fs":   ["file system"],
    "fts":  ["full text search"],
    "rps":  ["requests per second"],
    "k8s":  ["kubernetes"],
    "auth": ["authentication", "authorization"],
}


def _load_query_expansions() -> dict:
    """Merge POCKET_QUERY_EXPANSION_FILE (JSON token→phrase|[phrases]) over built-in map."""
    merged = {k: list(v) for k, v in _DEFAULT_QUERY_EXPANSIONS.items()}
    path = os.getenv("POCKET_QUERY_EXPANSION_FILE")
    if not path or not Path(path).exists():
        return merged
    try:
        import json
        data = json.loads(Path(path).read_text())
    except (ValueError, OSError):
        return merged
    if not isinstance(data, dict):
        return merged
    for token, phrases in data.items():
        if isinstance(phrases, str):
            phrases = [phrases]
        if isinstance(phrases, list):
            merged[str(token).lower()] = [str(p) for p in phrases]
    return merged


POCKET_QUERY_EXPANSION_MAP = _load_query_expansions()

# POCKET-504 semantic query router — opt-in, deterministic, offline (regex shape detection)
POCKET_QUERY_ROUTER = _truthy(os.getenv("POCKET_QUERY_ROUTER", ""))

POCKET_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
POCKET_SQLITE_DB.parent.mkdir(parents=True, exist_ok=True)
