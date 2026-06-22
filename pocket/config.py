import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent

# Configuration values
POCKET_SOURCE_DIR = Path(os.getenv("POCKET_SOURCE_DIR", str(BASE_DIR / "notes")))
POCKET_SQLITE_DB = Path(os.getenv("POCKET_SQLITE_DB", str(BASE_DIR / ".pocket" / "pocket_data.db")))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
# Expose the resolved embedding model to the lower-level pocketindex memo layer
# (which only reads env, staying decoupled from pocket.config). Folding the model
# into the source fingerprint means changing EMBEDDING_MODEL automatically
# invalidates memos and forces a clean re-embed at the new vector dimension.
os.environ["POCKET_EMBED_SIG"] = EMBEDDING_MODEL


def _truthy(val: str) -> bool:
    return str(val).strip().lower() in ("1", "true", "yes", "on")


# --- Knowledge-graph (GraphRAG) configuration (POCKET-404) ---
# The graph branch is opt-in: only when POCKET_GRAPH is truthy (or `pocket update
# --graph` is used) does the pipeline extract entities/relations. With it off the
# pipeline is exactly the vector/lexical path — zero extra cost or dependency.
POCKET_GRAPH = _truthy(os.getenv("POCKET_GRAPH", ""))
# Extraction backend: deterministic (default, offline) | ollama | airllm.
POCKET_LLM_PROVIDER = os.getenv("POCKET_LLM_PROVIDER", "deterministic")
POCKET_LLM_MODEL = os.getenv("POCKET_LLM_MODEL")  # backend-specific default if None
# Facts below this confidence are staged for HITL review, not committed directly.
POCKET_GRAPH_MIN_CONFIDENCE = float(os.getenv("POCKET_GRAPH_MIN_CONFIDENCE", "0.0"))

# --- Result diversity (MMR) configuration (POCKET-501) ---
# Off by default so the deterministic RRF ordering is unchanged unless opted in.
# When on, fused candidates are re-ranked with Maximal Marginal Relevance so
# near-duplicate chunks (e.g. several from the same file) don't crowd the top-k.
POCKET_MMR = _truthy(os.getenv("POCKET_MMR", ""))
# Trade-off knob in [0, 1]: 1.0 == pure relevance (no diversity penalty),
# 0.0 == pure diversity. 0.5 balances the two.
POCKET_MMR_LAMBDA = min(max(float(os.getenv("POCKET_MMR_LAMBDA", "0.5")), 0.0), 1.0)

# --- Cross-encoder reranker (precision pass after RRF fusion) ---
# Enable: POCKET_RERANKER=1
#
# After the RRF fusion step, a cross-encoder model re-scores the top
# POCKET_RERANKER_TOP_N candidates by attending to (query, chunk) jointly.
# Joint attention catches relevance signals (lexical overlap, entailment)
# that single-vector cosine similarity compresses away, lifting precision
# ~5-15 % on asymmetric queries at the cost of one forward pass per candidate.
#
# Uses sentence_transformers.CrossEncoder — already in the dependency tree.
# Default model: ms-marco-MiniLM-L-6-v2 (33 MB, runs on CPU in ~50 ms/query).
# Recommended upgrade: POCKET_RERANKER_MODEL=BAAI/bge-reranker-v2-m3 (568 MB,
# multilingual, MTEB reranking #1 as of 2025).
POCKET_RERANKER = _truthy(os.getenv("POCKET_RERANKER", ""))
POCKET_RERANKER_MODEL = os.getenv(
    "POCKET_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
# Candidate pool fed to the reranker; final results are still top-`limit`.
# Set higher for better recall at the cost of more inference time.
POCKET_RERANKER_TOP_N = int(os.getenv("POCKET_RERANKER_TOP_N", "20"))

# --- HyDE: Hypothetical Document Embeddings (arXiv:2212.10496) ---
# Enable: POCKET_HYDE=1
#
# Before the vector search, a local Ollama model generates a short hypothetical
# passage that would answer the query. That passage is embedded instead of the
# bare query, bridging the asymmetric query-document gap: real document chunks
# are longer and denser than a user's query, so encoding a document-like text
# lands the query vector much closer to true positive chunks in index space.
#
# Lexical (BM25) search always uses the original query so keyword recall is
# unaffected. Requires a running Ollama daemon; degrades gracefully (falls back
# to original query) when the daemon is unavailable.
POCKET_HYDE = _truthy(os.getenv("POCKET_HYDE", ""))
POCKET_HYDE_OLLAMA_MODEL = os.getenv("POCKET_HYDE_OLLAMA_MODEL", "qwen3:0.6b")
# Reuses the same host env-var used by the extraction backend so a single
# OLLAMA_HOST setting covers the whole stack.
POCKET_HYDE_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

# --- Semantic splitter (sentence-boundary, embedding-guided chunking) ---
# Enable: POCKET_SEMANTIC_SPLIT=1
#
# Replaces the fixed-size RecursiveSplitter for prose/markdown files with a
# SemanticSplitter that groups sentences into chunks wherever the cosine
# similarity between consecutive sentence embeddings drops below the threshold.
# Chunks produced this way are semantically coherent rather than arbitrarily
# truncated, which improves retrieval precision for dense prose notes.
#
# Code files (Python, JS, …) are never affected — they keep the language-aware
# RecursiveSplitter regardless of this flag.
POCKET_SEMANTIC_SPLIT = _truthy(os.getenv("POCKET_SEMANTIC_SPLIT", ""))
# Cosine similarity drop below this value triggers a chunk boundary.
# Lower values (0.5) produce fewer, larger chunks; higher (0.85) produces
# many small, maximally-coherent chunks.
POCKET_SEMANTIC_SPLIT_THRESHOLD = float(
    os.getenv("POCKET_SEMANTIC_SPLIT_THRESHOLD", "0.7")
)

# --- Weighted / tunable Reciprocal Rank Fusion (POCKET-502) ---
# RRF normally fuses every strategy with equal weight. These per-strategy
# weights scale each strategy's reciprocal-rank contribution, so a target where
# (say) the lexical index is more trustworthy than the semantic one can lean on
# it. Default 1.0 each == plain (unweighted) RRF, so behaviour is unchanged
# unless tuned. `pocket eval --tune` grid-searches these against the eval
# harness and writes the winner to POCKET_RRF_WEIGHTS_FILE, turning the guard
# into an optimizer; that file (when present) overrides the env defaults below.
_RRF_STRATEGIES = ("vector", "lexical", "graph")


def _clamp_weight(val) -> float:
    """A fusion weight is a non-negative float (0 disables a strategy)."""
    try:
        return max(float(val), 0.0)
    except (TypeError, ValueError):
        return 1.0


def _load_weight_overrides() -> dict:
    """Read tuned weights persisted by `pocket eval --tune`, if configured.

    POCKET_RRF_WEIGHTS_FILE points at a JSON object like
    ``{"vector": 1.5, "lexical": 2.0, "graph": 0.5}``. Missing file / bad JSON
    degrades to no override (the env defaults stand), so a stale path can never
    break startup.
    """
    path = os.getenv("POCKET_RRF_WEIGHTS_FILE")
    if not path or not Path(path).exists():
        return {}
    try:
        import json

        data = json.loads(Path(path).read_text())
    except (ValueError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {s: _clamp_weight(data[s]) for s in _RRF_STRATEGIES if s in data}


def _resolved_rrf_weights() -> dict:
    weights = {
        s: _clamp_weight(os.getenv(f"POCKET_RRF_{s.upper()}_WEIGHT", "1.0"))
        for s in _RRF_STRATEGIES
    }
    weights.update(_load_weight_overrides())
    return weights


POCKET_RRF_WEIGHTS = _resolved_rrf_weights()


# Ensure directories exist
POCKET_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
POCKET_SQLITE_DB.parent.mkdir(parents=True, exist_ok=True)
