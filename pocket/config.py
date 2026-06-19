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

# --- Knowledge-graph (GraphRAG) configuration (POCKET-404) ---
# The graph branch is opt-in: only when POCKET_GRAPH is truthy (or `pocket update
# --graph` is used) does the pipeline extract entities/relations. With it off the
# pipeline is exactly the vector/lexical path — zero extra cost or dependency.
def _truthy(val: str) -> bool:
    return str(val).strip().lower() in ("1", "true", "yes", "on")

POCKET_GRAPH = _truthy(os.getenv("POCKET_GRAPH", ""))
# Extraction backend: deterministic (default, offline) | ollama | airllm.
POCKET_LLM_PROVIDER = os.getenv("POCKET_LLM_PROVIDER", "deterministic")
POCKET_LLM_MODEL = os.getenv("POCKET_LLM_MODEL")  # backend-specific default if None
# Facts below this confidence are staged for HITL review, not committed directly.
POCKET_GRAPH_MIN_CONFIDENCE = float(os.getenv("POCKET_GRAPH_MIN_CONFIDENCE", "0.0"))

# Ensure directories exist
POCKET_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
POCKET_SQLITE_DB.parent.mkdir(parents=True, exist_ok=True)
