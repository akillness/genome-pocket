# genome-pocket 🧬

[![Build Status](https://github.com/akillness/genome-pocket/actions/workflows/ci.yml/badge.svg)](https://github.com/akillness/genome-pocket/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Self-contained engine](https://img.shields.io/badge/engine-self--contained-green.svg)](#-concept--architecture)
[![sqlite-vec](https://img.shields.io/badge/sqlite--vec-v0.1.9-blue.svg)](https://github.com/asg017/sqlite-vec)

Sequence your knowledge. Carry the whole map in your pocket.

**Pocket Knowledge Ops** is a local-first personal knowledge runtime powered by the **PocketIndex** declarative incremental ETL paradigm (an in-tree vendored engine inspired by CocoIndex's Source→Refine→Load→Serve model). It watches your local markdown notes, chunks them, generates vector embeddings using a local SentenceTransformer model, and stores them in a local SQLite database with `sqlite-vec` for semantic search.

---

## 🖼️ Concept & Architecture

![Pocket Concept](docs/images/pocket-architecture.svg)


Pocket operates on the core mental model of **Target = F(Source)**. All data processing is incremental ($\Delta$-only), ensuring that only modified files are reprocessed, and deleted files are automatically cleaned up from the database.

### Core Workflow — Source → Refine → Load → Serve
1. **Source (LocalFS):** Watches a local directory (e.g., `./notes`) for Markdown/text files **and recognized source-code files** (`.py`, `.rs`, `.ts`/`.js`, `.go`, `.java`, ...).
2. **Refine (data cleaning):** `TextRefiner` normalizes raw content (Unicode NFC, CRLF→LF, trailing/duplicate whitespace, excess blank lines) while keeping an offset map so lineage still points at the original source bytes. For code files it switches to an **indentation-preserving** pass so block structure (e.g. Python indentation) survives into the index.
3. **Transformation (PocketIndex Pipeline):**
   - Splits refined text into chunks using `RecursiveSplitter`. The splitter is **code-aware**: `detect_code_language()` maps the filename to a language and the splitter prefers that language's structural boundaries (class/def/fn/...), falling back to a recursive paragraph→sentence→line→word→char split for prose. `SeparatorSplitter` and `CustomLanguageConfig` are available for custom formats.
   - Generates embeddings using a local `SentenceTransformer` model (`all-MiniLM-L6-v2`).
   - Generates stable, deterministic IDs using `IdGenerator` to ensure lineage and idempotency.
4. **Load (SQLite + sqlite-vec + FTS5):** Stores chunk text, embeddings, and lineage metadata (file path, start/end offsets) in a local SQLite database. The same load mirrors chunk text into an FTS5 index so the target supports both vector and lexical (BM25) search.
5. **Serve (hybrid retrieval):** A single retrieval layer (`pocket/retrieval.py`) fuses vector + lexical results via Reciprocal Rank Fusion and is exposed three ways:
   - **CLI:** `pocket search "query" --mode hybrid|vector|lexical`
   - **MCP Server:** `pocket-mcp` for Claude Code / Cursor.
   - **REST API Server:** `pocket serve` / `pocket-api` (Starlette + uvicorn) with `/health`, `/search`, and `/lineage` endpoints.

---

## 📂 Project Structure

```text
genome-pocket/
├── .pocket/                  # Internal database storage (git-ignored)
│   └── pocket_data.db        # SQLite DB: chunk embeddings + lineage/memo state
├── docs/                     # Documentation
│   ├── architecture/         # System design and data flow
│   ├── decisions/            # Architecture Decision Records (ADRs)
│   ├── images/               # Concept diagrams and images
│   │   └── pocket-architecture.svg
│   └── planning/             # Roadmap and sprint backlogs

├── notes/                    # Local markdown notes directory (source)
├── pocketindex/              # Self-contained ETL engine (vendored, no pip dep)
│   ├── __init__.py           # App, lifespan, fn, map, mount_each, context
│   ├── connectors/           # localfs source + sqlite target (lineage/memo + FTS5)
│   ├── ops/                  # sentence_transformers embedder, text splitter, refiner
│   └── resources/            # file, chunk, deterministic id helpers
├── pocket/                   # Application source code
│   ├── __init__.py
│   ├── cli.py                # CLI commands (init, update, search)
│   ├── config.py             # Configuration & environment variables
│   ├── mcp_server.py         # MCP server interface
│   ├── pipeline.py           # ETL pipeline wiring (Source→Refine→Load)
│   ├── retrieval.py          # Hybrid retrieval (vector + lexical + RRF), shared by CLI/MCP/API
│   └── api_server.py         # REST API server (Starlette + uvicorn)
├── .env                      # Environment configuration
├── main.py                   # CLI entry point
├── pyproject.toml            # Project dependencies and scripts
└── README.md                 # Project README
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended)

### 2. Installation
Clone the repository and install in editable mode:
```bash
git clone https://github.com/akillness/genome-pocket.git
cd genome-pocket
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
POCKET_SOURCE_DIR=./notes
POCKET_SQLITE_DB=./.pocket/pocket_data.db
EMBEDDING_MODEL=all-MiniLM-L6-v2
```


### 4. Usage

#### Initialize the Notes Directory
```bash
pocket init
```

#### Run the Indexing Pipeline
Run in catch-up mode (processes all pending changes and exits):
```bash
pocket update
```

Run in live mode (watches for file changes in real-time, re-indexing on a polling interval):
```bash
pocket update -L                  # poll every 2s (default)
pocket update -L --interval 5     # poll every 5s
```

Every pass prints per-component processing statistics (adds / reprocesses /
unchanged / deletes / errors) so you can monitor and cross-check what the
incremental engine actually did against your logs:

```text
[pocketindex] run complete in 0.23s
  process_file: adds=1 reprocesses=0 unchanged=1 deletes=0 errors=0 in_progress=0
  total: adds=1 reprocesses=0 unchanged=1 deletes=0 errors=0 in_progress=0
```

#### Search the Knowledge Base
```bash
pocket search "What is Pocket?"               # hybrid (vector + lexical) by default
pocket search "vec_distance_cosine" --mode lexical   # exact keyword / symbol match
pocket search "how does incremental sync work" --mode vector


#### Inspect & Manage the Index
bash
pocket ls                                      # list indexed sources + chunk counts
pocket show                                    # summarize the index (sources/chunks/FTS)
pocket show notes/welcome.md                   # show one source's chunk lineage
pocket drop notes/welcome.md --yes             # evict one source's chunks + lineage
pocket drop --yes                              # reset the entire index (rebuild on next update)


#### Serve the REST API
```bash
pocket serve --host 127.0.0.1 --port 8000     # or: pocket-api
```

Endpoints:
- `GET /health` — liveness and index status.
- `GET /search?q=<query>&limit=5&mode=hybrid` — retrieval via query string.
- `POST /search` — JSON body `{"query": "...", "limit": 5, "mode": "hybrid"}`.
- `GET /lineage?file_path=<path>` — ordered chunk lineage for a source file.

```bash
curl "http://127.0.0.1:8000/search?q=pocket&mode=hybrid&limit=3"
curl -X POST http://127.0.0.1:8000/search -H 'Content-Type: application/json' \
     -d '{"query": "incremental sync", "mode": "vector"}'
```

---

## 🤖 MCP Server Integration

To connect Claude Code or Cursor to your Pocket knowledge base, add the following to your MCP configuration file (e.g., `mcp_config.json`):

```json
{
  "mcpServers": {
    "pocket": {
      "command": "uv",
      "args": ["run", "--package", "genome-pocket", "pocket-mcp"]
    }
  }
}
```

### Exposed Tools
- `search_knowledge(query: str, limit: int = 5, mode: str = "hybrid")`: Search the personal knowledge base using hybrid (vector + lexical) retrieval; `mode` is `hybrid`, `vector`, or `lexical`.
- `get_file_lineage(file_path: str)`: Retrieve the indexing history and lineage details for a specific source file.
- `list_concepts(concept: str = None)`: List key concepts and relationships (Sprint 2).
