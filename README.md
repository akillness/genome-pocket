# genome-pocket 🧬

[![Build Status](https://github.com/username/genome-pocket/actions/workflows/ci.yml/badge.svg)](https://github.com/username/genome-pocket/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![CocoIndex v1](https://img.shields.io/badge/CocoIndex-v1.0.10-orange.svg)](https://cocoindex.github.io/)
[![sqlite-vec](https://img.shields.io/badge/sqlite--vec-v0.1.9-blue.svg)](https://github.com/asg017/sqlite-vec)

Sequence your knowledge. Carry the whole map in your pocket.

**Pocket Knowledge Ops** is a local-first personal knowledge runtime powered by the **CocoIndex** declarative incremental ETL paradigm. It watches your local markdown notes, chunks them, generates vector embeddings using a local SentenceTransformer model, and stores them in a local SQLite database with `sqlite-vec` for semantic search.

---

## 🖼️ Concept & Architecture

![Pocket Concept](docs/images/pocket-architecture.svg)


Pocket operates on the core mental model of **Target = F(Source)**. All data processing is incremental ($\Delta$-only), ensuring that only modified files are reprocessed, and deleted files are automatically cleaned up from the database.

### Core Workflow
1. **Source (LocalFS):** Watches a local directory (e.g., `./notes`) for Markdown/text files.
2. **Transformation (CocoIndex Pipeline):**
   - Splits text into chunks using `RecursiveSplitter`.
   - Generates embeddings using a local `SentenceTransformer` model (`all-MiniLM-L6-v2`).
   - Generates stable, deterministic IDs using `IdGenerator` to ensure lineage and idempotency.
3. **Target (SQLite + sqlite-vec):** Stores chunk text, embeddings, and lineage metadata (file path, start/end offsets) in a local SQLite database.
4. **Retrieval (MCP Server):** Exposes the retrieval layer as a Model Context Protocol (MCP) server, allowing AI coding agents (like Claude Code or Cursor) to query the knowledge base directly.

---

## 📂 Project Structure

```text
genome-pocket/
├── .pocket/                  # Internal database storage (git-ignored)
│   ├── cocoindex.db          # CocoIndex internal state
│   └── pocket_data.db        # SQLite database with chunk embeddings
├── docs/                     # Documentation
│   ├── architecture/         # System design and data flow
│   ├── decisions/            # Architecture Decision Records (ADRs)
│   ├── images/               # Concept diagrams and images
│   │   └── pocket-architecture.svg
│   └── planning/             # Roadmap and sprint backlogs

├── notes/                    # Local markdown notes directory (source)
├── pocket/                   # Source code
│   ├── __init__.py
│   ├── cli.py                # CLI commands (init, update, search)
│   ├── config.py             # Configuration & environment variables
│   ├── mcp_server.py         # MCP server interface
│   └── pipeline.py           # CocoIndex ETL pipeline
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
git clone https://github.com/username/genome-pocket.git
cd genome-pocket
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
COCOINDEX_DB=./.pocket/cocoindex.db
POCKET_SOURCE_DIR=./notes
POCKET_SQLITE_DB=./.pocket/pocket_data.db
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

Run in live mode (watches for file changes in real-time):
```bash
pocket update -L
```

#### Search the Knowledge Base
```bash
pocket search "What is Pocket?"
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
- `search_knowledge(query: str, limit: int = 5)`: Search the personal knowledge base using semantic vector search.
- `get_file_lineage(file_path: str)`: Retrieve the indexing history and lineage details for a specific source file.
- `list_concepts(concept: str = None)`: List key concepts and relationships (Sprint 2).
