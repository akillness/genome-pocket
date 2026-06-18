# Local Environment & Setup Guide

This guide helps you set up the local development environment for **Pocket**.

---

## Prerequisites

- **Python:** Version 3.10 to 3.13
- **uv:** Fast Python package installer and resolver (recommended)
- **Docker:** Optional, required only if running SurrealDB or PostgreSQL locally

---

## Step-by-Step Setup

### 1. Clone the Repository
```bash
git clone <your-pocket-repo-url> pocket-ops
cd pocket-ops
```

### 2. Initialize the Environment
Create a virtual environment and install the dependencies in editable mode:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```bash
# Pocket internal database (stores PocketIndex engine state)
POCKET_SQLITE_DB=./.pocket/pocket_data.db

# Local storage path for notes and documents
POCKET_SOURCE_DIR=./notes

# Local SQLite database path for embeddings
POCKET_SQLITE_DB=./.pocket/pocket_data.db

# Optional: API keys for LLM extraction (if not using local models)
# OPENAI_API_KEY=your-key
# ANTHROPIC_API_KEY=your-key
```

### 4. Initialize the Source Directory
Create a directory to store your markdown notes and code files:
```bash
mkdir -p notes
echo "# My First Note\n\nPocket is a local-first personal Knowledge Ops runtime." > notes/welcome.md
```

### 5. Run the Pipeline
Run the indexing pipeline in catch-up mode to process the initial notes:
```bash
uv run pocket update
```

To run the pipeline in live mode (watching for file changes in real-time):
```bash
uv run pocket update -L
```

### 6. Verify the Index
Run a search query to verify that the welcome note was indexed:
```bash
uv run pocket search "What is Pocket?"
```
