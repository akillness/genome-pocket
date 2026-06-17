# Pocket Product Backlog

This backlog contains user stories and tasks categorized by priority. It serves as the source of truth for sprint planning.

---

## Epics

- **EPIC-1: DNA Core (Incremental ETL):** Build the core pipeline that syncs local files to vector/graph stores incrementally.
- **EPIC-2: Hybrid Retrieval:** Implement lexical, semantic, and graph-based search with semantic routing.
- **EPIC-3: MCP Integration:** Expose the knowledge base to AI agents via the Model Context Protocol.
- **EPIC-4: Knowledge Ops:** Add tracing, evaluation, and human-in-the-loop approval workflows.

---

## Backlog Items

### High Priority (Sprint 1 - 2 Target)

- [ ] **POCKET-101: Project Initialization & CLI Skeleton**
  - *User Story:* As a developer, I want to initialize the Pocket project structure and run a basic CLI command so that I can verify the environment.
  - *Tasks:* Set up `pyproject.toml`, CLI entrypoint, and `.env` loading.
- [ ] **POCKET-102: Local Filesystem Source Connector**
  - *User Story:* As a user, I want Pocket to watch my local markdown and code files so that changes are detected automatically.
  - *Tasks:* Configure `localfs.walk_dir` with live file watching.
- [ ] **POCKET-103: SQLite / LanceDB Target Setup**
  - *User Story:* As a system, I want to store chunk embeddings in a local database so that I don't rely on cloud database services.
  - *Tasks:* Set up SQLite with `sqlite-vec` or local LanceDB target schema.
- [ ] **POCKET-104: Incremental Chunking & Embedding Pipeline**
  - *User Story:* As a user, I want only modified files to be re-embedded so that I save local compute and API costs.
  - *Tasks:* Implement `@coco.fn(memo=True)` for file processing, chunking with `RecursiveSplitter`, and embedding generation.
- [ ] **POCKET-105: Lineage Metadata Storage**
  - *User Story:* As an auditor, I want to see the exact source file and character range for every chunk so that I can verify the source of truth.
  - *Tasks:* Store file path, start/end offsets, and source hash in the target database.

### Medium Priority (Sprint 3 - 4 Target)

- [ ] **POCKET-201: SurrealDB Graph Target Integration**
  - *User Story:* As a user, I want to extract concepts and relationships from my notes and store them in a graph database so that I can perform relational queries.
  - *Tasks:* Set up SurrealDB relation targets for entity-relationship extraction.
- [ ] **POCKET-202: Hybrid Retrieval Engine**
  - *User Story:* As an AI agent, I want to search using a combination of keyword, vector, and graph queries so that I get highly relevant context.
  - *Tasks:* Implement BM25 + Vector + Graph retrieval fusion.
- [ ] **POCKET-203: MCP Server Interface**
  - *User Story:* As a Claude Code user, I want to connect Claude to Pocket via MCP so that Claude can search my personal knowledge base.
  - *Tasks:* Build an MCP server exposing `search_knowledge` and `get_file_lineage` tools.
- [ ] **POCKET-204: Semantic Query Router**
  - *User Story:* As a system, I want to route queries to the best search strategy (e.g., code search vs. concept search) based on query intent.
  - *Tasks:* Implement a lightweight semantic router using local embeddings or LLM classification.

### Low Priority (Sprint 5+ Target)

- [ ] **POCKET-301: Local Tracing & Lineage UI**
  - *User Story:* As a user, I want a simple web UI to visualize how a query was routed and which source files contributed to the answer.
  - *Tasks:* Build a lightweight Streamlit or FastAPI/React UI.
- [ ] **POCKET-302: Human-in-the-Loop Approval Gate**
  - *User Story:* As a user, I want to approve or reject changes to my knowledge graph before they are committed so that I maintain high data quality.
  - *Tasks:* Implement an interactive CLI/UI prompt for graph updates.
- [ ] **POCKET-303: Automated Retrieval Evaluation**
  - *User Story:* As a developer, I want to run automated evaluations on my retrieval pipeline so that I can prevent regression when changing chunk sizes or models.
  - *Tasks:* Set up a local evaluation script using synthetic query-context pairs.
