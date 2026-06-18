# Pocket Roadmap

This document outlines the long-term roadmap for **Pocket**, structured across three major phases. The goal is to transition from a single-agent local prototype to a production-ready, multi-agent personal Knowledge Ops runtime.

```
Phase 1: Local Core & CLI (Sprint 1-2) ──> Phase 2: Hybrid Retrieval & MCP (Sprint 3-4) ──> Phase 3: Ops & Human-in-the-Loop (Sprint 5+)
```

---

## Phase 1: Local Core & CLI (Sprints 1 - 2)
*Focus: Establish the declarative incremental pipeline, local storage, and basic CLI.*

- [ ] **Core Pipeline Setup:** Integrate CocoIndex with local filesystem source (`localfs.walk_dir`).
- [ ] **Local Vector Target:** Configure SQLite with `sqlite-vec` or local LanceDB as the primary vector target.
- [ ] **Basic Chunking & Embedding:** Implement `RecursiveSplitter` and local embedding models (e.g., HuggingFace/SentenceTransformers or local Ollama embeddings).
- [ ] **CLI Interface:** Build a CLI tool (`pocket`) to initialize, update, and query the local index.
- [ ] **Lineage Tracking:** Ensure every chunk in the database stores its source file path, character offsets, and hash.

## Phase 2: Hybrid Retrieval & MCP Server (Sprints 3 - 4)
*Focus: Implement hybrid search, semantic routing, and expose Pocket to external AI agents.*

- [ ] **Graph Target Integration:** Add a local Graph DB target (e.g., SurrealDB or local Neo4j) to store conceptual relationships extracted from notes.
- [ ] **Hybrid Retrieval Engine:** Combine lexical search (BM25/FTS5), vector search, and graph traversal into a unified retrieval layer.
- [ ] **Semantic Routing:** Implement a router to classify queries and direct them to the appropriate retrieval strategy.
- [ ] **MCP Server Implementation:** Expose Pocket as a Model Context Protocol (MCP) server, allowing Claude Code, Cursor, or other agents to query the knowledge base.
- [ ] **Query Expansion:** Use local LLMs to expand user queries before retrieval.

## Phase 3: Ops, Tracing & Human-in-the-Loop (Sprints 5+)
*Focus: Add evaluation, tracing, failure analysis, and human approval workflows.*

- [~] **Tracing & Lineage UI:** Engine now emits per-component run statistics (adds/reprocesses/unchanged/deletes/errors) via `UpdateStats`/`ComponentStats`, surfaced on the CLI after every `pocket update`. A local web UI to visualize traces remains pending.
- [x] **Live Incremental Watching:** `pocket update -L [--interval N]` runs the pipeline on a polling loop, picking up new/edited/deleted files between passes (previously a no-op flag).
- [x] **Code-Aware Chunking:** `RecursiveSplitter` is now language-aware (`detect_code_language` + per-language structural separators, plus `SeparatorSplitter`/`CustomLanguageConfig`), `TextRefiner` has an indentation-preserving code path, and the source connector indexes recognized code files — so source code chunks on real syntax boundaries instead of arbitrary character counts.
- [ ] **Evaluation Framework:** Integrate a lightweight evaluation suite (e.g., Ragas or custom local evals) to measure retrieval precision and recall.
- [ ] **Human-in-the-Loop (HITL) Approval:** Add an approval step for indexing sensitive files or executing complex graph updates.
- [ ] **Multi-Agent Orchestration:** Introduce specialized retrieval agents (e.g., Code Agent, Note Agent, Web Agent) only when complexity warrants it.
