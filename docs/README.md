# Pocket Knowledge Ops Documentation

Welcome to the documentation for **Pocket**, a DNA-based Pocket Knowledge Ops runtime. Pocket is a local-first, privacy-preserving personal knowledge management system built on the core principles of **CocoIndex** (Declarative, Incremental, Lineage, and Human-in-the-Loop).

## Document Structure

This documentation is organized into the following sections to guide the project from planning to production:

```text
docs/
├── README.md                  # Documentation Index (This file)
├── planning/
│   ├── roadmap.md             # Long-term Roadmap
│   ├── backlog.md             # Product Backlog & User Stories
│   └── sprint-01.md           # Sprint 1 Action Plan (Ready for execution)
├── architecture/
│   ├── system-overview.md     # High-level Architecture & DNA Core
│   ├── data-flow.md           # Declarative Data Flow (Target = F(Source))
│   ├── retrieval-layer.md     # Hybrid Retrieval & Semantic Routing
│   ├── ops-layer.md           # Evaluation, Tracing, and Human-in-the-Loop
│   └── mcp-server.md          # Model Context Protocol (MCP) Integration
├── development/
│   ├── setup-guide.md         # Local Environment & Dependency Setup
│   ├── api-spec.md            # API & Interface Specifications
│   └── pocketindex-guide.md   # PocketIndex Integration & Best Practices
└── decisions/
    ├── adr-001-local-first.md # ADR 1: Local-First & Privacy Architecture
    └── adr-002-hybrid-db.md   # ADR 2: Vector + Graph Database Selection
```

---

## Core Mental Model: DNA-based Pocket Knowledge Ops

Pocket inherits the core "DNA" of CocoIndex to deliver a robust, low-overhead personal knowledge engine:

1. **Declarative (`Target = F(Source)`):** Define what the target knowledge state should be using Python. The engine guarantees synchronization.
2. **Incremental ($\Delta$-only):** Only process changed files, notes, or code. No full-batch re-indexing.
3. **Lineage (End-to-End):** Every retrieved chunk or answer traces back to its exact source byte.
4. **Hybrid Targets:** Vector DB (semantic) + Graph DB (conceptual relationships) + Relational metadata.
5. **MCP Connected:** Expose the entire Pocket knowledge base as an MCP server for Claude Code, Cursor, and other AI agents.
