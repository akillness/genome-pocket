# Ops Layer: Evaluation, Tracing & Human-in-the-Loop

This document outlines the design of the **Ops Layer** in Pocket, focusing on system reliability, explainability, and human-in-the-loop (HITL) validation.

---

## Tracing & Lineage

To ensure complete explainability, Pocket tracks the execution of every pipeline component and retrieval query.

### 1. Component Tracing
CocoIndex automatically tracks the execution history of all `@coco.fn` components. Pocket exposes this data to help developers diagnose pipeline failures:
- **Execution Logs:** Track which files were processed, skipped (memoized), or failed.
- **Dependency Graph:** Visualize the parent-child relationships between components (e.g., `process_file` -> `process_chunk`).

### 2. Retrieval Lineage
Every retrieved chunk returned to an AI agent contains a `lineage` metadata block:
```json
{
  "text": "Pocket is a local-first personal Knowledge Ops runtime...",
  "lineage": {
    "source_file": "notes/architecture.md",
    "char_start": 120,
    "char_end": 340,
    "source_hash": "a1b2c3d4..."
  }
}
```
This allows the agent to cite its sources precisely and lets the user click through to the exact line in their editor.

---

## Evaluation Framework

To prevent retrieval quality regressions, Pocket includes a lightweight local evaluation suite:

- **Synthetic Query Generation:** Uses a local LLM to generate test queries from a subset of notes.
- **Retrieval Metrics:** Measures **Precision@K**, **Recall@K**, and **Mean Reciprocal Rank (MRR)** against the synthetic dataset.
- **Regression Testing:** Runs automatically before major changes to chunk size, overlap, or embedding models are committed.

---

## Human-in-the-Loop (HITL) Approval

For sensitive operations or complex graph updates, Pocket introduces an interactive approval gate:

```
[ Pipeline Run ] ──> [ Detect Sensitive/Graph Change ] ──> [ Pause Execution ]
                                                                 │
                                                                 ▼
[ Resume/Abort ] <── [ User Approval (CLI/UI) ] <───────── [ Show Diff ]
```

### Implementation Pattern
Using a local CLI prompt or a web UI, Pocket pauses execution and displays a diff of the proposed changes:
- **Sensitive Files:** If a file marked as `private` is about to be indexed or sent to an external API, the system requests confirmation.
- **Graph Schema Changes:** If the pipeline extracts a new relationship type (e.g., `User` -> `knows` -> `User`), the user must approve the schema update.
