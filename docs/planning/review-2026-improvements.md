# Pocket Review & Improvement Backlog (spec-stack: Write layer)

This is the **Write** artifact of a spec-stack pass (Write → Freeze → Run): a
grounded review of the current codebase plus a prioritized, feature-oriented
improvement backlog. Each idea below names a concrete seam in the existing code
so the next loop can pick one up, freeze a seed, and verify it against
`pocket eval` / the test suite.

## Snapshot of what already exists (verified)

`genome-pocket` is well past prototype. Confirmed by reading the source and
running the suite (`bash run_tests.sh` → **91 tests pass offline in ~5 s**):

- **Incremental ETL** — `Target = F(Source)`, Δ-only, lineage + memo + deletion
  sweep (`pocketindex/`, `pocket/pipeline.py`).
- **Hybrid retrieval** — vector (sqlite-vec) + lexical (FTS5/BM25) + GraphRAG,
  fused by Reciprocal Rank Fusion in one shared path (`pocket/retrieval.py`),
  exposed identically via CLI / MCP / REST+WebUI.
- **Local-first knowledge graph** — extraction, entity resolution with audit
  trail, HITL approval gate, multi-hop traversal.
- **Ops** — `pocket eval` regression harness (Hit@k/MRR/MAP), run statistics,
  live watching, lifecycle commands, query-tracing Web UI.
- **Multimodal** — opt-in SigLIP2 text+image embedding into one shared space.

## Findings fixed in this pass

1. **Broken/stale test runner (fixed).** `run_tests.sh` invoked
   `tests.test_retrieval_api.TestGraphExtraction` — a class that lives in
   `tests/test_graph_unit.py` — so the script errored out partway. Worse, it
   drove tests via `python -m unittest`, which does **not** load the
   `tests/conftest.py` session-scoped `MockEmbedder` fixture, so it silently
   fell back to loading real model weights (slow, network-dependent) and skipped
   whole modules (`test_graph_unit`, `test_multimodal`, `TestRetrievalEvaluation`).
   Rewrote it to drive **pytest**, which auto-discovers every module, honors the
   offline mock, and can never go stale.
2. **Stale test count in README (fixed).** P0 row claimed "81 tests"; the suite
   is now 91.

## Feature shipped in this pass

- **`pocket search --json` (agent-native output).** The CLI now mirrors the REST
  `/search` and MCP `search_knowledge` surfaces by emitting
  `{query, mode, count, hits[]}` as pure JSON on stdout (status/diagnostics on
  stderr). This is the spec-stack "tools" contract: harness output an agent or
  pipeline can parse as evidence. (`pocket/cli.py`, 2 new tests.)
- **POCKET-501 · Result diversity (MMR) in fusion — DONE.** Fusion split into
  `_fuse_ranked` (full candidate pool) + `_fuse` (top-k); opt-in `_mmr_rerank`
  re-orders by `λ·relevance − (1−λ)·max-cosine-to-selected` using each
  candidate's stored embedding (`_fetch_embeddings`/`_cosine`). Off by default
  (`POCKET_MMR`/`POCKET_MMR_LAMBDA`, `pocket search --mmr/--no-mmr`); 6 new tests.
  (`pocket/retrieval.py`, `pocket/config.py`, `pocket/cli.py`.)

## Prioritized improvement backlog

### Retrieval quality (highest leverage — measurable via `pocket eval`)

- **POCKET-501 · Result diversity (MMR) in fusion.** ✅ Shipped this pass (see
  above). The next refinement here is feeding MMR a real query-vs-doc relevance
  term (today relevance = fused RRF score, redundancy = doc-doc cosine), and
  proving the Recall@k/MAP trade-off with `pocket eval` on a graded corpus.
- **POCKET-502 · Weighted / tunable RRF.** ✅ Shipped this pass. Per-strategy
  weights (`config.POCKET_RRF_WEIGHTS`, env `POCKET_RRF_{VECTOR,LEXICAL,GRAPH}_WEIGHT`)
  scale each strategy's RRF contribution in `_fold_ranked`/`_fuse`/`_fuse_ranked`/`search`
  (default 1.0 each == plain RRF). `evaluation.tune_weights` grid-searches them
  (`pocket eval --tune [--tune-metric] [--save-weights]`), never landing below the
  equal-weight baseline, and `save_weights`/`POCKET_RRF_WEIGHTS_FILE` feed the
  winner back into config. The next refinement is a smarter search (coordinate
  ascent / random search) once a graded corpus makes the lift measurable on
  hybrid queries (today's unit grid proves correctness, not yet a quality win).
  *Seam: `_fold_ranked`/`_fuse` + `evaluation.py`.*

- **POCKET-503 · Query expansion (roadmap, unbuilt).** Optionally expand the
  query (synonyms / local-LLM paraphrase) before `_gather`, reusing the existing
  `POCKET_LLM_PROVIDER` backends. Keep deterministic default (no expansion).
- **POCKET-504 · Semantic query router (roadmap, unbuilt).** Auto-pick the mode
  (code vs prose vs concept/graph) from query shape instead of forcing
  `--mode`, so `hybrid` callers get the right blend automatically.

### Engine parity (already on the README roadmap)

- **POCKET-P4 · State-diff delta writes.** Adopt `connectorkits.statediff`
  upsert/delete semantics to stop chunk accumulation on edits.
- **POCKET-P5 · Persistent memo store.** SQLite-backed `@fn(memo=True)` that
  survives process restarts.

### Ops & UX

- **POCKET-505 · HITL review in the Web UI.** Pending graph facts are reviewable
  only via CLI today; surface the `admin.list_pending/approve/reject` queue in
  the existing dependency-free Web UI.
- **POCKET-506 · Answer synthesis with citations.** Optional local-LLM RAG answer
  that cites the exact lineage offsets each claim came from (Pocket already
  carries byte-exact lineage end to end).
- **POCKET-507 · Snippet highlighting.** Use the stored offsets to highlight the
  matched span in `format_hits` / Web UI output.

### Housekeeping

- **POCKET-508 · Starlette TestClient deprecation.** `tests/test_retrieval_api.py`
  emits a `StarletteDeprecationWarning` (httpx vs httpx2). Track the upstream
  migration so the warning doesn't mask future ones.

## Suggested next loop

**POCKET-501 (MMR)** and **POCKET-502 (weighted RRF)** are now shipped: both sit
on the well-tested `_fuse` seam and are wired into the `pocket eval` harness. The
honest next loop is **measurement, not more knobs** — build a small graded corpus
(or a hand-written `gold.json`) so MMR's diversity trade-off and the tuner's
weight lift become real Recall@k/MAP numbers on hybrid queries instead of
unit-grid correctness. After that, **POCKET-503 (query expansion)** or
**POCKET-504 (semantic router)** are the next contained retrieval-quality bets.

