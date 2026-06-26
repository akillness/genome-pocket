# Pocket Review & Improvement Backlog (spec-stack: Write layer)

This is the **Write** artifact of a spec-stack pass (Write → Freeze → Run): a
grounded review of the current codebase plus a prioritized, feature-oriented
improvement backlog. Each idea below names a concrete seam in the existing code
so the next loop can pick one up, freeze a seed, and verify it against
`pocket eval` / the test suite.

## Snapshot of what already exists (verified)

`genome-pocket` is well past prototype. Confirmed by reading the source and
running the suite (`bash run_tests.sh` → **169 tests pass offline in ~11 s**):


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
- **Graded-corpus eval proof (POCKET-501/502 measurement loop) — DONE.** A
  shipped graded corpus (`eval/corpus/` + multi-relevant `eval/gold.json`) plus a
  deterministic offline `HashingEmbedder` turn the fusion features from "mechanics
  proven" into "quality measured": MMR raises Recall@3 0.5→1.0 on a redundancy
  query, and `tune_weights` lifts MAP over the equal-weight baseline on a hybrid
  query. The harness gained `evaluate(use_mmr=, mmr_lambda=)` and a
  `pocket eval --mmr/--no-mmr` flag so the diversity trade-off is measurable from
  the CLI. (`pocket/evaluation.py`, `pocket/cli.py`, `eval/`,
  `tests/test_eval_proof.py`; 5 new tests.)


## Prioritized improvement backlog

### Retrieval quality (highest leverage — measurable via `pocket eval`)

- **POCKET-501 · Result diversity (MMR) in fusion.** ✅ Shipped + measured. The
  Recall@k/MAP trade-off is now proven on a graded corpus (see the measurement
  loop above: Recall@3 0.5→1.0). The remaining refinement is feeding MMR a real
  query-vs-doc relevance term (today relevance = fused RRF score, redundancy =
  doc-doc cosine) and auto-picking `mmr_lambda` per query.

- **POCKET-502 · Weighted / tunable RRF.** ✅ Shipped this pass. Per-strategy
  weights (`config.POCKET_RRF_WEIGHTS`, env `POCKET_RRF_{VECTOR,LEXICAL,GRAPH}_WEIGHT`)
  scale each strategy's RRF contribution in `_fold_ranked`/`_fuse`/`_fuse_ranked`/`search`
  (default 1.0 each == plain RRF). `evaluation.tune_weights` grid-searches them
  (`pocket eval --tune [--tune-metric] [--save-weights]`), never landing below the
  equal-weight baseline, and `save_weights`/`POCKET_RRF_WEIGHTS_FILE` feed the
  winner back into config. The lift is now measured on a graded hybrid corpus
  (MAP beats the equal-weight baseline by down-weighting a misleading vector
  strategy). ✅ The search itself was refined this pass: `tune_weights(method=)`
  now offers a cheaper `coordinate` ascent (one strategy at a time, memoised,
  `pocket eval --tune --tune-method coordinate`) that reaches the grid optimum
  with strictly fewer `evaluate` calls on the 3-strategy hybrid surface. A
  remaining bet is random/Bayesian search over a finer-grained weight range.

  *Seam: `_fold_ranked`/`_fuse` + `evaluation.py`.*

- **POCKET-503 · Query expansion.** ✅ Shipped this pass. Opt-in, deterministic,
  offline: `retrieval._expand_query` appends synonym/acronym expansion terms from
  `config.POCKET_QUERY_EXPANSION_MAP` (built-in acronym map, overridable via a
  `POCKET_QUERY_EXPANSION_FILE` JSON) to the query before `_gather`, so both BM25
  and the embedding see the long form of an abbreviation. It only *adds* missing
  words (original tokens keep their rank/mass) and de-dupes, so the default
  (`POCKET_QUERY_EXPANSION=0`) is a strict no-op. Threaded through
  `evaluation.evaluate(..., use_expansion=)` and `pocket search/eval --expand`.
  The lift is measured on the graded corpus: a two-answer gold case pairs
  `db_journal.md` (spelled-out match) with `db_wal.md` (only reachable via the
  long form of `wal`), and expansion raises Recall@3 0.5→1.0 while the hit-rate
  floor holds. A local-LLM paraphrase backend (reusing `POCKET_LLM_PROVIDER`)
  remains a future, non-deterministic add-on on top of this deterministic core.
- **POCKET-504 · Semantic query router.** ✅ Shipped this pass. Opt-in,
  deterministic, offline: `retrieval._route_query` classifies a query's *shape*
  (pure regex/keyword, no model call) into a concrete mode — code-shaped queries
  (snake_case / camelCase identifiers, `foo()` calls, `::` scopes,
  `filename.ext`, code punctuation, backtick spans) → `lexical` exact-match;
  relationship / concept questions ("how does X relate to Y", "connection
  between …") → `graph` multi-hop; everything else → the `hybrid` blend.
  Exposed as `mode="auto"` (CLI `--mode auto`, `/search?mode=auto`,
  `/trace?mode=auto`); `_resolve_mode` downgrades a routed `graph` to `hybrid`
  when the target has no graph tables so routing never returns zero results.
  Setting `POCKET_QUERY_ROUTER=1` also auto-routes a plain `hybrid` call (the
  default mode), giving existing callers the right blend with no call-site
  change; default OFF keeps `hybrid` a fixed blend. Measured on the graded
  corpus: a new code-shaped gold case (`router_anchor.md` +
  `router_blend_a/b/c.md`, `mode="auto"`) routes to lexical and ranks the
  exact-match answer #1 (MAP=1.0) where plain hybrid lets a vector-favoured
  distractor outrank it — a ranking win at an unchanged Recall@k floor. A
  follow-up could learn the routing thresholds from labelled traffic instead of
  the hand-tuned shape rules.


### Engine parity (already on the README roadmap)

- **POCKET-P4 · State-diff delta writes.** ✅ Shipped this pass. `declare_row`
  now fingerprints each desired row, reads the stored fingerprint, and asks
  `connectorkits.statediff.diff` for the action (`insert`/`replace`/skip) so a
  reprocess rewrites only changed rows and stops churning the FTS index — the
  orphan-deletion half was already covered by `end_source`/`sweep`. Runs report
  `row_writes`/`row_skips` via `UpdateStats`.
- **POCKET-P5 · Logic-fingerprint memo keying.** ✅ Shipped this pass. The memo
  store was already SQLite-backed (`_pocket_memo` / `_pocket_extract_memo`) and
  already survived process restarts — `test_incremental_memoization` rebuilds a
  fresh `App`/target on the same DB between runs. The genuine remaining cocoindex
  C4 gap was that our memo key folded only source content + embedding signature,
  so editing a transform's *code* left unchanged files skipped on stale output.
  `mount_each` now folds `_logic_fingerprint(func)` (`inspect.getsource` →
  bytecode → qualname, hashed via cocoindex/SHA-256) into every memo key, so a
  pipeline-code edit invalidates stale memos and forces a clean reprocess.
- **POCKET-P6 · `full_reprocess` force-rebuild flag.** ✅ Shipped this pass.
  Mirrors cocoindex's `App.update_blocking(full_reprocess=True)`: a new
  `full_reprocess` keyword (surfaced as `pocket update --full-reprocess`) sets a
  `_FULL_REPROCESS` contextvar so `mount_each` bypasses the memo fast-path and
  re-runs every transform even on unchanged fingerprints — the on-demand escape
  hatch for changes the fingerprint can't see (schema/target-format edits). The
  P4 per-row state-diff still dedups physical writes, so a clean rebuild doesn't
  duplicate or churn rows; live mode forces only the catch-up pass. This closes
  the last cocoindex *critical* (C-series) gap.
- **POCKET-W2 · Push-style live mode.** ✅ Shipped this pass. Live indexing is now
  change-driven instead of blind interval polling: source connectors self-register
  via `pocketindex.register_source`, `LocalFS.signature()` exposes a cheap
  `(mtime, size)` map, and the live loop only re-runs the pipeline when an actual
  add/edit/delete is observed — an idle watch costs a single `stat` scan per
  interval rather than a full re-embedding pass, while edits are picked up promptly.
  Sources without a `signature()` fall back to interval polling so no change is ever
  silently missed. This closes the last cocoindex *workflow* gap; only the
  native-cocoindex migration PoC remains.

### Ops & UX

- **POCKET-505 · HITL review in the Web UI.** ✅ Shipped this pass. The pending
  queue (`admin.list_pending/approve_pending/reject_pending`), previously CLI-only,
  is now exposed over REST (`GET /pending`, `POST /pending/{approve,reject}` with a
  JSON `{ids?}` body) and in the dependency-free Web UI as a **Pending review**
  panel with per-fact and bulk approve/reject buttons. Because entity/relation ids
  are signed 64-bit hashes beyond JavaScript's 2**53 safe range, the REST layer
  speaks ids as decimal strings (stringify out, parse to int in) so the browser
  never silently corrupts them — proven by a >2**53 approve round-trip test. The
  mode picker also gained the POCKET-504 `auto` option.
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

**POCKET-501 (MMR)**, **POCKET-502 (weighted RRF + coordinate ascent)**,
**POCKET-503 (query expansion)**, and **POCKET-504 (semantic query router)** are
shipped *and now measured*: the graded corpus under `eval/` turns all four into
real Recall@k/MAP wins (`tests/test_eval_proof.py`), and `pocket eval --mmr` /
`--tune` / `--expand` plus `pocket search --mode auto` expose the trade-offs from
the CLI. The Ops & UX queue advanced too: **POCKET-505 (HITL review in the Web
UI)** now surfaces the pending-fact queue over REST and in the browser, and
**POCKET-P4 (state-diff delta writes)**, **POCKET-P5 (logic-fingerprint memo
keying)**, and **POCKET-P6 (`full_reprocess` flag)** closed the Engine-parity
backlog: `declare_row` uses `connectorkits.statediff.diff` to skip no-op writes,
`mount_each` folds a transform logic fingerprint into every memo key so a
pipeline-code edit reprocesses unchanged files instead of serving old-code
output, and `--full-reprocess` forces a clean rebuild on demand. **All cocoindex
*critical* (C1–C5) gaps are now green, and the W2 live-mode push gap is closed
too** — live indexing is change-driven (`register_source` + `LocalFS.signature()`),
so idle watches cost only a `stat` scan and edits are picked up promptly. The only
engine-parity item left is the eventual **native-cocoindex migration**
(`pocket/pipeline_native.py` PoC).
After that, **POCKET-506 (answer synthesis with citations)** and **POCKET-507
(snippet highlighting)** build
on the byte-exact lineage Pocket already carries. Smaller follow-ups still open: a
non-deterministic local-LLM paraphrase backend on top of the POCKET-503 core,
learning the POCKET-504 routing thresholds from labelled traffic, and
auto-selecting `mmr_lambda`/weights per query rather than a fixed grid.



