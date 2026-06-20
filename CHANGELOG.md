# Changelog

All notable changes to **genome-pocket** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Engine-parity hardening cycle, cross-checked against upstream `cocoindex` (1.0.11)
by installing it in an isolated venv and diffing its public API against the
vendored `pocketindex` engine.

### Added
- **Automated retrieval evaluation & regression guard (POCKET-303).** New
  `pocket/evaluation.py` plus a `pocket eval` command turn retrieval quality into
  a measurable, CI-gateable number so changing chunk sizes, the embedding model,
  or fusion weights can't silently regress recall. It computes standard IR metrics
  (Hit@k, MRR, Precision/Recall@k, MAP) over either a hand-written gold set
  (`--cases <json>`, `{query, relevant_files[, mode]}`) or **synthetic query/context
  pairs mined from the existing index** (`synthesize_cases()`): for each source file
  it picks the tokens most *distinctive* to it (lowest cross-file document
  frequency) and builds a self-labeled query whose only correct answer is that
  file, so no curated gold set is required and any indexing/chunking regression
  shows up as a dropped hit. `evaluate()` calls the real `retrieval.search()` (per
  case `mode`), so the harness can never drift from production retrieval.
  `save_baseline()`/`compare_to_baseline()` (with a `--tolerance`) record a run and
  fail `pocket eval --baseline <json>` (exit 1) on any metric that fell below it.
  Tests: metric primitives + lenient path matching, synthetic self-retrieval at
  perfect Hit@k/MRR, empty-index safety, gold-case JSON parsing + validation
  errors, baseline round-trip + tolerance-aware regression detection, and the CLI
  end-to-end (synthesize → save → pass → doctored-baseline fail) (6).
- **Local query-tracing & lineage web UI (POCKET-301 slice).** `pocket serve` now

  serves a single, dependency-free HTML page at `GET /` (`pocket/web_ui.py`) that
  visualizes *how a query was routed* and *which source files answered it*. A new
  `retrieval.routing_trace()` is the testable core: it reuses the same
  `_gather`/`_fuse` orchestration as `search()` (refactored into a shared `_gather`
  helper plus a `_MODE_STRATEGIES` routing table) and returns, per strategy
  (`vector`/`lexical`/`graph`), whether the chosen mode *activates* it, whether it
  is *available* on the target (FTS / graph tables present), and its candidate
  count — plus the fused hits, each tagged with the `contributors` that surfaced
  it. Exposed over HTTP as `GET /trace`, and the page lazy-loads per-file chunk
  lineage via the existing `/lineage` endpoint. No Streamlit/React/build step, in
  keeping with Pocket's local-first design. Tests: trace strategy/contributor
  annotation, lexical-mode routing isolation, missing-index empty trace, and the
  `/` + `/trace` endpoints (4).
- **Interactive review during `pocket update --graph` (POCKET-301 slice).**
  A new `--review` flag turns the graph build into a human-in-the-loop pass: after
  indexing, `_interactive_graph_review()` (in `pocket/cli.py`) walks the operator
  through every fact the confidence gate staged as `pending`, offering a bulk
  *approve-all / reject-all / each / skip* choice and, in *each* mode, a per-fact
  *approve / reject / leave-pending / quit* prompt. It reuses the same
  `pocket.admin` review API as `pocket graph review`, so the inline and post-hoc
  flows stay consistent; anything left unresolved stays pending for later. The flag
  is opt-in and a no-op without `--graph` or in live mode (both reported, not
  silent). Tests: approve-all commit, each-mode per-fact routing, quit mid-loop,
  skip leaves all pending, no-prompt when nothing is staged, plus CLI end-to-end
  `update --graph --review` and `--review`-without-`--graph` guard (7).
- **Human-in-the-loop graph approval gate (POCKET-302).** Low-confidence graph
  facts are no longer dropped — they are *staged*. `EntityNode`/`RelationEdge` gain
  a `status` column (`"approved"` | `"pending"`); the pipeline writes any node/edge
  below `POCKET_GRAPH_MIN_CONFIDENCE` (or with a staged endpoint) as `pending`
  instead of committing or discarding it. All graph reads in `pocket/retrieval.py`
  filter to approved facts via a `_status_clause()` helper (legacy graphs without
  the column degrade to an always-true predicate), so pending facts never surface
  in search, neighborhood, or concept listings until a human accepts them. New
  `pocket/admin.py` review API — `list_pending()`, `approve_pending(ids=None)`,
  `reject_pending(ids=None)` — is surfaced through a restructured `pocket graph`
  command group: `pocket graph review` lists/approves/rejects staged facts
  (`--approve`/`--reject <id>`, `--approve-all`/`--reject-all`), while
  `pocket graph <entity>` still routes to the neighborhood view for backward
  compatibility. Matches the uncertainty-guided KG-construction stance
  (arXiv:2605.26835). Tests: staging vs. commit by threshold, pending facts hidden
  from retrieval, approve/reject round-trips, specific-id approval, and CLI
  review + back-compat routing (7).
- **GraphRAG retrieval — entity-anchored multi-hop fusion (POCKET-404d).**
  `pocket/retrieval.py` adds `_graph_search()`, a third retriever that anchors the query
  to the nearest `entities` by name embedding, traverses one hop over `relations`, and
  collects the `source_chunk_ids` of every touched node/edge — surfacing real `embeddings`
  chunks (full lineage preserved) in graph-relevance order. `_fuse()` now blends it as a
  third Reciprocal Rank Fusion list (optional `graph_rows` arg keeps the two-list
  signature), and `RetrievalHit` gains a `graph_rank`. New `mode="graph"` (traversal only)
  and graph-aware `mode="hybrid"` (vector + lexical + graph); both stay backward compatible
  because graph only participates when the `entities` table exists. Surfaced through
  `pocket search --mode graph`, the existing `pocket graph <entity>` CLI, a new MCP
  `traverse_graph` tool, and the REST `/search?mode=graph` endpoint. The design follows the
  GraphRAG multi-hop pattern (arXiv:2606.00610 / 2606.17856). Tests: graph-mode anchoring,
  empty-without-graph guard, hybrid graph-signal fusion, and the `traverse_graph` tool (4).
- **Auditable entity-resolution merge rationale (POCKET-404c).**
  `pocketindex/ops/entity_resolution.py` now records *why* each merge happened: every
  accepted union carries a `MergeRecord` (`kept` / `merged` / `method` ∈
  `exact_name`|`embedding`|`llm` / `similarity` / `rationale`), and `ResolvedEntity`
  exposes the per-cluster `merges` audit trail. The optional `MergeAdjudicator` may now
  return `(bool, rationale)` so an LLM's merge justification is captured, not discarded —
  the verifiability requirement from arXiv:2606.01210 (don't trust opaque LLM ER
  decisions). The pipeline serializes the trail into a new `EntityNode.resolution` JSON
  column so a human can audit a merge through the same end-to-end lineage as nodes/edges.
  Also fixed a duplicate `by_norm` initialization in the blocking pass. Tests:
  `TestEntityResolutionRationale` (5) + resolution round-trip assertion in the graph
  target integration test.
- **Hardened JSON extraction prompt + extraction memoization (POCKET-404b).**
  `pocketindex/ops/extract.py` now pins the strict-JSON extraction prompt behind a
  `PROMPT_VERSION` constant with explicit JSON-only / grounding / verbatim-evidence /
  calibrated-confidence directives and a grounded few-shot exemplar, drawn from the
  2026 GraphRAG and structured-output literature. The strict-JSON-only contract and the
  single grounded exemplar are grounded in the small-model structured-output reliability
  benchmark (arXiv:2605.02363, where naive prompting yields 0% valid JSON), with the
  "format tax" caveat noted (arXiv:2604.03616); schema-agnostic typing/keys follow
  arXiv:2606.01208 / 2604.14862, calibrated confidence arXiv:2605.26835, verifiable
  evidence arXiv:2606.01210, and local-LLM viability arXiv:2605.20815. The Ollama/airLLM
  backends are wrapped in a
  new `MemoizingExtractor` keyed on `sha256(prompt_version, model_id, chunk_text)`,
  persisted via `SqliteExtractionStore` (`_pocket_extract_memo` table) so unchanged
  chunks under an unchanged prompt are never re-sent to the model across runs; bumping
  `PROMPT_VERSION` invalidates the cache. The deterministic backend stays unwrapped, so
  default runs gain no new table. Tests: `TestExtractionPromptAndMemo` (5).
- **Run statistics / monitoring (POCKET-401).** New `pocketindex/stats.py`
  (`UpdateStats` / `ComponentStats`) tracks adds / reprocesses / unchanged /
  deletes / errors per component. Stats are threaded through `mount_each` and
  printed on the CLI after every `pocket update`; `sweep()` now returns deletion
  counts.
- **Real live-mode watching (POCKET-402).** `pocket update -L [--interval N]`
  runs the pipeline on a polling loop (default 2s), picking up new / edited /
  deleted files between passes and stopping cleanly on Ctrl+C. Previously the
  `-L` flag was a no-op.
- **Code-aware splitting (POCKET-403).** Rewrote `pocketindex/ops/text.py` as a
  dependency-free mirror of the upstream text-ops surface: `detect_code_language`,
  `SeparatorSplitter`, `CustomLanguageConfig`, and a language-aware
  `RecursiveSplitter` that prefers per-language structural boundaries
  (class / def / fn / impl / ...) and falls back to a recursive
  paragraph→sentence→line→word→char split for prose. Chunks remain offset-exact
  and the `split(text, chunk_size, chunk_overlap)` signature stays
  backward-compatible. The `localfs` source connector now indexes recognized
  source-code files (`.py`, `.rs`, `.ts`/`.js`, `.go`, `.java`, ...) and the
  pipeline detects language by filename to route code vs. prose.
- **Lifecycle commands `ls` / `show` / `drop` (POCKET-405).** New CLI verbs that
  inspect and reset target state without re-running the pipeline: `pocket ls`
  lists indexed source files with chunk counts and offset spans; `pocket show`
  summarizes the whole index (sources / chunks / FTS status) or, given a path,
  prints that source's chunk lineage; `pocket drop [PATH] [--yes]` resets the
  entire index or evicts a single source's chunks, FTS mirror, and lineage/memo
  state (clearing the memo so a later `update` re-adds it). Backed by new
  read helpers `retrieval.list_sources` / `retrieval.target_stats` and a
  write-side `pocket/admin.py` (`drop_target` / `drop_source`).
- **Local-first knowledge graph — extraction, entity resolution & graph target
  (POCKET-404a).** `pocket update --graph` (or `POCKET_GRAPH=1`) now extracts a
  SQLite-resident knowledge graph alongside the vector/lexical index. New
  `pocketindex/ops/extract.py` turns chunks into `(entities, relations)` behind one
  `ExtractionModel` protocol with three **local** backends — `DeterministicExtractor`
  (default; no LLM, no network, no heavy deps, so the whole graph path is
  offline-testable), `OllamaExtractor` (local daemon), and **`AirLLMExtractor`**
  (in-process airLLM; an optional `genome-pocket[airllm]` extra that layer-shards a
  70B-class model onto a single 4GB GPU). **airLLM replaces LiteLLM** as the
  heavy-model backend: a hosted proxy is the wrong default for Pocket's privacy DNA.
  New `pocketindex/ops/entity_resolution.py` deduplicates entities via cost-effective
  blocking → cheap filters → optional LLM adjudication → label propagation over
  sqlite-vec embeddings (no faiss). The graph reuses the existing
  lineage/memo/sweep machinery, so editing a file re-extracts and deleting it sweeps
  its whole subgraph. `entities`/`relations` are modeled as `EntityNode`/`RelationEdge`
  in `pocket/pipeline.py`; retrieval gains `graph_neighborhood` + `pocket graph <entity>`;
  `admin.drop_target` clears the graph tables. With `--graph` off the pipeline is
  byte-for-byte unchanged — zero new cost or dependency for existing users.

### Changed
- **Indentation-preserving refine path.** `TextRefiner.refine(text, code=True)`
  preserves inline whitespace and indentation for code files (prose still
  collapses runs of whitespace), so block structure such as Python indentation
  survives into the index.

### Fixed
- Blank-line collapse in `TextRefiner` no longer swallows the following line's
  indentation.
- Removed a duplicated `IdGenerator` bullet in the README transformation steps.

### Docs
- **Graph target / KG-ops design spec (POCKET-201 / POCKET-404).** Added
  `docs/architecture/graph-target.md`: a local-first GraphRAG design (SQLite-resident
  `entities`/`relations` reusing the existing lineage/memo/sweep machinery, local-engine
  extraction via **airLLM/Ollama** — airLLM replacing LiteLLM so even a 70B-class
  extractor runs on-device, no hosted proxy — sqlite-vec blocked + LLM-adjudicated entity
  resolution, N-list RRF retrieval fusion, and HITL gating of low-confidence facts).
  Grounded in a live 2025–2026 arXiv survey and split POCKET-404 into 404a–404d.

### Tests
- Added `TestCodeAwareSplitting` (8 cases) and the integration test
  `test_code_file_lineage_and_boundaries`, plus `test_run_reports_stats` and
  `test_live_mode_picks_up_new_file`, and `TestLifecycleCommands` (6 cases)
  covering `ls`/`show`/`drop`, single-source eviction + re-index, and the CLI
  surface. Added `TestGraphExtraction` (6 cases — deterministic extraction,
  JSON validation, provider fallback, entity-resolution merging + optional
  adjudication) and `TestGraphTarget` (6 cases — `--graph` off creates no graph
  tables, entities/relations materialize and dedupe, idempotent re-runs, deletion
  sweeps the subgraph, neighborhood retrieval, and `drop` clears graph tables).
  Full suite (39 tests) passes via `bash run_tests.sh`.
