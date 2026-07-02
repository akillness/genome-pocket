# Changelog

All notable changes to **genome-pocket** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Engine-parity hardening cycle, cross-checked against upstream `cocoindex` (1.0.11)
by installing it in an isolated venv and diffing its public API against the
vendored `pocketindex` engine.

### Fixed
- **`POCKET_HYDE_OLLAMA_HOST` env var now actually read.** The README documented
  `POCKET_HYDE_OLLAMA_HOST` but `pocket/config.py` only consulted `OLLAMA_HOST`,
  making the documented variable a silent no-op. Config now resolves
  `POCKET_HYDE_OLLAMA_HOST` first and falls back to `OLLAMA_HOST`, then the
  `127.0.0.1:11434` default — backward compatible for existing `OLLAMA_HOST` users.
- **MCP `search_knowledge` accepts `mode="auto"` (serve-surface parity).** The
  CLI (`--mode auto`) and REST API (`?mode=auto`) already engaged the POCKET-504
  semantic router, but the MCP tool rejected `auto` — the only serve surface
  without it. The tool now accepts the same five modes
  (`auto|hybrid|vector|lexical|graph`) as the other two surfaces. Covered by new
  assertions in `tests/test_pipeline.py::test_mcp_tools`.
- **README drift corrected against the code.** Semantic-chunking threshold docs
  now state the real default (`0.7`) and the correct direction (higher floor →
  more/smaller chunks; the old text inverted it); the MCP tools list includes the
  fourth tool `traverse_graph`; usage covers `pocket update --full-reprocess`,
  `pocket update --graph --review`, and the `pocket graph review` HITL flow; the
  project-structure tree includes `pocketindex/stats.py`, `tests/`, `eval/`,
  `run_tests.sh`, and `CHANGELOG.md`.

### Changed
- **Retrieval and PocketIndex internals split into focused modules.** The public
  facades remain stable (`from pocket import retrieval`, `import pocketindex as
  pix`), but the previous monoliths are now organized by responsibility:
  `pocket/retrieval/{router,search,fusion,rerank,graph,inspect}.py` and
  `pocketindex/{app,context,runtime,memo}.py`. The split keeps old private helper
  re-exports used by the in-tree tests, adds `pocket.retrieval` to the explicit
  setuptools package list, and updates README/SVG diagrams to show the new module
  boundaries.

### Added
- **Push-style live mode (POCKET-W2, cocoindex live push).** Live indexing is
  now change-driven instead of blind interval polling. Source connectors
  self-register via `pocketindex.register_source`, and `LocalFS.signature()`
  exposes a cheap `(mtime_ns, size)` map of the indexable files. The live loop
  compares successive signatures and only re-runs the pipeline when an actual
  add, edit, or delete is observed — an idle watch now costs a single `stat`
  scan per interval instead of a full re-embedding pass, and edits are picked up
  promptly. Sources that don't expose a `signature()` fall back to the original
  interval polling so no change is ever silently missed. Tests:
  `test_live_mode_push_skips_run_when_sources_unchanged`,
  `test_live_mode_push_reruns_when_file_modified`, and
  `test_live_mode_push_reruns_when_file_deleted` in `tests/test_pipeline.py` —
  suite now 169. This closes the last workflow gap; only the native-cocoindex
  migration PoC remains.
- **`full_reprocess` force-rebuild flag (POCKET-P6, cocoindex C5).** Mirrors
  cocoindex's `App.update_blocking(full_reprocess=True)`: the new
  `full_reprocess` keyword on `App.update_blocking` / `run_async` (surfaced as
  `pocket update --full-reprocess`) sets a `_FULL_REPROCESS` contextvar that
  makes `mount_each` bypass the memo fast-path, so every transform re-runs even
  when its content+logic fingerprint is unchanged. This is the on-demand escape
  hatch for changes the fingerprint can't observe (e.g. a schema/target-format
  change). The per-row state-diff in the SQLite target (P4) still dedups physical
  writes, so a clean rebuild re-executes the logic without duplicating or
  churning rows, and pipeline state stays intact for the next incremental run.
  In live mode only the initial catch-up pass is forced; later polls revert to
  incremental. Tests: `test_full_reprocess_forces_rebuild_of_unchanged_files`
  (engine) and `test_update_cli_threads_full_reprocess_flag` (CLI wiring) in
  `tests/test_pipeline.py` — suite now 166. This closes the last cocoindex
  *critical* (C-series) gap; the remaining workflow gap (W2 live-push) is closed
  above, leaving only the native-cocoindex migration PoC.
- **Logic-fingerprint memo keying (POCKET-P5, cocoindex C4).** The memo store was
  already SQLite-backed (`_pocket_memo` / `_pocket_extract_memo`) and so already
  survived process restarts — verified by `test_incremental_memoization`, which
  rebuilds a fresh `App`/`TableTarget` on the same DB between runs. The genuine
  remaining gap versus cocoindex (whose persistent memo is keyed by a *logic
  fingerprint*) was that our memo key folded only source content and the embedding
  signature: editing a transform's code (e.g. chunking/extraction) left unchanged
  files skipped, silently serving output produced by the *old* code. `mount_each`
  now computes `_logic_fingerprint(func)` once per run — `inspect.getsource`,
  falling back to bytecode then qualified name, hashed via the cocoindex
  fingerprint with a SHA-256 fallback — and folds it into every memo key
  (alongside `POCKET_EMBED_SIG`) so a pipeline-code edit invalidates stale memos
  and forces a clean reprocess. Tests: two new in `tests/test_pipeline.py`
  (`test_logic_fingerprint_folds_into_memo_hash`, `test_logic_change_invalidates_memo`)
  — suite now 164.
- **State-diff delta writes in the SQLite target (POCKET-P4).** `TableTarget`
  previously re-UPSERTed *every* row a source re-emitted on reprocess and
  delete-reinserted its FTS5 companion row each time, so editing one paragraph of
  a file rewrote all of its chunks. `declare_row` now fingerprints the desired
  non-key values, reads the stored row's fingerprint, and asks
  `cocoindex.connectorkits.statediff.diff` for the write action — `insert` for a
  new key, `replace` when the stored row differs, and a no-op skip when it has
  already converged (with a built-in fallback when cocoindex is absent). Skipped
  rows are still attributed to their source so `end_source` never mistakes them
  for orphans and sweeps them. The orphan-deletion half of the cocoindex C2 gap
  (stale chunks accumulating on edits) was already handled by `end_source`/`sweep`;
  this pass closes the other half — write amplification and FTS churn on the rows
  that did *not* change. `UpdateStats`/`ComponentStats` gained `num_row_writes`
  and `num_row_skips` (surfaced in `__str__` as `row_writes`/`row_skips`) so a run
  reports how many physical rows it actually touched. Tests: three new in
  `tests/test_pipeline.py` (statediff `insert`/skip/`replace` decision,
  per-row skip-on-redeclare with orphan-survival, and an end-to-end forced
  reprocess that writes 0 / skips all rows) — suite now 162.
- **Graded-corpus eval proof for the fusion features (POCKET-501/502 follow-up).**
  Until now MMR and weighted RRF were proven only as mechanics — the offline
  `MockEmbedder` emits zero vectors, so no real cosine signal reached the
  retrieval path end to end. Added a shipped graded corpus (`eval/corpus/` +
  hand-labelled `eval/gold.json` with multi-relevant hybrid queries) and a
  deterministic offline `HashingEmbedder` (L2-normalised hashed bag of words, no
  model download) so the harness measures an actual quality delta. New measured
  results on hybrid queries: MMR raises Recall@3 from 0.5→1.0 on a query whose
  second distinct answer is buried by near-duplicate chunks, and
  `tune_weights` lifts MAP above the equal-weight baseline by down-weighting a
  vector strategy that favours keyword-dense distractors. The harness gained the
  plumbing to measure this: `evaluation.evaluate(..., use_mmr=, mmr_lambda=)` and
  `pocket eval --mmr/--no-mmr` let you A/B the MMR diversity trade-off from the
  command line instead of only via `POCKET_MMR`. Tests: `tests/test_eval_proof.py`
  (gold/corpus label integrity, retrievability floor, MMR Recall@k win, tuner
  MAP win + vector down-weighting, and the `--mmr` CLI path) — 5 new.

- **Coordinate-ascent weight search (POCKET-502 refinement).** `tune_weights`
  gained a `method=` parameter: the exhaustive `"grid"` (default, unchanged) or a
  cheaper `"coordinate"` ascent that optimises one strategy at a time, holding the
  others fixed, and repeats full passes until one yields no improvement
  (`max_passes`, default 3). Scored points are memoised so revisits are free, and
  `1.0` is still always probed so the result is never worse than plain RRF. On the
  3-strategy hybrid surface this reaches the grid's optimum with strictly fewer
  `evaluate` calls; exposed as `pocket eval --tune --tune-method {grid,coordinate}`.
  Tests: coordinate ascent matches the grid optimum more cheaply on both the
  graded corpus and a synthetic lexical suite, plus method validation and the new
  CLI flag — 4 new.

- **Query expansion (POCKET-503).** Opt-in, deterministic, offline. When
  `POCKET_QUERY_EXPANSION` is truthy (or `pocket search --expand` /
  `pocket eval --expand`), retrieval augments the query with synonym/acronym
  expansion terms from a built-in map (`config.POCKET_QUERY_EXPANSION_MAP`,
  overridable via a `POCKET_QUERY_EXPANSION_FILE` JSON) before fusion, so a query
  phrased with an abbreviation (`wal`) can still reach a document that only spells
  out the long form (`write ahead log`). The new `retrieval._expand_query` only
  *appends* missing words (BM25 rank and vector mass of the original tokens are
  preserved) and de-duplicates deterministically, so the default (flag off) is a
  strict no-op. The lift is measured on the graded corpus: a new two-answer gold
  case pairs `db_journal.md` (matches the spelled-out terms) with `db_wal.md`
  (whose only hook is the long form of `wal`, present in no document), and
  expansion raises Recall@3 from 0.5→1.0 by recovering the abbreviation-only file
  while the hit-rate floor holds either way. Threaded through
  `evaluation.evaluate(..., use_expansion=)` and exposed on the CLI as
  `pocket search/eval --expand/--no-expand`. Tests: `_expand_query` determinism /
  dedup / no-op / case-insensitivity / file override, the graded-corpus Recall@k
  win, and the `--expand` CLI path — 7 new.

- **Semantic query router (POCKET-504).** Opt-in, deterministic, offline. A new
  `mode="auto"` (and `pocket search --mode auto`, `GET /search?mode=auto`,
  `/trace?mode=auto`) picks a concrete retrieval mode from the query's *shape*
  instead of always fanning out every strategy: code-shaped queries (snake_case /
  camelCase identifiers, `foo()` calls, `::` scopes, `filename.ext`, code
  punctuation, backtick spans) route to **lexical** exact-match; relationship /
  concept questions ("how does X relate to Y", "connection between …") route to
  **graph** multi-hop; everything else keeps the **hybrid** blend. The classifier
  `retrieval._route_query` is a pure regex/keyword shape check (no model call), so
  routing is reproducible and unit-testable; `_resolve_mode` downgrades a routed
  `graph` to `hybrid` when the target has no graph tables, so routing never
  silently returns zero results. Setting `POCKET_QUERY_ROUTER=1` also auto-routes
  a plain `hybrid` call (the default mode), so existing callers get the right
  blend without changing their call site; default OFF keeps `hybrid` a fixed blend.
  The lift is measured on the graded corpus: a new code-shaped gold case
  (`router_anchor.md` + `router_blend_a/b/c.md`, `mode="auto"`) routes to lexical
  and ranks the exact-match answer #1 (MAP=1.0) where plain hybrid lets a
  vector-favoured distractor outrank it — a measured ranking win at an unchanged
  Recall@k floor. Tests: shape-classification battery, the auto-vs-hybrid MAP win
  + lexical-route equivalence, the `POCKET_QUERY_ROUTER` flag upgrade, the
  graph→hybrid fallback, the `--mode auto` CLI path, and `routing_trace` / `/trace`
  auto routing — 7 new.

- **HITL pending-review in the Web UI + REST (POCKET-505).** The confidence gate
  (`POCKET_GRAPH_MIN_CONFIDENCE`) stages low-confidence graph facts as
  `status="pending"`, but until now they were reviewable only through the CLI
  (`pocket graph review`). Surfaced the same `admin.list_pending/approve_pending/
  reject_pending` queue over HTTP and in the dependency-free Web UI: `GET /pending`
  lists staged entities/relations, and `POST /pending/approve` / `POST /pending/reject`
  commit or discard them (JSON `{"ids": [...]}` for specific facts, omit `ids` /
  send `null` for all). The Web UI gained a **Pending review** panel that loads the
  queue and exposes per-fact and bulk approve/reject buttons (and the mode picker
  now also lists the POCKET-504 `auto` router). Entity/relation ids are signed
  64-bit hashes that exceed JavaScript's safe-integer range (2**53), so the REST
  layer speaks ids as **decimal strings** (`_stringify_pending` on the way out,
  `_parse_ids` back to int on the way in) — the Python CLI/admin layer keeps using
  native ints. Tests: listing surfaces only pending rows with string ids and
  join-resolved relation endpoints, a >2**53 id survives the approve round-trip
  (a float64-rounded neighbour would miss it), reject deletes the rows and their
  FTS mirror, action/id validation (404 / 400), the missing-index 503 guard, and
  the Web UI markup — 6 new.

- **Weighted / tunable Reciprocal Rank Fusion (POCKET-502).** RRF used to fuse
  every strategy with equal weight; `_fold_ranked`/`_fuse`/`_fuse_ranked`/`search`
  now take per-strategy `weights` (`{vector, lexical, graph}`) so each strategy's
  reciprocal-rank contribution can be scaled (`weight·1/(RRF_K+rank)`). Defaults
  are 1.0 each (`config.POCKET_RRF_WEIGHTS`, from `POCKET_RRF_{VECTOR,LEXICAL,GRAPH}_WEIGHT`),
  so behaviour is identical to plain RRF until tuned; negative weights clamp to 0
  (disable, never invert). The eval harness becomes an **optimizer**:
  `evaluation.tune_weights` grid-searches the weights against real retrieval
  (`pocket eval --tune [--tune-metric M] [--save-weights FILE]`), varying only the
  strategies the cases' modes exercise, always probing 1.0 so it can never land
  below the equal-weight baseline, and breaking ties toward the baseline. The
  winner is persisted via `save_weights`/`load_weights`; pointing
  `POCKET_RRF_WEIGHTS_FILE` at that file feeds it back into `config` (overriding
  the env defaults), closing the tune→apply loop for search and eval alike.
  Tests: weight-resolution defaults/merge/clamp, equal-weight == plain RRF,
  up-weighting flips a fusion tie, zero weight disables without dropping the
  chunk, env + file config resolution, the tuner never-worse-than-baseline +
  persistence, unknown-metric guard, and the `--tune` CLI path (9).

- **Result diversity via MMR fusion re-ranking (POCKET-501).** `pocket.retrieval`
  can now re-rank the fused candidate pool with Maximal Marginal Relevance so
  near-duplicate chunks (e.g. several from the same file) stop crowding the
  top-k. Fusion was split into `_fuse_ranked` (full `(chunk_id, hit)` pool) and
  `_fuse` (the existing top-k); when MMR is on, `search()` pulls each candidate's
  stored float32 embedding (`_fetch_embeddings`) and picks greedily by
  `λ·relevance − (1−λ)·max-cosine-to-selected` (`_mmr_rerank`/`_cosine`). Off by
  default — the deterministic RRF order is unchanged unless opted in via
  `POCKET_MMR`/`POCKET_MMR_LAMBDA` (env) or `pocket search --mmr/--no-mmr`
  (per-query). Degrades safely: missing/zero embeddings count as non-redundant,
  so the relevance order is preserved. Tests: cosine signal + degenerate inputs,
  λ=1 reproduces relevance order, low λ promotes a diverse candidate over a
  near-duplicate, limit/empty handling, the end-to-end MMR search path, and the
  config-default routing (6).
- **Agent-native `pocket search --json`.** The CLI now mirrors the REST
  `/search` and MCP `search_knowledge` surfaces: with `--json` it emits
  `{query, mode, count, hits[]}` (each hit carrying full
  `file_path`/`text`/`start_offset`/`end_offset`/`score` lineage) as pure JSON on
  stdout, while status and "run update first" diagnostics go to stderr, so a
  calling agent or pipeline can parse `pocket search` output deterministically.
  Tests: JSON payload shape + lineage keys on a real lexical hit, and the
  empty-index path emitting `[]` on stdout with the hint on stderr (2).

### Fixed
- **Test runner was broken and bypassed the offline mock.** `run_tests.sh`
  invoked `tests.test_retrieval_api.TestGraphExtraction` (the class actually
  lives in `tests/test_graph_unit.py`), so the script errored partway through,
  and it drove tests via `python -m unittest`, which does not load the
  session-scoped `MockEmbedder` fixture in `tests/conftest.py` — silently loading
  real model weights and skipping whole modules (`test_graph_unit`,
  `test_multimodal`, `TestRetrievalEvaluation`). Rewrote it to drive pytest,
  which auto-discovers every module, honors the offline mock, and runs all 91
  tests in ~5 s. Updated the stale "81 tests" count in the README.

### Docs
- **Review & improvement backlog (spec-stack Write layer).** Added
  `docs/planning/review-2026-improvements.md`: a grounded review of the current
  codebase plus a prioritized, feature-oriented backlog (MMR fusion, weighted
  RRF, query expansion, semantic routing, HITL in the Web UI, citation
  synthesis) with concrete code seams for the next loop.
- **Product strategy & discovery doc (PM frameworks).** Added
  `docs/planning/pm-product-strategy-2026.md`: positioning, Jobs-to-be-Done,
  personas, a North Star + input-metric set, a Teresa Torres Opportunity
  Solution Tree mapping the engineering backlog to outcomes, a lightweight PRD
  for cited-answer synthesis (POCKET-506), and pretotyping/riskiest-assumption
  tests to validate before building. Also corrected a stale "89 tests" count in
  the README roadmap (suite is now 91).


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
- **Multimodal image embedding via SigLIP2 (opt-in).** A new transformers-native
  `SiglipEmbedder` (`pocketindex/ops/siglip_embedder.py`) maps text *and* images
  into SigLIP2's shared, L2-normalized space, so a text query matches stored image
  embeddings through the existing sqlite-vec single-vector + RRF path — no
  reranker or multi-vector machinery required. Enabled by setting
  `EMBEDDING_MODEL=google/siglip2-base-patch16-224` (any `siglip2` id); the default
  text-only path is unchanged. The `localfs` connector now lists image files
  (`.png/.jpg/.jpeg/.webp/.gif/.bmp/.tiff`); the pipeline routes them to a
  single-row, no-split image embedding pass *only* when the active embedder
  advertises `supports_image`, and the memo fingerprint hashes image bytes so an
  edited image re-embeds. Graph extraction stays text-only. Heavy deps
  (`transformers`/`torch`/`Pillow`) install via the new `multimodal` extra and are
  imported lazily, keeping the base install text-only. SigLIP2 is Apache-2.0.

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
### Added
- **Multimodal image embedding via SigLIP2 (opt-in).** A new transformers-native
  `SiglipEmbedder` (`pocketindex/ops/siglip_embedder.py`) maps text *and* images
  into SigLIP2's shared, L2-normalized space, so a text query matches stored image
  embeddings through the existing sqlite-vec single-vector + RRF path — no
  reranker or multi-vector machinery required. Enabled by setting
  `EMBEDDING_MODEL=google/siglip2-base-patch16-224` (any `siglip2` id); the default
  text-only path is unchanged. The `localfs` connector now lists image files
  (`.png/.jpg/.jpeg/.webp/.gif/.bmp/.tiff`); the pipeline routes them to a
  single-row, no-split image embedding pass *only* when the active embedder
  advertises `supports_image`, and the memo fingerprint hashes image bytes so an
  edited image re-embeds. Graph extraction stays text-only. Heavy deps
  (`transformers`/`torch`/`Pillow`) install via the new `multimodal` extra and are
  imported lazily, keeping the base install text-only. SigLIP2 is Apache-2.0.
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
- **Default embedding model → `Qwen/Qwen3-Embedding-0.6B` (POCKET-405).** Replaces
  `all-MiniLM-L6-v2` (384-d) with the Apache-2.0 Qwen3-Embedding-0.6B (1024-d),
  a 2026 open-weight MTEB-leading retriever. Both the indexing path
  (`SentenceTransformerEmbedder`) and the query path (`pocket.retrieval`) now apply
  the model's asymmetric prompt registry — documents use the empty `document`
  prompt, queries are wrapped in the `query` instruction — while symmetric models
  with no prompts (e.g. MiniLM) keep encoding plainly, so the swap is fully
  backward compatible. Override with `EMBEDDING_MODEL=...` as before.
- **Embedding-model-aware memoization.** The source fingerprint
  (`_compute_memo_hash`) now folds in the active embedding signature
  (`POCKET_EMBED_SIG`, set from `EMBEDDING_MODEL`). Switching models invalidates
  every memo so unchanged sources are re-embedded at the new vector dimension on
  the next `pocket update`, instead of leaving stale mixed-dimension vectors that
  would break `vec_distance_cosine`.
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
