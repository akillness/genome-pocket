# Changelog

All notable changes to **genome-pocket** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Engine-parity hardening cycle, cross-checked against upstream `cocoindex` (1.0.11)
by installing it in an isolated venv and diffing its public API against the
vendored `pocketindex` engine.

### Added
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

### Changed
- **Indentation-preserving refine path.** `TextRefiner.refine(text, code=True)`
  preserves inline whitespace and indentation for code files (prose still
  collapses runs of whitespace), so block structure such as Python indentation
  survives into the index.

### Fixed
- Blank-line collapse in `TextRefiner` no longer swallows the following line's
  indentation.
- Removed a duplicated `IdGenerator` bullet in the README transformation steps.

### Tests
- Added `TestCodeAwareSplitting` (8 cases) and the integration test
  `test_code_file_lineage_and_boundaries`, plus `test_run_reports_stats` and
  `test_live_mode_picks_up_new_file`, and `TestLifecycleCommands` (6 cases)
  covering `ls`/`show`/`drop`, single-source eviction + re-index, and the CLI
  surface. Full suite (27 tests) passes via `bash run_tests.sh`.
