# cocoindex Gap Analysis & Migration Roadmap

> **Status:** Living document — last updated by jeo agent (dev branch)  
> **Scope:** Compare genome-pocket's custom `pocketindex` engine against the real
> `cocoindex` 1.0.11 library and define a phased migration path.

---

## 1. Why pocketindex exists

`genome-pocket` was bootstrapped before `cocoindex` was pip-installable in the
project venv. A compatibility-shim (`pocketindex/`) was authored to expose the
same surface (`App`, `fn`, `map`, `mount_each`, `lifespan`, `ContextKey`,
`use_context`, `EnvironmentBuilder`) so `pocket/pipeline.py` could be written
against a stable API without taking a hard runtime dependency.

That shim now diverges from cocoindex 1.0.11 in ways that cause silent
correctness bugs and block several planned features.

---

## 2. Critical gaps (correctness)

| # | Missing capability | pocketindex behaviour | cocoindex 1.0.11 | Affected files |
|---|--------------------|-----------------------|------------------|----------------|
| C1 | **Content fingerprinting** | Always full-reprocess every file on every run | `connectorkits.fingerprint.fingerprint_bytes/str/object` | `pocketindex/__init__.py` `mount_each`, `pocketindex/connectors/localfs.py` |
| C2 | **State-diff / delta writes** | ✅ Done — `end_source`/`sweep` delete orphaned chunks, and `declare_row` now uses `connectorkits.statediff.diff` for a per-row `insert`/`replace`/skip decision so unchanged rows are not rewritten | `connectorkits.statediff.DiffAction`, `TrackingRecordTransition` | `pocketindex/connectors/sqlite.py` `TableTarget` |
| C3 | **`map()` concurrency** | Sequential `for` loop | `cocoindex.map()` fans out as concurrent async tasks | `pocketindex/__init__.py` `map()` |
| C4 | **`fn(memo=True)` scope** | Per-run in-memory dict; cleared on restart | LMDB-backed persistent memo, keyed by logic fingerprint | `pocketindex/__init__.py` `fn` decorator |
| C5 | **`full_reprocess` flag** | Not implemented | `App.update_blocking(full_reprocess=True)` | `pocketindex/__init__.py` `App.update_blocking` |

---

## 3. Workflow gaps

| # | Missing workflow | Impact | Target location |
|---|-----------------|--------|-----------------|
| W1 | **`list_concepts` MCP tool** is a stub | Cannot browse knowledge graph via MCP | `pocket/mcp_server.py` |
| W2 | **Live mode uses polling** (`asyncio.sleep`) | File edits detected only on next poll | `pocketindex/__init__.py` live loop |
| W3 | **No progress display** | Long index runs are silent | missing `show_progress` / `UpdateHandle` |
| W4 | **No GPU runner** | `fn(runner=cocoindex.GPU)` not available | `pocketindex/__init__.py` |
| W5 | **No multi-app registry** | `list_app_names()` not available | `pocketindex/__init__.py` |
| W6 | **`drop_blocking` incomplete** | Async cleanup not guaranteed | `pocket/admin.py` |
| W7 | **Test suite hangs** | model download blocks pytest | `tests/` — need `MockEmbedder` |

---

## 4. Test infrastructure gaps

| # | Problem | Fix | Status |
|---|---------|-----|--------|
| T1 | `SentenceTransformerEmbedder` downloads model on first import | `MockEmbedder` returning deterministic fixed-dim zeros | ✅ done (`tests/conftest.py`) |
| T2 | No unit tests for `pocketindex/` internals | Engine reconciliation/abort covered via pipeline tests | ✅ done (`tests/test_pipeline.py`) |
| T3 | No fingerprint integration tests | `_compute_memo_hash` exercised by `test_incremental_memoization` | ✅ done |
| T4 | Graph tests require a real pipeline run | `tests/test_graph_unit.py` — DeterministicExtractor + in-memory SQLite, no pipeline run needed | ✅ done |

| T5 | `pytest` not in dev dependencies | Added via `uv add --dev pytest`; needs `uv.lock` commit | ✅ done |
| T6 | FTS5 lexical index orphan reconciliation untested | `test_fts_index_reconciles_on_edit_and_delete` asserts BM25 stays in lockstep on edit/delete | ✅ done |


---

## 5. Migration path (phased)

### Phase 0 — Unblock tests (immediate, <=1 day)
1. Add `MockEmbedder` to `tests/conftest.py` — fixed 384-dim zero vector, no network.
2. Monkeypatch `SentenceTransformerEmbedder` in all test setUp methods.
3. Confirm `pytest tests/ -q` completes in < 30 s with all 39 tests green.

### Phase 1 — Fingerprinting + delta writes (1-2 days)
4. Use `cocoindex.connectorkits.fingerprint.fingerprint_bytes` in `pocketindex/connectors/localfs.py`.
5. Store per-file hash in `source_state` sidecar table; skip unchanged files.
6. Teach `sqlite.TableTarget` to diff emitted PKs vs stored PKs and issue deletes for orphans.

### Phase 2 — Concurrency + memo (2-3 days)
7. Replace `map()` sequential loop with `asyncio.gather`.
8. Add persistent memo store (SQLite sidecar) for `@fn(memo=True)` across restarts.

### Phase 3 — list_concepts MCP + GPU runner stub (1 day)
9. Implement `list_concepts` in `pocket/mcp_server.py`: query `entities` + `relations` tables, return JSON summary. Guard with `POCKET_GRAPH=1`.
10. Add `GPU` runner stub to `pocketindex/__init__.py` for API parity.

### Phase 4 — Native cocoindex PoC (1 sprint)
11. Create `pocket/pipeline_coco.py` replacing `pocketindex` imports with real `cocoindex`.
12. Wire `cocoindex.Settings(db_path=...)` and adapt `TableTarget` to native target handler.
13. Run side-by-side, compare output; retire `pocketindex/` once parity confirmed.

---

## 6. Priority order

```
P0  T1-T3   MockEmbedder — unblocks all other work
P1  C1      Fingerprinting — biggest correctness win; large repos re-index fully every run
P2  C2      State-diff delta writes — DONE (orphan sweep + per-row statediff skip)
P3  W1      list_concepts MCP — user-visible, ~30 min to implement
P4  C3      map() concurrency — throughput improvement for multi-file batches
P5  W2      Live mode push — polling is functionally correct, lower priority
P6  Phase4  Native cocoindex migration — defer until Phases 0-3 are green
```

---

## 7. cocoindex APIs available but unused

These cocoindex 1.0.11 features are installed and importable right now:

```python
from cocoindex.connectorkits.fingerprint import fingerprint_bytes, fingerprint_str
from cocoindex.connectorkits.statediff import DiffAction, TrackingRecordTransition
from cocoindex.connectorkits.async_adapters import sync_to_async_iter, async_to_sync_iter
from cocoindex.ops.text import RecursiveSplitter, SeparatorSplitter, detect_code_language
from cocoindex.ops.sentence_transformers import SentenceTransformerEmbedder
import cocoindex  # App, fn, map, mount, mount_each, lifespan, Settings, ...
```

`ops.litellm` requires `litellm`; `ops.entity_resolution` requires `faiss` — neither installed.

---

## 8. References

- `docs/architecture/system-overview.md` — big-picture model
- `docs/architecture/graph-target.md` — POCKET-404 graph branch spec
- `docs/architecture/retrieval-layer.md` — hybrid search design
- `docs/architecture/ops-layer.md` — HITL gate
- [cocoindex PyPI](https://pypi.org/project/cocoindex/) v1.0.11
