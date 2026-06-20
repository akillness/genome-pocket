# Graph Target & Knowledge-Graph Ops — Design Spec

**Status:** Draft (spec phase) · **Owners:** Pocket core · **Tracks:** POCKET-201 (graph
target), POCKET-404 (LLM & entity-resolution ops), Roadmap Phase 2 "Graph Target
Integration".

This document specifies the local-first knowledge-graph (KG) target for Pocket and the
two ops that feed it — LLM-based extraction and entity resolution. It is the prerequisite
design that POCKET-404 was blocked on ("depends on a not-yet-existing graph target").

---

## 1. Why now, and what the 2026 literature says

The graph target was deferred until the vector/lexical core (POCKET-101…105, 401…405) was
stable. It is unblocked now. The design below is grounded in a survey of recent (2025–2026)
work on GraphRAG and LLM-driven KG construction, pulled live from arXiv:

| Theme | Representative 2026 work | What we take from it |
|-------|--------------------------|----------------------|
| **GraphRAG is the dominant pattern** for multi-hop / knowledge-intensive QA | *MemGraphRAG* (arXiv:2606.00610), *FlowRAG* (arXiv:2606.17856), *PathRouter* (arXiv:2606.16409) | Build a real entity–relation graph, not just entity-keyword seeds; support multi-hop traversal as a first-class retrieval mode. |
| **Local / consumer-hardware GraphRAG is viable** | *GraphRAG on Consumer Hardware: Benchmarking Local LLMs* (arXiv:2605.20815) | Keep extraction optional and runnable against a local LLM (Ollama); never require a cloud LLM or a server graph DB. Matches Pocket's local-first/privacy DNA. |
| **Strict-JSON output is the reliability bottleneck on small/local models** | *When Correct Isn't Usable: Improving Structured Output Reliability in Small LMs* (arXiv:2605.02363), *The Format Tax* (arXiv:2604.03616), *Schema Key Wording as an Instruction Channel* (arXiv:2604.14862) | 7-9B models hit ~85% task accuracy but **0%** valid-JSON under naive prompting — so demand an explicit JSON-only contract + one grounded exemplar + descriptive schema keys. Note the "format tax": any format demand costs reasoning accuracy, so keep the schema tight and the parser tolerant (a freeform→reformat two-pass is a deferred option). |
| **Schema-agnostic + uncertainty-guided KG construction** | *Schema-Agnostic KG Construction via Hybrid Ontology Discovery* (arXiv:2606.01208), *Helicase: Uncertainty-Guided… Multi-Agent LLMs* (arXiv:2605.26835) | Don't hard-code an ontology. Let extraction propose entity/relation types, carry a confidence score, and gate low-confidence facts behind the HITL approval (POCKET-302). |
| **Cost-effective LLM entity resolution via blocking + graph refinement** | *Adaptive Graph Refinement and Label Propagation with LLMs for Cost-Effective Entity Resolution* (arXiv:2605.25814), *Structure-Guided Entity Resolution* (arXiv:2605.23597) | Use cheap embedding-based **blocking** to generate candidate pairs, reserve the LLM for adjudicating only ambiguous pairs, then propagate labels across the candidate graph. Don't run O(n²) LLM comparisons. |
| **Trust / verifiability of LLM ER decisions** | *Can we trust LLM Self-Explanations for Entity Resolution?* (arXiv:2606.01210) | Persist the evidence + rationale for each merge; surface it through the existing end-to-end lineage so a human can audit a merge before it is committed. |

**Net design stance:** a *local-first GraphRAG* layer — SQLite-resident graph, optional
local-LLM extraction, embedding-blocked + LLM-adjudicated entity resolution, every node
and edge lineage-tagged and (when low-confidence) gated behind human approval.

---

## 2. Scope

**In scope (this spec):**
- A SQLite-resident graph target (`entities`, `relations`) reusing the existing
  lineage/memo/sweep machinery of `pocketindex/connectors/sqlite.py`.
- An optional LLM extraction op (`pocketindex/ops/extract.py`, the `ops.litellm` parity
  called for in POCKET-404, re-homed onto local engines — airLLM/Ollama, not a hosted proxy).
- An entity-resolution op (`pocketindex/ops/entity_resolution.py`, the
  `ops.entity_resolution` parity).
- Pipeline wiring (`pocket/pipeline.py`) and retrieval/CLI/MCP integration points.

**Out of scope (follow-ups):** SurrealDB/Neo4j backends (the system-overview diagram lists
SurrealDB; we treat it as a *future* alternate connector, not a v1 dependency); full graph
visualization UI (POCKET-301); the evaluation framework (POCKET-303), though we define the
metrics it will consume.

---

## 3. Data model

Two new lineage-aware tables, materialized by the same connector contract as `embeddings`
(stable single-column PK, `_pocket_lineage_*` / `_pocket_memo_*` companions, sweep on
source deletion). Both are **derived from chunks**, so a chunk's source file remains the
ultimate provenance anchor.

### 3.1 `entities`

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | Stable id from canonical name + type (via `IdGenerator`), so re-extraction is idempotent. |
| `name` | TEXT | Canonical surface form (post-resolution). |
| `type` | TEXT | Proposed type (schema-agnostic; e.g. `Person`, `Concept`, `Tool`). |
| `aliases` | TEXT (JSON) | Merged surface forms from entity resolution. |
| `embedding` | BLOB | sentence-transformers vector of `name` (reuse `EMBEDDER`) — powers blocking + vector lookup. |
| `summary` | TEXT | Optional one-line LLM/aggregated description. |
| `confidence` | REAL | Min/mean extraction confidence; drives the HITL gate. |
| `source_file` | TEXT | First/primary chunk's source file (lineage anchor). |
| `source_chunk_ids` | TEXT (JSON) | All chunk ids that mention this entity. |

FTS mirror on `name`+`summary` (reuse `fts_text_column`) so entities are lexically
searchable like chunks.

### 3.2 `relations`

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | Stable id from `(subject_id, predicate, object_id)`. |
| `subject_id` | INTEGER | FK → `entities.id`. |
| `predicate` | TEXT | Relation type (schema-agnostic, lower-snake, e.g. `depends_on`). |
| `object_id` | INTEGER | FK → `entities.id`. |
| `evidence` | TEXT | Verbatim source span supporting the edge (verifiability). |
| `confidence` | REAL | Extraction confidence. |
| `source_file` | TEXT | Lineage anchor. |
| `source_chunk_id` | INTEGER | Chunk the edge was extracted from. |

No foreign-key constraints enforced at the SQLite level (the engine reconciles via sweep);
referential integrity is maintained by always declaring an edge's endpoints in the same
run.

### 3.3 Lineage & incrementality

Because nodes/edges are attributed to their originating chunk's `source_key`, the existing
`begin_source`/`end_source`/`sweep` flow already gives us: re-extract on file change, drop
stale facts when a file's content changes, and remove a file's whole subgraph on deletion —
**for free**, with no new reconciliation code. `pocket drop <file>` (POCKET-405) extends to
the new tables by adding them to `_COMPANION_TABLES` handling in `pocket/admin.py`.

---

## 4. Ops

### 4.1 LLM extraction op — `pocketindex/ops/extract.py`

Parity target: upstream `cocoindex.ops.litellm`, but **re-homed onto a local engine**.
Where upstream proxies a *hosted* API via LiteLLM, Pocket's privacy/local-first DNA makes a
hosted proxy the wrong default. We ship **airLLM** (`lyogavin/airllm`) as the local heavy-model
backend instead of LiteLLM: airLLM layer-sharded inference runs a 70B-class model on a single
4GB GPU (8GB for 405B) by streaming layers from disk, *without* quantization/distillation
accuracy loss — so a user can run a genuinely capable extractor entirely on their own machine.
A dependency-light op turns a chunk into `(entities, relations)`:

- **Provider abstraction:** an `ExtractionModel` protocol (`extract(text) -> Extraction`)
  with three built-in backends —
  `DeterministicExtractor` (default, **no LLM, no network, no deps** — a noun-phrase /
  co-occurrence extractor that makes the whole graph pipeline offline-testable, the 404a
  slice), `OllamaExtractor` (local HTTP daemon), and `AirLLMExtractor` (local in-process
  airLLM for users who want a large model on modest hardware). Selected via
  `POCKET_LLM_PROVIDER` (`deterministic` | `ollama` | `airllm`) / `POCKET_LLM_MODEL`
  (env-overridable, mirroring existing `EMBEDDING_MODEL`). airLLM/torch stay an **optional
  extra** (`pip install genome-pocket[airllm]`); importing the backend is lazy so the base
  install carries zero torch/transformers weight.
- **Hardened, schema-agnostic prompt** (POCKET-404b, per arXiv:2606.01208): the model
  proposes entity types and predicates; we do not pin an ontology. The prompt is pinned by a
  `PROMPT_VERSION` constant and instructs the model to emit STRICT JSON only (no prose/fence),
  ground every fact in the TEXT, attach a verbatim `evidence` span, calibrate `confidence`
  (don't flat-1.0), and use `lower_snake_case` predicates — with one grounded few-shot exemplar
  to lift JSON validity on small local models (the structured-output reliability benchmark
  arXiv:2605.02363, where naive prompting yields 0% valid JSON; local-LLM GraphRAG itself is
  viable per arXiv:2605.20815). Output is validated against a small dataclass schema;
  malformed output → drop the chunk's extraction, log, continue (same degrade-don't-crash posture
  as the FTS5 fallback). The deterministic backend emits the same dataclass shape so downstream
  code is backend-agnostic.
- **Confidence + evidence** (per arXiv:2606.01210 / 2605.26835): every entity and relation
  carries a `confidence ∈ [0,1]` and a verbatim `evidence` span. These feed the HITL gate
  and lineage.
- **Memoized** (POCKET-404b): extraction is the expensive step, so the LLM backends are wrapped
  in a `MemoizingExtractor` keyed on `sha256(prompt_version, model_id, chunk_text)`. A
  `SqliteExtractionStore` (table `_pocket_extract_memo` in the Pocket DB) persists the cache
  across runs, so unchanged chunks under an unchanged prompt are never re-sent to the model;
  bumping `PROMPT_VERSION` transparently invalidates every cached extraction. The content-hash
  key means the cache can never return stale data. The deterministic backend is pure/cheap and
  stays unwrapped, so default runs gain no new table.
- **Disabled by default:** `pocket update --graph` (or `POCKET_GRAPH=1`) opts in. With the
  flag off, the pipeline is exactly today's vector/lexical pipeline — zero new cost or
  dependency for existing users.


### 4.2 Entity-resolution op — `pocketindex/ops/entity_resolution.py`

Parity target: upstream `cocoindex.ops.entity_resolution` (faiss-backed dedup). We use the
already-vendored **sqlite-vec** instead of faiss (no new dependency), following the
cost-effective blocking → adjudication → propagation pipeline from arXiv:2605.25814:

1. **Blocking:** for each freshly extracted entity, vector-search existing `entities` for
   the top-K nearest by name embedding (cosine via sqlite-vec). Only these candidate pairs
   proceed — avoids O(n²).
2. **Cheap filters:** exact/normalized name match and type compatibility resolve the easy
   pairs with no LLM call.
3. **LLM adjudication (optional):** ambiguous pairs above a similarity threshold are sent to
   the extraction model for a yes/no merge decision **only when** `--graph` LLM mode is on;
   otherwise fall back to a conservative similarity threshold. Persist the decision +
   rationale (verifiability).
4. **Label propagation:** transitively merge within a candidate cluster (a–b, b–c ⇒ a–c),
   then pick a canonical name (longest/highest-confidence surface form) and fold the rest
   into `aliases`.

Resolution runs as a post-pass after extraction within a run, so edges can be rewired to
canonical entity ids before `end_source` commits the subgraph.

---

## 5. Pipeline wiring (`pocket/pipeline.py`)


process_file ─(memo)→ chunks ─→ process_chunk ──→ embeddings  (today)
                                      └─(if --graph)→ extract_graph ──→ entities/relations
                                                                    └→ resolve_entities (post-pass)


- Add `EntityNode` / `RelationEdge` dataclasses mirroring `ChunkEmbedding`.
- Mount two more targets via `sqlite.mount_table_target("entities" | "relations", …)`.
- A new `@pix.fn` `extract_graph(chunk, …)` calls the extraction op then declares nodes/edges.
- Guard the whole graph branch behind the `--graph` flag so default runs are unchanged.

---

## 6. Retrieval integration

GraphRAG (per arXiv:2606.00610 / 2606.17856) adds a **third retriever** alongside vector
and lexical, fused by the existing RRF in `pocket/retrieval.py`:

- **Entity anchoring:** embed the query, vector-search `entities` for seed nodes.
- **Multi-hop expansion:** 1–2 hop traversal over `relations` from the seeds, collecting the
  `source_chunk_ids` of touched nodes/edges.
- **Fusion:** feed those chunks as a third ranked list into `_fuse()` (extend RRF to N
  lists). Graph hits keep full lineage (source file + offsets + supporting edge evidence),
  preserving the citation guarantee.
- New `mode="graph"` and `mode="hybrid"` (now vector+lexical+graph). `mode` stays backward
  compatible: graph only participates when the `entities` table exists.

CLI/MCP/REST: `pocket search --mode graph`; a new `pocket graph <entity>` to print a node's
neighborhood; MCP tool `traverse_graph`. (Detailed CLI surface deferred to the
implementation ticket.)

---

## 7. Human-in-the-loop & ops

- **Approval gate (POCKET-302) — *delivered (graph slice)*:** entities/relations below a
  `POCKET_GRAPH_MIN_CONFIDENCE` threshold (or with a staged endpoint) are written with
  `status="pending"`, not committed, and stay out of every graph read until
  `pocket graph review` approves them (`--approve`/`--reject <id>`, `--approve-all`/
  `--reject-all`) — matching the ops-layer.md HITL design and the uncertainty-guided
  stance from arXiv:2605.26835. An interactive in-update prompt now also ships:
  `pocket update --graph --review` (POCKET-301 slice) walks the operator through the
  staged facts inline (bulk approve-all/reject-all/each/skip, with a per-fact loop in
  *each* mode) over the same `admin` review API.
- **Lineage:** every node/edge stores its source file, chunk id, evidence span, and (for
  merges) the resolution rationale — so the ops-layer "retrieval lineage" block extends to
  graph facts.
- **Stats:** extraction/resolution counts flow through the existing `UpdateStats` plumbing
  (POCKET-401), so `pocket update --graph` reports entities/edges added/merged/dropped.

---

## 8. Evaluation hooks (feeds POCKET-303)

Define, but don't yet build, the metrics the eval suite will track: extraction
precision/recall on a hand-labeled note subset, entity-resolution pairwise F1, and
multi-hop QA answer accuracy vs. the vector-only baseline (the comparison framing from the
consumer-hardware GraphRAG benchmark, arXiv:2605.20815).

---

## 9. Risks & decisions

| Decision | Rationale |
|----------|-----------|
| SQLite graph, not SurrealDB, for v1 | No new runtime/dependency; reuses lineage/memo/sweep verbatim; honors local-first DNA. SurrealDB/Neo4j become optional alternate connectors later. |
| Local engine (Ollama / airLLM) default, no hosted proxy | Privacy DNA + 2026 evidence that local GraphRAG is viable (arXiv:2605.20815). airLLM replaces LiteLLM so even a 70B-class extractor runs locally on a 4GB GPU — no data leaves the machine. |
| Graph branch opt-in (`--graph`) | Extraction is the one genuinely expensive/optional step; existing users pay nothing. |
| Blocking + selective LLM ER, not all-pairs | Cost-effective ER evidence (arXiv:2605.25814); reuses sqlite-vec, no faiss. |
| Confidence + evidence on every fact | Verifiability concerns (arXiv:2606.01210) and HITL gating. |

---

## 10. Proposed ticket split

POCKET-404 is too large as one item. Recommend splitting:

1. **POCKET-404a — Graph target schema:** `entities`/`relations` tables + dataclasses +
   `pocket/admin.py` + `retrieval.list_sources` extensions; no LLM yet (extraction stubbed
   by a deterministic noun-phrase extractor so the plumbing is testable offline).
2. **POCKET-404b — LLM extraction op:** `ops/extract.py` with Ollama/airLLM backends,
   memoization, schema-agnostic JSON + confidence/evidence.
3. **POCKET-404c — Entity-resolution op:** `ops/entity_resolution.py` (blocking →
   adjudication → propagation).
4. **POCKET-404d — GraphRAG retrieval:** N-list RRF fusion, `mode="graph"`, CLI/MCP surface. ✅
5. **POCKET-302 (graph slice) — HITL approval gate** for low-confidence facts (staging +
   `pocket graph review`). ✅

404a is self-contained and offline-testable (no LLM, no network) — the right first slice,
exactly as POCKET-405 was chosen for being low-risk and self-contained.
