# genome-pocket — Product Strategy & Discovery (2026)

PM companion to the engineering backlog in
[`review-2026-improvements.md`](review-2026-improvements.md). That doc answers
*"what can we build next and where's the seam."* This one answers the prior
questions a PM has to settle first: **who is this for, what job are they hiring
it to do, which outcome are we moving, and how do we de-risk before building.**
Frameworks applied: Jobs-to-be-Done, Teresa Torres' Opportunity Solution Tree,
North Star + input metrics, Alberto Savoia's pretotyping, and a positioning
statement. Everything below is grounded in the shipped code (CLI/MCP/REST,
hybrid RRF retrieval, byte-exact lineage, local-first), not aspiration.

---

## 1. Positioning (Moore "for / who / unlike" form)

> **For** developers and knowledge workers who pair with AI assistants,
> **who** need their private notes and code searchable by those assistants
> **without** shipping the corpus to a cloud index,
> **genome-pocket** is a local-first knowledge runtime
> **that** serves one hybrid-retrieval path (vector + lexical + graph, RRF-fused)
> identically over CLI, MCP, and REST, with byte-exact lineage on every hit.
> **Unlike** cloud RAG SaaS (Glean, Mem) or notebook tools (Notion AI), the
> corpus never leaves the machine and every answer is traceable to the exact
> source bytes.

The strategic wedge is **MCP**: as coding agents become the primary consumer of
personal knowledge, "the local index your agent can query" is a sharper, less
crowded position than "another notes app with search."

---

## 2. Jobs-to-be-Done

Primary job (functional):
> **When** I'm working with an AI assistant on a problem that depends on my own
> notes/code, **I want** the assistant to retrieve the exact, relevant passages
> from my private corpus, **so I can** get grounded answers without manually
> hunting for context or leaking my data to a third party.

Supporting jobs:
- *(Emotional)* "I want to **trust** what the assistant tells me" → satisfied by
  byte-exact lineage; the unbuilt gap is making that trust *visible* at answer
  time (see Opportunity O2).
- *(Functional)* "I want to keep the index **fresh** without thinking about it"
  → satisfied by Δ-only incremental ETL + file watcher.
- *(Functional)* "I want to find things by **meaning or by exact token**, and
  by **how concepts connect**" → satisfied by hybrid + GraphRAG.

JTBD reframes the roadmap: P4/P5 (engine parity) serve the *freshness* job;
the retrieval-quality and answer-synthesis backlog serves the *grounded-answer*
and *trust* jobs — which is where differentiated user value concentrates.

---

## 3. Personas (two, kept deliberately narrow)

| | **Ada — the agent-augmented developer** (primary) | **Nora — the research note-taker** (secondary) |
|---|---|---|
| Context | Lives in an editor with an MCP-capable agent | Large markdown vault, occasional CLI use |
| Hires Pocket to | Feed the agent cited context from her own repo+notes | Find half-remembered notes by meaning |
| Key surface | **MCP** (`search_knowledge`), `pocket search --json` | CLI + Web UI |
| Success signal | Agent answers cite *her* files, offline | Finds the note in one query |
| Top unmet need | Sees raw chunks, not a synthesized cited answer | No "why did this match" affordance |

Designing for Ada first is the bet: she has the acute, frequent, underserved
job, and she's reachable through the MCP ecosystem.

---

## 4. North Star & input metrics

**North Star (proposed):** *Grounded retrievals per active week* — a query that
returns ≥1 hit the user (or their agent) actually uses. It captures value
delivered (not vanity index size) and is local-first measurable via the existing
`pocket eval` / trace plumbing without phoning home.

| Input metric | Why it moves the NSM | Where it's measured today |
|---|---|---|
| **Retrieval quality** (Hit@k / MRR / MAP) | Bad recall → unused results | ✅ `pocket eval` harness |
| **Index freshness lag** | Stale index → wrong/missing hits | Δ-only ETL + watcher (P4/P5 reduce lag) |
| **Time-to-first-value** | Onboarding friction kills adoption | ⚠️ not instrumented — see RAT-1 |
| **Trust/citation rate** | Cited answers get acted on | ⚠️ needs O2 (answer synthesis) |

Two of four inputs are already measurable; the gaps (TTFV, trust) are exactly
the riskiest assumptions below.

---

## 5. Opportunity Solution Tree (Teresa Torres)


OUTCOME: increase "grounded retrievals per active week"
│
├─ O1 · Results don't surface the right/diverse passages
│     ├─ POCKET-501 MMR diversity in _fuse           (backlog, measurable)
│     ├─ POCKET-502 weighted/tunable RRF             (backlog, measurable)
│     └─ POCKET-504 semantic query router            (backlog)
│
├─ O2 · Users get chunks, not a trustworthy answer   ← highest user-value gap
│     ├─ POCKET-506 answer synthesis with citations  (backlog) ★ PRD below
│     └─ POCKET-507 snippet highlighting             (backlog)
│
├─ O3 · The agent (Ada) can't consume Pocket cleanly
│     ├─ pocket search --json                        ✅ shipped this cycle
│     └─ richer MCP tool surface (graph, lineage)    (candidate)
│
├─ O4 · Index goes stale / bloats on edits
│     ├─ POCKET-P4 state-diff delta writes           (roadmap, "next")
│     └─ POCKET-P5 persistent memo                    (roadmap)
│
└─ O5 · New users can't tell if it's working (onboarding)
      ├─ "pocket doctor" health/first-run check      (candidate)
      └─ HITL queue visible in Web UI (POCKET-505)   (backlog)


The tree shows the engineering backlog is well-stocked under O1/O4 but **O2 and
O5 are thin** — and O2 is where Ada's emotional "I want to trust it" job lives.
That's the strategic recommendation: the next *product* bet is O2, even though
O1 is the easiest *engineering* win.

---

## 6. PRD (lightweight) — O2 / POCKET-506: Cited answer synthesis

**Problem.** Pocket returns ranked chunks; Ada's agent (or Nora at the CLI) must
read and stitch them. The product already carries byte-exact lineage end to end
but throws away its biggest trust advantage by stopping at "here are passages."

**Hypothesis.** If Pocket can optionally compose a short answer *with inline
citations to exact source offsets*, grounded-retrieval rate and trust both rise,
and Pocket becomes a verifiable RAG endpoint rather than just a retriever.

**Scope (v1, opt-in, honors local-first DNA):**
- New `pocket answer "<q>"` (+ `/answer` REST, `answer_knowledge` MCP tool) that
  runs the existing hybrid retrieval, then asks a **local** LLM
  (`POCKET_LLM_PROVIDER`, already wired for graph extraction) to synthesize an
  answer where every claim carries a `[file_path:start-end]` citation drawn from
  the hits' lineage.
- Deterministic default stays retrieval-only; synthesis is explicitly invoked,
  so the offline test suite and the "no heavy deps in base" rule are untouched.
- `--json` emits `{answer, citations[], hits[]}` so Ada's agent parses it.

**Non-goals (v1):** multi-turn chat, cloud models, answer caching.

**Acceptance criteria:**
1. Every sentence in the answer maps to ≥1 citation resolvable to real stored
   offsets (no fabricated spans) — testable against a fixture corpus with a
   stub/mock provider.
2. With no provider configured, the command degrades to ranked hits + a clear
   stderr hint (mirrors the `search --json` empty-index pattern).
3. `pocket eval` retrieval numbers are unchanged (synthesis sits *after* fusion).

**Effort/risk:** medium. New seam, but reuses retrieval + lineage + the existing
local-LLM provider abstraction. Main risk is citation faithfulness → RAT-2.

---

## 7. Riskiest Assumption Tests / pretotyping (Savoia — test before building)

| ID | Assumption | Cheapest test | Kill / go signal |
|---|---|---|---|
| **RAT-1** | Ada actually wants cited *answers*, not just better chunk ranking | "Mechanical-Turk" pretotype: hand-write cited answers over real hits for 10 queries, show to 3–5 target devs | <3/5 prefer cited answer over raw chunks → invest in O1 ranking instead |
| **RAT-2** | A *local* small model can cite faithfully enough to trust | Spike: run synthesis on a fixture corpus, hand-grade citation faithfulness on 20 answers before any product wiring | <80% faithful → gate behind a verifier or defer O2 |
| **RAT-3** | MCP is the high-leverage surface (the wedge bet) | Instrument which surface drives retrievals once we can; until then, count MCP tool installs vs CLI usage in interviews | MCP negligible → re-weight toward CLI/Web UI personas |
| **RAT-4** | Onboarding (O5) is a real drop-off, not assumed | Watch 3 first-runs end-to-end; time to first useful hit | If TTFV is already fast → drop O5, don't build "doctor" |

RAT-2 is the gate for the PRD above: **run the faithfulness spike before
committing engineering to POCKET-506.** This is the pretotyping discipline —
validate the riskiest belief with the cheapest experiment first.

---

## 8. Recommendation (sequencing)

1. **Now (1 loop):** ship **POCKET-501 (MMR)** — contained, on a tested seam,
   proven by `pocket eval`. Low risk, moves O1, keeps momentum. (Engineering's
   pick, and it's correct as the *immediate* step.)
2. **Next (validate, then build):** run **RAT-1 + RAT-2** for O2. If both pass,
   the **POCKET-506 cited-answer** PRD is the differentiated product bet that
   turns a good retriever into a trustworthy local RAG endpoint for Ada.
3. **Parallel/background:** **P4** keeps freshness honest (foundational, low
   user-visibility) — schedule it but don't let it block the O2 product bet.

The throughline: engineering optimizes the *retriever* (O1); the product
opportunity is the *trustworthy cited answer* (O2), and the discipline is to
pretotype it (RAT-1/2) before spending the build.
