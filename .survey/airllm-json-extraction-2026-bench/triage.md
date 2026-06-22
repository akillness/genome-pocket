# Triage — airLLM JSON extraction prompt hardening, 2026 benchmark basis

**Bounded research question:** Which 2026 benchmarks/papers should the
local-LLM (airLLM / Ollama) strict-JSON knowledge-graph extraction prompt in
`pocketindex/ops/extract.py` (POCKET-404b) cite, and do they confirm or
challenge the current prompt design (JSON-only, grounded, evidence span,
calibrated confidence, single few-shot exemplar)?

**Problem.** The 404b prompt was hardened against the 2026 literature, but the
specific "lifts JSON validity for smaller models" reliability claim was only
backed by a *GraphRAG-on-consumer-hardware* paper (arXiv:2605.20815), which is
about local-LLM viability for GraphRAG, not specifically about JSON/structured
output reliability. The claim needed a benchmark that measures structured-output
reliability for small/local models directly.

**Audience.** genome-pocket maintainers wiring the graph-extraction op; the HITL
calibration work (POCKET-302); future maintainers auditing why the prompt is
shaped the way it is.

**Why now.** Current date 2026-06-19. A cluster of 2026 papers (Apr–Jun) now
benchmark exactly this: strict-JSON output reliability for 7–9B / open-weight
models, and the accuracy cost of demanding formatted output. The prompt should
cite the *on-target* benchmark rather than a general GraphRAG one, and should
record the one finding that pushes back on the single-pass strict-JSON design
("The Format Tax").

**Scope guard.** Research only — no model retraining, no decoder/grammar
constraint work. Output is citation reconciliation in code + docs plus this
reusable survey package.

**Evidence source.** arXiv API (`export.arxiv.org/api/query`), live, verified
each ID by fetching its real title + abstract. No GitHub/web-search lane tool
was available in this runtime; arXiv is the relevant platform for the
benchmark-discovery question, so it serves as the primary lane.
