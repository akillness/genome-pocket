# Solutions — 2026 benchmarks for strict-JSON / structured extraction

All IDs verified live against the arXiv API (real title + abstract fetched on
2026-06-19). Sorted by relevance to the airLLM JSON-extraction prompt.

## Primary (cite these in extract.py)

| Paper | arXiv | What it benchmarks | Bearing on our prompt |
|---|---|---|---|
| **When Correct Isn't Usable: Improving Structured Output Reliability in Small Language Models** | 2605.02363 | 3 × 7–9B models, 5 prompting strategies, "output accuracy" = correct **and** valid JSON. Naive prompting → 0% valid output despite ~85% task accuracy. | The direct, on-target evidence for the JSON-only system-prompt contract + minimal-reference/exemplar prompting. Replaces the GraphRAG paper as the citation for "lifts JSON validity on small models". |
| **The Format Tax** | 2604.03616 | Accuracy degradation from JSON/XML/LaTeX/Markdown output demands across open-weight models; locates most of the cost at the *prompt*, not the decoder. | The key caveat to record: a single-pass strict-JSON prompt pays a format tax. Our mitigation = one tight schema + grounded exemplar + tolerant parser; a future option is freeform-then-reformat two-pass extraction. |
| **Schema Key Wording as an Instruction Channel in Structured Generation under Constrained Decoding** | 2604.14862 | Schema-key tokens act as an implicit instruction channel under constrained decoding. | Supports descriptive `lower_snake_case` predicate / field names carrying task signal — already in rule 4 of the prompt. |

## Secondary (token / format-choice context, optional)

| Paper | arXiv | Note |
|---|---|---|
| Token-Oriented Object Notation vs JSON: A Benchmark of Plain and Constrained Decoding Generation | 2603.03306 | TOON cuts tokens vs JSON; relevant if airLLM token/VRAM budget becomes the bottleneck. We stay on JSON for parser simplicity. |
| Draft-Conditioned Constrained Decoding for Structured Generation in LLMs | 2603.03305 | Engine-level decoding technique; out of scope (we are prompt-level, not decoder-level). |
| Mitigating Bias in Locally Constrained Decoding via Tractable Proposals | 2606.01926 | Constrained-decoding bias; out of scope for a prompt-only change. |

## Already cited in graph-target.md §1 (verified real, kept)

- 2606.01208 — Schema-Agnostic KG Construction via Hybrid Ontology Discovery (schema-agnostic typing).
- 2605.26835 — Helicase: Uncertainty-Guided… Multi-Agent LLMs (calibrated confidence).
- 2606.01210 — Can we trust LLM Self-Explanations for Entity Resolution? (evidence/verifiability).
- 2605.20815 — GraphRAG on Consumer Hardware: Benchmarking Local LLMs (local-LLM viability; **no longer** the sole basis for the JSON-validity claim).

## Decision

Add 2605.02363 (primary JSON-reliability basis) and 2604.03616 (format-tax
caveat) to the `_EXTRACTION_PROMPT` header and `graph-target.md §1/§4.1`; keep
2604.14862 as supporting evidence for descriptive schema keys. Do not adopt
decoder-level or TOON changes — out of scope for a prompt hardening pass.
