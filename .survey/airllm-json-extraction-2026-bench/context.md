# Context — local-LLM structured-output extraction, 2026

**Workflow context.** Pocket runs knowledge-graph extraction locally (airLLM so
a 70B-class model fits a 4GB GPU, or Ollama), privacy-first, no cloud LLM. The
extractor must return a single strict-JSON object `{entities, relations}` with a
verbatim evidence span and a calibrated confidence per fact, so downstream
parsing (`parse_extraction_json`) and the HITL approval gate (POCKET-302) get
clean, auditable input.

**Affected users.** Anyone running `pocket update` with an LLM provider; small
local models are exactly the regime where "format-compliant JSON" is least
reliable.

**The core pain the literature documents.** Small/open-weight models routinely
produce *correct content in invalid JSON* — or pay an accuracy penalty just for
being told to emit JSON. Both failure modes hit the local-first design directly.

**Workarounds people use (from the papers):**
- System prompt + explicit "JSON only, no prose/fence" contract (vs naive
  prompting that collapses to 0% valid-output rate).
- One in-context grounded exemplar to teach the exact shape.
- Decouple reasoning from formatting: generate freeform, then reformat in a
  second pass (constrained decoding alone only fixes part of the gap).
- Constrained/grammar decoding (engine-level), schema-key wording as an implicit
  instruction channel, token-frugal serializations (TOON) for VRAM/token limits.

**Adjacent problems.** Confidence calibration for HITL gating; verbatim evidence
for merge auditability; schema-agnostic typing. These were already cited from
KG-construction / entity-resolution papers in `graph-target.md §1`; this survey
adds the missing *structured-output-reliability* axis.

**User voices (abstract excerpts, verbatim from arXiv):**
- *"NAIVE prompting (no system prompt) achieves up to 85% task accuracy on GSM8K
  but 0% output accuracy across all models and datasets."* — arXiv:2605.02363.
- *"Asking a large language model to respond in JSON should be a formatting
  choice, not a capability tax… format-requesting instructions alone cause most
  of the accuracy loss, before any decoder constraint is applied."* —
  arXiv:2604.03616 (The Format Tax).
- *"schema-key tokens also enter the autoregressive context and may guide
  generation."* — arXiv:2604.14862.
