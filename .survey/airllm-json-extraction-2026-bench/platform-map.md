# Platform map — where the structured-output reliability work lives (2026)

The "platform" axis here is the *layer at which JSON reliability is enforced*,
since that determines what genome-pocket can adopt at the prompt level vs what
needs engine support.

| Layer | Representative 2026 work | Adoptable in Pocket now? | Why |
|---|---|---|---|
| **Prompt / in-context** (system contract, exemplar, schema-key wording) | 2605.02363, 2604.03616, 2604.14862 | **Yes — adopted (404b)** | Pure prompt change; no runtime dependency. This is where our hardening lives. |
| **Two-pass (reason → reformat)** | 2604.03616 (The Format Tax principle) | Possible, deferred | Would halve the format tax but doubles local-LLM calls; revisit if extraction quality on small models is poor. |
| **Constrained / grammar decoding** | 2603.03305, 2606.01926, 2604.14862 | Engine-dependent | Ollama supports `format: json` / grammars; airLLM path does not expose a grammar hook today. Out of scope for a prompt pass. |
| **Serialization format choice (JSON vs TOON)** | 2603.03306 | No (stay JSON) | TOON saves tokens but adds a custom parser + one-shot teaching overhead; JSON keeps `parse_extraction_json` simple. Reconsider only under VRAM/token pressure. |

## Engine notes
- **airLLM** (default, 70B on 4GB GPU): prompt-level control only — makes the
  strict-JSON contract + grounded exemplar the *primary* reliability lever, which
  is exactly what 2605.02363 measures. No grammar/constrained-decoding hook.
- **Ollama** (alternative): could additionally pass `format=json`; not wired,
  tracked as a possible future enhancement, not part of 404b.
- **Deterministic backend**: no LLM, unaffected; stays unwrapped (no memo, no
  format tax).

## Cross-platform takeaway
At the only layer Pocket controls today (the prompt), the 2026 evidence is
consistent: an explicit JSON-only contract plus a single grounded exemplar is
the highest-leverage, lowest-cost reliability lever for small local models —
while acknowledging (Format Tax) that any format demand carries an accuracy cost
that a future two-pass design could reduce.
