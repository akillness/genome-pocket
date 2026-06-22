"""Automated retrieval evaluation (POCKET-303).

This module is the regression harness for Pocket's retrieval pipeline. It lets a
developer answer one question reliably: *did changing the chunk size, embedding
model, fusion weights, or graph extraction make retrieval worse?*

It does that with three pieces, all read-only over the same target the rest of
Pocket reads:

  * **Metrics** — pure functions computing standard information-retrieval scores
    (hit@k, recall@k, precision@k, reciprocal rank, average precision) from a
    ranked list of retrieved file paths against a set of relevant ones.
  * **Synthetic query/context pairs** — :func:`synthesize_cases` mines the
    *existing index* for the most distinctive tokens of each source file and
    turns them into a self-labeled query whose only correct answer is that file.
    No hand-curated gold set is needed, and a regression in chunking/indexing
    shows up immediately as a dropped hit. Hand-written cases can also be loaded
    from JSON via :func:`load_cases`.
  * **Baseline comparison** — :func:`evaluate` aggregates per-case scores into
    :class:`EvalMetrics`; :func:`compare_to_baseline` flags any metric that fell
    below a saved baseline (beyond a tolerance), so ``pocket eval`` can fail CI
    on a retrieval regression.

Everything calls :func:`pocket.retrieval.search`, so the evaluation exercises the
exact code path real queries take — it can never drift from production retrieval.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import sqlite3
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


import pocket.config as config
from pocket import retrieval

# Higher-is-better aggregate metrics, in report order. compare_to_baseline only
# flags a regression when one of these falls below the baseline.
METRIC_NAMES = (
    "hit_rate",
    "mrr",
    "precision_at_k",
    "recall_at_k",
    "mean_average_precision",
)

# Tokens too generic to make a distinctive synthetic query. Kept tiny and
# domain-neutral on purpose; distinctiveness is enforced by document frequency,
# this list just removes obvious stopwords that survive a small corpus.
_STOPWORDS = frozenset(
    """
    the a an and or but if then else for to of in on at by with from as is are
    was were be been being this that these those it its it's into over under
    not no yes can will would should could may might do does did has have had
    you your we our they their he she his her them us me my i so such than too
    very just about also more most some any all each every which who whom whose
    """.split()
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


@dataclass
class EvalCase:
    """One query and the set of source files that should answer it.

    ``relevant_files`` are matched against retrieved ``file_path`` values
    leniently (exact, path-suffix, or basename) so a gold set written with
    relative paths still matches an index that stores absolute paths.
    """

    query: str
    relevant_files: List[str]
    mode: str = "hybrid"
    note: str = ""


@dataclass
class CaseResult:
    """Per-case retrieval outcome and its scores."""

    query: str
    mode: str
    relevant_files: List[str]
    retrieved_files: List[str]
    hit: bool
    reciprocal_rank: float
    precision_at_k: float
    recall_at_k: float
    average_precision: float
    note: str = ""


@dataclass
class EvalMetrics:
    """Aggregate scores across an evaluation run (means over cases)."""

    n_cases: int
    k: int
    hit_rate: float
    mrr: float
    precision_at_k: float
    recall_at_k: float
    mean_average_precision: float
    cases: List[CaseResult] = field(default_factory=list)

    def to_dict(self, include_cases: bool = True) -> Dict:
        d = asdict(self)
        if not include_cases:
            d.pop("cases")
        return d


# --- Metric primitives (pure, no I/O) --------------------------------------


def _matches(retrieved: str, relevant: str) -> bool:
    """Whether a retrieved path satisfies a (possibly relative) relevant path."""
    if retrieved == relevant:
        return True
    r = retrieved.replace("\\", "/")
    rel = relevant.replace("\\", "/")
    if r.endswith("/" + rel) or rel.endswith("/" + r):
        return True
    return os.path.basename(r) == os.path.basename(rel) and bool(os.path.basename(r))


def _relevance_flags(retrieved: List[str], relevant: List[str]) -> List[bool]:
    """For each retrieved path, whether it matches any relevant path."""
    return [any(_matches(got, rel) for rel in relevant) for got in retrieved]


def reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """1 / (rank of the first relevant hit), or 0.0 if none is retrieved."""
    for idx, is_rel in enumerate(_relevance_flags(retrieved, relevant), start=1):
        if is_rel:
            return 1.0 / idx
    return 0.0


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Fraction of the top-k retrieved paths that are relevant."""
    if k <= 0:
        return 0.0
    return sum(_relevance_flags(retrieved[:k], relevant)) / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Fraction of the relevant paths that appear in the top-k retrieved."""
    if not relevant:
        return 0.0
    found = sum(any(_matches(got, rel) for got in retrieved[:k]) for rel in relevant)
    return found / len(relevant)


def average_precision(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Average precision over the top-k ranked results.

    Precision is averaged at each rank where a *new* relevant file appears,
    normalized by the number of relevant files (capped at k). This rewards
    ranking relevant files higher, not just retrieving them.
    """
    if not relevant or k <= 0:
        return 0.0
    seen: set = set()
    hits = 0
    summed = 0.0
    for idx, got in enumerate(retrieved[:k], start=1):
        matched = next((rel for rel in relevant if _matches(got, rel)), None)
        if matched is not None and matched not in seen:
            seen.add(matched)
            hits += 1
            summed += hits / idx
    denom = min(len(relevant), k)
    return summed / denom if denom else 0.0


# --- Synthetic query/context pairs -----------------------------------------


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _read_corpus(db_path: Path) -> Dict[str, str]:
    """Return {file_path: concatenated chunk text} for the whole index."""
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT file_path, text FROM embeddings ORDER BY file_path, start_offset"
        ).fetchall()
    finally:
        conn.close()
    corpus: Dict[str, str] = {}
    for file_path, text in rows:
        corpus[file_path] = corpus.get(file_path, "") + " " + (text or "")
    return corpus


def synthesize_cases(
    db_path: Optional[Path] = None,
    mode: str = "lexical",
    per_file: int = 1,
    n_terms: int = 4,
    min_token_len: int = 4,
) -> List[EvalCase]:
    """Mine the index for self-labeled query/context pairs.

    For every indexed source file, picks the tokens that are most *distinctive*
    to it — appearing in the fewest other files, ranked by rarity then length —
    and joins ``n_terms`` of them into a query whose single correct answer is
    that file. Because the labels come from the corpus itself, this needs no
    hand-curated gold set and turns any indexing/chunking regression into a
    dropped hit.

    ``mode`` defaults to ``"lexical"``: distinctive-token queries are a direct,
    deterministic probe of the BM25 index regardless of which embedding model is
    installed. Pass ``"hybrid"``/``"vector"`` to evaluate the semantic path
    against the real model on your machine.

    Returns an empty list when the index is missing or empty.
    """
    db_path = Path(db_path or config.POCKET_SQLITE_DB)
    if not db_path.exists():
        return []
    corpus = _read_corpus(db_path)
    if not corpus:
        return []

    # Document frequency: in how many files does each token appear?
    doc_freq: Dict[str, int] = {}
    file_tokens: Dict[str, List[str]] = {}
    for fp, text in corpus.items():
        toks = [
            t
            for t in _tokenize(text)
            if len(t) >= min_token_len and t not in _STOPWORDS
        ]
        file_tokens[fp] = toks
        for tok in set(toks):
            doc_freq[tok] = doc_freq.get(tok, 0) + 1

    cases: List[EvalCase] = []
    for fp, toks in file_tokens.items():
        if not toks:
            continue
        # Unique tokens for this file, most distinctive first: lowest corpus
        # document frequency, then longer (more specific) tokens.
        uniq = sorted(
            set(toks),
            key=lambda t: (doc_freq.get(t, 0), -len(t), t),
        )
        # Build up to `per_file` queries from disjoint slices of distinctive
        # tokens so multiple cases per file don't all repeat the same term.
        made = 0
        for start in range(0, len(uniq), max(n_terms, 1)):
            if made >= per_file:
                break
            terms = uniq[start : start + n_terms]
            if not terms:
                break
            cases.append(
                EvalCase(
                    query=" ".join(terms),
                    relevant_files=[fp],
                    mode=mode,
                    note=f"synthetic: distinctive tokens of {os.path.basename(fp)}",
                )
            )
            made += 1
    return cases


# --- Loading hand-written cases --------------------------------------------


def load_cases(path: Path) -> List[EvalCase]:
    """Load gold query/context cases from a JSON file.

    Accepts either a top-level list of case objects or ``{"cases": [...]}``.
    Each case needs ``query`` and ``relevant_files``; ``mode`` and ``note`` are
    optional. Raises ``ValueError`` on a malformed file so failures are loud.
    """
    raw = json.loads(Path(path).read_text())
    items = raw.get("cases", []) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError("eval cases file must be a list or {'cases': [...]}")
    cases: List[EvalCase] = []
    for i, obj in enumerate(items):
        if not isinstance(obj, dict) or "query" not in obj:
            raise ValueError(f"case #{i} missing required 'query' field")
        rel = obj.get("relevant_files")
        if not isinstance(rel, list) or not rel:
            raise ValueError(f"case #{i} needs a non-empty 'relevant_files' list")
        cases.append(
            EvalCase(
                query=str(obj["query"]),
                relevant_files=[str(x) for x in rel],
                mode=str(obj.get("mode", "hybrid")),
                note=str(obj.get("note", "")),
            )
        )
    return cases


# --- Runner & baseline comparison ------------------------------------------


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(
    cases: List[EvalCase],
    db_path: Optional[Path] = None,
    k: int = 5,
    model_name: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    use_mmr: Optional[bool] = None,
    mmr_lambda: Optional[float] = None,
) -> EvalMetrics:
    """Run every case through real retrieval and aggregate the scores.

    Each case is searched with its own ``mode`` (via :func:`retrieval.search`),
    top-``k`` results are scored against its relevant files, and the per-case
    scores are averaged into an :class:`EvalMetrics`. The individual
    :class:`CaseResult` rows are retained on ``.cases`` for drill-down.

    ``weights`` is passed straight to :func:`retrieval.search` so the harness
    can score a candidate set of fusion weights (POCKET-502); ``None`` uses the
    configured defaults.

    ``use_mmr``/``mmr_lambda`` are likewise forwarded so the harness can
    measure the MMR diversity trade-off (POCKET-501) — i.e. score the same gold
    cases with and without Maximal Marginal Relevance re-ranking and read the
    Recall@k/MAP delta — instead of only toggling it through global config.
    ``None`` (the default) follows ``config.POCKET_MMR`` / ``POCKET_MMR_LAMBDA``.
    """
    db_path = Path(db_path or config.POCKET_SQLITE_DB)
    results: List[CaseResult] = []
    for case in cases:
        hits = retrieval.search(
            case.query,
            limit=k,
            db_path=db_path,
            model_name=model_name,
            mode=case.mode,
            weights=weights,
            use_mmr=use_mmr,
            mmr_lambda=mmr_lambda,
        )


        retrieved_files = [h.file_path for h in hits]
        rr = reciprocal_rank(retrieved_files, case.relevant_files)
        results.append(
            CaseResult(
                query=case.query,
                mode=case.mode,
                relevant_files=list(case.relevant_files),
                retrieved_files=retrieved_files,
                hit=rr > 0.0,
                reciprocal_rank=rr,
                precision_at_k=precision_at_k(retrieved_files, case.relevant_files, k),
                recall_at_k=recall_at_k(retrieved_files, case.relevant_files, k),
                average_precision=average_precision(
                    retrieved_files, case.relevant_files, k
                ),
                note=case.note,
            )
        )

    return EvalMetrics(
        n_cases=len(results),
        k=k,
        hit_rate=_mean([1.0 if r.hit else 0.0 for r in results]),
        mrr=_mean([r.reciprocal_rank for r in results]),
        precision_at_k=_mean([r.precision_at_k for r in results]),
        recall_at_k=_mean([r.recall_at_k for r in results]),
        mean_average_precision=_mean([r.average_precision for r in results]),
        cases=results,
    )


@dataclass
class Regression:
    """One aggregate metric that fell below the baseline beyond tolerance."""

    metric: str
    baseline: float
    current: float

    @property
    def delta(self) -> float:
        return self.current - self.baseline


def save_baseline(path: Path, metrics: EvalMetrics) -> None:
    """Persist a run's aggregate metrics as a regression baseline (JSON)."""
    Path(path).write_text(
        json.dumps(metrics.to_dict(include_cases=False), indent=2, sort_keys=True)
    )


def load_baseline(path: Path) -> Dict:
    """Load a previously saved baseline metrics dict."""
    return json.loads(Path(path).read_text())


def compare_to_baseline(
    current: EvalMetrics,
    baseline: Dict,
    tolerance: float = 0.0,
) -> List[Regression]:
    """Return the metrics that regressed versus a baseline.

    A metric regresses when ``current < baseline - tolerance``. ``tolerance``
    (>= 0) absorbs noise, e.g. ``0.01`` ignores drops of one point. Metrics
    absent from the baseline are skipped (newly added metrics never fail).
    """
    out: List[Regression] = []
    cur = current.to_dict(include_cases=False)
    for name in METRIC_NAMES:
        if name not in baseline:
            continue
        base_val = float(baseline[name])
        cur_val = float(cur[name])
        if cur_val < base_val - tolerance:
            out.append(Regression(metric=name, baseline=base_val, current=cur_val))
    return out


def format_report(metrics: EvalMetrics, show_cases: bool = False) -> str:
    """Render an evaluation run as human-readable text for the CLI."""
    lines = [
        f"Retrieval evaluation — {metrics.n_cases} case(s), k={metrics.k}",
        "-" * 48,
        f"  Hit@{metrics.k}:        {metrics.hit_rate:.4f}",
        f"  MRR:            {metrics.mrr:.4f}",
        f"  Precision@{metrics.k}:  {metrics.precision_at_k:.4f}",
        f"  Recall@{metrics.k}:     {metrics.recall_at_k:.4f}",
        f"  MAP@{metrics.k}:        {metrics.mean_average_precision:.4f}",
    ]
    if show_cases:
        lines.append("-" * 48)
        for r in metrics.cases:
            mark = "OK " if r.hit else "MISS"
            top = r.retrieved_files[0] if r.retrieved_files else "(none)"
            lines.append(
                f"  [{mark}] rr={r.reciprocal_rank:.3f} "
                f"q={r.query!r} -> {os.path.basename(top)}"
            )
    return "\n".join(lines)


# --- Weighted-RRF tuning (POCKET-502) --------------------------------------

# Default candidate weights to grid-search per strategy. 0.0 lets the optimizer
# drop a strategy entirely; 1.0 (always probed) reproduces plain RRF, so the
# baseline is always inside the grid and the tuner can never pick something
# worse than the equal-weight default.
DEFAULT_WEIGHT_GRID = (0.0, 0.5, 1.0, 2.0)

_STRATEGY_ORDER = ("vector", "lexical", "graph")
_EQUAL_WEIGHTS = {s: 1.0 for s in _STRATEGY_ORDER}


@dataclass
class WeightTrial:
    """One evaluated point in the weight grid: its weights and target score."""

    weights: Dict[str, float]
    score: float


@dataclass
class TuningResult:
    """Outcome of a weighted-RRF grid search (POCKET-502)."""

    metric: str
    baseline_weights: Dict[str, float]
    baseline_score: float
    best_weights: Dict[str, float]
    best_score: float
    tuned_strategies: List[str]
    trials: List[WeightTrial] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        """Whether the best point beats the equal-weight baseline."""
        return self.best_score > self.baseline_score

    @property
    def delta(self) -> float:
        return self.best_score - self.baseline_score

    def to_dict(self, include_trials: bool = False) -> Dict:
        d = asdict(self)
        d["improved"] = self.improved
        d["delta"] = self.delta
        if not include_trials:
            d.pop("trials")
        return d


def _deviation(weights: Dict[str, float]) -> float:
    """Total distance from the equal-weight baseline (tie-breaker, lower wins)."""
    return sum(abs(weights[s] - 1.0) for s in _STRATEGY_ORDER)


def tune_weights(
    cases: List[EvalCase],
    db_path: Optional[Path] = None,
    k: int = 5,
    model_name: Optional[str] = None,
    values=DEFAULT_WEIGHT_GRID,
    metric: str = "mean_average_precision",
) -> TuningResult:
    """Grid-search per-strategy RRF weights, returning the best (POCKET-502).

    Turns the eval harness from a guard into an optimizer: every weight
    combination is scored with the *same* real retrieval path :func:`evaluate`
    uses, and the combination maximising ``metric`` is returned alongside the
    equal-weight baseline so the caller can see the lift before persisting it.

    Only strategies actually exercised by the cases' modes are varied (others
    are pinned to 1.0), so a lexical-only suite searches a 1-D grid instead of a
    wasteful 3-D one. ``1.0`` is always probed, so the result is never worse than
    plain RRF. Ties are broken toward the baseline (smallest weight deviation),
    keeping the recommendation conservative.
    """
    if metric not in METRIC_NAMES:
        raise ValueError(
            f"metric must be one of {METRIC_NAMES}, got {metric!r}"
        )

    # Clean the grid: clamp to non-negative, dedupe (order-preserving), and
    # guarantee 1.0 is present so the baseline point exists.
    vals: List[float] = []
    for v in values:
        fv = max(float(v), 0.0)
        if fv not in vals:
            vals.append(fv)
    if 1.0 not in vals:
        vals.append(1.0)

    # Which strategies are worth varying for this case set?
    tuned = [
        s
        for s in _STRATEGY_ORDER
        if any(s in retrieval._MODE_STRATEGIES.get(c.mode, ()) for c in cases)
    ]

    baseline_score = getattr(
        evaluate(cases, db_path=db_path, k=k, model_name=model_name,
                 weights=dict(_EQUAL_WEIGHTS)),
        metric,
    )

    trials: List[WeightTrial] = []
    best_weights = dict(_EQUAL_WEIGHTS)
    best_score = baseline_score
    best_dev = 0.0
    for combo in itertools.product(vals, repeat=len(tuned)):
        weights = dict(_EQUAL_WEIGHTS)
        weights.update(dict(zip(tuned, combo)))
        if weights == _EQUAL_WEIGHTS:
            score = baseline_score  # already measured; don't pay for it twice
        else:
            score = getattr(
                evaluate(cases, db_path=db_path, k=k, model_name=model_name,
                         weights=weights),
                metric,
            )
        trials.append(WeightTrial(weights=dict(weights), score=score))
        dev = _deviation(weights)
        if score > best_score or (score == best_score and dev < best_dev):
            best_score, best_weights, best_dev = score, dict(weights), dev

    return TuningResult(
        metric=metric,
        baseline_weights=dict(_EQUAL_WEIGHTS),
        baseline_score=baseline_score,
        best_weights=best_weights,
        best_score=best_score,
        tuned_strategies=tuned,
        trials=trials,
    )


def save_weights(path: Path, weights: Dict[str, float]) -> None:
    """Persist tuned fusion weights as JSON for ``POCKET_RRF_WEIGHTS_FILE``."""
    out = {s: float(weights[s]) for s in _STRATEGY_ORDER if s in weights}
    Path(path).write_text(json.dumps(out, indent=2, sort_keys=True))


def load_weights(path: Path) -> Dict[str, float]:
    """Load fusion weights previously written by :func:`save_weights`."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("weights file must be a JSON object")
    return {s: float(data[s]) for s in _STRATEGY_ORDER if s in data}


# ---------------------------------------------------------------------------
# RAGAS-style LLM-as-judge evaluation (POCKET-601)
# ---------------------------------------------------------------------------

@dataclass
class JudgeMetrics:
    """Aggregate metrics from a retrieval evaluation run with an LLM judge.

    Wraps the standard :class:`EvalMetrics` and adds ``mean_context_relevance``:
    the average score (0–1) assigned by a local Ollama model when asked how
    relevant each retrieved chunk is to its query.  Higher is better.
    """

    base: EvalMetrics
    mean_context_relevance: float  # mean over all cases of per-chunk relevance
    n_judged: int                  # number of cases for which judge ran
    judge_model: str               # Ollama model used as judge

    def to_dict(self) -> Dict:
        d = self.base.to_dict(include_cases=False)
        d.update(
            {
                "mean_context_relevance": self.mean_context_relevance,
                "n_judged": self.n_judged,
                "judge_model": self.judge_model,
            }
        )
        return d


def _ollama_relevance_score(
    query: str,
    context: str,
    host: str,
    model: str,
) -> float:
    """Ask a local Ollama model how relevant *context* is to *query* (0–1).

    The score is used as a proxy for RAGAS Context Relevance — a fraction of
    retrieved sentences that are actually needed to answer the question.  A
    local judge avoids cloud dependencies while still providing a signal beyond
    keyword overlap.  Returns 0.5 (neutral) on any Ollama error so a failed
    daemon doesn't invalidate the whole eval run.
    """
    import urllib.error
    import urllib.request

    prompt = (
        f"Query: {query}\n\nPassage:\n{context[:800]}\n\n"
        "Rate how relevant this passage is to the query on a scale from "
        "0.0 (completely irrelevant) to 1.0 (perfectly relevant). "
        "Reply with a single decimal number only, nothing else."
    )
    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": False}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        raw = body.get("response", "").strip()
        m = re.search(r"-?\d+\.?\d*", raw)
        val = float(m.group()) if m else 0.5
        return min(max(val, 0.0), 1.0)
    except Exception:  # noqa: BLE001 — any Ollama failure → neutral score
        return 0.5


def evaluate_with_judge(
    cases: List["EvalCase"],
    k: int = 5,
    db_path: Optional[Path] = None,
    ollama_host: Optional[str] = None,
    ollama_model: Optional[str] = None,
) -> JudgeMetrics:
    """Evaluate retrieval with standard IR metrics **plus** an Ollama LLM judge.

    Runs :func:`evaluate` for the standard IR metrics (Hit@k, MRR, …) and then
    makes one additional Ollama call per retrieved chunk to score its relevance
    to the query (0–1).  The mean over all cases is reported as
    ``mean_context_relevance`` in the returned :class:`JudgeMetrics`.

    Requires a running Ollama daemon; if unavailable, judge scores default to
    0.5 (neutral) so the standard IR metrics are still reported correctly.

    Parameters
    ----------
    cases:
        Evaluation cases — same as :func:`evaluate`.
    k:
        Rank cutoff for both standard metrics and the judge pass.
    db_path:
        SQLite database path; defaults to ``config.POCKET_SQLITE_DB``.
    ollama_host:
        Ollama daemon base URL; defaults to ``config.POCKET_HYDE_OLLAMA_HOST``
        (both share the same OLLAMA_HOST env-var).
    ollama_model:
        Ollama model to use as judge; defaults to ``config.POCKET_HYDE_OLLAMA_MODEL``.
    """
    base = evaluate(cases, k=k, db_path=db_path)
    host = ollama_host or config.POCKET_HYDE_OLLAMA_HOST
    model = ollama_model or config.POCKET_HYDE_OLLAMA_MODEL

    relevance_scores: List[float] = []
    n_judged = 0

    for case in cases:
        hits = retrieval.search(case.query, limit=k, mode=case.mode, db_path=db_path)
        if not hits:
            continue
        chunk_scores = [
            _ollama_relevance_score(case.query, h.text, host, model)
            for h in hits
        ]
        relevance_scores.append(sum(chunk_scores) / len(chunk_scores))
        n_judged += 1

    mean_rel = (
        sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
    )
    return JudgeMetrics(
        base=base,
        mean_context_relevance=mean_rel,
        n_judged=n_judged,
        judge_model=model,
    )


def format_judge_report(metrics: JudgeMetrics) -> str:
    """Render a :class:`JudgeMetrics` result as human-readable text."""
    lines = [
        format_report(metrics.base),
        "",
        f"LLM-as-judge context relevance (model: {metrics.judge_model})",
        "-" * 48,
        f"  Mean context relevance: {metrics.mean_context_relevance:.4f}  "
        f"(over {metrics.n_judged} case(s))",
    ]
    return "\n".join(lines)