"""Graded-corpus eval proof for fusion features (POCKET-501 / POCKET-502).

The unit tests for MMR (``TestMmrRerank``) and weighted RRF (``TestWeightedFusion``)
prove the *mechanics* with hand-injected vectors. They cannot prove a *quality
win*, because the session-wide ``MockEmbedder`` emits all-zero vectors, so a
real cosine signal never reaches the retrieval path end to end.

This module closes that gap. It installs a deterministic, offline
``HashingEmbedder`` (bag-of-words hashed into a fixed space, L2-normalised — no
network, no model download, fully reproducible), indexes the shipped graded
corpus under ``eval/corpus/``, and measures the two fusion features through the
real :func:`pocket.evaluation.evaluate` / :func:`tune_weights` harness on the
hand-labelled ``eval/gold.json`` cases:

  * **MMR (POCKET-501)** recovers a second, distinct relevant file that
    near-duplicate chunks bury under plain fusion → Recall@k rises.
  * **Weighted RRF (POCKET-502)** down-weights a misleading vector strategy and
    lifts MAP on a hybrid query where the keyword index is more trustworthy.

Because the embedder is deterministic the measured deltas are stable, so these
are real regression tests over the fusion quality, not illustrative scripts.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import pathlib
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import pocketindex as pix

_EVAL_DIR = pathlib.Path(__file__).resolve().parent.parent / "eval"
_CORPUS_DIR = _EVAL_DIR / "corpus"
_GOLD = _EVAL_DIR / "gold.json"

_EMBED_DIM = 256
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


class HashingEmbedder:
    """Deterministic offline embedder: L2-normalised hashed bag of words.

    Tokens are hashed into ``_EMBED_DIM`` buckets and counted, then the vector is
    normalised. Two documents that share vocabulary get a high cosine similarity
    and disjoint documents get ~0 — a *real* (if crude) semantic signal that,
    unlike ``MockEmbedder``'s zero vector, lets MMR diversity and vector-vs-lexical
    disagreement actually manifest. Same interface the ingestion and query sides
    expect (async ``embed`` + sync ``encode``).
    """

    def __init__(self, model_name: str = "hash", **kwargs):
        self.model_name = model_name
        self.embedding_dim = _EMBED_DIM

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(_EMBED_DIM, dtype=np.float32)
        for tok in _TOKEN_RE.findall(text.lower()):
            bucket = int(hashlib.sha1(tok.encode()).hexdigest(), 16) % _EMBED_DIM
            v[bucket] += 1.0
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        return v

    async def embed(self, text: str) -> np.ndarray:
        return self._vec(text)

    def encode(self, text, *, normalize_embeddings: bool = False, **kw):
        if isinstance(text, list):
            return np.vstack([self._vec(t) for t in text])
        return self._vec(text)


class TestGradedCorpusEvalProof(unittest.TestCase):
    """Measure the fusion features on the shipped graded corpus, offline."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db_path = pathlib.Path(cls.tmp.name) / "proof.db"

        cls._old_db = os.environ.get("POCKET_SQLITE_DB")
        os.environ["POCKET_SQLITE_DB"] = str(cls.db_path)
        # Reload config + retrieval so they bind the proof DB path, then point
        # both the ingestion-side embedder factory and the query-side model at
        # the deterministic HashingEmbedder (overriding conftest's MockEmbedder
        # for the lifetime of this class).
        importlib.reload(sys.modules["pocket.config"])
        from pocket import retrieval, evaluation
        importlib.reload(retrieval)
        importlib.reload(evaluation)
        cls.retrieval = retrieval
        cls.evaluation = evaluation

        cls._patches = [
            patch(
                "pocketindex.ops.sentence_transformers.SentenceTransformerEmbedder",
                new=HashingEmbedder,
            ),
            patch("pocket.retrieval._get_model", new=lambda name: HashingEmbedder()),
        ]
        for p in cls._patches:
            p.start()

        from pocket.pipeline import app_main

        app = pix.App(
            "pocket_eval_proof",
            app_main,
            sourcedir=_CORPUS_DIR,
            db_path=cls.db_path,
        )
        app.update_blocking(live=False, report_to_stdout=False)

    @classmethod
    def tearDownClass(cls):
        for p in cls._patches:
            p.stop()
        if cls._old_db is not None:
            os.environ["POCKET_SQLITE_DB"] = cls._old_db
        else:
            os.environ.pop("POCKET_SQLITE_DB", None)
        importlib.reload(sys.modules["pocket.config"])
        cls.tmp.cleanup()

    def _gold(self):
        return self.evaluation.load_cases(_GOLD)

    def _case(self, needle: str):
        case = next((c for c in self._gold() if needle in c.query), None)
        self.assertIsNotNone(case, f"gold.json must contain a case matching {needle!r}")
        return case

    # --- corpus / label integrity ------------------------------------------

    def test_gold_labels_reference_existing_corpus_files(self):
        """Every relevant_file in gold.json must exist in eval/corpus/."""
        cases = self._gold()
        self.assertGreaterEqual(len(cases), 4, "gold set should be non-trivial")
        corpus_files = {p.name for p in _CORPUS_DIR.glob("*.md")}
        self.assertTrue(corpus_files, "corpus directory must contain markdown files")
        for case in cases:
            for rel in case.relevant_files:
                self.assertIn(
                    os.path.basename(rel),
                    corpus_files,
                    f"gold relevant file {rel!r} missing from eval/corpus/",
                )

    def test_every_gold_case_retrieves_a_relevant_file(self):
        """A retrievability floor: each gold query hits within top-k."""
        metrics = self.evaluation.evaluate(self._gold(), db_path=self.db_path, k=5)
        self.assertEqual(
            metrics.hit_rate,
            1.0,
            "every gold query should surface at least one relevant file in top-5",
        )

    # --- POCKET-501: MMR diversity is a measurable Recall@k win -------------

    def test_mmr_recovers_a_buried_second_relevant_file(self):
        """MMR lifts Recall@k by demoting near-duplicates that bury answer #2.

        The diversity case has two distinct relevant files; near-duplicate
        paraphrases of the first crowd the top-3 so plain fusion finds only one
        (Recall@3 = 0.5). MMR penalises the redundant copies, so the second
        distinct file surfaces (Recall@3 = 1.0).
        """
        case = self._case("cache eviction policy")
        self.assertEqual(
            len(case.relevant_files), 2, "diversity case needs two distinct answers"
        )
        base = self.evaluation.evaluate(
            [case], db_path=self.db_path, k=3, use_mmr=False
        )
        mmr = self.evaluation.evaluate(
            [case], db_path=self.db_path, k=3, use_mmr=True, mmr_lambda=0.5
        )
        self.assertLessEqual(
            base.recall_at_k,
            0.5,
            "plain fusion must bury the second relevant file (redundancy crowds top-k)",
        )
        self.assertEqual(
            mmr.recall_at_k, 1.0, "MMR must recover both distinct relevant files"
        )
        self.assertGreater(
            mmr.recall_at_k,
            base.recall_at_k,
            "MMR must strictly improve Recall@k on the redundant query",
        )

    # --- POCKET-502: weight tuning is a measurable MAP win -----------------

    def test_weight_tuning_lifts_map_and_downweights_misleading_vector(self):
        """The tuner finds weights that beat equal-weight RRF on a hybrid query.

        On the weighted-fusion case the vector strategy ranks keyword-dense
        distractors above the answer (whose only hook is a rare token BM25
        rewards). The grid search should land on weights with strictly higher
        MAP than the equal-weight baseline, and those weights must down-weight
        the misleading vector strategy relative to lexical.
        """
        case = self._case("zebrastripe")
        result = self.evaluation.tune_weights(
            [case],
            db_path=self.db_path,
            k=5,
            metric="mean_average_precision",
        )
        self.assertTrue(
            result.improved, "tuning must beat the equal-weight RRF baseline here"
        )
        self.assertGreater(
            result.best_score,
            result.baseline_score,
            "best MAP must strictly exceed the equal-weight baseline MAP",
        )
        self.assertLess(
            result.best_weights["vector"],
            result.best_weights["lexical"],
            "tuner should down-weight the misleading vector strategy vs lexical",
        )
        # 1.0 is always probed, so the winner can never be worse than plain RRF.
        self.assertGreaterEqual(result.best_score, result.baseline_score)

    def test_coordinate_ascent_matches_grid_cheaper(self):
        """Coordinate ascent reaches the grid's optimum with fewer evaluations.

        The same hybrid case has a separable weight surface, so optimizing one
        strategy at a time (coordinate ascent) lands on the same best MAP as the
        exhaustive grid while scoring strictly fewer combinations — the point of
        the cheaper search. The quality property (down-weighting the misleading
        vector strategy) must survive too.
        """
        case = self._case("zebrastripe")
        grid = self.evaluation.tune_weights(
            [case], db_path=self.db_path, k=5,
            metric="mean_average_precision", method="grid",
        )
        coord = self.evaluation.tune_weights(
            [case], db_path=self.db_path, k=5,
            metric="mean_average_precision", method="coordinate",
        )
        self.assertTrue(coord.improved, "coordinate ascent must beat the baseline too")
        self.assertEqual(
            coord.best_score,
            grid.best_score,
            "coordinate ascent should reach the grid's optimum on this separable case",
        )
        self.assertLess(
            len(coord.trials),
            len(grid.trials),
            "coordinate ascent must score fewer combinations than the full grid",
        )
        self.assertLess(
            coord.best_weights["vector"],
            coord.best_weights["lexical"],
            "coordinate winner must also down-weight the misleading vector strategy",
        )

    def test_coordinate_ascent_cli_flag(self):
        """`pocket eval --tune --tune-method coordinate` runs and reports."""
        import pocket.cli as cli_module
        from click.testing import CliRunner

        importlib.reload(cli_module)
        runner = CliRunner()
        res = runner.invoke(
            cli_module.cli,
            ["eval", "--cases", str(_GOLD), "--tune",
             "--tune-method", "coordinate"],
        )
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("via coordinate search", res.output)

    # --- POCKET-503: query expansion is a measurable Recall@k win -----------

    def test_query_expansion_recovers_abbreviation_only_match(self):
        """Expansion recovers a relevant file an abbreviation alone can't reach.

        The expansion case has two relevant files: db_journal.md (matches the
        query's spelled-out terms) and db_wal.md (whose only hook is the long
        form of the acronym 'wal', which appears in no document). Without
        expansion BM25 has no 'wal' token to match and the hashed embedding
        shares no vocabulary with db_wal.md, so only db_journal.md surfaces
        (Recall@3 = 0.5). Turning expansion on rewrites 'wal' -> 'write ahead
        log', so db_wal.md is retrieved too (Recall@3 = 1.0).
        """
        case = self._case("wal journaling")
        self.assertEqual(
            len(case.relevant_files), 2, "expansion case needs two relevant files"
        )
        base = self.evaluation.evaluate(
            [case], db_path=self.db_path, k=3, use_expansion=False
        )
        expanded = self.evaluation.evaluate(
            [case], db_path=self.db_path, k=3, use_expansion=True
        )
        self.assertLessEqual(
            base.recall_at_k,
            0.5,
            "without expansion the abbreviation-only file must stay buried",
        )
        self.assertEqual(
            expanded.recall_at_k,
            1.0,
            "expansion must recover the abbreviation-only relevant file",
        )
        self.assertGreater(
            expanded.recall_at_k,
            base.recall_at_k,
            "expansion must strictly improve Recall@k on the abbreviation query",
        )
        self.assertEqual(
            base.hit_rate,
            1.0,
            "the spelled-out file is always found, so the hit-rate floor holds",
        )

    def test_query_expansion_cli_flag_raises_measured_recall(self):
        """`pocket eval --cases ... --expand` reports a higher Recall@k.

        Proves the CLI threads the expansion toggle into the harness. Only the
        'wal' gold case carries an acronym in the built-in expansion map, so the
        expansion is a no-op for the other cases and the aggregate Recall@3 rises
        solely because the abbreviation-only file is recovered.
        """
        import pocket.cli as cli_module
        from click.testing import CliRunner

        importlib.reload(cli_module)
        runner = CliRunner()
        args = ["eval", "--cases", str(_GOLD), "--mode", "hybrid", "--k", "3"]
        plain = runner.invoke(cli_module.cli, args + ["--no-expand"])
        expanded = runner.invoke(cli_module.cli, args + ["--expand"])
        self.assertEqual(plain.exit_code, 0, plain.output)
        self.assertEqual(expanded.exit_code, 0, expanded.output)

        def _recall(output: str) -> float:
            m = re.search(r"Recall@3:\s+([\d.]+)", output)
            self.assertIsNotNone(m, f"no Recall@3 line in:\n{output}")
            return float(m.group(1))

        self.assertGreater(
            _recall(expanded.output),
            _recall(plain.output),
            "the --expand flag must measurably raise aggregate Recall@3",
        )

    # --- POCKET-504: semantic query router picks the mode from query shape --

    def test_router_classifies_query_shapes(self):
        """The router maps query *shape* to a concrete mode, deterministically.

        Pure function (no DB): code-shaped queries go to lexical exact-match,
        relationship/concept questions go to graph multi-hop, and plain prose
        keeps the hybrid blend. These are the routing branches POCKET-504 adds.
        """
        route = self.retrieval._route_query
        for q in (
            "parse_payload schema validation rules",  # snake_case identifier
            "parsePayload request handler",           # camelCase identifier
            "where is config.py loaded",              # filename.ext
            "what calls connect()",                   # function call
            "resolve ns::symbol scope",               # C++/Rust scope
            "fix the `init()` path",                  # backtick code span
        ):
            self.assertEqual(route(q), "lexical", f"{q!r} should route to lexical")
        for q in (
            "how does cache eviction relate to expiry",
            "connection between bfs and dijkstra",
            "what is the relationship between wal and durability",
        ):
            self.assertEqual(route(q), "graph", f"{q!r} should route to graph")
        for q in (
            "cache eviction policy entry removal",
            "breadth first search neighbor frontier",
            "time to live cache expiry lifetime",
        ):
            self.assertEqual(route(q), "hybrid", f"{q!r} should route to hybrid")

    def test_router_auto_beats_hybrid_on_code_query(self):
        """Auto routing a code query to lexical lifts MAP over the hybrid blend.

        The gold 'parse_payload' case is code-shaped, so the router picks
        lexical. The rare 'parse_payload' token sits only in router_anchor.md
        (BM25 ranks it #1), while the keyword-dense router_blend_* distractors
        dominate the diluted bag-of-words vector — so plain hybrid lets a
        distractor outrank the answer (MAP=0.5). Auto drops the misleading vector
        strategy and puts the answer first (MAP=1.0). Recall@k stays 1.0 either
        way, so this is a measured *ranking* win, not a recall floor.
        """
        case = self._case("parse_payload")
        self.assertEqual(case.mode, "auto", "router case must use mode=auto")
        self.assertEqual(
            self.retrieval._route_query(case.query),
            "lexical",
            "the code-shaped query must route to lexical",
        )

        EvalCase = self.evaluation.EvalCase
        hybrid_case = EvalCase(
            query=case.query, relevant_files=list(case.relevant_files), mode="hybrid"
        )
        lexical_case = EvalCase(
            query=case.query, relevant_files=list(case.relevant_files), mode="lexical"
        )
        auto = self.evaluation.evaluate([case], db_path=self.db_path, k=5)
        hybrid = self.evaluation.evaluate([hybrid_case], db_path=self.db_path, k=5)
        lexical = self.evaluation.evaluate([lexical_case], db_path=self.db_path, k=5)

        # Auto must reproduce exactly the lexical route it selected.
        self.assertEqual(
            auto.cases[0].retrieved_files,
            lexical.cases[0].retrieved_files,
            "auto must return the same hits as the lexical mode it routed to",
        )
        # ...and rank the answer first, unlike the hybrid blend.
        self.assertEqual(
            auto.cases[0].reciprocal_rank, 1.0, "auto must rank the answer #1"
        )
        self.assertLess(
            hybrid.cases[0].reciprocal_rank,
            1.0,
            "plain hybrid must bury the answer below a vector-favoured distractor",
        )
        self.assertGreater(
            auto.mean_average_precision,
            hybrid.mean_average_precision,
            "routing to lexical must strictly beat hybrid MAP on the code query",
        )
        self.assertEqual(
            auto.recall_at_k, 1.0, "the answer is still retrieved (recall floor holds)"
        )
        self.assertEqual(
            hybrid.recall_at_k,
            1.0,
            "hybrid also retrieves the answer — the win is ranking, not recall",
        )

    def test_router_flag_upgrades_plain_hybrid(self):
        """POCKET_QUERY_ROUTER auto-routes a plain `hybrid` call without --mode.

        With the flag off, `mode="hybrid"` stays a fixed blend (a distractor
        ranks first). With the flag on, the same plain `hybrid` call is routed
        exactly like an explicit `mode="auto"` and surfaces the answer first.
        """
        q = "parse_payload validation"

        with patch.object(self.retrieval.config, "POCKET_QUERY_ROUTER", False):
            plain = self.retrieval.search(q, limit=5, db_path=self.db_path, mode="hybrid")
        with patch.object(self.retrieval.config, "POCKET_QUERY_ROUTER", True):
            routed = self.retrieval.search(q, limit=5, db_path=self.db_path, mode="hybrid")
        auto = self.retrieval.search(q, limit=5, db_path=self.db_path, mode="auto")

        self.assertTrue(plain and routed and auto, "all three searches must return hits")
        self.assertEqual(
            [h.file_path for h in routed],
            [h.file_path for h in auto],
            "flag-on hybrid must route identically to explicit mode=auto",
        )
        self.assertTrue(
            routed[0].file_path.endswith("router_anchor.md"),
            "routed hybrid must rank the exact-match answer first",
        )
        self.assertFalse(
            plain[0].file_path.endswith("router_anchor.md"),
            "un-routed hybrid must let a distractor rank first (proving the upgrade matters)",
        )

    def test_router_graph_query_falls_back_when_graph_absent(self):
        """A graph-routed query degrades to hybrid when the DB has no graph tables.

        The offline corpus is built without graph extraction, so routing a
        relationship query to `graph` would return nothing. `_resolve_mode` must
        fall back to hybrid so auto routing never silently yields zero results.
        """
        q = "how does cache eviction relate to expiry"
        self.assertEqual(
            self.retrieval._route_query(q), "graph", "relationship query routes to graph"
        )
        hits = self.retrieval.search(q, limit=5, db_path=self.db_path, mode="auto")
        self.assertTrue(
            hits, "graph route must fall back to hybrid on a graph-less database"
        )

    def test_router_cli_search_auto_mode(self):
        """`pocket search ... --mode auto` routes and surfaces the answer."""
        import pocket.cli as cli_module
        from click.testing import CliRunner

        importlib.reload(cli_module)
        runner = CliRunner()
        res = runner.invoke(
            cli_module.cli,
            ["search", "parse_payload validation",
             "--mode", "auto", "--limit", "3"],
        )
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("router_anchor.md", res.output)



    # --- CLI plumbing: `pocket eval --mmr` measures the trade-off ----------

    def test_cli_eval_mmr_flag_raises_measured_recall(self):
        """`pocket eval --cases ... --mmr` reports a higher Recall@k than --no-mmr.

        Proves the CLI threads the MMR toggle into the harness so the diversity
        trade-off is measurable from the command line, not only via env config.
        """
        import pocket.cli as cli_module
        from click.testing import CliRunner

        importlib.reload(cli_module)
        runner = CliRunner()
        args = ["eval", "--cases", str(_GOLD), "--mode", "hybrid", "--k", "3"]
        plain = runner.invoke(cli_module.cli, args + ["--no-mmr"])
        diverse = runner.invoke(cli_module.cli, args + ["--mmr"])
        self.assertEqual(plain.exit_code, 0, plain.output)
        self.assertEqual(diverse.exit_code, 0, diverse.output)

        def _recall(output: str) -> float:
            m = re.search(r"Recall@3:\s+([\d.]+)", output)
            self.assertIsNotNone(m, f"no Recall@3 line in:\n{output}")
            return float(m.group(1))

        self.assertGreater(
            _recall(diverse.output),
            _recall(plain.output),
            "the --mmr flag must measurably raise aggregate Recall@3 on the gold set",
        )


if __name__ == "__main__":
    unittest.main()
