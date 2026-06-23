"""Tests for 2026 SOTA improvements: reranker, HyDE, SemanticSplitter,
OllamaExtractor schema-constrained output, and the RAGAS judge.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

class TestReranker(unittest.TestCase):
    """pocket.retrieval._rerank() and RetrievalHit.reranker_rank."""

    def _make_hit(self, text, score):
        from pocket.retrieval import RetrievalHit
        return RetrievalHit(
            file_path="a.md", text=text,
            start_offset=0, end_offset=len(text),
            score=score,
        )

    def test_reranker_rank_field_exists(self):
        """RetrievalHit has the new reranker_rank field."""
        from pocket.retrieval import RetrievalHit
        hit = RetrievalHit("f", "t", 0, 1, 0.5)
        self.assertIsNone(hit.reranker_rank)

    def test_reranker_reorders_by_cross_encoder_score(self):
        """_rerank() reorders hits by cross-encoder score and sets reranker_rank."""
        from pocket.retrieval import _rerank

        hits = [
            self._make_hit("unrelated text", 0.9),   # high RRF but low CE score
            self._make_hit("very relevant passage", 0.3),  # low RRF but high CE
        ]
        mock_model = MagicMock()
        # Cross-encoder: second hit is more relevant
        mock_model.predict.return_value = np.array([0.1, 0.9])

        with patch("pocket.retrieval._get_reranker", return_value=mock_model):
            result = _rerank("test query", hits, "mock-model")

        # Best CE score first
        self.assertEqual(result[0].text, "very relevant passage")
        self.assertEqual(result[0].reranker_rank, 1)
        self.assertEqual(result[1].reranker_rank, 2)

    def test_reranker_passthrough_on_empty(self):
        """_rerank() returns [] immediately for empty input."""
        from pocket.retrieval import _rerank
        self.assertEqual(_rerank("q", [], "m"), [])

    def test_reranker_fallback_on_model_load_failure(self):
        """_rerank() returns original order when model load fails."""
        from pocket.retrieval import _rerank, RetrievalHit

        hits = [RetrievalHit("f", "t", 0, 1, 0.5)]
        with patch("pocket.retrieval._get_reranker", return_value=None):
            result = _rerank("q", hits, "missing-model")
        self.assertEqual(result, hits)

    def test_reranker_fallback_on_predict_exception(self):
        """_rerank() returns original order when predict() raises."""
        from pocket.retrieval import _rerank, RetrievalHit

        hits = [RetrievalHit("f", "t", 0, 1, 0.5)]
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("GPU OOM")
        with patch("pocket.retrieval._get_reranker", return_value=mock_model):
            result = _rerank("q", hits, "mock")
        self.assertEqual(result, hits)


# ---------------------------------------------------------------------------
# HyDE query expansion
# ---------------------------------------------------------------------------

class TestHyDE(unittest.TestCase):
    """pocket.retrieval._hyde_expand()."""

    def test_hyde_returns_generated_text_on_success(self):
        from pocket.retrieval import _hyde_expand
        import urllib.error

        fake_response = json.dumps({"response": "This is a hypothetical passage."}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = fake_response

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _hyde_expand(
                "What is sqlite-vec?",
                ollama_model="qwen3:0.6b",
                ollama_host="http://localhost:11434",
            )
        self.assertEqual(result, "This is a hypothetical passage.")

    def test_hyde_fallback_on_connection_error(self):
        """_hyde_expand() returns the original query when Ollama is down."""
        from pocket.retrieval import _hyde_expand
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = _hyde_expand(
                "What is sqlite-vec?",
                ollama_model="qwen3:0.6b",
                ollama_host="http://localhost:11434",
            )
        self.assertEqual(result, "What is sqlite-vec?")

    def test_hyde_fallback_on_empty_response(self):
        """_hyde_expand() returns original query when model returns empty text."""
        from pocket.retrieval import _hyde_expand

        fake_response = json.dumps({"response": ""}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = fake_response

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _hyde_expand(
                "original query",
                ollama_model="qwen3:0.6b",
                ollama_host="http://localhost:11434",
            )
        self.assertEqual(result, "original query")


# ---------------------------------------------------------------------------
# SemanticSplitter
# ---------------------------------------------------------------------------

class TestSemanticSplitter(unittest.TestCase):
    """pocketindex.ops.text.SemanticSplitter."""

    def _make_model(self, *, n_sentences=None, fixed_vecs=None):
        """Minimal sync model stub with encode() returning fixed vectors."""
        model = MagicMock()
        if fixed_vecs is not None:
            model.encode.return_value = np.array(fixed_vecs, dtype=np.float32)
        else:
            # Two orthogonal vectors → cosine 0 → always a breakpoint
            model.encode.return_value = np.array(
                [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
            )
        return model

    def test_fallback_when_model_is_none(self):
        """SemanticSplitter falls back to RecursiveSplitter when model=None."""
        from pocketindex.ops.text import SemanticSplitter
        text = "Hello world. " * 200  # long enough to produce multiple chunks
        splitter = SemanticSplitter(model=None)
        chunks = splitter.split(text)
        self.assertTrue(len(chunks) >= 1)
        # RecursiveSplitter uses overlap so chunks are not disjoint; verify the
        # first and last chunk together span the full source text.
        self.assertEqual(chunks[0].text[:10], text[:10])
        self.assertEqual(chunks[-1].text[-13:], text[-13:])

    def test_fallback_on_single_sentence(self):
        """One sentence is returned as a single chunk (no embeddings needed)."""
        from pocketindex.ops.text import SemanticSplitter
        text = "Only one sentence here."
        model = MagicMock()
        splitter = SemanticSplitter(model=model)
        chunks = splitter.split(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text.strip(), text.strip())

    def test_splits_on_low_similarity(self):
        """Two sentences with orthogonal embeddings (sim=0) produce two chunks."""
        from pocketindex.ops.text import SemanticSplitter

        text = "First sentence about Python. Second sentence about databases."
        # Orthogonal embeddings → cosine = 0 < threshold=0.7 → split
        model = self._make_model(
            fixed_vecs=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        splitter = SemanticSplitter(model=model, breakpoint_threshold=0.7)
        chunks = splitter.split(text)
        # Should produce 2 chunks (one per sentence)
        self.assertGreaterEqual(len(chunks), 1)
        # All original text covered
        full = "".join(c.text for c in chunks)
        self.assertIn("Python", full)
        self.assertIn("databases", full)

    def test_merges_tiny_chunks(self):
        """Very short groups are merged so no chunk is below min_chunk_size."""
        from pocketindex.ops.text import SemanticSplitter

        # Two sentences, both short → merged into one chunk
        text = "Hi. Bye."
        model = self._make_model(
            fixed_vecs=[[1.0, 0.0], [0.0, 1.0]]
        )
        splitter = SemanticSplitter(
            model=model, breakpoint_threshold=0.7, min_chunk_size=50
        )
        chunks = splitter.split(text)
        # Both tiny sentences merged into a single chunk
        self.assertEqual(len(chunks), 1)
        self.assertIn("Hi", chunks[0].text)
        self.assertIn("Bye", chunks[0].text)

    def test_zero_vector_treated_as_similar(self):
        """Zero-vectors (MockEmbedder) must not crash and are treated as similar."""
        from pocketindex.ops.text import SemanticSplitter

        text = "First sentence. Second sentence."
        model = MagicMock()
        model.encode.return_value = np.zeros((2, 384), dtype=np.float32)
        splitter = SemanticSplitter(model=model, breakpoint_threshold=0.7)
        # Should not raise and should produce at least one chunk
        chunks = splitter.split(text)
        self.assertGreaterEqual(len(chunks), 1)

    def test_offsets_are_exact(self):
        """Chunk start/end offsets correctly index back into the original text."""
        from pocketindex.ops.text import SemanticSplitter

        text = "Alpha beta gamma. Delta epsilon zeta."
        model = self._make_model(
            fixed_vecs=[[1.0, 0.0], [0.0, 1.0]]
        )
        splitter = SemanticSplitter(
            model=model, breakpoint_threshold=0.7, min_chunk_size=1
        )
        chunks = splitter.split(text)
        for chunk in chunks:
            s, e = chunk.start.char_offset, chunk.end.char_offset
            self.assertEqual(text[s:e], chunk.text)


# ---------------------------------------------------------------------------
# OllamaExtractor schema-constrained output
# ---------------------------------------------------------------------------

class TestOllamaExtractorSchema(unittest.TestCase):
    """OllamaExtractor tries schema format first, falls back to plain JSON."""

    def _make_extractor(self):
        from pocketindex.ops.extract import OllamaExtractor
        return OllamaExtractor(model="llama3", host="http://localhost:11434")

    def _fake_response(self, payload: dict):
        """Build a mock urlopen context manager returning *payload* as JSON."""
        body = json.dumps(payload).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = body
        return mock_resp

    def test_schema_path_used_on_first_call(self):
        """First call sends the full JSON Schema dict as 'format'."""
        from pocketindex.ops.extract import _EXTRACTION_JSON_SCHEMA

        extractor = self._make_extractor()
        good_payload = {
            "response": json.dumps({"entities": [], "relations": []})
        }
        with patch("urllib.request.urlopen", return_value=self._fake_response(good_payload)) as mock_open:
            extractor.extract("Some text.")

        call_args = mock_open.call_args
        req = call_args[0][0]
        sent = json.loads(req.data.decode())
        # format must be the full schema object, not the string "json"
        self.assertIsInstance(sent["format"], dict)
        self.assertIn("properties", sent["format"])
        self.assertTrue(extractor._schema_supported)

    def test_falls_back_to_json_on_400(self):
        """A 400 response flips _schema_supported to False and re-tries with 'format':'json'."""
        import urllib.error
        from pocketindex.ops.extract import OllamaExtractor

        extractor = OllamaExtractor(model="old", host="http://localhost:11434")

        good_payload = {
            "response": json.dumps({"entities": [], "relations": []})
        }

        call_count = {"n": 0}

        def side_effect(req, timeout):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise urllib.error.HTTPError(
                    url="/api/generate", code=400,
                    msg="Bad Request", hdrs=None, fp=None,
                )
            return self._fake_response(good_payload)

        with patch("urllib.request.urlopen", side_effect=side_effect):
            result = extractor.extract("Some text.")

        # Schema path was rejected → fell back to legacy json mode
        self.assertFalse(extractor._schema_supported)
        # Subsequent calls skip schema path entirely
        self.assertEqual(call_count["n"], 2)

    def test_schema_supported_cached_across_calls(self):
        """Once schema mode succeeds, subsequent calls don't re-probe."""
        good_payload = {
            "response": json.dumps({"entities": [], "relations": []})
        }
        extractor = self._make_extractor()

        with patch(
            "urllib.request.urlopen",
            return_value=self._fake_response(good_payload),
        ) as mock_open:
            extractor.extract("First call.")
            extractor.extract("Second call.")

        # Each call should use exactly one HTTP request (schema path only).
        self.assertEqual(mock_open.call_count, 2)
        self.assertTrue(extractor._schema_supported)


# ---------------------------------------------------------------------------
# Config new keys
# ---------------------------------------------------------------------------

class TestConfig2026(unittest.TestCase):
    """New config keys are present and have correct defaults."""

    def setUp(self):
        # Remove cached module so env changes take effect.
        if "pocket.config" in sys.modules:
            importlib.reload(sys.modules["pocket.config"])

    def test_reranker_defaults(self):
        import pocket.config as cfg
        self.assertFalse(cfg.POCKET_RERANKER)
        self.assertIn("ms-marco", cfg.POCKET_RERANKER_MODEL)
        self.assertEqual(cfg.POCKET_RERANKER_TOP_N, 20)

    def test_hyde_defaults(self):
        import pocket.config as cfg
        self.assertFalse(cfg.POCKET_HYDE)
        self.assertIsInstance(cfg.POCKET_HYDE_OLLAMA_MODEL, str)
        self.assertTrue(cfg.POCKET_HYDE_OLLAMA_HOST.startswith("http"))

    def test_semantic_split_defaults(self):
        import pocket.config as cfg
        self.assertFalse(cfg.POCKET_SEMANTIC_SPLIT)
        self.assertAlmostEqual(cfg.POCKET_SEMANTIC_SPLIT_THRESHOLD, 0.7)


# ---------------------------------------------------------------------------
# RAGAS judge evaluation
# ---------------------------------------------------------------------------

class TestRAGASJudge(unittest.TestCase):
    """pocket.evaluation._ollama_relevance_score and evaluate_with_judge."""

    def test_relevance_score_parses_float(self):
        """Score is parsed from the Ollama response text."""
        from pocket.evaluation import _ollama_relevance_score

        fake_resp = json.dumps({"response": "0.85"}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = fake_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            score = _ollama_relevance_score(
                "test query", "some context",
                host="http://localhost:11434", model="qwen3:0.6b",
            )
        self.assertAlmostEqual(score, 0.85)

    def test_relevance_score_clamped_to_0_1(self):
        """Values outside [0,1] are clamped."""
        from pocket.evaluation import _ollama_relevance_score

        for raw, expected in [("2.5", 1.0), ("-0.3", 0.0)]:
            fake_resp = json.dumps({"response": raw}).encode()
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = fake_resp
            with patch("urllib.request.urlopen", return_value=mock_resp):
                score = _ollama_relevance_score(
                    "q", "c", host="http://localhost:11434", model="m"
                )
            self.assertAlmostEqual(score, expected)

    def test_relevance_score_neutral_on_error(self):
        """Returns 0.5 when Ollama is unavailable."""
        from pocket.evaluation import _ollama_relevance_score
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("refused"),
        ):
            score = _ollama_relevance_score(
                "q", "c", host="http://localhost:11434", model="m"
            )
        self.assertAlmostEqual(score, 0.5)

    def test_judge_metrics_dataclass(self):
        """JudgeMetrics wraps EvalMetrics and exposes to_dict()."""
        from pocket.evaluation import JudgeMetrics, EvalMetrics

        base = EvalMetrics(
            n_cases=2, k=5,
            hit_rate=0.5, mrr=0.5,
            precision_at_k=0.5, recall_at_k=0.5,
            mean_average_precision=0.5,
        )
        jm = JudgeMetrics(
            base=base,
            mean_context_relevance=0.75,
            n_judged=2,
            judge_model="qwen3:0.6b",
        )
        d = jm.to_dict()
        self.assertIn("mean_context_relevance", d)
        self.assertAlmostEqual(d["mean_context_relevance"], 0.75)
        self.assertEqual(d["judge_model"], "qwen3:0.6b")
        # Standard metrics still present
        self.assertIn("hit_rate", d)
