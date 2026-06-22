"""Shared pytest fixtures and auto-patches for genome-pocket tests.

MockEmbedder
------------
The real SentenceTransformerEmbedder downloads a ~90 MB model on first use.
This module installs a session-scoped autouse patch that replaces it with
MockEmbedder (fixed 384-dim zero vector) for the entire test suite.

The patch targets:
  - pocketindex.ops.sentence_transformers.SentenceTransformerEmbedder
  - sentence_transformers.SentenceTransformer  (pocket.retrieval lru_cache)
  - pocket.retrieval._get_model                (bypasses lru_cache entirely)

Works for both pytest-native tests and unittest.TestCase subclasses because
it is a session-scoped autouse fixture that starts before any test class.
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

_EMBED_DIM = 384


class MockEmbedder:
    """Deterministic fake embedder: fixed zero vector, no network."""

    def __init__(self, model_name: str = "mock", **kwargs):
        self.model_name = model_name
        self.embedding_dim = _EMBED_DIM

    async def embed(self, text: str) -> np.ndarray:
        return np.zeros(_EMBED_DIM, dtype=np.float32)

    def encode(self, text, *, normalize_embeddings: bool = False, **kw):
        """Sync encode path used by pocket.retrieval._get_model."""
        if isinstance(text, list):
            return np.zeros((_EMBED_DIM,), dtype=np.float32)
        return np.zeros(_EMBED_DIM, dtype=np.float32)


_mock_embedder_instance = MockEmbedder()


@pytest.fixture(scope="session", autouse=True)
def _patch_sentence_transformers():
    """Session-wide autouse: swap every embedding path to MockEmbedder."""
    patches = [
        patch(
            "pocketindex.ops.sentence_transformers.SentenceTransformerEmbedder",
            new=MockEmbedder,
        ),
        patch(
            "sentence_transformers.SentenceTransformer",
            new=lambda *a, **kw: _mock_embedder_instance,
        ),
        patch(
            "pocket.retrieval._get_model",
            new=lambda model_name: _mock_embedder_instance,
        ),
    ]
    started = [p.start() for p in patches]
    yield started
    for p in patches:
        try:
            p.stop()
        except RuntimeError:
            pass


@pytest.fixture
def mock_embedder():
    """Per-test fixture that returns the MockEmbedder instance."""
    return _mock_embedder_instance
