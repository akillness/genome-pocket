"""Embedding backends for PocketIndex + a registry to pick one by model id.

Two model *types* live behind one contract:

  * :class:`SentenceTransformerEmbedder` — text-only (default; Qwen3-Embedding,
    MiniLM, ...).
  * :class:`~pocketindex.ops.siglip_embedder.SiglipEmbedder` — multimodal
    SigLIP2 (text + image into one shared space; opt-in).

Rather than scatter ``if is_siglip_model(...)`` branches across the pipeline and
the retrieval layer, every model family is one row in :data:`EMBEDDER_BACKENDS`.
:func:`resolve_backend` is the single dispatch point both sides consume, so a new
family is added by appending one entry instead of editing several call sites.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import sentence_transformers
from sentence_transformers import SentenceTransformer

from pocketindex.ops.siglip_embedder import SiglipEmbedder, is_siglip_model


class SentenceTransformerEmbedder:
    # Text-only backend: image files are not routed here.
    supports_image = False

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        # Instruction-aware models (e.g. Qwen3-Embedding) ship a prompt registry.
        # Documents are encoded with the (possibly empty) "document" prompt so the
        # asymmetric query/document encoding matches the model's training recipe.
        prompts = getattr(self.model, "prompts", None) or {}
        self._doc_prompt = "document" if "document" in prompts else None

    async def embed(self, text: str):
        # Generate embedding for the text (passage/document side).
        kwargs = {"normalize_embeddings": True}
        if self._doc_prompt is not None:
            kwargs["prompt_name"] = self._doc_prompt
        return self.model.encode(text, **kwargs)


@dataclass(frozen=True)
class EmbedderBackend:
    """One embedding model family.

    ``matches`` classifies a hub id; ``indexer`` builds the ingestion-side
    embedder (shared async ``embed`` contract + ``supports_image`` flag);
    ``query_model`` builds the retrieval-side encoder. The factories resolve the
    embedder classes by module-global name at call time so the test suite's
    MockEmbedder patch stays effective.
    """

    name: str
    matches: Callable[[str], bool]
    indexer: Callable[[str], object]
    query_model: Callable[[str], object]


# Ordered registry: first entry whose ``matches`` accepts the model id wins; the
# last entry is the text-only catch-all. Add a model family = append a row here.
EMBEDDER_BACKENDS: list[EmbedderBackend] = [
    EmbedderBackend(
        name="siglip2-multimodal",
        matches=is_siglip_model,
        indexer=lambda name: SiglipEmbedder(name),
        query_model=lambda name: SiglipEmbedder(name),
    ),
    EmbedderBackend(
        name="sentence-transformers-text",
        matches=lambda _name: True,
        indexer=lambda name: SentenceTransformerEmbedder(name),
        # Resolve the class off the live module at call time (not a captured
        # ``from`` binding) so the test suite's SentenceTransformer patch is
        # honored even when pocket.retrieval is reloaded for DB isolation.
        query_model=lambda name: sentence_transformers.SentenceTransformer(name),
    ),
]


def resolve_backend(model_name: str) -> EmbedderBackend:
    """Return the first registry entry whose ``matches`` accepts ``model_name``."""
    for backend in EMBEDDER_BACKENDS:
        if backend.matches(model_name):
            return backend
    # Unreachable while the registry keeps its catch-all text entry, but guard
    # explicitly so a future edit that drops it fails loudly instead of silently.
    raise ValueError(f"no embedding backend registered for {model_name!r}")


def build_embedder(model_name: str):
    """Pick the ingestion-side embedding backend for ``model_name``.

    SigLIP2 ids get the transformers-native multimodal :class:`SiglipEmbedder`
    (text + image, shared space); everything else uses the sentence-transformers
    text backend. Both expose the same async ``embed(text)`` contract, so the
    pipeline is agnostic to which one it received.
    """
    return resolve_backend(model_name).indexer(model_name)
