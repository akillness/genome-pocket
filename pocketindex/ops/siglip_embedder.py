"""SigLIP2 multimodal embedder for PocketIndex.

SigLIP2 (``google/siglip2-*``) is a dual-encoder that maps text and images into a
single shared, L2-normalized space, so a plain cosine/dot product compares a text
query against an image embedding. That lets the existing sqlite-vec single-vector
+ RRF retrieval path stay exactly as-is while gaining image search — the model is
opt-in via ``EMBEDDING_MODEL=google/siglip2-base-patch16-224`` (or any siglip2 id).

Heavy deps (``transformers``, ``torch``, ``Pillow``) are imported lazily so the
base, text-only install is unaffected when this backend is not selected.
"""
from __future__ import annotations

import io
import pathlib
from functools import cached_property

import numpy as np


def is_siglip_model(model_name: str) -> bool:
    """True for SigLIP / SigLIP2 hub ids (the multimodal shared-space family)."""
    return "siglip" in model_name.lower()


class SiglipEmbedder:
    """Encode text and images into SigLIP2's shared embedding space.

    Exposes the same async ``embed(text)`` contract as ``SentenceTransformerEmbedder``
    plus ``embed_image(path)`` for the image ingestion pass and a synchronous
    ``encode_query(text)`` used by the (sync) retrieval layer. ``supports_image``
    lets the pipeline decide whether to route image files to this embedder.
    """

    supports_image = True

    def __init__(self, model_name: str):
        self.model_name = model_name

    @cached_property
    def _runtime(self):
        # Lazy, cached load: importing transformers/torch and pulling weights only
        # happens the first time an embedding is actually requested.
        import torch
        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name).eval()
        return torch, processor, model

    def _normalize(self, out) -> np.ndarray:
        torch, _, _ = self._runtime
        # transformers >=5 returns a pooled-output wrapper from get_*_features;
        # earlier versions return the tensor directly. Handle both.
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            out = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state.mean(dim=1)
        if out.ndim == 2:
            out = out[0]
        vec = torch.nn.functional.normalize(out, dim=-1)
        return vec.detach().cpu().numpy().astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """Synchronous text -> shared-space vector (retrieval/query side)."""
        torch, processor, model = self._runtime
        inputs = processor(
            text=[text], return_tensors="pt", padding="max_length", max_length=64
        )
        with torch.no_grad():
            return self._normalize(model.get_text_features(**inputs))

    def encode_image(self, image_bytes: bytes) -> np.ndarray:
        """Synchronous image bytes -> shared-space vector (document side)."""
        torch, processor, model = self._runtime
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            return self._normalize(model.get_image_features(**inputs))

    async def embed(self, text: str) -> np.ndarray:
        # Text document side shares the query encoder (SigLIP is symmetric across
        # modalities — no separate document prompt).
        return self.encode_query(text)

    async def embed_image(self, path: pathlib.Path) -> np.ndarray:
        with open(path, "rb") as f:
            return self.encode_image(f.read())
