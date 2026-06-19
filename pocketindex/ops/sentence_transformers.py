"""SentenceTransformer embedder for PocketIndex."""
from sentence_transformers import SentenceTransformer

from pocketindex.ops.siglip_embedder import SiglipEmbedder, is_siglip_model


def build_embedder(model_name: str):
    """Pick the embedding backend for ``model_name``.

    SigLIP2 ids get the transformers-native multimodal :class:`SiglipEmbedder`
    (text + image, shared space); everything else uses the sentence-transformers
    text backend. Both expose the same async ``embed(text)`` contract, so the
    pipeline is agnostic to which one it received.
    """
    if is_siglip_model(model_name):
        return SiglipEmbedder(model_name)
    return SentenceTransformerEmbedder(model_name)


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
