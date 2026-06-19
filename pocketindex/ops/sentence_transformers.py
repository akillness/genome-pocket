"""SentenceTransformer embedder for PocketIndex."""
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
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
