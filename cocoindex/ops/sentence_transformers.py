"""SentenceTransformer embedder for CocoIndex."""
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    async def embed(self, text: str):
        # Generate embedding for the text
        return self.model.encode(text, normalize_embeddings=True)
