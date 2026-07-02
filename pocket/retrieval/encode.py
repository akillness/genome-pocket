from functools import lru_cache

@lru_cache(maxsize=4)
def _get_model(model_name: str):
    """Cache the embedding model so repeated queries don't reload weights.

    Delegates model-type selection to the shared embedder registry
    (:func:`pocketindex.ops.sentence_transformers.resolve_backend`) so the query
    side can never drift from the ingestion side. Returns a multimodal
    :class:`SiglipEmbedder` for siglip2 ids (so a text query is encoded into the
    shared image/text space), or a plain SentenceTransformer for text models.
    """
    from pocketindex.ops.sentence_transformers import resolve_backend

    return resolve_backend(model_name).query_model(model_name)


def _encode_query(model, text: str):
    """Encode query-side text into the index's vector space.

    - Multimodal SigLIP2 (``encode_query``): text -> shared image/text space so the
      query can match stored image embeddings.
    - Instruction-aware text models (e.g. Qwen3-Embedding) define a ``query`` prompt
      that must wrap the query for the asymmetric retrieval recipe.
      Symmetric models such as all-MiniLM expose no prompts and are encoded plainly.
    """
    if hasattr(model, "encode_query"):
        return model.encode_query(text)
    kwargs = {"normalize_embeddings": True}
    if "query" in (getattr(model, "prompts", None) or {}):
        kwargs["prompt_name"] = "query"
    return model.encode(text, **kwargs)
