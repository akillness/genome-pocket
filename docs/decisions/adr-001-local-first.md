# ADR 001: Local-First & Privacy Architecture

## Status
Proposed

## Context
Pocket is designed as a personal Knowledge Ops runtime. Personal notes, codebases, and experimental logs contain highly sensitive, proprietary, and private information. Relying on cloud-based SaaS models for embedding generation, vector storage, and LLM reasoning poses significant privacy risks and can incur high subscription or API costs.

## Decision
We will adopt a **local-first, privacy-preserving architecture** as the default configuration for Pocket:

1. **Local Embeddings:** We will use local embedding models (e.g., `all-MiniLM-L6-v2` or `bge-small-en-v1.5`) running on the user's CPU/GPU via `sentence-transformers` or `onnxruntime`.
2. **Local Vector Database:** We will use SQLite with the `sqlite-vec` extension or a local LanceDB instance for vector storage. This avoids the need to run a separate database server (like Qdrant or pgvector) for simple local setups.
3. **Local LLM Integration:** For query expansion and concept extraction, we will support local LLM runtimes (e.g., Ollama or Llama.cpp) via standard OpenAI-compatible APIs.
4. **Opt-in Cloud Fallbacks:** Cloud services (like OpenAI, Anthropic, or Pinecone) will only be used if explicitly configured by the user in their `.env` file.

## Consequences

### Pros
- **Privacy:** No personal data is sent to third-party APIs by default.
- **Cost:** Zero API costs for embedding generation and vector search.
- **Offline Capability:** The system works completely offline, making it ideal for local development and note-taking.
- **Low Latency:** Local database queries and embeddings avoid network roundtrips.

### Cons
- **Resource Usage:** Running embedding models and local LLMs consumes local CPU, GPU, and memory resources.
- **Model Quality:** Local small models may have lower retrieval accuracy compared to state-of-the-art cloud models (e.g., OpenAI's `text-embedding-3-large`). We will mitigate this using hybrid search and reciprocal rank fusion.
- **Installation Complexity:** Installing native extensions like `sqlite-vec` or setting up local GPU acceleration (CUDA/MPS) can be challenging on some platforms. We will provide detailed setup guides and pre-built binaries where possible.
