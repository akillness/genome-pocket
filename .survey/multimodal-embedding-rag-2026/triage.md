# Triage — Multimodal Embedding 업그레이드 & RAG 구조 점검 (2026)

> Survey slug: `multimodal-embedding-rag-2026`
> Date: 2026 기준 스캔 (모델 사실은 HuggingFace API로 실측 확인, 본 repo 코드 실독)
> Bounded research question:
> **"genome-pocket 의 텍스트 전용 임베딩(all-MiniLM-L6-v2 / 384d)을 2026년 오픈웨이트 멀티모달 임베딩으로 교체할 때, 어떤 모델이 SOTA·실사용 가능 수준이며, 현재 RAG 파이프라인 구조는 최신 트렌드 대비 어디가 강하고 어디가 부족한가?"**

## Problem (무엇이 문제인가)
- 현재 검색 엔진은 **텍스트 전용**이다. `pocket/retrieval.py` 와 `pocketindex/ops/sentence_transformers.py` 가
  `SentenceTransformer("all-MiniLM-L6-v2")` 단일 인코더에 묶여 있고, 임베딩은 `model.encode(text, normalize_embeddings=True)` 호출 하나다.
- 노트/문서 안의 **이미지, 다이어그램, 표, PDF 스캔, 차트**는 임베딩되지 않는다. README의 컨셉 다이어그램(`docs/images/pocket-architecture.svg`)처럼 시각 정보가 핵심인 자료가 검색 대상에서 누락된다.
- `all-MiniLM-L6-v2`(2021)는 384차원 영어 중심 모델로, 2026년 기준으로는 다국어·long-context·instruction 측면에서 명백히 구형이다.
- 벡터 저장은 sqlite-vec BLOB + `vec_distance_cosine` **브루트포스 풀스캔**이며(`_vector_search`), ANN 인덱스(vec0 가상테이블)·리랭커·쿼리 확장이 없다.

## Audience (누가 영향을 받나)
- **로컬-퍼스트 개인 지식 운영자** (genome-pocket의 1차 사용자): 마크다운 + 코드 + 첨부 이미지/PDF를 한 인덱스로 검색하고 싶은 사람.
- **MCP/REST 소비자**: Claude Code·Curs 등에서 `search_knowledge` 를 호출하는 에이전트. 멀티모달 검색이 되면 "이 다이어그램이 뭐냐"류 질의가 가능해진다.
- **유지보수자**: 임베딩 차원·차원 가변성·인덱스 마이그레이션·오프라인 동작 보장을 떠안는 사람.

## Why now (왜 지금인가)
- 2025~2026 사이 **오픈웨이트 멀티모달 임베딩**이 closed API(Voyage, Cohere) 수준에 근접하거나 일부 벤치마크에서 추월했다. 실측 다운로드/좋아요로 채택이 확인된다:
  - `jinaai/jina-embeddings-v4` (2025-05 공개, arXiv:2506.18902) — 526k+ DL, 526 likes. 단일 모델로 text·image·문서스크린샷·멀티벡터 지원.
  - `vidore/colqwen2.5-v0.2` — 112k DL, ColPali 계열 late-interaction 문서 검색 SOTA.
  - `nomic-ai/colnomic-embed-multimodal-7b/3b`, `Alibaba-NLP/gme-Qwen2-VL-*`, `TIGER-Lab/VLM2Vec-*`, `google/siglip2-*`, `jinaai/jina-clip-v2` 모두 실 채택 중.
- 동시에 RAG **구조 트렌드**도 이동했다: 단순 dense top-k → (1) hybrid + **reranker(cross-encoder)**, (2) **late-interaction(ColBERT/ColPali)**, (3) **문서를 이미지로 직접 임베딩(OCR 파이프라인 제거)**, (4) GraphRAG/contextual chunking. genome-pocket은 hybrid+RRF·opt-in GraphRAG까지는 왔지만 reranker·late-interaction·멀티모달은 비어 있다.
- 즉, **모델 교체 적기**(차원/모달리티 재설계가 어차피 인덱스 재빌드를 요구)이면서, 같은 작업에서 **구조 점검**을 함께 하면 마이그레이션 1회로 두 부채를 정리할 수 있다.

## Scope / Non-goals
- In-scope: 오픈웨이트·HF 제공 멀티모달 임베딩 후보 정리(SOTA/실사용), 본 repo RAG 구조 vs 2026 SOTA 구조 장단점 비교, 교체·도입 시 구조적 제약(차원, 오프라인, 스토리지) 식별.
- Out-of-scope(이번 survey): 실제 코드 구현/PR, 벤치마크 재현 실행, closed API 상용 계약. (이건 후속 plan/team 단계.)
