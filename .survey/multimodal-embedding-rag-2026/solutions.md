# Solutions — 2026 오픈웨이트 멀티모달 임베딩 & RAG 구조 후보

> 모델 사실(존재/채택)은 HuggingFace API로 실측(다운로드·likes·공개일). 벤치마크 수치는 각 모델 카드/논문의 자가보고이며 재현 실행은 안 함 — 채택·적합성 판단용 신호로만 사용.

## A. 멀티모달 임베딩 모델 후보 (오픈웨이트 / HF)

| 모델 | 유형 | 모달리티 | 라이선스 성격 | 임베딩 형태 | HF 채택(실측) | genome-pocket 적합도 |
|---|---|---|---|---|---|---|
| **jinaai/jina-embeddings-v4** (2025-05, arXiv:2506.18902) | VLM 기반 통합 임베더 | text, image, 문서 스크린샷, 테이블/차트 | 비상용 제한(라이선스 확인 필요) | single-vector **+ multi-vector(late-interaction)**, Matryoshka 차원 | 526k DL / 526 likes | **★ 1순위 후보** — 한 모델로 텍스트+문서이미지, 차원 가변(MRL) → 기존 단일-벡터 경로와 호환 가능 |
| **jinaai/jina-clip-v2** | CLIP 계열 dual-encoder | text, image (다국어) | **비상용 CC-BY-NC-4.0** (HF 실측 확인 — 상용 배포 불가) | single-vector(공유 공간) | 73k DL / 334 likes | 가벼운 멀티모달 시작점. text↔image 직접 cosine, 기존 `vec_distance_cosine` 그대로 재사용. **단 상용 시 라이선스 차단** |
| **vidore/colqwen2.5-v0.2** (ColPali 계열) | late-interaction 문서 검색 | 문서 페이지 **이미지** | **MIT** (실측 확인), Qwen2.5-VL 기반 | **multi-vector(per-patch)** | 112k DL / 99 likes | 문서/PDF "OCR 없이" 검색 SOTA. 단 multi-vector → MaxSim 점수화·저장 재설계 필요 |
| **nomic-ai/colnomic-embed-multimodal-7b / 3b** | late-interaction 멀티모달 | text, image, 문서 | Apache 친화(확인 필요) | multi-vector | 25k / 18k DL | ColPali 대안, 다국어·오픈성 강점. 동일하게 multi-vector 인프라 요구 |
| **nomic-ai/nomic-embed-multimodal-3b** | dense 멀티모달 | text, image, 문서 | 오픈 | single-vector | 2k DL / 29 likes | single-vector라 도입 부담 적음 |
| **Alibaba-NLP/gme-Qwen2-VL-2B / 7B-Instruct** | VLM 기반 instruct 임베더 | text, image, text+image | 오픈(Qwen) | single-vector | 9.2k / 1.3k DL, 134 likes | instruction-aware, 2B는 로컬 친화. MTEB/멀티모달 강세 |
| **TIGER-Lab/VLM2Vec-Full / V2.0** | VLM→임베딩(MMEB 학습) | text, image, video(V2) | 오픈 | single-vector | 337k DL | 범용 멀티모달 임베딩 벤치(MMEB) 기반, 광범위 태스크 |
| **OpenSearch-AI/Ops-MM-embedding-v1-2B / 7B** | 멀티모달 임베더 | text, image, 문서 | 오픈 | single-vector | 2.7k / 0.9k DL | 신규, 검색 특화 |
| **google/siglip2-so400m-* / base-*** | SigLIP2 dual-encoder | text, image | 오픈(웨이트 공개) | single-vector | 1.0M+ DL | 견고한 image-text 베이스라인, 경량·고처리량. 문서이미지엔 약함(자연이미지 강) |
| **(참고) 텍스트 전용 업그레이드: Qwen/Qwen3-Embedding-0.6B/4B/8B** | 텍스트 임베더 | text(다국어, instruct) | Apache 2.0 | single-vector, MRL | 9.1M / 2.0M / 2.1M DL | 멀티모달이 과하면 **텍스트만이라도** MiniLM 대체. 0.6B는 로컬 친화 SOTA |

### Closed/배제 (오프라인·오픈웨이트 요건 불충족, 비교 기준선으로만)
- **Voyage multimodal-3**, **Cohere Embed v4(멀티모달)**, **OpenAI** 임베딩 — API 전용. 품질 기준선이지만 local-first 정체성과 충돌하여 본 repo 대상에서 제외.

### 선택 가이드 (genome-pocket 제약: local-first/offline, sqlite-vec single-vector 경로)
- **최소 변경 / 즉시 멀티모달**: `jina-clip-v2` 또는 `nomic-embed-multimodal-3b`(single-vector). 기존 `vec_distance_cosine` 풀스캔·RRF 그대로, 인덱스 차원만 교체.
- **품질 1순위 통합**: `jina-embeddings-v4`(single-vector 모드로 시작, 후속에 multi-vector 옵션). text+문서이미지 한 모델.
- **문서/PDF 헤비**: `colqwen2.5` / `colnomic` — 단 multi-vector(MaxSim) 저장·점수화 신규 인프라 필요(인접 문제 #1·#2).
- **멀티모달 보류·텍스트만 현대화**: `Qwen3-Embedding-0.6B`(Apache, MRL 384/512/1024 등) 로 MiniLM 교체 → 저위험 quick win.

## B. RAG 구조: 2026 SOTA 트렌드 vs genome-pocket 현황

| 구조 요소 | 2026 SOTA 트렌드 | genome-pocket 현황 | 장점(현행) | 단점/격차 |
|---|---|---|---|---|
| 1차 검색 | dense + sparse **hybrid** | ✅ vector(sqlite-vec) + FTS5 BM25 | 이미 hybrid, lineage 보존, 3경로 공유 | — |
| 융합 | RRF / weighted | ✅ RRF(k=60) | 단순·견고, 논문 표준값 | 가중·learned fusion 없음 |
| **리랭킹** | retrieve→**cross-encoder rerank** (2단) | ❌ 없음 | — | 정밀도 상한 낮음. RRF 직후 재점수 슬롯 부재 |
| **Late-interaction** | ColBERT/ColPali multi-vector(MaxSim) | ❌ single-vector만 | 저장·연산 가벼움 | 문서이미지·세밀 매칭 SOTA 불가 |
| **멀티모달** | 통합 임베더 / 문서=이미지 | ❌ 텍스트 전용 | 단순 | 이미지/PDF/표/차트 검색 불가 |
| ANN 인덱스 | HNSW/IVF (vec0 등) | ❌ 브루트포스 `vec_distance_cosine` 풀스캔 | 정확·구현 단순, 소규모 OK | 코퍼스↑·multi-vector 시 병목 |
| 청킹 | **contextual/semantic chunking** | ✅ code-aware RecursiveSplitter | 구조 인식 우수 | LLM contextual-chunk·문서레이아웃 미적용 |
| 쿼리 처리 | query rewrite/expansion, HyDE | ❌ FTS 토큰 OR만 | 결정적·빠름 | 의미 확장 없음 |
| 구조적 검색 | **GraphRAG** | ✅ opt-in GraphRAG 분기 | 동일 incremental lineage 재사용, HITL 게이트 | deterministic 추출 기본 → 관계 품질 제한 |
| 평가 | RAG eval/trace loop | ✅ ops-layer eval/trace/HITL 문서화 | 운영 성숙 | 멀티모달/리랭크 평가 미포함 |

### 장단점 요약
- **강점(유지)**: hybrid+RRF, 엄밀한 lineage(소스 바이트 오프셋 인용), incremental ETL(Δ-only/memo/삭제 sweep), 단일 retrieval 경로 공유, opt-in GraphRAG, 오프라인 테스트(MockEmbedder). → 2026 기준으로도 **인프라 골격은 견고**.
- **격차(보강)**: (1) 멀티모달 임베딩 진입점 자체가 없음, (2) 리랭커 2단 부재, (3) late-interaction/multi-vector 표현 불가, (4) ANN 부재로 스케일·멀티벡터 비용, (5) 임베딩 테이블에 모델/차원/모달 메타데이터 없어 마이그레이션 안전장치 부재.

## C. 권장 도입 순서 (survey 결론 — 후속 plan 단계 입력)
1. **저위험 quick win**: 텍스트 임베더를 `Qwen3-Embedding-0.6B`(Apache, MRL)로 교체 + `embeddings`에 model/dim/modality 컬럼 추가 + `pocket reindex` 마이그레이션 경로. (멀티모달 전 안전 토대.)
2. **멀티모달 1단계(single-vector)**: 이미지/문서이미지 Source 커넥터 + `jina-clip-v2` 또는 `nomic-embed-multimodal-3b`로 동일 cosine 경로에 통합. 기존 RRF 재사용.
3. **정밀도 2단계(reranker)**: RRF 결과 상위 K를 cross-encoder(또는 jina-v4 multi-vector MaxSim)로 재점수.
4. **문서 헤비 옵션(late-interaction)**: `colqwen2.5`/`colnomic` + multi-vector 저장(MaxSim) + ANN(vec0/HNSW) — 코퍼스 성장 시.
5. 각 단계는 **opt-in**(GraphRAG 패턴 차용)으로 두어 local-first·offline 정체성과 가벼움 유지.
