# Platform Map — 어디서 무엇을, 어떤 제약으로

> 멀티모달 임베딩은 "모델 + 서빙 런타임 + 벡터 저장"의 3축이 함께 결정된다. genome-pocket 제약(local-first/offline, sqlite-vec, single-vector 경로)을 기준으로 매핑.

## 1. 모델 배포 플랫폼 (어디서 가중치를 받나)
| 플랫폼 | 역할 | 본 repo 관점 비고 |
|---|---|---|
| **HuggingFace Hub** | 1차 공급원 (가중치·model card·라이선스) | 모든 후보가 HF에 존재(실측). `sentence-transformers`/`transformers`로 로드. 오프라인 캐시(`HF_HUB_OFFLINE=1`) 가능 |
| **GGUF 미러** (예: jina-v4-text-GGUF, gme-Qwen2-VL-GGUF) | 양자화·CPU/llama.cpp 서빙 | "포켓" 경량 요구에 부합. 멀티모달 GGUF는 아직 제한적(텍스트/일부만) |
| vLLM 전용 빌드 (예: jina-embeddings-v4-vllm-retrieval) | 고처리량 GPU 서빙 | 서버형 배포 시; 로컬 노트북 기본 경로는 아님 |

## 2. 서빙 런타임 (어떻게 인코딩하나)
| 런타임 | 적합 모델군 | 제약 |
|---|---|---|
| **sentence-transformers** (현행) | jina-clip-v2, siglip2, nomic 일부, Qwen3-Embedding | 현재 `SentenceTransformerEmbedder`와 인터페이스 호환 → 교체 비용 최소 |
| **transformers (AutoModel/VLM)** | jina-v4, gme-Qwen2-VL, VLM2Vec, colqwen, colnomic | VLM 로딩·이미지 전처리 필요. `embed()` 시그니처를 (text|image) 멀티모달로 확장해야 함 |
| **llama.cpp / GGUF** | 양자화된 텍스트 임베더, 일부 VLM | CPU·오프라인 강점, 멀티모달 커버리지 미성숙 |
| **Ollama** | 로컬 임베딩 서버 | repo가 이미 GraphRAG에서 ollama provider 사용 → 임베딩에도 재활용 패턴 가능 |

### 모델 크기 vs 로컬 현실 (오프라인 제약 핵심)
- **경량(노트북/CPU 가능)**: jina-clip-v2, siglip2-base, nomic-embed-multimodal-3b, gme-Qwen2-VL-2B, Qwen3-Embedding-0.6B.
- **중량(GPU 권장)**: jina-embeddings-v4, gme-7B, colqwen2.5(3B VLM), colnomic-7b, VLM2Vec-Full.
- 결론: 기본값은 **경량 single-vector**, 대형/late-interaction은 **opt-in 환경변수**로. (GraphRAG의 opt-in 패턴 재사용.)

## 3. 벡터 저장 / 검색 (어디에 넣나)
| 옵션 | single-vector | multi-vector(late-interaction) | ANN | 본 repo 적합 |
|---|---|---|---|---|
| **sqlite-vec (현행)** | ✅ BLOB + `vec_distance_cosine` 풀스캔 | ✗ 네이티브 MaxSim 없음(앱단 구현 필요) | vec0 가상테이블로 부분 지원 | 1·2단계(single-vector)에 그대로 사용 |
| sqlite-vec + 앱단 MaxSim | △ | △ (토큰벡터 다행 저장 후 집계) | △ | colpali 도입 시 커스텀 필요 |
| 외부 ANN(FAISS/HNSWlib) | ✅ | △ | ✅ | 스케일 시 고려, local-first 유지 가능 |
| 전용 벡터DB(Qdrant/Milvus 등) | ✅ | ✅ multi-vector 네이티브 | ✅ | "포켓" 경량 정체성과 충돌 → 비권장 |

## 4. 모달리티 진입점 (무엇을 임베딩하나) — 현재 vs 필요
| 모달리티 | 현재 Source 커넥터 | 필요 작업 |
|---|---|---|
| 마크다운/텍스트/코드 | ✅ localfs | 유지 |
| 이미지(png/jpg/svg) | ❌ | localfs 확장 또는 image 커넥터 + 이미지 전처리 |
| PDF/스캔 문서 | ❌ (외부 OCR 수작업 우회) | 페이지→이미지 렌더 후 colpali/jina-v4로 직접 임베딩(OCR 제거) |
| 표/차트 | ❌ | 문서이미지 경로에 포함 |
| 오디오/비디오 | ❌ | VLM2Vec-V2(video) 등 후속 범위 |

## 5. 라이선스/상용성 게이트 (도입 전 반드시 확인) — HF API 실측(2026 스캔)
- **상용 안전(실측 확인)**: `Qwen/Qwen3-Embedding-0.6B`=**apache-2.0**, `google/siglip2-base`=**apache-2.0**, `vidore/colqwen2.5-v0.2`=**MIT**. → 상용 배포 가능.
- **비상용 확정(상용 차단, 실측 확인)**: `jinaai/jina-clip-v2`=**CC-BY-NC-4.0**. 상용 제품엔 사용 불가 → 상용이면 siglip2/nomic/colqwen 계열로 대체.
- **카드에 SPDX 미표기(도입 전 검증 필수)**: `jinaai/jina-embeddings-v4`, `nomic-ai/colnomic-embed-multimodal-3b` — HF API에 명시 license 필드 없음(연구/커스텀 라이선스 가능). 상용이면 반드시 모델 카드 원문 확인.
- **Closed 제외**: Voyage·Cohere·OpenAI 멀티모달 임베딩(API 전용, offline 불가) — 본 repo 정체성과 불일치.

## 6. 핸드오프 체크 (plan 단계로 넘길 때)
- [ ] 기본 모델/대형 옵션 2-tier 결정 (예: 기본 jina-clip-v2, 옵션 jina-v4/colqwen).
- [ ] `embeddings` 스키마에 model_id·dim·modality 컬럼 + `pocket reindex` 마이그레이션.
- [ ] `SentenceTransformerEmbedder.embed()` 를 (text|image) 멀티모달 시그니처로 확장하거나 멀티모달 임베더 op 신설.
- [ ] reranker(opt-in) 슬롯을 `retrieval.search()` RRF 뒤에 배치.
- [ ] multi-vector 채택 시 sqlite-vec MaxSim·ANN 전략 ADR 작성.
