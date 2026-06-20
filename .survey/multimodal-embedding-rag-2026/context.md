# Context — 워크플로우·사용자·우회책·인접 문제

## 현재 워크플로우 (코드 실독 기반)
`README.md` Source→Refine→Load→Serve + 실제 모듈:

1. **Source** — `pocketindex/connectors/localfs`: `./notes` 의 마크다운/텍스트/코드 파일 감시. (이미지·PDF·오디오 커넥터 없음 → 멀티모달 진입점 부재.)
2. **Refine** — `pocketindex/ops/refine.py`: `TextRefiner`(텍스트) / 코드용 indentation-preserving 패스. 오프셋 맵으로 lineage 보존.
3. **Transform** — `RecursiveSplitter`(code-aware 청킹) → `SentenceTransformerEmbedder.embed()` (`pocketindex/ops/sentence_transformers.py`, 5줄) → `IdGenerator`.
4. **Load** — `pocketindex/connectors/sqlite.py`: 임베딩을 `BLOB`(`sqlite_vec.serialize_float32`)로 저장, 텍스트를 FTS5에 미러링. **차원 비고정**(스키마가 `Annotated[...] → BLOB`로 일반화, `_create_table`), vec0 가상테이블·ANN 인덱스 없음.
5. **Serve** — `pocket/retrieval.py`:
   - vector: `vec_distance_cosine(embedding, ?) ORDER BY distance ASC LIMIT ?` → **O(N) 풀스캔**.
   - lexical: FTS5 `bm25()`.
   - fusion: RRF(`RRF_K=60`), 각 전략 `limit*4` over-fetch.
   - CLI / MCP(`mcp_server.py`) / REST(`api_server.py`) 3경로 공유.
6. **(opt-in) GraphRAG** — `POCKET_GRAPH=1`: entity/relation 추출(deterministic/ollama/airllm), `graph_neighborhood`/`list_graph_concepts`.

## 영향받는 사용자 & 현재 한계
- 임베딩 모델 호출이 `model.encode(text, ...)` 텍스트 인자 고정 → **이미지/문서 이미지 임베딩 불가**.
- `_get_model`(`lru_cache`)이 쿼리측에서 `config.EMBEDDING_MODEL` 단일 모델만 로드 → **쿼리·문서 비대칭 인코더**(예: ColPali류 멀티벡터)나 **모달리티별 인코더 라우팅** 표현 불가.
- 검색이 단일 벡터 cosine 1순위 → **리랭킹/late-interaction 점수화 슬롯 없음**.
- 차원이 코드에 박혀있진 않지만(BLOB) `vec_distance_cosine`는 길이 일치 가정 → **모델 교체 시 전체 재인덱스 필수**(혼재 차원 불가).

## 현행 우회책 (현재 사용자가 겪는 workaround)
- 이미지/다이어그램은 **검색 불가** → 사용자는 파일명·주변 텍스트(alt-text/캡션)에만 의존해 lexical로 우회.
- PDF/스캔 문서는 **외부 OCR로 텍스트화 후** notes에 넣어야 인덱싱됨(파이프라인 외부 수작업).
- 더 강한 다국어/긴 문서 검색이 필요하면 사용자가 `EMBEDDING_MODEL` 을 더 큰 SentenceTransformer로 바꾸지만, **차원 변경 시 인덱스 깨짐** → `pocket drop --yes` 후 재빌드 수동 처리.
- 정밀도 부족분은 `limit`을 키워 후처리 read로 메우는 식.

## 인접 문제 (같이 풀면 이득)
1. **ANN 부재** — 노트 수천 청크까지는 풀스캔이 괜찮지만, 멀티모달(이미지+문서이미지)로 코퍼스가 커지면 O(N) cosine은 병목. 멀티벡터(late-interaction)는 N×토큰 으로 더 심함 → 도입 시 인덱싱/저장 전략 재설계 동반 필요.
2. **리랭커 슬롯 없음** — 2026 RAG 표준은 retrieve→rerank 2단. RRF 뒤에 cross-encoder/멀티벡터 재점수 단계가 비어 있음.
3. **차원/모달 메타데이터 미기록** — `embeddings` 테이블에 모델명·차원·모달리티 컬럼이 없어 모델 혼재/마이그레이션 안전장치 부재.
4. **오프라인 보장 vs 모델 크기** — 프로젝트 정체성이 local-first/offline(테스트는 `MockEmbedder`). 멀티모달 SOTA(3B~7B VLM 임베더)는 GPU·VRAM 요구가 커서 "포켓"의 가벼움과 충돌 → tiered(소형 기본 / 대형 옵션) 설계가 필요.

## User voices (대표 페르소나 관점, repo 신호 기반)
- *"내 노트의 다이어그램·스크린샷도 검색되면 좋겠다"* — 멀티모달 인덱싱 요구. (현재 불가, README가 SVG 다이어그램을 핵심 자산으로 둠.)
- *"PDF를 OCR 없이 그냥 넣고 싶다"* — ColPali/jina-v4류 "문서를 이미지로 임베딩" 트렌드와 직결.
- *"노트북에서 오프라인으로 돌아야 한다"* — 모델 선택의 1차 제약. closed API(Voyage-multimodal-3, Cohere Embed v4) 배제 근거.
- *"모델 바꿀 때 인덱스가 안 깨졌으면"* — 차원/모델 메타데이터 + 마이그레이션 명령 필요.
