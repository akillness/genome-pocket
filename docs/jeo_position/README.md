# jeo_position — Codex Impact Workshop 개발자 포지션 정리

> 사회혁신가를 지원하는 개발자(정장영) 입장에서, 워크숍 전 준비·업무 범위 타진·3시간 기술 도전과제를
> **PM 프레임워크(pm-skills)** 와 **제품 네이밍(product-name)** 관점으로 정리한 작업 패키지.

출처: [Codex Impact Workshop 사전 참고 문서](https://docs.google.com/document/d/1Y3ptCqBKA_zA1Ayna1a_St8eHFjllxP1Z4h8KLLPsvg/edit) (이동이·정장영)

## 한 줄 요약

서울환경연합(애드보커시 단체)이 **성과지표인 "언론보도"를 자동 수집·분류·정성평가**하고
미디어 소통 통계로 활용할 수 있는 **내부 도구/대시보드**를 함께 만든다.
활동가가 직접 모니터링·아카이빙하는 반복 업무 시간을 줄이고, "보도 횟수"를 넘어 "보도의 깊이"를 데이터로 본다.

## 문서 구성

| 문서 | 내용 | 반영 관점 |
|------|------|-----------|
| [00-context-brief.md](00-context-brief.md) | 워크숍 맥락, 페르소나, JTBD, North Star, 가정 지도 | pm-skills (Discovery/Strategy) |
| [01-preparation.md](01-preparation.md) | 워크숍 전·당일 준비 체크리스트, 25분 싱크 질문, 환경 셋업 | pm-skills (Discovery) |
| [02-scope.md](02-scope.md) | 타진 가능한 업무 범위, Opportunity Solution Tree, RICE 우선순위, 범위 티어 | pm-skills (Discovery/Execution) |
| [03-tech-challenges-3h.md](03-tech-challenges-3h.md) | 3시간 MVP 시나리오, 시간대별 플랜, 기술 스택 후보, 리스크 | pm-skills (Execution) |
| [04-product-naming.md](04-product-naming.md) | 결과물 후보 이름 5종 + 근거 | product-name |
| [05-credits-and-api-experiments.md](05-credits-and-api-experiments.md) | 참가자 지원 크레딧(Codex $100 + OpenAI API $50), API별 활용 매핑, 비용 가드레일, 3시간 실험 플랜 | pm-skills (Execution) |

## 핵심 사실 (사전 문서에서 확정)

- **사회혁신가**: 이동이 / 서울환경연합 사무처장 / 기후·환경 섹터 / 현장 활동가가 주 사용자.
- **과제 분류**: 성과평가·의사결정 도구형.
- **현재 방식**: 구글 알리미 → 협업툴 잔디 채널로 실시간 뉴스 수신. **축적·기록 체계 없음 → 성과 데이터 휘발.**
- **실패 경험**: 앱스스크립트로 키워드 기사를 구글시트에 기입 시도 → 기간 설정·키워드 정확도 문제로 실패.
- **데이터 준비 상태**: "바로 가져올 수 있음", 팀원과 원본 공유 가능.
- **개발자 강점(매칭 메모)**: 멀티모달 RAG 파이프라인, API 서버, Slack Bot 기반 에이전트 경험.
- **기대 협업 방식**: 함께 방법을 찾고, 난이도에 맞춰 쉽게 설명해 주는 사람.
