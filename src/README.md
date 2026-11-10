# src/ 모듈 지도

각 폴더의 담당과 들어갈 파일입니다. 파일 이름은 제안이며, 담당자가 바꾸면 이 표를 같이 고칩니다. 모듈 사이 입출력은 [docs/INTERFACES.md](../docs/INTERFACES.md)를 따릅니다.

| 폴더 | 담당 | 들어갈 것 (제안 파일 이름) |
|---|---|---|
| `config/` | B | 설정 로더 (`settings.py`), `ChatOllama` 생성 함수 (`llm.py`), 호출 수·토큰 콜백 (`callbacks.py`) |
| `tools/` | B | BM25 검색 (`bm25_search.py`), 계산기 (`calculator.py`), 공통 오류 처리 |
| `baselines/` | B | P0 · P0-raw (`p0.py`), P1 ReAct (`p1_react.py`), 공통 답 형식 지시문 (`prompts.py`) |
| `meba/` | A · C · D | 아래 표 |
| `meba/conditional/` | C · D | 근거 장부 상세화 (`ledger.py`, C), 간결 출력 (`concise.py`, D), 동적 압축량 (`dynamic_m.py`, C) |
| `eval/` | A · D | 채점기 (`scorer.py`, A), 측정 로거 (`logger.py`, A), 러너 (`runner.py`, A), VRAM 샘플러 (`vram.py`, A), 결과표·McNemar (`tables.py`, `stats.py`, D) |

## `meba/` 세부

| 파일 (제안) | 부품 | 담당 |
|---|---|---|
| `router.py` | 라우터 (첫 호출, 수학 풀이 / 검색 선택) | A |
| `judge.py` | 근거 판정 + 답 호출, JSON 스키마, 대조군 파서 | C |
| `requery.py` | 다중 홉 재검색 (빠진 정보 → 후속 검색어) | C |
| `compress.py` | 검색 결과 압축 (제목 + 앞뒤 1문장 창, BM25 상위 m) | C |
| `budget.py` | 출력 상한 적용, 잘림 감지·재시도, 호출 예산 집계 | D |
| `loop_breaker.py` | 루프 끊기 | D |
| `graph.py` | LangGraph 조립, 부품 켜기/끄기(절제 변형) | A (통합) |

절제 변형은 별도 코드 복사본을 만들지 않고, `graph.py`에서 부품별 켜기/끄기 설정으로 만듭니다.

## 공통 규칙

- 모델 태그·temperature·상한 값은 `config/`를 통해서만 읽습니다.
- 파이프라인 함수는 `qid`, `question`만 받습니다. 문항 유형을 받지 않습니다.
- 단위 테스트는 `tests/`에 같은 이름으로 둡니다 (예: `tests/test_scorer.py`). 채점기와 압축은 GPU 없이 테스트할 수 있으니 꼭 테스트를 붙입니다.
