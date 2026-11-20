# 진행 현황 (progress)

> **기준 문서:** [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — 작업 단위·기한·선행 조건은 이 문서를 따른다. 여기서는 상태만 기록한다.
> **마지막 갱신:** 2026-11-20 (`main` @ `3aceac0`). A-2·TBD #4·#5 행만 갱신했고, 나머지 행은 각 담당이 PR에서 갱신한다.

## 갱신 규칙

- 상태는 **`main`에 머지된 것**을 기준으로 적는다. 브랜치에만 있는 작업은 `🔄 진행 중`.
- 작업을 지우거나 완료 조건을 바꾸지 않는다. 바꿀 일이 있으면 IMPLEMENTATION_PLAN을 먼저 고친다.
- 각자 자기 담당 행을 PR에서 같이 갱신한다. 근거 칸에는 파일이나 PR 번호를 적는다.
- 상태 표시: ✅ 완료 · 🔄 진행 중 · ⬜ 미착수 · ❓ 저장소로 확인 불가(담당자 확인 필요)

## 한눈에 보기

| 구간 | 기간 | 상태 |
|---|---|---|
| 1주차 | 11/11 ~ 11/17 | 진행 중. 동결·설정·토크나이저·압축·채점기 완료. BM25·예산 래퍼 행은 각 담당 갱신 대기 |
| 2주차 | 11/18 ~ 11/24 | 미착수 |
| 3주차 | 11/25 ~ 11/27 | 미착수 |
| 측정 | 11/28 ~ 12/03 | 미착수 |
| 마무리 | 12/04 | 미착수 |

테스트: 전체 254개 통과 (2026-11-20, `pytest -q`, A-3 브랜치 기준. `main`은 161개)

## 지금 막혀 있는 것

| 항목 | 기한 | 막고 있는 작업 | 담당 |
|---|---|---|---|
| TBD #6 라운드 간 결과가 다를 때 대표값 규칙 | 11/16 (지남) | PR D-2 | D |

TBD #4·#5는 팀 3인 동의로 확정되어 [INTERFACES.md](INTERFACES.md) §6과 [EVALUATION.md](EVALUATION.md) §3에 반영됐다.

## 착수 전 선결 사항 · TBD

| # | 항목 | 상태 | 근거 |
|---|---|---|---|
| 1 | 동결 규칙: SQuAD 문맥 분리, `doc_id` 형식, `data.seed` | ✅ | INTERFACES §1·§2, EVALUATION §1. 시드 20261113 |
| 2 | `configs/` 포맷·모델 태그·`keep_alive` | ✅ | `configs/default.yaml` (yaml, `qwen2.5:7b-instruct-q4_K_M`, `keep_alive: -1`) |
| 3 | 공용 토크나이저·불용어 규칙 | ✅ | INTERFACES §3, `src/tools/tokenizer.py` |
| 7 | 의존성 파일·Python 버전 | ✅ | `requirements.txt` 한 곳에서 관리, Python 3.10+, `pyproject.toml`은 pytest 설정만 |
| 4 | P0 기권 문구 | ✅ | P0에도 기권 문구를 넣는다. INTERFACES §6, EVALUATION §3.3 (`4ede049`) |
| 5 | 답 추출 접두어·GSM8K 숫자 규칙 | ✅ | `Final answer:` 접두어, 마지막 숫자 규칙. EVALUATION §3.1·§3.2 (`4ede049`) |
| 6 | 대표값 규칙 | ⬜ | 위 "막혀 있는 것" 참고 |
| - | 스켈레톤 커밋 | ✅ | `c9335bb` + `data/frozen/.gitkeep` |
| - | 브랜치 이름 규칙 | ✅ | 각자 영문 이름 (`YongMin`, `HyunSeong`, `HwanHo`) |
| - | `main` 보호 설정, 팀원 초대 | ❓ | GitHub 설정이라 저장소로 확인 불가 |
| - | 공통 타입 `CallRecord`·`PipelineResult` | ✅ | `src/common/types.py` |

## 역할별 작업

### A · 고용민

| 작업 | 기한 | 상태 | 근거 · 비고 |
|---|---|---|---|
| PR A-0 스켈레톤·의존성·규격 합의 | 11/11 | ✅ | PR #1·#2·#6. `main` 보호·팀원 초대는 ❓ |
| PR A-1 데이터·코퍼스 동결 | 11/13 | ✅ | PR #6, 동결 커밋 `d95beb4`. 교차 확인 완료([data/README.md](../data/README.md)) |
| PR A-2 채점기 | 11/17 | ✅ | `src/eval/scorer.py`, `tests/test_scorer.py` (PR #12, `24b8e0a`). EM/F1·수치·기권 채점, 공통 답 추출 |
| PR A-3 로거·VRAM·러너 | 11/19 | 🔄 | `src/eval/{vram,logger,runner}.py`, `scripts/run_eval.py`, `tests/test_eval_io.py`·`tests/test_runner.py`·`tests/test_run_eval.py` (`YongMin` 브랜치, PR 대기). Antigravity 교차 검증(11/20, CPU 전용 환경) 지적 반영 완료. 파이프라인 등록(`PIPELINE_FACTORIES`)은 `"모듈:함수"`(생성 함수)와 `"call:모듈:함수"`(B-3의 `run_p0(qid, question, *, settings)` 형태) 두 형식을 받는다. M은 A-5b가 채운다. 실환경 검증 완료(11/20, Colab T4 + Ollama `qwen2.5:7b-instruct-q4_K_M`): 테스트 262개 통과, `--verify`로 로그 필드·이어 돌기·`RUN_META.json` 확인, `data/frozen` 무변경 |
| PR A-4 라우터 | 11/20 | ⬜ | 선행 B-2b, D-1 |
| PR A-5a 그래프 골격 | 11/23 | ⬜ | |
| PR A-5b M 통합 | 11/25 | ⬜ | |

### B · 이현성

| 작업 | 기한 | 상태 | 근거 · 비고 |
|---|---|---|---|
| PR B-1 설정·ChatOllama·콜백 | 11/13 | ✅ | `configs/default.yaml`, `src/config/{settings,llm,callbacks}.py`, `tests/test_config.py` (PR #3) |
| PR B-2a 공용 토크나이저 | 11/13 | ✅ | `src/tools/tokenizer.py`, `tests/test_tools.py` (PR #5) |
| PR B-2b BM25 검색·계산기 | 11/16 | ✅ | `src/tools/{bm25_search,calculator,__init__}.py`, `tests/test_tools.py` (PR #9) |
| PR B-3 P0·P0-raw·P1 | 11/18 | ✅ | `src/baselines/{prompts,p0,p1_react,__init__}.py`, `tests/test_baselines.py` |
| 실험 B-4 `k` 확정·P1 dev 로그 | 11/20 | ⬜ | |
| PR B-5 Colab 노트북 (A에서 이관 제안) | 11/26 | ⬜ | 이관은 팀 확인 필요 |

### C · 조환호

| 작업 | 기한 | 상태 | 근거 · 비고 |
|---|---|---|---|
| PR C-1 문장 창 압축 | 11/17 | ✅ | `src/meba/compress.py`, `tests/test_compress.py` (PR #4) |
| 실험 C-2 `k`×`m` 측정 | 11/19 | ✅ | [기록](experiments/2026-11-15-c-compress-m.md), `scripts/measure_compress_m.py`. 권장 m: k=3 → 6, k=5 → 7, k=8 → 10 |
| PR C-3 근거 판정·재검색 | 11/23 | ⬜ | 선행 D-1 (없으면 mock 상한 래퍼) |
| 실험 C-4 JSON vs 태그 | 11/25 | ⬜ | |
| PR C-5 조건부 `ledger`·`dynamic_m` | 11/27 | ⬜ | |

### D · 심동섭

| 작업 | 기한 | 상태 | 근거 · 비고 |
|---|---|---|---|
| PR D-1 출력 상한·예산·루프 끊기 | 11/17 | ⬜ | B-1 ✅로 착수 가능. A-4·A-5a·C-3의 선행 |
| PR D-2 결과표·McNemar | 11/20 | ⬜ | TBD #6 필요 |
| 분석 D-3 P1 로그 분석 | 11/23 | ⬜ | 선행 B-4 |
| 실험 D-4 `cap.judge` 민감도 | 11/27 | ⬜ | 선행 A-5b |
| PR D-5 조건부 `concise` | 11/27 | ⬜ | |

## 본 측정 및 마무리

| 단계 | 기간 | 상태 |
|---|---|---|
| M 구성 동결 | 11/27 | ⬜ |
| 본 비교 측정 (3라운드) | 11/28 ~ 11/30 | ⬜ |
| 절제 측정 (1라운드) | 12/01 ~ 12/02 | ⬜ |
| 결과표·통계 | 12/03 | ⬜ |
| 보고서·발표 자료 | 11/28 ~ 12/04 | ⬜ |
| 버퍼 | 12/05 ~ 12/07 | ⬜ |
| 발표 | 12/08 | ⬜ |

## 알려진 임시값

| 항목 | 현재 값 | 확정 예정 |
|---|---|---|
| `retrieval.k` | 3 (`configs/default.yaml`의 임시값) | 실험 B-4 (11/20) |
| `compress.m` | 3 (임시값, 보존율 70~77%로 기준 미달) | 실험 C-2 완료. B-4에서 k가 정해지면 k=3 → 6, k=5 → 7, k=8 → 10으로 반영 |
| 의존성 `langchain>=1.0.0` | P1 `create_agent` 지원 위해 `requirements.txt` 상향 완료 | PR B-3에서 반영 |

## 기타

- 프로젝트 스킬(`.agents/skills/`, `.claude/skills/`, `skills-lock.json`)은 설치만 되어 있고 아직 커밋하지 않았다.
