# M 통합 dev 실행(A-5b)과 근거 판정 프롬프트 v2

- 날짜: 2026-11-26
- 담당: A (통합). 근거 판정(`src/meba/judge.py`)은 C 담당 모듈이라 PR 설명에 수정 이유를 쓴다
- 관련 부품 / 설정 키: 근거 판정 지시문 (`_DECISION_HELP`, `_MISSING_HELP`, `_REASONING_HELP`), `judge.format`=json
- 실행 로그: `results/runs/dev/r2/m.jsonl` (v1), `results/runs/dev/r3/m.jsonl` (v2), 비교 대상 P1은 `results/runs/dev/r1/p1.jsonl`
- 실행: `notebooks/verify_a5b.ipynb` §8(r2), §13~15(진단·r3·비교) (Colab T4)
- git 커밋: r2 = `69b824e`, r3 = (v2 커밋, 실행 후 기입)
- 환경: `qwen2.5:7b-instruct-q4_K_M`, temperature 0, `num_ctx` 8192, `retrieval.k` 5, `compress.m` 7, `compress.window` 1, `budget.max_requery` 2, `cap.judge` 256

## 질문

M을 통째로 개발 세트에 돌렸을 때 SPEC §1의 가설(P1보다 정확도는 높고 호출 수·시간은 P1 이하)이 맞는가. 맞지 않으면 어느 부품이 원인이고, 그 부품을 고치면 나아지는가.

## 1단계: v1로 돈 M (r2) — 가설 확인

| 그룹 | 문항 | 정답률 | 기권률 | 호출/문항 | 시간/문항 |
|---|---|---|---|---|---|
| GSM8K | 20 | 0.85 | 0.00 | 1.30 | 7.8s |
| HotpotQA | 20 | 0.15 | 0.65 | 2.90 | 5.1s |
| SQuAD 답 있음 | 10 | 0.20 | 0.60 | 3.10 | 5.4s |
| SQuAD 답할 수 없음 | 10 | 0.70 | 0.70 | 2.50 | 3.6s |
| **전체** | 60 | **0.483** | 과잉 기권 0.38 | **2.32** | **5.81s** |

호출·시간은 워밍업 1문항을 뺀 59문항 평균이다(EVALUATION §4). 완주 60/60이며 오류, 시간 초과, 예산 소진, 잘림, 파싱 실패는 모두 0이다. 판정 값 분포는 RETRIEVE 35, ABSTAIN 26, ANSWER 14다.

| 가설 | M (r2) | P1 (r1) | 판정 |
|---|---|---|---|
| 정확도 P1보다 높음 | 0.483 | 0.350 | 맞음. 다만 M만 정답 14, P1만 정답 6이라 McNemar 정확 p ≈ 0.12로 60문항에서는 유의하지 않다 |
| 호출 수 P1 이하 | 2.32 | 1.75 | **틀림** |
| 시간 P1 이하 | 5.81s | 9.02s | 규칙상 맞음. 그러나 P1 평균은 시간 초과 4건(120초, 워밍업 제외 3건)이 끌어올린 값이다. P1의 중앙값은 2.77s다 |

P1 쪽을 읽을 때 주의할 점이 있다. 시간 초과 문항은 호출 기록을 수거하지 못해 `n_calls=0`으로 남는다(EVALUATION §4). 그래서 P1의 호출 수 1.75는 실제보다 낮다. 또 이 P1 로그는 설정 스냅샷이 다른 예전 라운드다. B-5 리허설에서 같은 설정으로 다시 돌린 P1과 비교해야 최종 수치가 된다.

## 2단계: 원인 — 근거는 있는데 판정이 답을 확정하지 못한다

SPEC §7의 유형별 목표 호출 수는 SQuAD 2회, HotpotQA 3~4회다. 그런데 SQuAD 답 있음이 **3.1회**다. 한 문단짜리 질문에서 재검색을 하고 있다는 뜻이다. 호출 초과와 과잉 기권은 같은 원인에서 나온다. 판정이 RETRIEVE → RETRIEVE → ABSTAIN으로 예산(4회)을 다 쓰고 기권하는 경로가 많다.

- **근거는 모였다.** HotpotQA 근거 문장 보존율(`evidence_kept`)은 0.90이다. M의 첫 검색어는 질문 원문이고 검색·압축은 모델을 부르지 않는 결정적 과정이다. 그래서 1차 근거는 로그 없이도 다시 만들 수 있다. 실제 설정(`k=5, m=7`)으로 다시 만들어 보면, 답이 있는 검색 문항 30개 중 **24개는 1차 근거에 이미 정답 문자열이 있다**. 그러나 r2 정답은 5개다. 적어도 19문항은 근거에 답이 있는데도 놓쳤다.
- **실패 모양** (C-4 로그 `results/runs/dev/c4_judge_format/json.jsonl`로 근거를 다시 만들어 확인):
  - 빠진 정보 칸에 답 자체를 쓰고 재검색한다. `hotpotqa-0040`은 근거에 "a disciple of Aristotle and Theophrastus"가 있는데 후속 검색어로 `Aristotle`를 썼다. `hotpotqa-0046`(Montana State University)과 `squad2-0044`(First Amendment)도 같다.
  - 답이 원문 그대로 있어도 기권한다. `squad2-0041`의 근거는 "Shirley left garrisons at Oswego"인데 RETRIEVE → ABSTAIN했다. `squad2-0047`(lower levels of inequality)도 같다.
  - 두 문단을 합쳐야 하는 질문에서 기권한다. `hotpotqa-0056`은 두 문단 모두 "restaurant chain"이라고 쓰는데 기권했다.
- C-4 "함께 발견한 것 1"과 같은 결과다. 그 실험은 `k=3, m=3`, 이번 실행은 `k=5, m=7`이다. 근거 양이 늘어도 판정의 문제는 그대로 남았다.

결론: 가설의 틀(라우터 + 같은 호출 안의 판정)은 유지한다. 틀린 것은 **판정 지시문의 경계**다. ANSWER 기준("the evidence states the answer")이 너무 좁고, RETRIEVE 기준이 "답을 댈 수 있는가"와 무관하게 적혀 있다.

## 3단계: 근거 판정 프롬프트 v2 (r3)

| 조건 | 설명 |
|---|---|
| 기준 (v1, r2) | `69b824e`의 판정 지시문 |
| 변경 (v2, r3) | 아래 세 자리만 바꿨다. ABSTAIN 정의, 공통 답 형식 지시문(`CONTENT_INSTRUCTION`), 스키마·키 순서, 라우터는 그대로다 |

| 자리 | v1 | v2 |
|---|---|---|
| ANSWER | the evidence states the answer. | the evidence gives the answer, in one sentence or by combining facts from the passages. Choose this whenever you can name the answer from the evidence, even if it is worded differently from the question. |
| RETRIEVE | the evidence is related to the question, but one specific fact needed for the answer is not in it and could be found by another search. | you cannot name the answer yet, because one specific fact needed for it is not in the evidence and could be found by another search. |
| 빠진 정보 | …(used as the next search query) | 같음 + Never write a candidate answer here |
| 서술 | one or two sentences on whether the evidence answers the question. | one or two sentences: quote the part of the evidence that answers the question, or say which fact is missing. |

- 태그 형식(대조군)도 같은 서술 지시를 쓴다. 두 형식이 출력 형식만 달라야 C-4 비교가 성립하므로 테스트로 묶었다 (`tests/test_judge.py`).
- 세 자리를 한 번에 바꿨으므로 어느 문장이 효과를 냈는지는 이 실험으로 가를 수 없다. M 구성 동결(11/27) 전에 한 번 더 돌릴 시간만 있어서 묶어서 시험한다.

바뀐 것 외에 모든 조건(모델 태그, 설정 스냅샷, 개발 세트·코퍼스 해시)이 같은지 확인한다: [ ] (r3 `RUN_META.json`의 `settings`가 r2와 같은지 실행 후 체크)

### 목표 지표와 채택 기준 (실행 전에 정함)

- 목표 지표: 과잉 기권률, 문항당 호출 수 (워밍업 제외)
- 채택 기준 (SPEC §6과 같음): 유형별 정답 수가 줄지 않고 목표 지표가 좋아지면 v2 채택
- 따로 볼 것: SQuAD 답할 수 없음의 정답 수. 기권을 덜 하게 만들면 여기가 먼저 나빠진다
- 확인: 수학 경로는 바뀌지 않았으므로 GSM8K 결과가 r2와 같아야 한다

### 결과 (r3 실행 후 기입)

| 유형 | v1 정답 | v2 정답 | 새로 맞힘 | 새로 틀림 | 순정답 수 |
|---|---|---|---|---|---|
| HotpotQA (20) | 3 | | | | |
| SQuAD 답 있음 (10) | 2 | | | | |
| SQuAD 답할 수 없음 (10) | 7 | | | | |
| GSM8K (20) | 17 | | | | |

| 지표 | v1 | v2 | 변화 |
|---|---|---|---|
| 과잉 기권률 | 0.38 | | |
| 호출 수/문항 | 2.32 | | |
| 시간/문항 | 5.81s | | |
| 판정 분포 (ANSWER / RETRIEVE / ABSTAIN) | 14 / 35 / 26 | | |
| 근거에 정답 있는데 정답 아님 (답 있는 검색 문항) | ≥ 19 (§13에서 확정) | | |
| 잘림 · 파싱 실패 · 예산 소진 | 0 · 0 · 0 | | |

정답→오답으로 바뀐 문항: (qid와 짧은 원인)

## 결정

(r3 실행 후) 채택 / 기각 / 재실험 — 이유 한두 문장.

## 이 결과를 읽을 때 주의할 점

- 개발 세트 60문항이다. 정확도 차이의 유의성은 평가 세트 120문항에서 본다.
- "근거에 정답 문자열 있음"이 그 근거만으로 답할 수 있다는 보장은 아니다. 특히 HotpotQA 다리형이 그렇다.
- 호출 수 가설에는 구조적인 하한이 있다. 라우터가 모든 문항에서 1회를 쓰고 검색 문항은 판정 1회가 더 든다. 그래서 모든 검색 문항이 첫 판정에서 끝나도 평균은 (20×1 + 40×2)/60 ≈ 1.67이다. P1의 1.75(실제보다 낮게 기록됨)보다 낮아지려면 재검색이 거의 없어야 한다. 판정을 고쳐도 호출 수 가설은 아슬아슬하다.

## 후속 작업

- (A) r3 결과를 위 표에 채우고 결정을 적는다. 채택하면 M 구성 동결(11/27)에 v2를 넣는다.
- (B) B-5 리허설에서 같은 설정 스냅샷으로 P0·P0-raw·P1을 새 라운드로 다시 돌린다. P1의 시간 초과 4건(`hotpotqa-0040`, `hotpotqa-0049`, `squad2-0045`, `squad2-0051`)과 GSM8K `token repeat limit` 오류 2건의 원인도 함께 본다.
- (B/A) `RunMeta`는 이어 돌 때 설정 스냅샷만 비교하고 git 커밋은 비교하지 않는다. 프롬프트처럼 설정 밖에 있는 변경은 같은 라운드에 섞일 수 있다. 막을지 팀에서 정한다.
- (C) 판정 지시문 변경 리뷰. v2를 채택하면 C-4의 형식 비교를 v2로 다시 돌릴지 정한다.
