# M 통합 dev 실행(A-5b)과 근거 판정 프롬프트 v2

- 날짜: 2026-11-26
- 담당: A (통합). 근거 판정(`src/meba/judge.py`)은 C 담당 모듈이라 PR 설명에 수정 이유를 쓴다
- 관련 부품 / 설정 키: 근거 판정 지시문 (`_DECISION_HELP`, `_MISSING_HELP`, `_REASONING_HELP`), `judge.format`=json
- 실행 로그: `results/runs/dev/r2/m.jsonl` (v1), `results/runs/dev/r3/m.jsonl` (v2), `results/runs/dev/r4/m.jsonl` (v3), 비교 대상 P1은 `results/runs/dev/r1/p1.jsonl`
- 실행: `notebooks/verify_a5b.ipynb` §8(r2), §13~15(진단·r3·비교), §18~19(r4·비교) (VS Code + Colab T4 커널)
- git 커밋: r2 = `69b824e`, r3 = `9d8f0a4`, r4 = `5924f88`
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

### 결과 (r3, 커밋 `9d8f0a4`)

| 유형 | v1 정답 | v2 정답 | 새로 맞힘 | 새로 틀림 | 순정답 수 |
|---|---|---|---|---|---|
| HotpotQA (20) | 3 | 5 | 2 | 0 | +2 |
| SQuAD 답 있음 (10) | 2 | 5 | 4 | 1 | +3 |
| SQuAD 답할 수 없음 (10) | 7 | 4 | 0 | 3 | **−3** |
| GSM8K (20) | 17 | 17 | 0 | 0 | 0 (답 60/60 동일) |

| 지표 | v1 | v2 | 변화 |
|---|---|---|---|
| 정답 | 29/60 | 31/60 | +2 (McNemar 정확 p = 0.754) |
| 과잉 기권률 | 0.38 | 0.18 | −0.20 |
| 호출 수/문항 | 2.32 | 1.90 | −18% |
| 시간/문항 | 5.81s | 4.83s | −17% |
| 판정 분포 (ANSWER / RETRIEVE / ABSTAIN) | 14 / 35 / 26 | 27 / 10 / 13 | 재검색 고리 감소 |
| 1차 근거에 정답 있는 24문항의 첫 판정 (ANSWER / RETRIEVE / ABSTAIN) | 7 / 12 / 5 | 17 / 4 / 3 | |
| 근거에 정답 있는데 정답 아님 (답 있는 검색 문항) | 19 (기권 14, 오답 5) | 15 (기권 6, 오답 9) | |
| 오류 · 예산 소진 | 0 · 0 | 0 · 0 | |

정답→오답으로 바뀐 문항
- `squad2-0050`, `squad2-0053`, `squad2-0059` (답할 수 없음): 근거에 비슷하지만 질문과 한 부분이 어긋나는 문장이 있다. 각각 순서(영양분 획득 "전"의 역할 ↔ 영양분 획득 수단으로 진화), 행동(motivate ↔ suppress), 역할(participants ↔ observers)이 다르다. v2는 그 문장을 답으로 냈다.
- `squad2-0043` (답 있음): 정답 문장의 일부(`every year between 1346 and 1671`)만 답해 EM이 틀렸다. 기권과는 무관하고, 답의 범위 문제다.

## 결정 (v2)

**채택 기준 미달 → 재실험(v3).** 목표 지표(과잉 기권 0.38 → 0.18, 호출 2.32 → 1.90)는 좋아졌다. 그러나 SQuAD 답할 수 없음 정답이 7 → 4로 줄어, 실행 전에 정한 "유형별 정답 수가 줄지 않음"을 지키지 못했다. 기준을 사후에 바꾸지 않고, 실패 원인(질문과 어긋나는 비슷한 문장을 답으로 냄)만 겨냥한 v3를 같은 기준으로 시험한다.

## 4단계: 근거 판정 프롬프트 v3 (r4)

| 조건 | 설명 |
|---|---|
| 기준선 | v1 (r2, 현재 M). 채택 판정은 v1과 비교한다 |
| 변경 (v3, r4) | v2에 두 가지만 더했다. ANSWER·RETRIEVE·빠진 정보 지시는 v2 그대로다 |

| 자리 | v2 | v3 |
|---|---|---|
| ABSTAIN | the evidence does not contain the answer and another search would not help. | the evidence does not contain the answer, **or it only says something similar that does not fit the question (for example a different action, role, order of events or condition)**, and another search would not help. |
| 서술 | quote the part of the evidence that answers the question, or say which fact is missing. | quote the part of the evidence that answers the question **and check that it fits every detail of the question**, or say which fact is missing. |

### 채택 기준 (실행 전에 정함)

- v3가 **유형별 정답 수에서 v1보다 줄지 않고**, 과잉 기권률과 호출 수(워밍업 제외)가 v1보다 좋아지면 채택한다. 노트북 §19가 이 세 조건을 자동으로 판정한다.
- 셋 다 통과하지 못하면 v1(현재 M)과 v2 중 무엇으로 M 구성을 동결할지(11/27) 팀이 정한다. 이 실험에서 v2를 사후 채택하지 않는다.
- 같은 개발 세트 60문항으로 세 번째 시험하는 프롬프트다. 결과는 개발 세트에 맞춰진 쪽으로 기울 수 있으므로 효과는 평가 세트 결과로만 주장한다.

### 결과 (r4, 커밋 `5924f88`)

r2·r3·r4의 `RUN_META.json` 설정 스냅샷이 모두 같다. 세 라운드는 판정 지시문(커밋)만 다르다.

| 유형 | v1 정답 | v2 정답 | v3 정답 |
|---|---|---|---|
| HotpotQA (20) | 3 | 5 | 3 |
| SQuAD 답 있음 (10) | 2 | 5 | 5 |
| SQuAD 답할 수 없음 (10) | 7 | 4 | 4 |
| GSM8K (20) | 17 | 17 | 17 (답 60/60 동일) |
| **전체** | 29 | **31** | 29 |

| 지표 | v1 | v2 | v3 |
|---|---|---|---|
| 과잉 기권률 | 0.38 | 0.18 | 0.26 |
| 호출 수/문항 | 2.32 | 1.90 | 1.78 |
| 시간/문항 (평균 · 중앙값) | 5.81s | 4.83s · 3.16s | 4.59s · 3.26s |
| 판정 분포 (ANSWER / RETRIEVE / ABSTAIN) | 14 / 35 / 26 | 27 / 10 / 13 | 23 / 1 / 17 |

- v1 → v3: 새로 맞힘 4, 새로 틀림 4 (p = 1.00). v2 → v3: 새로 맞힘 1, 새로 틀림 3 (p = 0.63)
- **v3가 겨냥한 세 문항은 고쳐지지 않았다.** `squad2-0050`·`0053`·`0059`를 v3도 답했다. 대신 v2가 맞히던 답 있음 문항 3개(`hotpotqa-0046`, `hotpotqa-0050`, `squad2-0041`)를 기권했다. RETRIEVE도 거의 사라졌다(10 → 1).

답이 있는 검색 문항 30개의 실패 단계 (로그에서 근거를 복원해 분류):

| 단계 | v1 | v2 | v3 | 책임 부품 |
|---|---|---|---|---|
| 정답 | 5 | 10 | 8 | |
| 근거에 정답 없음 | 6 | 5 | 6 | 검색·압축 |
| 근거에 정답 있는데 기권 | 14 | 6 | 8 | 근거 판정 |
| 근거에 정답 있는데 답 범위가 어긋남 (F1 ≥ 0.5) | 4 | 5 | 5 | 답 형식 (모든 파이프라인 공통) |
| 근거에 정답 있는데 오답 (F1 < 0.5) | 1 | 4 | 3 | 근거 판정 |

답 범위 어긋남은 `Pac-12 Conference` vs `Pac-12`, `Bhabananda Deka` vs `Prof Bhabananda Deka` 같은 경우다. 세 버전에서 거의 같은 문항이므로 판정 지시문과 무관하다.

## 결정 (v3)

**채택 기준 미달 → 기각.** 과잉 기권과 호출 수는 v1보다 좋아졌지만, SQuAD 답할 수 없음이 7 → 4로 줄었다.

v2·v3 모두 기준을 통과하지 못했으므로, 미리 정한 대로 **M 구성 동결(11/27) 때 v1과 v2 중 하나를 팀이 고른다.** 판단 재료는 아래와 같다.

| | v1 (현재 M, 기준을 통과한 유일한 버전) | v2 |
|---|---|---|
| 정답 | 29 | 31 |
| SQuAD 답할 수 없음 | 7 | 4 |
| 과잉 기권 | 0.38 | 0.18 |
| 호출 수 (P1 r1: 1.75) | 2.32 | 1.90 |
| 근거 | 실행 전 규칙의 기본값 | 가설 지표(정확도·호출 수)에 더 가깝다. 고르면 "규칙에서 벗어난 선택"으로 이유를 남긴다 |

**이 실험이 보여 준 것:** 지시문으로는 7B 모델의 기권 **임계값**만 움직였고, 답이 있는 근거와 함정 근거를 가르는 **판별력**은 늘지 않았다.
- 문구를 바꿀 때마다 답 있음과 답할 수 없음이 서로 맞바뀌었다.
- 세 버전의 전체 정답은 29 → 31 → 29로, 차이가 우연 범위 안이다.
- 판정 지시문 튜닝은 여기서 멈춘다. 같은 60문항으로 더 고치면 개발 세트 과적합이 된다.

## 이 결과를 읽을 때 주의할 점

- 개발 세트 60문항이다. 정확도 차이의 유의성은 평가 세트 120문항에서 본다.
- "근거에 정답 문자열 있음"이 그 근거만으로 답할 수 있다는 보장은 아니다. 특히 HotpotQA 다리형이 그렇다.
- 호출 수 가설에는 구조적인 하한이 있다. 라우터가 모든 문항에서 1회를 쓰고 검색 문항은 판정 1회가 더 든다. 그래서 모든 검색 문항이 첫 판정에서 끝나도 평균은 (20×1 + 40×2)/60 ≈ 1.67이다. P1의 1.75(실제보다 낮게 기록됨)보다 낮아지려면 재검색이 거의 없어야 한다. 판정을 고쳐도 호출 수 가설은 아슬아슬하다.

## 후속 작업

- (A) r3 결과를 위 표에 채우고 결정을 적는다. 채택하면 M 구성 동결(11/27)에 v2를 넣는다.
- (B) B-5 리허설에서 같은 설정 스냅샷으로 P0·P0-raw·P1을 새 라운드로 다시 돌린다. P1의 시간 초과 4건(`hotpotqa-0040`, `hotpotqa-0049`, `squad2-0045`, `squad2-0051`)과 GSM8K `token repeat limit` 오류 2건의 원인도 함께 본다.
- (B/A) `RunMeta`는 이어 돌 때 설정 스냅샷만 비교하고 git 커밋은 비교하지 않는다. 프롬프트처럼 설정 밖에 있는 변경은 같은 라운드에 섞일 수 있다. 막을지 팀에서 정한다.
- (C) 판정 지시문 변경 리뷰. v2를 채택하면 C-4의 형식 비교를 v2로 다시 돌릴지 정한다.
