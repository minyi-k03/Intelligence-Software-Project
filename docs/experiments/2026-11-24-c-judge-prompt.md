# 근거 판정 프롬프트 수정안 (과잉 기권 줄이기) — 기각

- 날짜: 2026-11-24
- 담당: C (조환호)
- 관련 부품 / 설정 키: 근거 판정 (`src/meba/judge.py`의 판정 지시문). 공통 문구(`CONTENT_INSTRUCTION`)와 설정은 바꾸지 않았다
- 실행 로그: `results/runs/dev/c_judge_prompt/` (`baseline/`, `revised/`, `compare_json.md`)
- 실행: `notebooks/dev_c_judge_prompt.ipynb` → `python -m scripts.compare_judge_format --formats json` (Colab T4, Ollama 0.40.17)
- git 커밋: 기준선 `c7fe41c`(PR #21 병합 직후 `main`), 수정안 `8f7a449`(`HwanHo`). 수정안은 `0422208`에서 되돌렸다
- 데이터: `data/frozen/dev.jsonl` (`8f18f76d…`), `data/frozen/corpus.jsonl`

## 결론

**수정안을 기각한다.** 목표였던 과잉 기권이 거의 줄지 않았고(57% → 53%), 정답 수가 12 → 10으로 줄었으며 특히 SQuAD 답할 수 없음이 7 → 5로 나빠졌다. 채택 기준을 넘지 못했다.

이후 팀은 용민(A)이 같은 문제를 M 전체 파이프라인에서 시험한 **v2로 M 구성을 동결**했다(PR #24, [2026-11-26-a-m-dev-judge-prompt.md](2026-11-26-a-m-dev-judge-prompt.md)). 두 실험의 결론은 같다: **판정 문구는 기권 문턱만 옮기고, 답이 있는 문항과 없는 문항을 더 잘 구분하지는 못한다.** 같은 개발 세트로 반복 튜닝하지 않기 위해 이 수정안을 v2의 후보(v4)로 다시 돌리지 않는다.

## 질문

C-4([2026-11-23-c-judge-format.md](2026-11-23-c-judge-format.md))에서 근거에 정답이 있는데도 기권하는 문항이 많았다(JSON, 답이 있는 30문항 중 11). 판정 지시문만 고쳐서 이 과잉 기권을 줄일 수 있는가.

## 수정안을 만든 근거 (C-4 JSON 로그 분석, k=3·m=3)

답이 있는 30문항 중 기권 20문항을 원인별로 나눴다. "근거에 정답 있음"은 압축이 남긴 문장 창에 정답 문자열이 들어 있는지로 셌다.

| 원인 | 문항 | 예 |
|---|---|---|
| 1차 근거에 정답이 있는데 첫 판정에서 바로 기권 | 3 | 비교형 질문(`hotpotqa-0051` "director and screenwriter")처럼 답이 문장에 그대로 적혀 있지 않은 경우 |
| 1차 근거에 정답이 있는데 정답 자체를 빠진 정보로 적고 재검색 → 기권 | 3 | `hotpotqa-0040`(정답 Aristotle)의 후속 검색어가 `Aristotle` |
| 1차 근거에 정답이 있는데 확인하려고 재검색 → 마지막 판정에서 기권 | 4 | `hotpotqa-0048`, `squad2-0041` |
| 재검색으로 정답을 찾고도 마지막 판정에서 기권 | 1 | `hotpotqa-0053` |
| 근거에 정답이 없음 (검색·압축 부족) | 9 | 당시 임시값 `m=3`의 영향. 판정 문구로 고칠 대상이 아님 |

## 비교 조건

| 조건 | 설명 |
|---|---|
| 기준선 | 원래 판정 지시문 v1 (C-3에서 작성, `c7fe41c`) |
| 변경 | 수정안 (`8f7a449`). 판정 전용 문구만 바꿈 |
| 같은 것 | 공통 문구(PR #21 이후 `CONTENT_INSTRUCTION`), `retrieval.k` 5, `compress.m` 7, `compress.window` 1, `cap.judge` 256, `budget.max_requery` 2, JSON 형식, 개발 세트 검색형 40문항(HotpotQA 20, SQuAD 답 있음 10, 답할 수 없음 10) |
| 경로 | C-4와 같은 하네스(`scripts/compare_judge_format.py`). 그래프 없이 검색 → 압축 → 근거 판정 → 재검색을 직접 잇고, 라우터는 부르지 않는다. 문항은 러너의 `run_item`(120초 제한, 워밍업 표시)으로 돌렸다 |

바뀐 것(판정 지시문) 외에 모든 조건이 같은지 `RUN_META.json`으로 확인했다: [x]
(수정안 실행의 `git_dirty`가 `true`인 것은 Colab 작업 폴더에 먼저 끝난 기준선 결과 폴더가 있었기 때문으로 보인다. 코드 변경은 없었다)

### 바꾼 문구

| 자리 | 기준선 (v1) | 수정안 |
|---|---|---|
| ANSWER | the evidence states the answer. | the evidence states the answer, **or the answer can be worked out by combining or comparing facts** in the evidence. |
| RETRIEVE | the evidence is related to the question, but one specific fact needed for the answer is not in it and could be found by another search. | **the answer cannot be found or worked out from the evidence yet**, but the evidence names a specific person, place, work or thing whose missing fact another search could find. |
| ABSTAIN | the evidence does not contain the answer and another search would not help. | the evidence does not contain the answer and another search is **unlikely** to find it, **for example the evidence is on the topic but never mentions what the question asks.** |
| 첫 판정 추가 지시 | 없음 | If you can already name the answer from the evidence, choose ANSWER. **Do not search again only to confirm an answer you can already give.** |
| 마지막 판정 추가 지시 | 없음 | **No more searches are possible.** Choose ANSWER if the evidence states the answer or it can be worked out from the evidence; otherwise choose ABSTAIN. |
| 빠진 정보 | the missing fact, naming the people, places, works or terms involved (used as the next search query) | what is still unknown, **written as a short search query** about a person, place, work or thing named in the question or evidence. **Do not put a guessed answer in it** |
| 서술 | one or two sentences on whether the evidence answers the question. | one or two sentences: **point to the evidence that gives the answer**, or say what is missing. |

## 목표 지표와 채택 기준

- 목표 지표: 과잉 기권률(답이 있는 문항에서 기권한 비율) 감소
- 채택 기준: 유형별 정답 수가 줄지 않고 과잉 기권률이 줄 것 (조건부 부품의 기준 SPEC §6을 준용)

## 결과

판정 호출/문항과 출력 토큰/문항은 첫 문항(워밍업)을 뺀 39문항 평균이다. 나머지는 40문항 전체다. 두 실행 모두 잘림·파싱 실패·시간 초과·예산 소진·오류 0이다.

| 그룹 | 프롬프트 | 문항 | 정확도 | 텍스트 F1 | 과잉 기권률 | 판정 호출/문항 | 출력 토큰/문항 |
|---|---|---|---|---|---|---|---|
| 전체 | 기준선 | 40 | 30% | 0.29 | 57% | 1.85 | 116 |
| 전체 | 수정안 | 40 | 25% | 0.30 | 53% | 1.28 | 81 |
| HotpotQA | 기준선 | 20 | 15% | 0.22 | 60% | 1.89 | 125 |
| HotpotQA | 수정안 | 20 | 10% | 0.19 | 60% | 1.32 | 89 |
| SQuAD 답 있음 | 기준선 | 10 | 20% | 0.42 | 50% | 2.10 | 126 |
| SQuAD 답 있음 | 수정안 | 10 | 30% | 0.52 | 40% | 1.40 | 83 |
| SQuAD 답할 수 없음 | 기준선 | 10 | 70% | - | - | 1.50 | 90 |
| SQuAD 답할 수 없음 | 수정안 | 10 | 50% | - | - | 1.10 | 65 |

| 유형별 정답 수 | 기준선 | 수정안 | 변화 |
|---|---|---|---|
| HotpotQA | 3 | 2 | −1 |
| SQuAD 답 있음 | 2 | 3 | +1 |
| SQuAD 답할 수 없음 | 7 | 5 | −2 |
| 전체 | 12 | 10 | −2 |

같은 문항 비교: 둘 다 정답 9, 기준선만 정답 3, 수정안만 정답 1, 둘 다 오답 27 (McNemar 정확 검정 p = 0.62).

첫 판정 분포 (40문항)

| 프롬프트 | ANSWER | RETRIEVE | ABSTAIN |
|---|---|---|---|
| 기준선 | 9 | 22 | 9 |
| 수정안 | 16 | 10 | 14 |

### 정답이 바뀐 문항

| 문항 | 정답 | 기준선 | 수정안 |
|---|---|---|---|
| `hotpotqa-0046` (bridge) | Montana State University | RETRIEVE → ANSWER, 정답 | RETRIEVE → ABSTAIN, 오답 |
| `squad2-0045` (답 있음) | very rare | RETRIEVE → RETRIEVE → ABSTAIN, 오답 | ANSWER, 정답 |
| `squad2-0050` (답할 수 없음) | UNANSWERABLE | RETRIEVE → ABSTAIN, 정답 | ANSWER "acquiring nutrients", 오답 |
| `squad2-0053` (답할 수 없음) | UNANSWERABLE | ABSTAIN, 정답 | ANSWER "to suppress an Islamic rebellion", 오답 |

### 답이 있는 30문항의 결과 분류

| | 기준선 | 수정안 |
|---|---|---|
| 정답 | 5 | 5 |
| 오답 (근거에 정답 있음 / 없음) | 7 / 1 | 8 / 1 |
| 기권 (근거에 정답 있음 / 없음) | 12 / 5 | 11 / 5 |
| 첫 판정이 바로 ABSTAIN | 6 | 10 |

(위 표의 정답 5는 텍스트 문항만 센 것이다. 전체 정답 12·10에는 답할 수 없음 정답 7·5가 들어 있다.)

## 왜 실패했나

1. **목표를 건드리지 못했다.** 근거에 정답이 있는데 기권한 문항이 12 → 11로 거의 그대로다.
2. **재검색 대신 바로 기권하게 됐다.** RETRIEVE 조건을 좁히자(근거에 나온 대상의 사실만 찾을 것) 첫 판정 RETRIEVE가 22 → 10으로 줄고, 답이 있는 문항에서 첫 판정이 바로 ABSTAIN인 경우가 6 → 10으로 늘었다. 판정 호출 −31%, 출력 토큰 −30%는 이 변화 때문이라 비용 개선으로 보지 않는다.
3. **답을 내라는 쪽으로 문턱을 옮기자 답할 수 없음 문항에서 답을 지어냈다.** `squad2-0050`, `squad2-0053`. ABSTAIN 예시("주제는 맞지만 묻는 것을 언급하지 않을 때")를 넣었지만 막지 못했다.

## 이 결과를 읽을 때 주의할 점

- 바뀐 문항이 4개라 차이는 우연 범위 안이다(p = 0.62). "수정안이 나쁘다고 증명됐다"가 아니라 "나아졌다는 근거가 없고 채택 기준을 넘지 못했다"로 읽는다.
- 기준선은 v1이다. 팀이 동결한 v2와는 직접 비교하지 않았다. 용민의 실험은 M 전체 파이프라인(라우터 포함, 60문항)이고 이 실험은 검색형 40문항 하네스라, 숫자를 서로 옮겨 비교하지 않는다.
- 동작 확인(2문항, `results/runs/_scratch/`)에서 첫 문항이 120초 시간 초과로 오답이 됐다. 모델을 처음 메모리에 올리는 시간으로 추정한다(확인은 못 함). 본 실행 두 번은 시간 초과 0이다. 본 측정의 첫 문항(워밍업)은 정확도에 들어가므로, 측정 전에 모델을 한 번 불러 두는 것을 A에게 제안한다.

## 결정

**기각.** `src/meba/judge.py`는 수정안 이전으로 되돌렸고(`0422208`), 이후 PR #24의 v2를 그대로 쓴다. 이 실험으로 바뀐 코드·설정은 없다.

후속 작업:

- (C) 남은 과잉 기권은 판정 문구가 아니라 근거 장부 상세화(C-5)처럼 구조를 바꾸는 쪽에서 다룬다. v2 실험 기록도 같은 판단이다
- (A) 본 측정 전 모델 미리 불러 두기 (첫 문항 시간 초과 방지)
- 재현: `notebooks/dev_c_judge_prompt.ipynb`는 기준선 `c7fe41c`, 수정안 `8f7a449`를 직접 지정해 돌린다
