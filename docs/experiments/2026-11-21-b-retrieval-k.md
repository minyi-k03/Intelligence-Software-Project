# 검색 결과 수 k 확정 및 10문항 생성 속도·P1 개발 세트 실측 (실험 B-4)

- 날짜: 2026-11-21 (P1 개발 세트 로그 재생성: 2026-11-23)
- 담당: B (이현성)
- 관련 부품 / 설정 키: BM25 문단 검색(`retrieval.k`), 검색 결과 압축(`compress.m` — 실험 C-2 표 연동), 기준선 P1 ReAct
- 실행 로그: `results/runs/dev/r1/p1.jsonl` (`python -m scripts.measure_retrieval_k` 및 Colab T4 `python -B -m scripts.run_eval --bundle dev --round r1 --pipelines p1`)
- git 커밋: `HyunSeong` @ `3acd6fb` (`RUN_META.json.git_commit = 3acd6fb57a3514489e8df0db0f9e6a8bc6bd781b`, `git_dirty = false`)
- 데이터: `data/frozen/dev.jsonl` (`8f18f76d…`), `data/frozen/corpus.jsonl` (`d562adc1…`)

## 질문

1. 개발 세트 검색 대상 40문항(HotpotQA 20 + SQuAD 2.0 20)에서 1차 검색 재현율과 문맥 길이를 고려할 때 가장 적합한 `retrieval.k`와 그에 대응하는 `compress.m`([실험 C-2](2026-11-15-c-compress-m.md))은 얼마인가?
2. 확정한 설정(`k`, `m`)으로 Colab GPU(T4)에서 10문항 실측 시 문항당 소요 시간과 1라운드 예상 시간은 얼마이며, P1 개발 세트 60문항 로그는 규격대로 생성되는가?

## 1단계: `retrieval.k` 오프라인 검색 재현율 측정 (모델 호출 없음)

재현 명령:
```bash
python -m scripts.measure_retrieval_k
```

바뀐 것 외에 모든 조건(동결 개발 세트·코퍼스 해시, 공용 토크나이저 `BM25_K1=1.5`, `BM25_B=0.75`)이 같은지 확인했다: [x]

### 1.1 후보 `k`별 1차 검색 재현율 (개발 세트 검색 문항 40개, 코퍼스 653문단)

| `k` | 전체 `any_gold` (40) | 전체 `all_gold` (40) | 정답 문단 회수율 (60) | SQuAD 2.0 (20) | HotpotQA 1개+ (20) | HotpotQA 2개 모두 (20) | bridge 2개 (10) | comparison 2개 (10) |
|---|---|---|---|---|---|---|---|---|
| 3 | 38/40 (95.0%) | 27/40 (67.5%) | 47/60 (78.3%) | 18/20 (90.0%) | 20/20 (100.0%) | 9/20 (45.0%) | 6/10 | 3/10 |
| **5** | **40/40 (100.0%)** | **31/40 (77.5%)** | **51/60 (85.0%)** | **20/20 (100.0%)** | **20/20 (100.0%)** | **11/20 (55.0%)** | **6/10** | **5/10** |
| 8 | 40/40 (100.0%) | 34/40 (85.0%) | 54/60 (90.0%) | 20/20 (100.0%) | 20/20 (100.0%) | 14/20 (70.0%) | 6/10 | 8/10 |
| 10 | 40/40 (100.0%) | 37/40 (92.5%) | 57/60 (95.0%) | 20/20 (100.0%) | 20/20 (100.0%) | 17/20 (85.0%) | 7/10 | 10/10 |

- `any_gold`: 정답 근거 문단 중 **최소 1개 이상**이 상위 `k`개 안에 들어온 문항 비율 (단일 홉 및 다중 홉 1차 출발점 확보율)
- `all_gold`: 정답 근거 문단이 **모두** 1차 검색 상위 `k`개 안에 들어온 문항 비율 (HotpotQA는 정답 문단 2개 모두)

### 1.2 문맥 길이 및 실험 C-2 권장 `compress.m` 결합 효과 (40문항 평균)

| `k` | C-2 권장 `m` | P1 관찰문 글자 수 | M 압축 전 글자 수 | M 압축 후 글자 수 | 글자 수 감소율 | HotpotQA 끝단 문장 보존 (47) |
|---|---|---|---|---|---|---|
| 3 | 6 | 1609 | 1600 | 1148 | 28.3% | 32/47 (68.1%) |
| **5** | **7** | **2616** | **2602** | **1373** | **47.2%** | **34/47 (72.3%)** |
| 8 | 10 | 4368 | 4346 | 1876 | 56.8% | 37/47 (78.7%) |
| 10 | - | 5780 | 5752 | - | - | - |

### 1.3 `k` · `m` 분석 및 선택 근거

1. **`k=3` (임시값) 기각 이유:**
   - 단일 문단만 찾으면 되는 **SQuAD 2.0에서조차 2문항(10%)을 놓치며(`18/20`)**, HotpotQA `comparison` 유형은 1차 검색에서 두 대상을 함께 찾는 비율이 `3/10`에 불과하다.
   - 또한 [실험 C-2](2026-11-15-c-compress-m.md)에서 확인했듯 `k=3, m=6`은 문단 대부분을 그대로 남기므로 압축 효과가 작다.
2. **`k=5, m=7` 채택 이유 (최적 균형점):**
   - `k=5`에서 **SQuAD 2.0 재현율 `20/20 (100.0%)`** 및 **전체 문항 1차 홉 확보율(`any_gold`) `40/40 (100.0%)`** 에 도달한다.
   - HotpotQA `bridge` 유형의 1차 검색 2문단 동시 회수율은 `k=3, 5, 8` 모두 `6/10`으로 동일하다(두 번째 문단은 1차 문단 안의 징검다리 엔티티를 알아야 찾을 수 있어 M의 **재검색** 부품이 담당하는 영역이다).
   - `k=5`에 실험 C-2 권장값 **`compress.m=7`** 을 결합하면 40문항 평균 문맥 길이가 `2602자 → 1373자`(**47.2% 감소**, HotpotQA 20문항 기준 40% 감소)로 줄어, `k=3`의 비압축 길이(`1600자`)보다도 더 짧은 입력으로 더 높은 근거 재현율을 얻는다.
3. **`k=8, m=10` 대비 이점:**
   - `k=8`은 압축이 없는 비교 기준선 **P1(ReAct)** 에서 검색 1회당 관찰문이 **평균 4,368자**로 급증하여, 다중 스텝(최대 7스텝) 반복 시 `num_ctx=8192` 압박과 추론 지연을 크게 유발한다.

### 1.4 스모크 테스트 중 발견한 공통 지시문(`CONTENT_INSTRUCTION`) 도구 호출 억제 현상 및 문구 정제

Colab(T4, `qwen2.5:7b-instruct-q4_K_M`, `temperature=0`)에서 초안 문구로 스모크 테스트를 실행한 결과, `CONTENT_INSTRUCTION`의 `"Do not explain."` + `"If the question cannot be answered from the available information, reply exactly: UNANSWERABLE"` 조합이 **P1의 도구 호출(`bm25_search`)과 수학 추론을 100% 억제**해 1턴째에 조기 기권(`calls=0`)하는 현상을 확인했다.

[INTERFACES.md §6](../INTERFACES.md)(*"아래 문구는 출발점이며, 문구를 다듬을 때도 의미 조건(번호 목록)은 유지합니다"*)에 따라 개발 세트 샘플 4문항(`hotpotqa-0040`, `squad2-0040`, `squad2-0050`, `gsm8k-0040`)에서 3가지 조건을 대조 실험했다:

| 조건 | `hotpotqa-0040` (Aristotle) | `squad2-0040` (after 1279) | `squad2-0050` (UNANSWERABLE) | `gsm8k-0040` (64) |
|---|---|---|---|---|
| ① 초안 `CONTENT_INSTRUCTION` + 기본 `P1_TOOL_GUIDANCE` (4문항 대조) | `calls=0` → `UNANSWERABLE` (조기 기권) | `calls=0` → `UNANSWERABLE` (조기 기권) | `calls=0` → `UNANSWERABLE` (미검색 기권) | `calls=0` → `128` (오답) |
| ② 초안 `CONTENT_INSTRUCTION` 유지 + `P1_TOOL_GUIDANCE`만 강화 (4문항 대조) | `calls=0` → `UNANSWERABLE` (조기 기권) | `calls=0` → `UNANSWERABLE` (조기 기권) | `calls=0` → `UNANSWERABLE` (미검색 기권) | `calls=0` → `128` (오답) |
| ③ 1차 정제 `CONTENT_INSTRUCTION` (`"inside the final answer line"`) + 강화 `P1_TOOL_GUIDANCE` (4문항 대조) | `calls=1` → `Aristotle` (정답) | `calls=1` → `1279` (검색 성공) | `calls=2` → `acquiring nutrients` (정상 검색 수행) | `64` (정답) |
| **④ 최종 커밋 `CONTENT_INSTRUCTION` (`3acd6fb`, `"in the final answer"`) + 강화 `P1_TOOL_GUIDANCE` (`r1/p1.jsonl` 실측)** | **`order_idx=0` 워밍업 `timeout` (120s)** *(나머지 HotpotQA 19문항 평균 1.89회 호출)* | **`searches=1, n_calls=2` → `1279` (검색 성공)** | **`searches=2, n_calls=2` → `acquiring nutrients` (정상 검색 수행)** | **`n_calls=2` → `64` (정답)** |

- **결론:** `P1_TOOL_GUIDANCE`만 바꿔서는(조건 ②) 초안 `CONTENT_INSTRUCTION`의 `"Do not explain."`과 `"from the available information"` 조합이 도구 호출을 계속 가로막는다(초기 프롬프트에 지문이 주어지지 않는 P0·P1 1턴째에 모델이 `"available information"`을 '현재 프롬프트에 주어진 지문'으로 해석해 검색 없이 곧바로 `UNANSWERABLE`을 출력함). 반면 M의 근거 판정(`src/meba/judge.py`)은 시스템 프롬프트 상단에 `"You answer a question using only the evidence passages given to you. Do not use outside knowledge."`와 `ABSTAIN` 조건(`"the evidence does not contain the answer..."`)을 이미 명시하고 있다. 따라서 줄 형식(`Final answer:` 줄 vs JSON `answer` 필드)에 무관하게 P0·P1·M 근거 판정에 모두 호환되도록 `"Do not include explanations in the final answer."` 및 `"If the question cannot be answered, reply exactly: UNANSWERABLE"`로 다듬어 `src/baselines/prompts.py`와 `docs/INTERFACES.md` §6에 반영했다.

---

## 2단계: Colab GPU(T4) 스모크 테스트 · 10문항 속도 실측 · P1 개발 세트 로그

- 측정 환경: Google Colab Tesla T4 (15,360 MiB), Driver `580.82.07`, CUDA `13.0`, `qwen2.5:7b-instruct-q4_K_M`, 최대 VRAM **`5,065 MiB`**

### 2.1 `create_agent` 스모크 테스트 (`--verify 3`)

- 실행 명령: `python -m scripts.run_eval --verify 3 --set dev --pipelines p0,p0_raw,p1`
- 확인 항목:
  - [x] `langchain>=1.0.0` `create_agent` 정상 동작 및 도구(`bm25_search`, `calculator`) 호출 확인
  - [x] `MetricsCallbackHandler` 호출당 1회 기록(중복 카운트 없음) 확인
  - [x] `GraphRecursionError` 및 타임아웃 예외 안전 처리 확인

### 2.2 10문항 실측 (생성 속도 및 라운드 시간 추정)

- 실행 명령: `python -B -m scripts.run_eval --verify 10 --round r10 --set dev --pipelines p0,p1` (앞 10문항: `hotpotqa-0040`~`0049` — **HotpotQA 전용**, 첫 1문항 워밍업 제외 9문항 비용 집계)

| 파이프라인 | 10문항 정답 (HotpotQA) | 문항당 평균 시간(초) | 문항당 중앙값 시간(초) | 문항당 평균 호출 수 | 문항당 평균 출력 토큰 | 비고 |
|---|---|---|---|---|---|---|
| P0 | 0/10 | 0.37s | 0.28s | 1.00 | 7.2 | 120문항 단순 외삽 약 0.7분 |
| P1 (`k=5`) | 3/10 | 3.54s | 2.71s | 2.00 | 65.4 | HotpotQA 정상 문항 기준 |

- **평가 세트 1라운드(120문항) 소요 시간 추정 (근거 분리):**
  - 최종 프롬프트(`3acd6fb`) 기준 60문항 P1 실측(§2.3)에서 워밍업(`hotpotqa-0040`, 120s 타임아웃) 및 비용 집계 구간 내 120초 타임아웃 3건(`hotpotqa-0049`, `squad2-0045`, `squad2-0051`)을 제외한 정상 56문항 평균 시간은 **3.08초/문항**(타임아웃 3건 포함 59문항 평균 9.02초/문항)으로, 10문항 HotpotQA 실측(3.54초/문항)과 비슷한 수준이다.
  - 따라서 **P1 120문항 추정 시간**은 **`정상 문항 시간 + 예상 타임아웃 수 × 120초`** 로 분리해 계산한다:
    - 정상 문항만 기준: `120 × 3.08초 ≈ 6.2분` (HotpotQA 10문항 3.54초 기준 약 `7.1분`)
    - 타임아웃 6~8건 가정(개발 세트 비율: 워밍업 제외 3/59 = 5.1%, 전체 4/60 = 6.7% 적용 시 약 6~8건): `6.2분 + (12~16분) ≈ 18.2~22.2분`
  - **P0**는 `120 × 0.37초 ≈ 0.7분`(P0-raw 포함 시 약 2~3분 내외)이며, **M은 아직 실측하지 않았으므로 A-5b 통합 후 개발 세트 측정 결과로 본 비교 1라운드 총 소요 시간을 갱신한다.**

### 2.3 P1 개발 세트 60문항 결과 요약 (`results/runs/dev/r1/p1.jsonl`, `3acd6fb`)

- 실행 명령: `python3 -B -m scripts.run_eval --bundle dev --round r1 --pipelines p1` (`git_commit = 3acd6fb5…`, `git_dirty = false`, `ollama = 0.40.17`, 최대 VRAM `5,065 MiB`)

| 유형 | 문항 수 | 정답 수 (정확도) | 평균 텍스트 F1 (`answer_kind="text"`) | 답할 수 없음 정답률 (`unanswerable`) | 평균 호출 수 | 평균 출력 토큰 | 평균 최대 입력 토큰 | 평균 시간(초) | 비고 |
|---|---|---|---|---|---|---|---|---|---|
| **전체** | 60 (워밍업 제외 59) | **21/60 (35.0%)** | **0.360** (30문항) | **4/10 (40.0%)** | 1.75 | 58.7 | 916.2 | 9.02s (정상 56문항 3.08s) | 타임아웃 4건(워밍업 1+본측정 3), Ollama 반복 중단 2건, 최대 VRAM 5,065 MiB |
| **HotpotQA** | 20 (비용 19) | **4/20 (20.0%)** | **0.273** (20문항) | - | 1.89 | 72.0 | 1290.6 | 9.72s (정상 18문항 3.60s) | `hotpotqa-0040`(워밍업)·`hotpotqa-0049` 120s 타임아웃 2건 |
| **SQuAD 2.0** | 20 | **8/20 (40.0%)** | **0.533** (`answerable` 10문항, EM 4/10) | **4/10 (40.0%)** | 1.55 | 38.3 | 1018.2 | 14.11s (정상 18문항 2.35s) | `squad2-0045`, `squad2-0051` 120s 타임아웃 2건 |
| **GSM8K** | 20 | **9/20 (45.0%)** | - (`numeric`) | - | 1.80 | 66.5 | 458.5 | 3.26s | `gsm8k-0051`, `gsm8k-0054` Ollama 반복 중단 오류 2건 (`token repeat limit reached`) |

- **P1 개발 세트 로그 핵심 관찰 (D-3 인계 사항):**
  1. **HotpotQA (20.0%, 텍스트 F1 0.273):** `hotpotqa-0040`(워밍업)·`hotpotqa-0049` 2건이 120초 타임아웃에 걸렸고, 정상 문항에서도 1차 검색 후 바로 답하는 경우가 많아 다중 홉 2번째 근거 문단을 놓치며, 압축 없이 5개 문단 전체를 넣으므로 최대 입력 토큰이 평균 `1,290.6`토큰에 달한다.
  2. **SQuAD 2.0 (전체 40.0% — `answerable` 4/10 [40.0%, 텍스트 F1 0.533], `unanswerable` 4/10 [40.0%]):** 출력 토큰 하드 상한(`num_predict`)과 루프 제어기가 없는 표준 P1에서 **`squad2-0045`, `squad2-0051` 2문항이 120초 타임아웃**에 걸려 평균 시간을 정상 18문항 `2.35s`에서 `14.11s`로 끌어올렸다. 또한 답할 수 없음 문항 10개 중 6개에서 함정 문단에 속아 오답을 내어 기권 정확도가 `4/10 (40.0%)`에 머물렀다.
  3. **GSM8K (45.0%):** 출력 상한이 없는 P1에서 **`gsm8k-0051`, `gsm8k-0054` 2문항이 Ollama 반복 중단 오류(`ResponseError: prediction aborted, token repeat limit reached`)** 로 실패했다(A-4 라우터 실험의 `gsm8k-0053` 사례와 동일 유형 — D-3 오류 유형 분석 및 M 출력 상한 효과 비교 대상).

## 결정

- **채택:** `retrieval.k = 5`, `compress.m = 7` (실험 C-2 권장값 연동), 공통 지시문 `CONTENT_INSTRUCTION` 및 `P1_TOOL_GUIDANCE` 정제 문구 채택
- **후속 작업:**
  - `configs/default.yaml`의 `retrieval.k: 3 → 5`, `compress.m: 3 → 7` 반영 완료
  - `results/runs/dev/r1/p1.jsonl` 및 `RUN_META.json` 커밋 → D-3(P1 로그 분석) 인계
