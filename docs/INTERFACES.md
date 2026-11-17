# 모듈 간 규격

> **상태: 제안 초안.** 11/12 동결 전에 팀 회의에서 확정합니다. 확정 뒤 바꾸려면 이 문서를 먼저 고치는 PR을 내고, 영향을 받는 역할 담당자의 리뷰를 받습니다.
> 표에서 `TBD`는 담당자가 정해서 채울 칸입니다.

4명이 11/13부터 따로 만들고 11/24에 합치기 때문에, 서로 주고받는 데이터의 모양을 먼저 고정합니다. 여기 없는 내부 구현은 각자 자유입니다.

---

## 1. 동결 데이터 파일 (`data/frozen/`) — 담당 A

| 파일 | 내용 |
|---|---|
| `eval.jsonl` | 평가 세트 120문항 |
| `dev.jsonl` | 개발 세트 60문항 |
| `corpus.jsonl` | 공용 코퍼스 |
| `MANIFEST.json` | 원본 데이터셋·split·시드·문항 수·생성 스크립트 커밋 |

### 1.1 문항 (`eval.jsonl`, `dev.jsonl`) — 한 줄에 한 문항

| 필드 | 타입 | 설명 |
|---|---|---|
| `qid` | str | 레포 내부 문항 ID. 예: `hotpotqa-0007`, `squad2-0031`, `gsm8k-0012` |
| `type` | str | `hotpotqa` \| `squad2` \| `gsm8k` — **에이전트에 넘기지 않는다** |
| `subtype` | str \| null | HotpotQA: `bridge` \| `comparison`, SQuAD 2.0: `answerable` \| `unanswerable`, GSM8K: null |
| `question` | str | 질문 원문 |
| `answers` | list[str] | 정답 후보. 답할 수 없음 문항은 `["UNANSWERABLE"]` |
| `answer_kind` | str | `text` \| `numeric` \| `unanswerable` — 채점 방식 선택용 |
| `gold_doc_ids` | list[str] | 정답 근거 문단의 `doc_id` (GSM8K는 빈 리스트) |
| `supporting_facts` | list[[str, int]] | HotpotQA 정답 근거 문장 `[title, 문장 번호]` (근거 문장 보존율 측정용) |
| `source_id` | str | 원본 데이터셋의 문항 ID (추적용). GSM8K는 원본에 ID가 없어 `gsm8k-test-<test split 행 번호>` |

- `qid` 번호는 유형별로 평가 세트 → 개발 세트 순서로 이어 붙인다(평가 `0000`~`0039`, 개발 `0040`~`0059`). 두 세트에서 `qid`가 겹치지 않는다.
- SQuAD 2.0 답할 수 없음 문항의 `gold_doc_ids`에는 원래 문맥(함정 문단)을 기록한다.
- GSM8K `answers`는 `####` 뒤 값에서 쉼표를 뺀 문자열이다.

### 1.2 코퍼스 문단 (`corpus.jsonl`)

| 필드 | 타입 | 설명 |
|---|---|---|
| `doc_id` | str | `hp::<title>` (같은 제목에 본문이 다른 문단이 있으면 `hp::<title>::2`, …) · `sq::<article title>::<paragraph_idx>` (idx = 같은 제목 안에서 데이터셋에 등장한 순서) |
| `title` | str | 문단 제목 (압축의 문장 창에 붙인다) |
| `sentences` | list[str] | 문장 단위로 나눈 본문. HotpotQA는 원본 문장 분할을 그대로 쓰고(각 문장 앞뒤 공백만 제거, 번호는 `supporting_facts`와 같다), SQuAD 2.0은 정규식 + 약어 보호로 나눈다(`scripts/freeze_data.py`의 `split_sentences`) |
| `text` | str | `sentences`를 공백 한 칸으로 이어 붙인 본문 (BM25 문단 검색용) |
| `origin` | str | `hotpotqa` \| `squad2` |

중복 문단(같은 제목·같은 본문)은 하나로 합칩니다.

### 1.3 해시

세 파일(`eval.jsonl`, `dev.jsonl`, `corpus.jsonl`)의 바이트 SHA-256을 `data/README.md`의 표에 기록합니다. 측정 러너는 시작할 때 해시를 다시 계산해 다르면 멈춥니다.

---

## 2. 설정 (`configs/`) — 담당 B (값의 출처는 SPEC)

모든 코드는 이 키를 설정에서 읽습니다. 코드 안에 숫자를 직접 쓰지 않습니다.

| 키 | 값 | 출처 |
|---|---|---|
| `model.tag` | `qwen2.5:7b-instruct-q4_K_M` (Qwen2.5-7B-Instruct 4-bit Ollama 태그) | SPEC §2 |
| `model.temperature` | 0 | ADR-0003 |
| `model.num_ctx` | 8192 | SPEC §2 |
| `model.keep_alive` | `-1` (측정 세션 중 메모리 상시 유지) | EVALUATION §4 |
| `budget.max_calls` | 4 | SPEC §7 |
| `budget.max_requery` | 2 | SPEC §7 |
| `cap.first` | 512 | SPEC §7 |
| `cap.finalize` | 128 | SPEC §7 |
| `cap.judge` | 256 | SPEC §7 |
| `cap.retry_multiplier` | 2 | SPEC §7 |
| `retrieval.k` | TBD (1주차 B 측정) | SPEC §7 |
| `compress.m` | TBD (1주차 C 측정) | SPEC §7 |
| `compress.window` | 1 (앞뒤 1문장) | SPEC §5 |
| `p1.max_steps` | 7 | SPEC §3 |
| `run.timeout_s` | 120 | EVALUATION §4 |
| `data.seed` | 20261113 (`MANIFEST.json`의 `seed`와 같아야 한다) | ADR-0001 |

---

## 3. 도구 — 담당 B

P1과 M이 **같은 구현**을 씁니다. 오류도 같은 방식으로 처리합니다(예외를 던져 루프를 멈추지 않고, 오류 문자열을 관찰 결과로 돌려준다).

| 도구 | 입력 | 출력 |
|---|---|---|
| BM25 검색 | `query: str`, `k: int` | 문단 리스트 `[{doc_id, title, sentences, text, score}]` (점수 내림차순) |
| 계산기 | `expression: str` | 결과 문자열. 실패 시 `"ERROR: <사유>"` |

BM25 공용 토크나이저(`src/tools/tokenizer.py`): `tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]` (NFKC 정규화·소문자·숫자 쉼표 제거·`[^\W_]+` 단어 추출·영어 불용어 `ENGLISH_STOPWORDS` 제거 및 전부 불용어 시 폴백), 공유 상수 `BM25_K1 = 1.5`, `BM25_B = 0.75`. 검색(B)과 압축(C)이 이 토크나이저와 상수를 함께 씁니다.

---

## 4. 파이프라인 공통 인터페이스

모든 파이프라인(P0, P0-raw, P1, M, 절제 변형)은 같은 모양으로 호출되고 같은 모양을 돌려줍니다. 러너(A)는 이 인터페이스만 압니다.

**입력**: `qid: str`, `question: str` — 이 둘만. `type`, `subtype`, `answers`는 넘기지 않습니다.

**출력 `PipelineResult`**

| 필드 | 타입 | 설명 |
|---|---|---|
| `answer` | str | 최종 답 문자열. **접두어 없이 답만** 넣는다(`extract_final_answer` 결과, §6). 기권이면 `"UNANSWERABLE"` |
| `abstained` | bool | `answer`가 정규화 후 `UNANSWERABLE`과 정확히 같은가. P0·P1·M 모두 `src.eval.scorer.is_abstained`로 정한다. 판정 ABSTAIN뿐 아니라 `decision=ANSWER`인데 답이 `UNANSWERABLE`인 경우도 True이며, 이때 `answer`는 `"UNANSWERABLE"`로 기록한다. `decisions`에는 모델이 낸 값을 그대로 남긴다. **빈 답은 기권으로 바꾸지 않고 오답으로 둔다** |
| `budget_exhausted` | bool | 호출 예산을 다 써서 끝났는가 (기권과 따로) |
| `route` | str \| null | M: `math` \| `search`, 그 외 null |
| `decisions` | list[str] | 근거 판정 결과 순서. 예: `["RETRIEVE", "ANSWER"]` |
| `n_calls` | int | 예산에 들어가는 모델 호출 수 |
| `n_retries` | int | 잘림 재시도 수 (예산 제외, 비용 포함) |
| `calls` | list[CallRecord] | 호출별 기록 (아래) |
| `retrieved_doc_ids` | list[list[str]] | 검색 회차별 결과 문단 ID |
| `kept_windows` | list[list[str]] | 압축 회차별 남긴 문장 창 ID (`doc_id#문장번호`) |
| `loop_breaks` | int | 루프 끊기가 발동한 횟수 |
| `error` | str \| null | 처리하지 못한 예외 메시지 |

**`CallRecord`**

| 필드 | 타입 | 설명 |
|---|---|---|
| `step` | str | `p0` \| `p1_step` \| `router` \| `finalize` \| `judge` \| `ledger` 등 |
| `cap` | int \| null | 적용한 출력 상한 |
| `input_tokens` | int | LangChain 콜백/usage 기준 |
| `output_tokens` | int | 〃 |
| `truncated` | bool | 출력 상한에 걸렸는가 |
| `is_retry` | bool | 잘림 재시도 호출인가 |
| `parse_ok` | bool \| null | 구조화 출력 파싱 성공 여부 (해당 없으면 null) |
| `latency_s` | float | 호출 소요 시간 |

---

## 5. 근거 판정 + 답 출력 형식 — 담당 C (상한은 D)

Ollama 구조화 출력(`format`에 JSON 스키마)을 씁니다. **키 순서가 중요합니다** — 서술이 맨 앞, 답이 맨 뒤.

```json
{
  "reasoning": "1~2 sentences about whether the evidence answers the question.",
  "decision": "ANSWER | RETRIEVE | ABSTAIN",
  "missing": "what information is still needed (empty unless RETRIEVE)",
  "answer": "short answer (empty unless ANSWER)"
}
```

- 마지막 판정 호출에서는 스키마의 `decision` 값에서 `RETRIEVE`를 뺍니다.
- 대조군: 같은 내용을 자유 텍스트 + 마지막 줄 태그(예: `DECISION: ANSWER`) + 정규식으로 받는 방식. 잘림률·파싱 실패율을 비교합니다.
- 조건부 부품 "근거 장부 상세화"는 이 스키마에 항목별 `FOUND` / `MISSING` 필드를 추가하는 방식으로 확장합니다 (TBD: C).

---

## 6. 공통 답 형식 지시문 — 담당 B · C 합의

> 결정(TBD #4·#5, 팀 3인 동의): P0에도 기권 문구를 넣고, 최종 답은 `Final answer:` 한 줄로 끝낸다. 모든 문구는 영어이며 **문항 유형을 알려 주지 않는 일반 표현**만 쓴다.

지시문은 두 부분이고, 상수는 `src/baselines/prompts.py`에 하나씩 둡니다(B). 아래 문구는 출발점이며, 문구를 다듬을 때도 **의미 조건**(번호 목록)은 유지합니다.

**① 내용 문장** — P0 · P1 · M 근거 판정이 같은 상수를 그대로 씁니다.

```text
Answer with a short answer only: a few words, a name, a date, or a number.
Do not explain. If the answer is a number, write only the number.
If the question cannot be answered from the available information,
reply exactly: UNANSWERABLE
```

**② 형식 줄** — 자유 서술로 답하는 호출(P0 · P1 · M 수학 경로)에만 붙입니다.

```text
End your reply with a last line of the form:
Final answer: <short answer or UNANSWERABLE>
```

| 파이프라인 | 지시문 |
|---|---|
| P0-raw | 없음 (질문만). 기권 불가, 참고용이라 결과표에 그대로 표시 |
| P0 | ① + ② (최소 지시문이지만 기권 문구와 형식 줄은 포함) |
| P1 | ① + ② (최종 답 단계) |
| M 수학 경로 | ① + ② |
| M 근거 판정 | ①만. 답은 §5 JSON의 `answer` 필드로 받으므로 `Final answer:` 줄을 쓰지 않는다 |

의미 조건: (1) 짧은 답만, 설명 금지 (2) 숫자 답이면 숫자만 (3) 답할 수 없으면 정확히 `UNANSWERABLE` (4) 자유 서술 호출은 마지막 줄을 `Final answer: …`로 끝냄.

**최종 답 추출**: 모든 파이프라인은 같은 함수 `src.eval.scorer.extract_final_answer`를 씁니다. 규칙은 [EVALUATION §3.1](EVALUATION.md)입니다. 파이프라인은 이 함수의 결과(접두어 없는 답)를 `PipelineResult.answer`에 넣습니다. 채점기도 같은 함수를 다시 적용하며, 이미 추출된 답에 다시 적용해도 값이 바뀌지 않습니다.

---

## 7. 측정 로그 (`results/runs/`) — 담당 A

### 7.1 경로

```
results/runs/<bundle>/<round>/<pipeline>.jsonl
  bundle   : main | ablation | dev
  round    : r1 | r2 | r3
  pipeline : p0 | p0_raw | p1 | m | m_minus_router | m_minus_requery | ...
results/runs/<bundle>/<round>/RUN_META.json
```

### 7.2 문항별 로그 한 줄 = 메타 + `PipelineResult` + 채점

| 추가 필드 | 설명 |
|---|---|
| `qid`, `type`, `subtype`, `answer_kind` | 채점·집계용 (로그에는 유형을 남긴다). 집계에서 "텍스트 F1"처럼 문항 종류를 가를 때는 `answer_kind`로 거른다 |
| `pipeline`, `bundle`, `round`, `order_idx` | 라운드 안 실행 순서 |
| `gold` | 정답 후보 |
| `em`, `f1` | 텍스트 문항(`answer_kind=text`)은 SQuAD 정규화 EM·F1. **답할 수 없음 문항(`unanswerable`)은 EM과 F1이 같은 값**(기권하면 1.0, 아니면 0.0, SQuAD v2 방식). 수치 문항은 null |
| `numeric_correct` | 수치 문항 (그 밖의 문항은 null) |
| `correct` | 주 지표 기준 정답 여부 (텍스트는 EM, 답할 수 없음은 기권 여부, 수치는 수치 비교) |
| `timeout` | 120초 초과 여부 (초과 시 `correct=false`) |
| `latency_s` | 문항 입력 ~ 최종 답 |
| `max_input_tokens` | 이 문항에서 호출별 입력 토큰의 최대값 |
| `evidence_kept` | 근거 문장 보존 여부 (HotpotQA, 압축 사용 시) |
| `is_warmup` | 첫 1문항(워밍업) 여부 — 집계에서 제외 |

### 7.3 `RUN_META.json`

git 커밋, 세 데이터 파일 해시, 설정 전체 스냅샷, GPU 이름·드라이버, Ollama 버전, 모델 태그, 실행 순서, 시작·종료 시각, 최대 VRAM(nvidia-smi 0.5초 간격 최대치).

---

## 8. 용어 → 코드 이름

| 용어 (GLOSSARY) | 코드 이름 |
|---|---|
| 라우터 | `router` |
| 수학 경로 / 검색 경로 | `math` / `search` |
| 마무리 호출 | `finalize` |
| 근거 판정 | `judge` |
| 판정 ANSWER / RETRIEVE / ABSTAIN | `ANSWER` / `RETRIEVE` / `ABSTAIN` |
| 다중 홉 재검색 | `requery` |
| 후속 검색어 | `followup_query` |
| 빠진 정보 | `missing` |
| 검색 결과 압축 | `compress` |
| 문장 창 | `window` |
| 출력 상한 | `cap` |
| 잘림 / 잘림 재시도 | `truncated` / `retry` |
| 루프 끊기 | `loop_breaker` |
| 호출 예산 / 예산 소진 | `budget` / `budget_exhausted` |
| 근거 장부 상세화 | `ledger` |
| 간결 출력 | `concise` |
| 동적 압축량 | `dynamic_m` |
| 절제 변형 | `m_minus_<부품>` |
