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
  bundle   : main | ablation | dev   (검증 모드는 _scratch — git 제외)
  round    : r1 | r2 | r3
  pipeline : p0 | p0_raw | p1 | m | m_minus_router | m_minus_requery | ...
results/runs/<bundle>/<round>/RUN_META.json
```

- `main`·`ablation`은 평가 세트(`eval.jsonl`), `dev`는 개발 세트(`dev.jsonl`)만 읽는다. 짝이 맞지 않으면 러너가 멈춘다. `_scratch`는 둘 다 허용한다.
- 라운드마다 파이프라인 실행 순서를 한 칸씩 돌려 바꾼다 (`r1`은 기본 순서, `r2`는 한 칸, `r3`은 두 칸). 본 비교의 4개 파이프라인에서 M이 3라운드에 서로 다른 순번에 놓인다.
- 실행: `python -m scripts.run_eval --bundle <묶음> --round <라운드>`. 검증 모드는 `--verify N`(앞 N문항, `_scratch`). 세션이 끊기면 같은 명령을 다시 실행하면 이어 돈다.
- 러너는 `scripts/run_eval.py`의 `PIPELINE_FACTORIES`에 등록된 `모듈:함수`로 파이프라인을 만든다. 함수는 `Settings`를 받아 `(qid, question) -> PipelineResult` 호출 가능 객체를 돌려준다. (제안: B·A가 자기 파이프라인을 등록할 때 확인)

### 7.2 문항별 로그 한 줄 = 메타 + `PipelineResult` + 채점

| 추가 필드 | 설명 |
|---|---|
| `qid`, `type`, `subtype`, `answer_kind` | 채점·집계용 (로그에는 유형을 남긴다). 집계에서 "텍스트 F1"처럼 문항 종류를 가를 때는 `answer_kind`로 거른다 |
| `pipeline`, `bundle`, `round` | 파이프라인 이름, 묶음, 라운드 |
| `order_idx` | 이 파이프라인 안에서 문항 순번(0부터, 문항 세트 파일 순서). 파이프라인 사이의 실행 순서는 `RUN_META.json`의 `pipeline_order` |
| `session_idx` | 이 문항을 돈 세션 번호(0부터) = `RUN_META.json`의 `sessions` 인덱스. 이어 돌기로 생긴 세션은 1 이상 |
| `gold` | 정답 후보 |
| `em`, `f1` | 텍스트 문항(`answer_kind=text`)은 SQuAD 정규화 EM·F1. **답할 수 없음 문항(`unanswerable`)은 EM과 F1이 같은 값**(기권하면 1.0, 아니면 0.0, SQuAD v2 방식). 수치 문항은 null |
| `numeric_correct` | 수치 문항 (그 밖의 문항은 null) |
| `correct` | 주 지표 기준 정답 여부 (텍스트는 EM, 답할 수 없음은 기권 여부, 수치는 수치 비교) |
| `timeout` | 120초 초과 여부 (초과 시 `correct=false`). 시간 초과 문항은 `answer=""`, `error="timeout"`이고, 작업을 중단하므로 `calls`는 비어 있다. `latency_s`는 제한 시간(120) 이상의 값이다 |
| `latency_s` | 문항 입력 ~ 최종 답. 러너가 직접 잰다 |
| `max_input_tokens` | 이 문항에서 호출별 입력 토큰의 최대값 (호출 기록이 없으면 0) |
| `evidence_kept` | 근거 문장 보존 여부. 검색된 문단 안의 근거 문장(`supporting_facts`)이 모두 남긴 문장 창(`compress.window` 반경)에 덮이면 true (실험 C-2의 보존율과 같은 기준: 검색이 못 찾은 문단의 근거 문장은 세지 않는다). HotpotQA가 아니거나, 압축 단계가 없거나(`kept_windows`가 비어 있음: P0·P1 등), 검색된 근거 문장이 없으면 null. `m_minus_compress`는 문단 전체를 창으로 남기므로(`full_paragraphs`) 검색된 근거 문장이 있으면 늘 true다 |
| `is_warmup` | 파이프라인마다 첫 1문항(`order_idx=0`)이 워밍업이다. **시간·비용 집계에서만 제외**하고, 정확도·McNemar는 워밍업 문항을 포함한 전체 문항으로 낸다 (A 제안, D 확인 필요) |

로그 한 줄에는 `PipelineResult`의 모든 필드(`answer`, `abstained`, `calls` 등)가 최상위에 그대로 펼쳐져 있고, `calls`의 각 원소는 `CallRecord` 필드를 가진 객체다. 파이프라인이 예외를 던지면 러너는 멈추지 않고 `answer=""`, `error="<예외 종류>: <메시지>"`인 오답 한 줄을 남긴다.

### 7.3 `RUN_META.json`

git 커밋, 세 데이터 파일 해시, 설정 전체 스냅샷, GPU 이름·드라이버, Ollama 버전, 모델 태그, 실행 순서, 시작·종료 시각, 최대 VRAM(nvidia-smi 0.5초 간격 최대치).

| 키 | 내용 |
|---|---|
| `bundle`, `round`, `item_set` | 묶음, 라운드, 읽은 문항 세트(`eval`\|`dev`) |
| `model_tag`, `settings` | 모델 태그, 설정 전체 스냅샷(`Settings.to_dict()`) |
| `data_hashes` | `eval.jsonl`·`dev.jsonl`·`corpus.jsonl`의 SHA-256 (시작 전에 `data/README.md` 표와 비교해 통과한 값) |
| `pipeline_order` | 이 라운드의 파이프라인 실행 순서 |
| `git_commit`, `git_dirty`, `gpu`(`name`, `driver`), `ollama_version` | 가장 최근 세션의 환경. 알 수 없으면 null |
| `max_vram_mib_by_pipeline`, `max_vram_mib` | 파이프라인별 최대 VRAM(MiB)과 그 최댓값. 파이프라인마다 샘플러를 따로 돌린다. `nvidia-smi`가 없으면 null |
| `started_at`, `finished_at` | 처음 시작과 마지막 종료 시각(UTC). 중간에 끊긴 라운드는 `finished_at`이 null |
| `sessions` | 세션(시작~끊김/종료)마다 시작·종료 시각과 환경. 이어 돌 때 GPU·커밋이 달라졌는지 여기서 확인한다 |

- `nvidia-smi`의 `memory.used`는 GPU 전체 사용량(다른 프로세스 포함)이다. GPU가 여러 장이면 가장 큰 값을 쓴다.
- 이어 돌 때 `settings`·`data_hashes`·`pipeline_order`·`bundle`·`round`·`item_set`이 처음과 다르면 러너가 멈춘다 (한 라운드에 조건이 섞이므로 새 라운드로 돌린다).
- 시간 초과한 작업은 취소 신호를 받는다. 모델 호출에서 막혀 신호가 닿지 않고 대기 시간(60초)을 넘기면 다음 문항의 측정이 오염되므로 러너가 `StuckWorkerError`로 멈춘다. 그 문항은 기록되지 않아 같은 명령으로 이어 돌면 다시 돈다.
- 이어 돈 세션의 첫 문항은 워밍업으로 표시하지 않는다(`order_idx=0`만 표시). 세션이 끊긴 뒤 모델이 다시 로딩되면 그 문항의 시간이 튈 수 있으므로, 시간 집계에서는 파이프라인마다 `session_idx`가 1 이상인 세션의 첫 문항(가장 작은 `order_idx`)을 이상치 후보로 확인한다.
- 동결 해시는 CLI가 파이프라인을 만들기 전(코퍼스·검색 인덱스를 읽기 전)에 한 번, `run_round`가 기록을 시작하기 전에 한 번 확인한다.

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
