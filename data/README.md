# data/

| 폴더 | 내용 | git |
|---|---|---|
| `raw/` | 원본 데이터셋 다운로드·캐시 | 제외 |
| `frozen/` | 동결된 평가 세트·개발 세트·공용 코퍼스 | **커밋, 동결 후 수정 금지** |

형식은 [docs/INTERFACES.md §1](../docs/INTERFACES.md), 구성 규칙은 [docs/EVALUATION.md §1~2](../docs/EVALUATION.md)를 따릅니다.

## 원본 데이터

| 유형 | 원본 | 설정 · split | 쓰는 필드 |
|---|---|---|---|
| HotpotQA | [Hugging Face `hotpot_qa`](https://huggingface.co/datasets/hotpot_qa) | `distractor` · validation | question, answer, type(bridge/comparison), context(10문단), supporting_facts |
| SQuAD 2.0 | Hugging Face `squad_v2` | validation | question, context, answers(빈 리스트면 답할 수 없음) |
| GSM8K | Hugging Face `gsm8k` | `main` · test | question, answer(`####` 뒤 값이 정답) |

정확한 데이터셋 ID와 리비전(커밋 해시)은 동결 스크립트에서 로드한 값을 `frozen/MANIFEST.json`에 기록합니다.

## 동결 절차 (11/12, 담당 A)

1. 동결 스크립트(`scripts/`)로 원본을 `raw/`에 받는다.
2. 고정 시드로 평가 세트와 개발 세트를 겹치지 않게 뽑는다 (SQuAD 2.0은 문맥 단위 분리 — EVALUATION §1 제안).
3. 두 세트 문항의 문단으로 공용 코퍼스를 만들고 중복을 제거한다.
4. `frozen/`에 `eval.jsonl`, `dev.jsonl`, `corpus.jsonl`, `MANIFEST.json`을 쓴다.
5. 세 파일의 SHA-256을 계산해 아래 표에 적고, 팀원 한 명이 다른 환경에서 같은 해시가 나오는지 확인한다.
6. 한 커밋으로 올린다: `chore(data): freeze eval/dev/corpus`

## 동결 기록

| 파일 | 문항/문단 수 | SHA-256 |
|---|---|---|
| `frozen/eval.jsonl` | 120 | |
| `frozen/dev.jsonl` | 60 | |
| `frozen/corpus.jsonl` | | |

- 동결 일시: 
- 동결 커밋: 
- 시드: 
- 교차 확인자: 

> 이 표가 채워진 뒤로 `frozen/`의 변경은 "평가 세트 사후 변경" 감점 사유입니다. 오류를 발견하면 고치지 말고 팀에 알린 뒤 ADR로 처리 방침을 기록합니다.
