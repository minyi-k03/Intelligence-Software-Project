# 설계 결정 기록 (ADR)

"왜 이렇게 했지?"에 답하는 문서입니다. 교수님이 설계 근거를 중요하게 보시므로, 보고서·발표의 근거 절은 여기서 가져옵니다.

| 번호 | 결정 | 상태 |
|---|---|---|
| [0001](0001-mixed-type-eval-set.md) | 평가 세트는 세 유형을 섞은 120문항, 유형은 숨기고 같은 채점기로 | accepted |
| [0002](0002-shared-corpus-bm25.md) | 공용 코퍼스 + BM25 검색, 모델 없는 압축 | accepted |
| [0003](0003-greedy-decoding.md) | 모든 호출은 temperature 0 | accepted |
| [0004](0004-m-architecture-call-budget.md) | M은 라우터 + 근거 판정 루프의 단일 에이전트, 호출 예산 4회 | accepted |
| [0005](0005-round-based-measurement.md) | 본 비교 3라운드, 절제 1라운드, 서로 다른 세션 | accepted |
| [0006](0006-core-and-conditional-parts.md) | 세 제안을 핵심 6 + 조건부 3으로 통합 | accepted |

## 새 ADR을 쓰는 때

아래 세 가지가 모두 해당할 때만 씁니다. 하나라도 아니면 커밋 메시지나 실험 기록으로 충분합니다.

1. 되돌리기 어렵다 (바꾸면 측정을 다시 해야 하는 등)
2. 맥락 없이 보면 의아하다 (읽는 사람이 "왜 이렇게?"라고 물을 것)
3. 실제 대안이 있었고, 이유가 있어 하나를 골랐다

예: P0에 기권 문구를 넣을지, 출력 상한 초기값을 바꿀지, 조건부 부품을 기각할지.

## 형식

파일 이름은 다음 번호 + 슬러그(`0007-short-slug.md`). 한 문단이면 충분합니다.

```md
---
status: proposed | accepted | superseded by ADR-NNNN
date: YYYY-MM-DD
---

# 결정을 한 줄로

맥락, 결정, 이유를 1~3문장으로.

## Considered Options      ← 버린 대안이 기억할 가치가 있을 때만
## Consequences            ← 뒤따르는 영향이 뻔하지 않을 때만
```
