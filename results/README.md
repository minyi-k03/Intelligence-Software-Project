# results/

| 폴더 | 내용 |
|---|---|
| `runs/` | 문항별 `.jsonl` 로그와 `RUN_META.json` |
| `tables/` | 결과표, 절제표, McNemar 교차표 (D가 로그에서 생성) |

## `runs/` 구조

```
runs/
├── main/       ← 본 비교 묶음 (P0 · P0-raw · P1 · M), r1 ~ r3
├── ablation/   ← 절제 묶음 (M + 절제 변형), r1
└── dev/        ← 개발 세트 실험
```

자세한 경로 규칙과 필드는 [docs/INTERFACES.md §7](../docs/INTERFACES.md)을 따릅니다.

- 정식 측정 로그(`main/`, `ablation/`)는 커밋합니다. 보고서의 모든 숫자는 이 로그에서 다시 계산할 수 있어야 합니다.
- 디버그 실행은 `runs/_scratch/`에 두면 git에서 제외됩니다.
- 측정이 끝난 로그는 고치지 않습니다. 다시 돌렸으면 새 라운드로 남기고 이유를 기록합니다.
