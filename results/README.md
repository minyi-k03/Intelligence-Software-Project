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

## 결과표 생성

레포 루트에서 `python scripts/build_tables.py`를 실행하면 `results/tables/`에
`main`, `ablation`, `side_effects`, `mcnemar` 표를 각각 CSV와 Markdown으로 저장합니다.
정식 집계에는 본 비교 3라운드와 M 및 핵심 절제 변형 6개를 포함한 절제 1라운드의 평가 세트 120문항이 필요합니다.
여기서 호출 수 비율은 재시도를 포함한 총 호출 수의 비율입니다. CSV의 빈 칸과 Markdown의
대시는 적용 대상이나 관측값이 없거나, 비율의 분모가 0인 경우를 뜻합니다.

소수 문항으로 작성한 모의 로그는 다음과 같이 검증합니다. 모든 로그에 같은 문항을 넣습니다.

```sh
python scripts/build_tables.py --runs-dir /tmp/mock-runs --verification --output-dir /tmp/mock-tables
```
