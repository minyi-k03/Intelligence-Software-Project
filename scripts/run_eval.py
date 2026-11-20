"""측정 러너 실행: 한 라운드를 돌려 `results/runs/<bundle>/<round>/`에 로그를 남긴다.

실행 (레포 루트에서):
    python -m scripts.run_eval --bundle main --round r1
    python -m scripts.run_eval --bundle dev --round r1 --pipelines p0,p1
    python -m scripts.run_eval --verify 3          # 검증 모드: 앞 3문항만, results/runs/_scratch/

- 시작 전에 동결 파일 해시를 `data/README.md` 표와 비교하고, 다르면 멈춘다.
- 문항마다 즉시 저장한다. 세션이 끊기면 같은 명령을 다시 실행하면 이어서 돈다.
- 설정을 바꿔 이어 돌리면 한 라운드에 조건이 섞이므로 거부한다 (새 라운드로 돌린다).
- 정식 묶음(`main`·`ablation`)은 평가 세트, `dev`는 개발 세트를 읽는다.

파이프라인은 `PIPELINE_FACTORIES`에 `이름 -> "모듈:함수"`로 등록한다. 함수는 설정을 받아
`(qid, question) -> PipelineResult` 호출 가능 객체를 돌려준다. 설정을 키워드(`settings=`)로 받는
`run_*(qid, question, *, settings=None)` 함수는 `"call:모듈:함수"`로 등록하면 러너가 설정을 묶어 준다
(docs/INTERFACES.md §7.1).
"""

from __future__ import annotations

import argparse
import functools
import importlib
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

from src.config.settings import REPO_ROOT, Settings, get_settings
from src.eval.logger import FrozenDataError, RunMetaMismatch, verify_frozen_hashes
from src.eval.runner import DEFAULT_ITEM_SET, SCRATCH_BUNDLE, Pipeline, run_round

PipelineFactory = Callable[[Settings], Pipeline]

# 항목이 이 접두어로 시작하면 "생성 함수"가 아니라 `fn(qid, question, *, settings=...)` 형태의
# 실행 함수다. 러너가 설정을 미리 묶어 `(qid, question)` 호출 객체로 바꿔 쓴다.
CALL_PREFIX = "call:"

# 이름 -> "모듈:함수" (생성 함수) 또는 "call:모듈:함수" (실행 함수).
# 파이프라인 PR(B-3 기준선, A-5b M, 절제 변형)이 자기 항목을 등록한다.
PIPELINE_FACTORIES: dict[str, str] = {
    "p0": "call:src.baselines.p0:run_p0",
    "p0_raw": "call:src.baselines.p0:run_p0_raw",
    "p1": "call:src.baselines.p1_react:run_p1",
    "m": "src.meba.graph:build_m",
}

DEFAULT_PIPELINES = ("p0", "p0_raw", "p1", "m")


def load_factory(name: str, registry: dict[str, str | PipelineFactory]) -> PipelineFactory:
    """등록된 이름의 파이프라인 생성 함수를 불러온다.

    Args:
        name: 파이프라인 이름.
        registry: 이름 -> "모듈:함수"(생성 함수), "call:모듈:함수"(실행 함수) 문자열 또는 생성 함수.

    Returns:
        설정을 받아 `(qid, question)` 호출 객체를 돌려주는 함수. 실행 함수 항목은 설정을
        `settings=`로 묶는 생성 함수로 감싸 돌려준다.

    Raises:
        SystemExit: 등록되지 않았거나 모듈·함수를 찾을 수 없을 때 (아직 구현 전인 파이프라인 포함).
    """
    entry = registry.get(name)
    if entry is None:
        raise SystemExit(f"등록되지 않은 파이프라인: {name!r} (등록된 것: {', '.join(registry)})")
    if callable(entry):
        return entry
    target = entry.removeprefix(CALL_PREFIX)
    module_name, _, func_name = target.partition(":")
    try:
        func = getattr(importlib.import_module(module_name), func_name)
    except (ImportError, AttributeError) as exc:
        raise SystemExit(f"파이프라인 {name!r}을 불러올 수 없다 ({entry}): {exc}") from exc
    if entry.startswith(CALL_PREFIX):
        return lambda settings: functools.partial(func, settings=settings)
    return func


def main(
    argv: Optional[Sequence[str]] = None,
    registry: Optional[dict[str, str | PipelineFactory]] = None,
    frozen_dir: Optional[Path] = None,
    readme_path: Optional[Path] = None,
) -> None:
    """CLI 진입점.

    Args:
        argv: 인자 목록. 없으면 `sys.argv`.
        registry: 파이프라인 등록부. 없으면 `PIPELINE_FACTORIES` (테스트·수동 확인용으로 바꿀 수 있다).
        frozen_dir: 동결 폴더. 없으면 `data/frozen/` (테스트용).
        readme_path: 해시 표가 있는 README. 없으면 `data/README.md` (테스트용).

    Raises:
        SystemExit: 인자가 잘못됐거나, 동결 해시가 다르거나, 파이프라인을 못 불러오거나, 이어 돌기 조건이 다를 때.
    """
    frozen_dir = frozen_dir or REPO_ROOT / "data" / "frozen"
    readme_path = readme_path or REPO_ROOT / "data" / "README.md"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", choices=[*DEFAULT_ITEM_SET], help="결과 묶음")
    parser.add_argument("--round", default="r1", dest="round_name", help="라운드 이름 (r1, r2, r3)")
    parser.add_argument("--pipelines", help="쉼표로 구분한 파이프라인 이름. main·dev 기본값: p0,p0_raw,p1,m")
    parser.add_argument("--set", dest="item_set", choices=["eval", "dev"], help="문항 세트 (기본: 묶음에 맞춤)")
    parser.add_argument("--verify", type=int, metavar="N", help=f"검증 모드: 앞 N문항만, {SCRATCH_BUNDLE} 묶음에 기록")
    parser.add_argument("--config", default=None, help="설정 파일 (기본: configs/default.yaml)")
    parser.add_argument("--runs-dir", default=None, help="결과 폴더 (기본: results/runs)")
    args = parser.parse_args(argv)

    bundle = SCRATCH_BUNDLE if args.verify is not None else args.bundle
    if bundle is None:
        parser.error("--bundle 또는 --verify N이 필요하다")
    if args.verify is not None and args.verify <= 0:
        parser.error("--verify는 1 이상이어야 한다")
    if args.pipelines:
        names = [n.strip() for n in args.pipelines.split(",") if n.strip()]
    elif bundle == "ablation":
        parser.error("ablation 묶음은 --pipelines로 M과 절제 변형 이름을 직접 정해야 한다")
    else:
        names = list(DEFAULT_PIPELINES)
    if not names:
        parser.error("--pipelines에 파이프라인 이름이 하나도 없다")
    if len(set(names)) != len(names):
        parser.error("--pipelines에 같은 이름이 두 번 들어 있다")

    # 파이프라인을 만들 때 코퍼스(검색 인덱스)를 읽으므로, 그 전에 동결 데이터를 확인한다.
    # run_round도 기록 직전에 한 번 더 확인한다.
    try:
        verify_frozen_hashes(frozen_dir, readme_path)
    except FrozenDataError as exc:
        raise SystemExit(f"동결 데이터 해시 확인 실패: {exc}") from exc

    registry = PIPELINE_FACTORIES if registry is None else registry
    settings = get_settings(args.config)
    pipelines = {name: load_factory(name, registry)(settings) for name in names}

    written = 0

    def progress(row: dict) -> None:
        """로그 한 줄을 쓸 때마다 진행 상황을 stderr에 한 줄 찍는다.

        Args:
            row: 방금 기록한 문항 로그 한 줄.
        """
        nonlocal written
        written += 1
        flag = "timeout" if row["timeout"] else ("ok" if row["correct"] else "wrong")
        print(
            f"[{row['pipeline']} {row['round']}] {row['order_idx'] + 1:>3} {row['qid']} "
            f"{flag} {row['latency_s']:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    # 상대 경로는 실행 위치가 아니라 레포 루트 기준으로 푼다 (Colab과 로컬 모두에서 같게).
    runs_dir = REPO_ROOT / (args.runs_dir or "results/runs")
    round_path = runs_dir / bundle / args.round_name
    try:
        meta = run_round(
            bundle=bundle,
            round_name=args.round_name,
            item_set=args.item_set or DEFAULT_ITEM_SET[bundle],
            settings=settings,
            pipelines=pipelines,
            frozen_dir=frozen_dir,
            readme_path=readme_path,
            runs_dir=runs_dir,
            limit=args.verify,
            on_row=progress,
        )
    except RunMetaMismatch as exc:
        raise SystemExit(f"{exc}\n다른 조건으로 다시 돌리려면 --round를 바꾸거나 {round_path} 를 지운다.") from exc
    if written == 0:
        print(
            f"새로 돈 문항이 없다: {round_path} 는 이미 끝난 라운드다. "
            "다시 돌리려면 --round를 바꾸거나 그 폴더를 지운다.",
            file=sys.stderr,
        )
    print(f"완료: {meta['bundle']}/{meta['round']} 최대 VRAM(MiB) {meta['max_vram_mib_by_pipeline']}")


if __name__ == "__main__":
    main()
