"""측정 러너: 묶음·라운드 실행, 라운드별 순서 교체, 워밍업 표시, 타임아웃, 이어 돌기.

파이프라인은 `(qid, question) -> PipelineResult` 하나만 안다 (INTERFACES §4).
문항 유형(`type`)·정답은 파이프라인에 넘기지 않고 채점·로그에서만 쓴다.

Reference: docs/INTERFACES.md §4·§7, docs/EVALUATION.md §3·§4.
"""

from __future__ import annotations

import ctypes
import json
import re
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

from src.common.types import PipelineResult
from src.config.settings import Settings
from src.eval.logger import (
    RunLog,
    RunMeta,
    collect_environment,
    verify_frozen_hashes,
)
from src.eval.scorer import score_item
from src.eval.vram import VramSampler
from src.meba.compress import covered_sentences

Pipeline = Callable[[str, str], PipelineResult]

# 시간 초과한 작업을 멈춘 뒤, 진행 중인 모델 호출이 끝나기를 기다리는 최대 시간(초).
# 호출마다 출력 상한(`cap`)이 있어 보통 이 안에 끝난다. 설계 수치가 아니라 러너 내부 안전 장치다.
DEFAULT_DRAIN_S = 60.0

_ROUND_RE = re.compile(r"r(\d+)")


class _Cancelled(BaseException):
    """시간 초과한 작업에 보내는 취소 신호. `except Exception`에 잡히지 않게 BaseException이다."""


class StuckWorkerError(RuntimeError):
    """시간 초과한 작업이 취소 신호에도 끝나지 않아 다음 문항 측정을 오염시킬 수 있을 때."""


def round_order(pipelines: list[str], round_name: str) -> list[str]:
    """라운드마다 파이프라인 실행 순서를 한 칸씩 돌려 바꾼다.

    `r1`은 주어진 순서 그대로, `r2`는 한 칸, `r3`은 두 칸 민다. 본 비교의 4개 파이프라인·3라운드에서
    각 파이프라인이 서로 다른 순번에 놓인다.

    Args:
        pipelines: 기본 실행 순서.
        round_name: `r1`, `r2`, `r3` 같은 라운드 이름.

    Returns:
        이 라운드의 실행 순서 (새 리스트).

    Raises:
        ValueError: 라운드 이름이 `r<번호>` 형식이 아니거나 번호가 0일 때.
    """
    match = _ROUND_RE.fullmatch(round_name)
    if not match or int(match.group(1)) < 1:
        raise ValueError(f"라운드 이름은 r1, r2, r3 형식이어야 한다: {round_name!r}")
    shift = (int(match.group(1)) - 1) % len(pipelines)
    return pipelines[shift:] + pipelines[:shift]


def _gold_sentences(item: dict[str, Any]) -> set[tuple[str, int]]:
    """HotpotQA 문항의 근거 문장을 (doc_id, 문장 번호)로 바꾼다.

    제목이 같은 문단이 여럿이면(`hp::<title>::2`) `gold_doc_ids`에서 찾아 쓴다.

    Args:
        item: `gold_doc_ids`와 `supporting_facts`(`[title, 문장 번호]` 목록)가 있는 동결 문항.

    Returns:
        {(doc_id, 문장 번호)}.
    """
    by_title: dict[str, str] = {}
    for doc_id in item["gold_doc_ids"]:
        title = re.sub(r"::\d+$", "", doc_id.removeprefix("hp::"))
        by_title.setdefault(title, doc_id)
    return {(by_title.get(title, f"hp::{title}"), idx) for title, idx in item["supporting_facts"]}


def _evidence_kept(item: dict[str, Any], result: PipelineResult, window: int) -> Optional[bool]:
    """압축이 남긴 문장 창이 근거 문장을 모두 덮는가.

    검색이 못 찾은 문단의 근거 문장은 세지 않는다(압축이 잃은 것이 아니므로, 실험 C-2와 같은 기준).

    Args:
        item: 동결 문항 한 줄.
        result: 파이프라인 결과.
        window: 압축에 쓴 `compress.window`.

    Returns:
        True/False. HotpotQA가 아니거나, 압축을 안 썼거나, 검색된 근거 문장이 없으면 None.
    """
    kept = [w for round_windows in result.kept_windows for w in round_windows]
    if item["type"] != "hotpotqa" or not kept:
        return None
    retrieved = {d for round_ids in result.retrieved_doc_ids for d in round_ids}
    present = {s for s in _gold_sentences(item) if s[0] in retrieved}
    if not present:
        return None
    return present <= covered_sentences(kept, window)


def _cancel(thread: threading.Thread) -> None:
    """스레드에 `_Cancelled`를 던진다 (파이썬 코드를 실행하는 시점에 발동하는 최선의 시도)."""
    if thread.ident is not None:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread.ident), ctypes.py_object(_Cancelled)
        )


def _empty_result(error: str) -> PipelineResult:
    """파이프라인이 결과를 못 냈을 때(예외·시간 초과) 채우는 빈 결과."""
    return PipelineResult(
        answer="",
        abstained=False,
        budget_exhausted=False,
        route=None,
        decisions=[],
        n_calls=0,
        n_retries=0,
        error=error,
    )


def _call_with_timeout(
    pipeline: Pipeline, qid: str, question: str, timeout_s: float, drain_s: float
) -> tuple[PipelineResult, bool, float]:
    """파이프라인을 별도 스레드에서 돌리고, 제한 시간을 넘기면 멈추게 한다.

    시간 초과한 작업은 취소 신호를 받고, 진행 중이던 모델 호출이 끝날 때까지(최대 `drain_s`)
    기다린 뒤 돌아온다. 그래야 다음 문항의 시간·VRAM이 이전 문항의 호출에 오염되지 않는다.

    Args:
        pipeline: `(qid, question) -> PipelineResult`.
        qid: 문항 ID.
        question: 질문 원문.
        timeout_s: 제한 시간(초).
        drain_s: 시간 초과 뒤 작업이 멈추기를 기다리는 최대 시간(초).

    Returns:
        (결과, 시간 초과 여부, 걸린 시간(초)). 예외나 시간 초과면 `error`가 채워진 빈 결과.
        시간 초과한 문항의 호출 기록은 수거할 수 없어 `calls`가 비어 있다.
    """
    outcome: dict[str, Any] = {}  # 작업 스레드가 `result` 또는 `error`를 남기는 곳

    def target() -> None:
        """파이프라인을 부르고 결과나 예외를 `outcome`에 남긴다."""
        try:
            outcome["result"] = pipeline(qid, question)
        except _Cancelled:
            pass
        except Exception as exc:  # 파이프라인의 처리 못 한 예외는 오답 한 줄로 남긴다
            outcome["error"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=target, name=f"pipeline-{qid}", daemon=True)
    start = time.perf_counter()
    worker.start()
    worker.join(timeout_s)
    elapsed = time.perf_counter() - start

    if worker.is_alive():
        _cancel(worker)
        worker.join(drain_s)
        if worker.is_alive():
            # 모델 호출에서 막혀 취소 신호가 닿지 않는 작업. 계속 돌면 다음 문항의 시간·VRAM이
            # 오염되므로 경고로 넘기지 않고 라운드를 멈춘다 (이 문항은 기록되지 않아 이어 돌면 다시 돈다).
            raise StuckWorkerError(
                f"{qid}: 시간 초과한 작업이 {drain_s}초 안에 멈추지 않았다. "
                "Ollama 상태를 확인한 뒤 같은 명령으로 이어 돌린다."
            )
        return _empty_result("timeout"), True, max(elapsed, float(timeout_s))
    if "error" in outcome:
        return _empty_result(outcome["error"]), False, elapsed
    if "result" not in outcome:  # 작업이 결과도 예외도 남기지 않고 끝난 경우 (예: SystemExit)
        return _empty_result("pipeline ended without a result"), False, elapsed
    return outcome["result"], False, elapsed


def run_item(
    item: dict[str, Any],
    pipeline: Pipeline,
    *,
    pipeline_name: str,
    bundle: str,
    round_name: str,
    order_idx: int,
    timeout_s: float,
    window: int,
    is_warmup: bool,
    drain_s: float = DEFAULT_DRAIN_S,
) -> dict[str, Any]:
    """문항 하나를 파이프라인으로 돌리고 로그 한 줄을 만든다 (INTERFACES §7.2).

    파이프라인에는 `qid`, `question`만 넘긴다. 시간은 러너가 문항 입력부터 최종 답까지 직접 잰다.
    예외와 시간 초과는 멈추지 않고 오답 한 줄로 남긴다.

    Args:
        item: 동결 문항 한 줄.
        pipeline: `(qid, question) -> PipelineResult`.
        pipeline_name: 로그에 적을 파이프라인 이름 (`p0`, `m` 등).
        bundle: `main` | `ablation` | `dev` | `_scratch`.
        round_name: `r1` | `r2` | `r3`.
        order_idx: 이 파이프라인 안에서의 문항 순번(0부터).
        timeout_s: 문항당 제한 시간(`run.timeout_s`).
        window: 압축에 쓴 `compress.window` (`evidence_kept` 계산용).
        is_warmup: 워밍업 문항인가.
        drain_s: 시간 초과 뒤 작업이 멈추기를 기다리는 최대 시간(초).

    Returns:
        로그 한 줄: 문항 메타 + `PipelineResult` 필드 + 채점 필드 + 측정 필드.
    """
    result, timed_out, latency_s = _call_with_timeout(
        pipeline, item["qid"], item["question"], timeout_s, drain_s
    )
    # 시간 초과·예외는 answer가 빈 문자열이라 채점기가 오답으로 본다.
    score = score_item(item, result.answer)
    return {
        "qid": item["qid"],
        "type": item["type"],
        "subtype": item["subtype"],
        "answer_kind": item["answer_kind"],
        "pipeline": pipeline_name,
        "bundle": bundle,
        "round": round_name,
        "order_idx": order_idx,
        "gold": item["answers"],
        **asdict(result),
        **score,
        "timeout": timed_out,
        "latency_s": round(latency_s, 4),
        "max_input_tokens": max((c.input_tokens for c in result.calls), default=0),
        "evidence_kept": _evidence_kept(item, result, window),
        "is_warmup": is_warmup,
    }


SCRATCH_BUNDLE = "_scratch"

# 묶음이 읽는 문항 세트. 정식 묶음은 이 짝만 허용하고, `_scratch`(검증용, git 제외)는 둘 다 허용한다.
# `_scratch`의 값은 CLI가 `--set`을 생략했을 때 쓰는 기본값이다.
DEFAULT_ITEM_SET = {"main": "eval", "ablation": "eval", "dev": "dev", SCRATCH_BUNDLE: "dev"}


def _check_bundle(bundle: str, item_set: str) -> None:
    """묶음과 문항 세트의 짝이 맞는지 확인한다.

    정식 묶음(`main`·`ablation`)에 개발 세트 결과가 섞이거나 그 반대가 되면 커밋되는 로그가 오염된다.

    Args:
        bundle: 결과 묶음 이름.
        item_set: 문항 세트 이름 (`eval` | `dev`).

    Raises:
        ValueError: 알 수 없는 묶음·세트이거나 짝이 맞지 않을 때.
    """
    if item_set not in ("eval", "dev"):
        raise ValueError(f"item_set은 eval 또는 dev여야 한다: {item_set!r}")
    if bundle not in DEFAULT_ITEM_SET:
        raise ValueError(f"알 수 없는 묶음: {bundle!r} ({', '.join(DEFAULT_ITEM_SET)})")
    if bundle != SCRATCH_BUNDLE and DEFAULT_ITEM_SET[bundle] != item_set:
        raise ValueError(f"묶음 {bundle!r}은 {DEFAULT_ITEM_SET[bundle]} 세트만 쓴다 (받은 값: {item_set!r})")


def load_items(path: Path) -> list[dict[str, Any]]:
    """동결 문항 `.jsonl`을 파일 순서대로 읽는다.

    Args:
        path: `eval.jsonl` 또는 `dev.jsonl`.

    Returns:
        문항 리스트.
    """
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_round(
    *,
    bundle: str,
    round_name: str,
    item_set: str,
    settings: Settings,
    pipelines: dict[str, Pipeline],
    frozen_dir: Path,
    readme_path: Path,
    runs_dir: Path,
    limit: Optional[int] = None,
    environment: Optional[dict[str, Any]] = None,
    sampler_factory: Callable[[], VramSampler] = VramSampler,
    drain_s: float = DEFAULT_DRAIN_S,
    on_row: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """한 라운드를 실행한다: 파이프라인을 라운드별 순서로 하나씩 문항 전체에 돌린다.

    시작 전에 동결 파일 해시를 확인하고(다르면 아무것도 쓰지 않고 멈춘다), 문항마다 즉시 로그에
    쓴다. 같은 라운드를 다시 부르면 이미 기록된 문항은 건너뛰고 이어 돈다. 파이프라인마다 VRAM
    샘플러를 따로 돌려 파이프라인별 최대 VRAM을 `RUN_META.json`에 남긴다. 워밍업(`is_warmup`)은
    파이프라인마다 첫 문항 하나다.

    Args:
        bundle: `main` | `ablation` | `dev` | `_scratch`.
        round_name: `r1` | `r2` | `r3`.
        item_set: `eval` | `dev`. 정식 묶음은 짝이 맞아야 한다.
        settings: 설정 (타임아웃·압축 창·스냅샷을 여기서 읽는다).
        pipelines: 이름 -> 파이프라인. 딕셔너리 순서가 기본 실행 순서다.
        frozen_dir: `data/frozen/`.
        readme_path: 해시 표가 있는 `data/README.md`.
        runs_dir: `results/runs/`.
        limit: 앞에서부터 이 수만큼만 돈다 (검증 모드).
        environment: 환경 정보. 없으면 `collect_environment()`로 모은다.
        sampler_factory: VRAM 샘플러 생성 함수 (테스트에서 가짜로 바꾼다).
        drain_s: 시간 초과한 작업이 멈추기를 기다리는 최대 시간(초).
        on_row: 로그 한 줄을 쓴 직후 부르는 함수 (진행 표시용).

    Returns:
        `RUN_META.json` 내용.

    Raises:
        ValueError: 묶음·세트·라운드 이름이 올바르지 않을 때.
        FrozenDataError: 동결 파일 해시가 기록과 다를 때.
        RunMetaMismatch: 이어 돌기인데 설정·데이터·순서 등이 처음과 다를 때.
    """
    _check_bundle(bundle, item_set)
    order = round_order(list(pipelines), round_name)
    data_hashes = verify_frozen_hashes(frozen_dir, readme_path)
    items = load_items(frozen_dir / f"{item_set}.jsonl")
    if limit is not None:
        items = items[:limit]

    round_dir = runs_dir / bundle / round_name
    meta = RunMeta(round_dir / "RUN_META.json")
    meta.begin(
        bundle=bundle,
        round_name=round_name,
        item_set=item_set,
        settings=settings.to_dict(),
        data_hashes=data_hashes,
        order=order,
        environment=environment if environment is not None else collect_environment(),
    )

    vram: dict[str, Optional[int]] = {}
    try:
        for name in order:
            log = RunLog(round_dir / f"{name}.jsonl")
            pending = [(i, item) for i, item in enumerate(items) if item["qid"] not in log.done_qids]
            if not pending:
                continue
            sampler = sampler_factory()
            sampler.start()
            try:
                for order_idx, item in pending:
                    row = run_item(
                        item,
                        pipelines[name],
                        pipeline_name=name,
                        bundle=bundle,
                        round_name=round_name,
                        order_idx=order_idx,
                        timeout_s=settings.run.timeout_s,
                        window=settings.compress.window,
                        is_warmup=order_idx == 0,
                        drain_s=drain_s,
                    )
                    log.append(row)
                    if on_row is not None:
                        on_row(row)
            finally:
                sampler.stop()
                vram[name] = sampler.peak_mib
    except BaseException:
        meta.record_vram(vram)  # 끊겨도 그때까지의 VRAM은 남긴다 (종료 시각은 비워 둔다)
        raise
    meta.finish(vram_by_pipeline=vram)
    return meta.data
