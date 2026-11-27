"""기본 M과 간결 M을 같은 개발 라운드에서 비교하고 수학 원문을 보존한다.

실행: python -m scripts.run_concise_experiment --verify 2
      python -m scripts.run_concise_experiment
      python -m scripts.run_concise_experiment --summarize
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence

from src.config.settings import REPO_ROOT, get_settings
from src.config.llm import ChatOllama
from src.eval.logger import RunLog, verify_frozen_hashes
from scripts.run_cap_sensitivity import ReasonTrace, TracingModel, _read_jsonl, summarize


_FINAL_LINE = re.compile(r"Final answer:[ \t]*\S.*", re.IGNORECASE)


def _text(content: Any) -> str:
    """ChatOllama 응답 content의 텍스트만 꺼낸다.

    Args:
        content: 문자열 또는 텍스트 블록 목록.

    Returns:
        이어 붙인 텍스트, 없으면 빈 문자열.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part if isinstance(part, str) else str(part.get("text") or "")
            for part in content if isinstance(part, (str, dict))
        )
    return ""


def _has_final_line(raw: str) -> bool:
    """자유 텍스트 출력의 마지막 줄이 공통 최종 답 형식인지 판정한다.

    Args:
        raw: 모델 응답 원문.

    Returns:
        마지막 비어 있지 않은 줄이 `Final answer: <답>`이면 True.
    """
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return bool(lines and _FINAL_LINE.fullmatch(lines[-1]))


def _trace_for(row: dict[str, Any], trace: Optional[dict[str, Any]]) -> Optional[list[dict[str, Any]]]:
    """정식 호출 기록과 길이가 맞는 완성된 보조 추적만 돌려준다.

    Args:
        row: 정식 문항 로그.
        trace: 같은 qid의 보조 로그, 없을 수 있다.

    Returns:
        순서가 맞는 호출 기록 목록, 아니면 None.
    """
    if trace and trace.get("trace_complete") and len(trace.get("calls", [])) == len(row["calls"]):
        return trace["calls"]
    return None


def _format_counts(rows: list[dict[str, Any]], traces: list[dict[str, Any]]) -> dict[str, int]:
    """수학의 최종 답 줄과 검색의 최종 JSON 파싱을 문항별로 센다.

    Args:
        rows: 한 파이프라인의 정식 문항 로그.
        traces: 모델 원문 보조 로그.

    Returns:
        형식 확인 문항 수, 오류 문항 수, 미확인 문항 수.
    """
    by_trace = {trace["qid"]: trace for trace in traces}
    checked = errors = unknown = 0
    for row in rows:
        if row["timeout"] or row["error"]:
            unknown += 1
            continue
        if row["route"] == "search":
            judges = [call for call in row["calls"] if call["step"] == "judge"]
            if not judges or not isinstance(judges[-1]["parse_ok"], bool):
                unknown += 1
            else:
                checked += 1
                errors += judges[-1]["parse_ok"] is False
        elif row["route"] == "math":
            events = _trace_for(row, by_trace.get(row["qid"]))
            answers = [event for event in events or [] if event["step"] in {"router", "finalize"}]
            raw = _text(answers[-1].get("raw_output")) if answers else ""
            if not raw:
                unknown += 1
            else:
                checked += 1
                errors += not _has_final_line(raw)
        else:
            unknown += 1
    return {"checked": checked, "errors": errors, "unknown": unknown}


def _reason_counts(rows: list[dict[str, Any]], traces: list[dict[str, Any]]) -> dict[str, int]:
    """모든 단계의 명시적 종료 사유로 확인된 잘림과 미확인을 센다.

    Args:
        rows: 정식 문항 로그.
        traces: 보조 모델 응답 로그.

    Returns:
        종료 사유 확인·미확인 호출 수와 확인된 잘림 수.
    """
    by_trace = {trace["qid"]: trace for trace in traces}
    known = unknown = truncated = 0
    for row in rows:
        events = _trace_for(row, by_trace.get(row["qid"]))
        for index, _call in enumerate(row["calls"]):
            reason = events[index].get("done_reason") if events else None
            if reason is None:
                unknown += 1
            else:
                known += 1
                truncated += reason in {"length", "max_tokens"}
    return {"checked": known, "unknown": unknown, "confirmed_truncations": truncated}


def compare(
    baseline: list[dict[str, Any]],
    concise: list[dict[str, Any]],
    baseline_traces: list[dict[str, Any]],
    concise_traces: list[dict[str, Any]],
) -> dict[str, Any]:
    """동일 문항의 기본 M과 간결 M을 쌍비교한다.

    Args:
        baseline: 기본 M 문항 로그.
        concise: 간결 M 문항 로그.
        baseline_traces: 기본 M 모델 원문·종료 사유.
        concise_traces: 간결 M 모델 원문·종료 사유.

    Returns:
        출력 감소율, 유형별 순정답 수, 형식·잘림 및 수학 원문 점검 대상.

    Raises:
        ValueError: 문항 집합·유형·워밍업 조건이 맞지 않을 때.
    """
    base_by = {row["qid"]: row for row in baseline}
    concise_by = {row["qid"]: row for row in concise}
    if len(base_by) != len(baseline) or len(concise_by) != len(concise) or set(base_by) != set(concise_by):
        raise ValueError("두 조건의 문항 ID가 일치하지 않거나 중복됐다")
    for qid in base_by:
        left, right = base_by[qid], concise_by[qid]
        if left["type"] != right["type"] or left["is_warmup"] != right["is_warmup"]:
            raise ValueError(f"유형 또는 워밍업 조건이 다른 문항: {qid}")
    base_stats = summarize(baseline, baseline_traces)
    concise_stats = summarize(concise, concise_traces)
    before = base_stats["mean_output_tokens"]
    after = concise_stats["mean_output_tokens"]
    reduction = (before - after) / before * 100 if before else None
    kinds = sorted({row["type"] for row in baseline})
    by_type = {}
    for kind in kinds:
        improved = sum(not base_by[qid]["correct"] and concise_by[qid]["correct"]
                       for qid in base_by if base_by[qid]["type"] == kind)
        regressed = sum(base_by[qid]["correct"] and not concise_by[qid]["correct"]
                        for qid in base_by if base_by[qid]["type"] == kind)
        by_type[kind] = {"improved": improved, "regressed": regressed,
                         "net_correct": improved - regressed,
                         "baseline_correct": base_stats["by_type"][kind]["correct"],
                         "concise_correct": concise_stats["by_type"][kind]["correct"]}
    base_trace_by = {trace["qid"]: trace for trace in baseline_traces}
    concise_trace_by = {trace["qid"]: trace for trace in concise_traces}
    math_review = []
    for qid, row in base_by.items():
        if row["route"] != "math":
            continue
        left = _trace_for(row, base_trace_by.get(qid))
        right = _trace_for(concise_by[qid], concise_trace_by.get(qid))
        if left is not None and right is not None:
            math_review.append(qid)
    return {
        "items": len(baseline),
        "baseline": base_stats,
        "concise": concise_stats,
        "output_reduction_pct": reduction,
        "by_type": by_type,
        "baseline_format": _format_counts(baseline, baseline_traces),
        "concise_format": _format_counts(concise, concise_traces),
        "baseline_reasons": _reason_counts(baseline, baseline_traces),
        "concise_reasons": _reason_counts(concise, concise_traces),
        "math_raw_review_qids": math_review,
        "math_step_preservation": "manual_review_required",
    }


def run_experiment(base_config: str, round_name: str, *, limit: Optional[int]) -> dict[str, Any]:
    """같은 세션의 개발 라운드에서 두 파이프라인을 실행한다.

    Args:
        base_config: 공통 설정 경로.
        round_name: 실험 전용 라운드.
        limit: 검증 모드 앞 N문항, 없으면 개발 세트 전체.

    Returns:
        저장된 로그의 쌍비교 결과.
    """
    if ChatOllama is None:
        raise RuntimeError("langchain-ollama가 없어 D-5 실험을 시작할 수 없다")
    from src.config.llm import create_chat_ollama
    from src.eval.runner import run_round
    from src.meba.conditional.concise import build_m_concise
    from src.meba.graph import build_m

    settings = get_settings(base_config)
    traces = {"m": ReasonTrace(), "m_concise": ReasonTrace()}

    def make_factory(name: str) -> Any:
        """파이프라인 하나의 모델 원문 관찰 생성 함수를 만든다."""
        def factory(**kwargs: Any) -> TracingModel:
            """공통 ChatOllama와 콜백을 그대로 쓰고 응답만 관찰한다."""
            return TracingModel(create_chat_ollama(**kwargs), kwargs["callbacks"][0], traces[name])
        return factory

    def tracked(name: str, base: Any) -> Any:
        """워커 스레드의 문항 ID를 모델 원문 기록에 연결한다."""
        def pipeline(qid: str, question: str) -> Any:
            """한 문항을 공통 그래프로 실행한다."""
            traces[name].begin(qid)
            return base(qid, question)
        return pipeline

    pipelines = {
        "m": tracked("m", build_m(settings, llm_factory=make_factory("m"))),
        "m_concise": tracked("m_concise", build_m_concise(settings, llm_factory=make_factory("m_concise"))),
    }
    bundle = "_scratch" if limit is not None else "dev"
    round_dir = REPO_ROOT / "results" / "runs" / bundle / round_name
    verify_frozen_hashes(REPO_ROOT / "data" / "frozen", REPO_ROOT / "data" / "README.md")
    trace_logs = {name: RunLog(round_dir / f"{name}_reason_trace.jsonl") for name in pipelines}

    def on_row(row: dict[str, Any]) -> None:
        """정식 문항 로그 다음에 같은 문항의 모델 원문을 즉시 기록한다."""
        name = row["pipeline"]
        events = traces[name].take(row["qid"])
        trace_logs[name].append({
            "qid": row["qid"], "calls": events,
            "trace_complete": not row["timeout"] and row["error"] is None and len(events) == len(row["calls"]),
        })

    run_round(
        bundle=bundle, round_name=round_name, item_set="dev", settings=settings,
        pipelines=pipelines, frozen_dir=REPO_ROOT / "data" / "frozen",
        readme_path=REPO_ROOT / "data" / "README.md",
        runs_dir=REPO_ROOT / "results" / "runs", limit=limit, on_row=on_row,
    )
    return compare(
        _read_jsonl(round_dir / "m.jsonl"), _read_jsonl(round_dir / "m_concise.jsonl"),
        _read_jsonl(trace_logs["m"].path), _read_jsonl(trace_logs["m_concise"].path),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """설정된 개발 라운드를 실행하거나 기존 로그만 다시 집계한다.

    Args:
        argv: CLI 인자. 없으면 명령행에서 읽는다.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", default="configs/d5_concise.json")
    parser.add_argument("--verify", type=int, metavar="N")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)
    if args.verify is not None and args.verify <= 0:
        parser.error("--verify는 1 이상이어야 한다")
    plan = json.loads((REPO_ROOT / args.plan).read_text(encoding="utf-8"))
    base_config, round_name = plan["base_config"], plan["round"]
    if not isinstance(base_config, str) or not isinstance(round_name, str) or re.fullmatch(r"r[1-9]\d*", round_name) is None:
        parser.error("base_config 또는 round 설정이 잘못됐다")
    if args.summarize:
        bundle = "_scratch" if args.verify is not None else "dev"
        round_dir = REPO_ROOT / "results" / "runs" / bundle / round_name
        result = compare(
            _read_jsonl(round_dir / "m.jsonl"), _read_jsonl(round_dir / "m_concise.jsonl"),
            _read_jsonl(round_dir / "m_reason_trace.jsonl"),
            _read_jsonl(round_dir / "m_concise_reason_trace.jsonl"),
        )
    else:
        result = run_experiment(base_config, round_name, limit=args.verify)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
