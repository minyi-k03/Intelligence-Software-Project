"""동일한 M 개발 세트를 `cap.judge` 조건별로 실행하고 실제 종료 사유를 기록한다.

실행: python -m scripts.run_cap_sensitivity --verify 2
      python -m scripts.run_cap_sensitivity
      python -m scripts.run_cap_sensitivity --summarize

각 조건은 `configs/d4_cap_sensitivity.json`의 독립 라운드를 사용한다. 정식 문항 로그는
기존 `run_round`가 즉시 저장하고, 종료 사유·원문은 같은 폴더의 보조 JSONL에 저장한다.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, Sequence

from src.config.settings import REPO_ROOT, Settings, get_settings
from src.config.llm import ChatOllama
from src.eval.logger import RunLog, verify_frozen_hashes


def load_conditions(path: Path) -> tuple[str, list[tuple[int, str]]]:
    """실험 설정에서 기본 설정 경로와 `(judge 상한, 라운드)` 목록을 읽는다.

    Args:
        path: 레포 루트 기준 실험 JSON 설정 파일.

    Returns:
        기본 설정 경로와 상한·라운드 순서쌍.

    Raises:
        ValueError: 중복·잘못된 상한이나 라운드가 있을 때.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    base = raw.get("base_config")
    entries = raw.get("conditions")
    if not isinstance(base, str) or not base or not isinstance(entries, list) or not entries:
        raise ValueError("base_config와 비어 있지 않은 conditions가 필요하다")
    conditions: list[tuple[int, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("각 조건은 객체여야 한다")
        cap, round_name = entry.get("judge_cap"), entry.get("round")
        if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
            raise ValueError(f"judge_cap은 양의 정수여야 한다: {cap!r}")
        if not isinstance(round_name, str) or re.fullmatch(r"r[1-9]\d*", round_name) is None:
            raise ValueError(f"라운드 이름이 잘못됐다: {round_name!r}")
        conditions.append((cap, round_name))
    if len({cap for cap, _ in conditions}) != len(conditions):
        raise ValueError("judge_cap이 중복됐다")
    if len({round_name for _, round_name in conditions}) != len(conditions):
        raise ValueError("라운드 이름이 중복됐다")
    return base, conditions


def settings_for_cap(base: Settings, cap: int) -> Settings:
    """공통 설정에서 근거 판정 상한 하나만 교체한 설정을 돌려준다.

    Args:
        base: 기본 설정.
        cap: 실험 설정에서 읽은 판정 출력 상한.

    Returns:
        다른 설정값이 모두 같은 `Settings`.
    """
    return replace(base, cap=replace(base.cap, judge=cap))


def _done_reason(response: Any) -> Optional[str]:
    """ChatOllama 응답의 명시적인 모델 종료 사유를 읽는다.

    Args:
        response: `ChatOllama.invoke`의 AIMessage.

    Returns:
        종료 사유, 없으면 None. 토큰 상한 도달만으로 추정하지 않는다.
    """
    metadata = getattr(response, "response_metadata", None)
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("done_reason") or metadata.get("finish_reason")
    return str(value) if value is not None else None


class ReasonTrace:
    """문항별 호출의 종료 사유와 원문을 정식 로그 옆에 기록할 때 쓰는 메모리 버퍼."""

    def __init__(self) -> None:
        self._current = threading.local()
        self._lock = threading.Lock()
        self._events: dict[str, list[dict[str, Any]]] = {}

    def begin(self, qid: str) -> None:
        """워커 스레드의 현재 문항을 설정한다.

        Args:
            qid: 문항 ID.
        """
        self._current.qid = qid
        with self._lock:
            self._events[qid] = []

    def record(self, response: Any, metrics: Any) -> None:
        """LangChain 호출 하나가 끝나면 모델 응답의 종료 사유를 보존한다.

        Args:
            response: `ChatOllama.invoke`의 응답.
            metrics: 공통 LangChain 토큰 집계 콜백.
        """
        qid = getattr(self._current, "qid", None)
        if qid is None:
            raise RuntimeError("문항 ID 없이 모델이 호출됐다")
        event = {
            "step": metrics.current_step,
            "cap": metrics.cap,
            "is_retry": metrics.is_retry,
            "done_reason": _done_reason(response),
            "raw_output": getattr(response, "content", None),
        }
        with self._lock:
            self._events[qid].append(event)

    def take(self, qid: str) -> list[dict[str, Any]]:
        """정식 문항 로그를 쓴 뒤 보조 추적 기록을 꺼낸다.

        Args:
            qid: 문항 ID.

        Returns:
            이 문항에서 끝난 모델 호출의 기록 목록.
        """
        with self._lock:
            return self._events.pop(qid, [])


class TracingModel:
    """공통 ChatOllama 응답을 관찰하되 모델 호출·집계 경로는 바꾸지 않는다."""

    def __init__(self, model: Any, metrics: Any, trace: ReasonTrace) -> None:
        self._model = model
        self._metrics = metrics
        self._trace = trace

    def invoke(self, model_input: Any, **kwargs: Any) -> Any:
        """ChatOllama를 호출한 뒤 그 응답의 종료 사유와 원문을 기록한다.

        Args:
            model_input: 공통 모델 입력.
            kwargs: 공통 호출 옵션.

        Returns:
            모델의 원래 응답.
        """
        response = self._model.invoke(model_input, **kwargs)
        self._trace.record(response, self._metrics)
        return response


def summarize(rows: list[dict[str, Any]], traces: list[dict[str, Any]]) -> dict[str, Any]:
    """정식 로그와 모델 종료 사유를 대응시켜 조건 하나의 지표를 계산한다.

    Args:
        rows: `run_round`가 쓴 문항별 결과.
        traces: 같은 문항의 보조 종료 사유 기록.

    Returns:
        정확도·출력 토큰·형식 오류·확인된 잘림 및 미확인 수.

    Raises:
        ValueError: 문항 ID가 중복되거나 추적 기록이 중복됐을 때.
    """
    qids = [row["qid"] for row in rows]
    trace_qids = [row["qid"] for row in traces]
    if len(qids) != len(set(qids)) or len(trace_qids) != len(set(trace_qids)):
        raise ValueError("문항 또는 종료 사유 추적에 중복 qid가 있다")
    by_trace = {row["qid"]: row for row in traces}
    cost = [row for row in rows if not row["is_warmup"]]
    total_output = sum(call["output_tokens"] for row in cost for call in row["calls"])
    types = sorted({row["type"] for row in rows})
    judge_calls = [call for row in rows for call in row["calls"] if call["step"] == "judge"]
    cost_judge_calls = [call for row in cost for call in row["calls"] if call["step"] == "judge"]
    judge_with_cap = [call for call in judge_calls if isinstance(call.get("cap"), int)]
    cap_reached = sum(call["output_tokens"] >= call["cap"] for call in judge_with_cap)
    format_checked = [call for call in judge_calls if isinstance(call["parse_ok"], bool)]
    reason_known = 0
    reason_truncated = 0
    reason_unknown = 0
    cap_only_flags = 0
    truncation_cases: list[dict[str, Any]] = []
    for row in rows:
        trace = by_trace.get(row["qid"])
        calls = row["calls"]
        events = trace.get("calls", []) if trace else []
        complete = bool(trace and trace.get("trace_complete") and len(events) == len(calls))
        for index, call in enumerate(calls):
            if call["step"] != "judge":
                continue
            event = events[index] if complete else None
            reason = event.get("done_reason") if event and event.get("step") == "judge" else None
            if reason is None:
                reason_unknown += 1
            else:
                reason_known += 1
                reason_truncated += reason in {"length", "max_tokens"}
                if reason in {"length", "max_tokens"}:
                    truncation_cases.append({
                        "qid": row["qid"], "cap": call["cap"],
                        "output_tokens": call["output_tokens"], "is_retry": call["is_retry"],
                    })
                cap_only_flags += bool(call["truncated"] and reason not in {"length", "max_tokens"})
    total_latency = sum(row["latency_s"] for row in cost)
    return {
        "items": len(rows),
        "correct": sum(bool(row["correct"]) for row in rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows) if rows else None,
        "by_type": {
            kind: {
                "correct": sum(bool(row["correct"]) for row in rows if row["type"] == kind),
                "items": sum(row["type"] == kind for row in rows),
                "accuracy": (
                    sum(bool(row["correct"]) for row in rows if row["type"] == kind)
                    / sum(row["type"] == kind for row in rows)
                ),
            }
            for kind in types
        },
        "cost_items": len(cost),
        "output_tokens": total_output,
        "mean_output_tokens": total_output / len(cost) if cost else None,
        "judge_output_tokens": sum(call["output_tokens"] for call in cost_judge_calls),
        "judge_output_calls": len(cost_judge_calls),
        "judge_mean_output_tokens": (
            sum(call["output_tokens"] for call in cost_judge_calls) / len(cost_judge_calls)
            if cost_judge_calls else None
        ),
        "judge_cap_reached": cap_reached,
        "judge_cap_checked": len(judge_with_cap),
        "judge_cap_reached_rate": cap_reached / len(judge_with_cap) if judge_with_cap else None,
        "total_latency_s": total_latency,
        "mean_latency_s": total_latency / len(cost) if cost else None,
        "timeouts": sum(bool(row["timeout"]) for row in rows),
        "format_errors": sum(call["parse_ok"] is False for call in format_checked),
        "format_checked": len(format_checked),
        "format_error_rate": (
            sum(call["parse_ok"] is False for call in format_checked) / len(format_checked)
            if format_checked else None
        ),
        "confirmed_truncations": reason_truncated,
        "judge_reasons_checked": reason_known,
        "judge_reasons_unknown": reason_unknown,
        "confirmed_truncation_rate": reason_truncated / reason_known if reason_known else None,
        "confirmed_truncation_cases": truncation_cases,
        "cap_only_truncated_flags": cap_only_flags,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """완성된 JSONL 로그를 읽는다.

    Args:
        path: 로그 경로.

    Returns:
        문항별 사전 목록.
    """
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compare_conditions(
    runs: dict[int, tuple[list[dict[str, Any]], dict[str, Any]]],
) -> dict[str, Any]:
    """같은 문항을 256 상한과 대응시키고 잘림 없는 정확도 유지 후보를 찾는다.

    Args:
        runs: 출력 상한별 `(문항 로그, 조건 요약)`.

    Returns:
        정오답 변화와 보수적인 적합 여부, 토큰 효율상 권장 후보.

    Raises:
        ValueError: 256 기준이 없거나 문항·유형·워밍업이 다를 때.
    """
    if 256 not in runs:
        raise ValueError("256 기준 조건이 필요하다")
    baseline_rows, baseline = runs[256]
    baseline_by = {row["qid"]: row for row in baseline_rows}
    if len(baseline_by) != len(baseline_rows):
        raise ValueError("256 기준에 중복 qid가 있다")
    comparisons: dict[int, dict[str, Any]] = {}
    for cap, (rows, stats) in runs.items():
        by_qid = {row["qid"]: row for row in rows}
        if len(by_qid) != len(rows) or set(by_qid) != set(baseline_by):
            raise ValueError(f"{cap} 조건의 문항 ID가 256과 다르다")
        for qid, row in by_qid.items():
            reference = baseline_by[qid]
            if row["type"] != reference["type"] or row["is_warmup"] != reference["is_warmup"]:
                raise ValueError(f"{cap} 조건의 문항 유형·워밍업이 다르다: {qid}")
        improved = [qid for qid in by_qid if by_qid[qid]["correct"] and not baseline_by[qid]["correct"]]
        regressed = [qid for qid in by_qid if baseline_by[qid]["correct"] and not by_qid[qid]["correct"]]
        accuracy_maintained = (
            stats["correct"] >= baseline["correct"]
            and all(stats["by_type"][kind]["correct"] >= reference["correct"]
                    for kind, reference in baseline["by_type"].items())
        )
        no_confirmed_truncation = (
            stats["judge_reasons_checked"] > 0
            and stats["judge_reasons_unknown"] == 0
            and stats["confirmed_truncations"] == 0
        )
        comparisons[cap] = {
            "improved_qids": improved,
            "regressed_qids": regressed,
            "accuracy_maintained": accuracy_maintained,
            "no_confirmed_truncation": no_confirmed_truncation,
            "eligible": (
                accuracy_maintained and no_confirmed_truncation
                and stats["mean_output_tokens"] is not None
            ),
        }
    eligible = [cap for cap, result in comparisons.items() if result["eligible"]]
    recommended = (
        min(eligible, key=lambda cap: (runs[cap][1]["mean_output_tokens"], cap))
        if eligible else None
    )
    return {"baseline_cap": 256, "conditions": comparisons, "recommended_cap": recommended}


def run_condition(settings: Settings, round_name: str, *, limit: Optional[int]) -> dict[str, Any]:
    """M 조건 하나를 독립 라운드로 실행하고 정식·종료 사유 로그를 쓴다.

    Args:
        settings: `cap.judge`만 교체한 설정.
        round_name: 이 조건 전용 라운드.
        limit: 검증 모드의 앞 N문항, 없으면 개발 세트 전체.

    Returns:
        실행 결과 요약.
    """
    if ChatOllama is None:
        raise RuntimeError("langchain-ollama가 없어 D-4 실험을 시작할 수 없다")
    from src.config.llm import create_chat_ollama
    from src.eval.runner import run_round
    from src.meba.graph import build_m

    trace = ReasonTrace()

    def factory(**kwargs: Any) -> TracingModel:
        """공통 ChatOllama에 모델 종료 사유 관찰만 덧댄다."""
        metrics = kwargs["callbacks"][0]
        return TracingModel(create_chat_ollama(**kwargs), metrics, trace)

    base = build_m(settings, llm_factory=factory)

    def pipeline(qid: str, question: str) -> Any:
        """문항 ID를 추적에 연결하고 원래 M을 실행한다."""
        trace.begin(qid)
        return base(qid, question)

    bundle = "_scratch" if limit is not None else "dev"
    runs_dir = REPO_ROOT / "results" / "runs"
    round_dir = runs_dir / bundle / round_name
    verify_frozen_hashes(REPO_ROOT / "data" / "frozen", REPO_ROOT / "data" / "README.md")
    trace_log = RunLog(round_dir / "m_reason_trace.jsonl")

    def on_row(row: dict[str, Any]) -> None:
        """정식 문항 로그가 저장된 직후 보조 종료 사유를 즉시 저장한다."""
        events = trace.take(row["qid"])
        trace_log.append({
            "qid": row["qid"],
            "calls": events,
            "trace_complete": not row["timeout"] and row["error"] is None and len(events) == len(row["calls"]),
        })

    run_round(
        bundle=bundle,
        round_name=round_name,
        item_set="dev",
        settings=settings,
        pipelines={"m": pipeline},
        frozen_dir=REPO_ROOT / "data" / "frozen",
        readme_path=REPO_ROOT / "data" / "README.md",
        runs_dir=runs_dir,
        limit=limit,
        on_row=on_row,
    )
    return summarize(_read_jsonl(round_dir / "m.jsonl"), _read_jsonl(trace_log.path))


def main(argv: Optional[Sequence[str]] = None) -> None:
    """설정된 출력 상한 조건을 순서대로 실행하거나 저장된 로그를 집계한다.

    Args:
        argv: CLI 인자. 없으면 명령행에서 읽는다.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", default="configs/d4_cap_sensitivity.json")
    parser.add_argument("--verify", type=int, metavar="N")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)
    if args.verify is not None and args.verify <= 0:
        parser.error("--verify는 1 이상이어야 한다")
    base_path, conditions = load_conditions(REPO_ROOT / args.plan)
    base = None if args.summarize else get_settings(base_path)
    bundle = "_scratch" if args.verify is not None else "dev"
    completed: dict[int, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for cap, round_name in conditions:
        path = REPO_ROOT / "results" / "runs" / bundle / round_name
        if args.summarize:
            rows = _read_jsonl(path / "m.jsonl")
            result = summarize(rows, _read_jsonl(path / "m_reason_trace.jsonl"))
        else:
            assert base is not None
            settings = settings_for_cap(base, cap)
            result = run_condition(settings, round_name, limit=args.verify)
            rows = _read_jsonl(path / "m.jsonl")
        completed[cap] = (rows, result)
        print(json.dumps({"cap_judge": cap, "round": round_name, **result}, ensure_ascii=False))
    print(json.dumps({"comparison": compare_conditions(completed), "verify_only": args.verify is not None},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
