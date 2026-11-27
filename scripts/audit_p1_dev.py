"""개발 세트 P1 재측정에서 최종 출력 형식과 반복 도구 요청을 집계한다.

실행: python -m scripts.audit_p1_dev --verify 2
      python -m scripts.audit_p1_dev --round r1
      python -m scripts.audit_p1_dev --round r1 --summarize

기존 B-4 결과에 없던 원문과 도구 요청을 별도 개발 라운드에 기록한다.
공통 PipelineResult 및 정식 측정 로그의 스키마는 바꾸지 않는다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:  # 표준 라이브러리만 있는 환경에서도 집계 함수 단위 검증 가능
    BaseCallbackHandler = object  # type: ignore[assignment,misc]

from src.baselines.p1_react import run_p1
from src.config.settings import REPO_ROOT, get_settings
from src.eval.logger import RunLog, RunMeta, collect_environment, verify_frozen_hashes
from src.eval.runner import load_items, run_item


class P1AuditCallback(BaseCallbackHandler):
    """LangChain 모델 응답에서 도구 이름과 인자를 순서대로 보존한다."""

    def __init__(self) -> None:
        self.model_calls = 0
        self.tool_requests: list[dict[str, Any]] = []
        self.invalid_requests = False
        self.raw_final_output: Optional[str] = None

    def on_llm_end(self, response: Any, **_kwargs: Any) -> None:
        """모델 호출 하나의 도구 요청을 기록한다. 토큰 집계 콜백과 독립적이다."""
        self.model_calls += 1
        try:
            generation = response.generations[0][0]
            message = generation.message
            calls = getattr(message, "tool_calls", None) or []
            for call in calls:
                name = call.get("name")
                args = call.get("args")
                if not isinstance(name, str) or not isinstance(args, (dict, str)):
                    self.invalid_requests = True
                    continue
                self.tool_requests.append({"name": name, "args": args})
            if getattr(message, "invalid_tool_calls", None):
                self.invalid_requests = True
        except (AttributeError, IndexError, KeyError, TypeError):
            self.invalid_requests = True


def _normalize_value(value: Any) -> Any:
    """인자 문자열의 공백·대소문자와 객체 키 순서를 정규화한다."""
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    return value


def _request_key(request: dict[str, Any]) -> tuple[str, str]:
    """도구 이름과 정규화한 인자를 비교 가능한 키로 만든다."""
    args = request["args"]
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    return (
        str(request["name"]),
        json.dumps(_normalize_value(args), ensure_ascii=False, sort_keys=True),
    )


def has_final_answer_line(raw_output: str) -> bool:
    """마지막 비어 있지 않은 줄이 요구된 최종 답 형식인지 판정한다."""
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    return bool(lines and re.fullmatch(r"Final answer:[ \t]*\S.*", lines[-1], flags=re.IGNORECASE))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """형식 오류율과 반복 호출·반복 루프 비율의 분자·분모를 계산한다."""
    format_rows = [
        row for row in rows
        if not row.get("timeout") and not row.get("error")
        and isinstance(row.get("raw_final_output"), str)
        and bool(row["raw_final_output"].strip())
    ]
    bad_format = sum(not has_final_answer_line(row["raw_final_output"]) for row in format_rows)

    history_rows = [row for row in rows if row.get("trace_complete") is True]
    duplicate_requests = 0
    total_requests = 0
    loop_questions = 0
    for row in history_rows:
        seen: set[tuple[str, str]] = set()
        duplicates_this_item = 0
        for request in row["tool_requests"]:
            total_requests += 1
            key = _request_key(request)
            if key in seen:
                duplicates_this_item += 1
            seen.add(key)
        duplicate_requests += duplicates_this_item
        loop_questions += int(duplicates_this_item > 0)

    return {
        "items": len(rows),
        "format_errors": bad_format,
        "format_checked": len(format_rows),
        "format_error_rate": bad_format / len(format_rows) if format_rows else None,
        "duplicate_requests": duplicate_requests,
        "tool_requests_checked": total_requests,
        "repeat_call_rate": duplicate_requests / total_requests if total_requests else None,
        "loop_questions": loop_questions,
        "history_checked": len(history_rows),
        "repeat_loop_rate": loop_questions / len(history_rows) if history_rows else None,
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """완성된 JSONL 행을 읽고 중복 문항을 거부한다."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    qids = [row["qid"] for row in rows]
    if len(qids) != len(set(qids)):
        raise ValueError(f"중복 문항이 있다: {path}")
    return rows


def _run_dir(round_name: str, verify: Optional[int]) -> Path:
    """정식 개발 재측정과 검증 실행의 저장 위치를 구분한다."""
    if verify is not None:
        return REPO_ROOT / "results" / "runs" / "_scratch" / "d3_p1" / round_name
    return REPO_ROOT / "results" / "audits" / "p1_d3" / round_name


def main(argv: Optional[Sequence[str]] = None) -> None:
    """동결 개발 문항을 이어서 돌리거나 저장된 감사 로그를 집계한다."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--round", default="r1", choices=("r1", "r2", "r3"))
    parser.add_argument("--verify", type=int, metavar="N", help="앞 N개 개발 문항만 검증")
    parser.add_argument("--summarize", action="store_true", help="기존 재측정 로그 집계만 출력")
    parser.add_argument("--config", default="configs/p1_d3.yaml", help="설정 파일 (레포 루트 기준)")
    args = parser.parse_args(argv)
    if args.verify is not None and args.verify <= 0:
        parser.error("--verify는 1 이상이어야 한다")

    run_dir = _run_dir(args.round, args.verify)
    log_path = run_dir / "p1_audit.jsonl"
    if args.summarize:
        if not log_path.is_file():
            parser.error(f"재측정 로그가 없다: {log_path}")
        print(json.dumps(summarize(_read_rows(log_path)), ensure_ascii=False, indent=2))
        return

    frozen_dir = REPO_ROOT / "data" / "frozen"
    data_hashes = verify_frozen_hashes(frozen_dir, REPO_ROOT / "data" / "README.md")
    cfg = get_settings(args.config)
    items = load_items(frozen_dir / "dev.jsonl")
    if args.verify is not None:
        items = items[: args.verify]
    meta = RunMeta(run_dir / "RUN_META.json")
    meta.begin(
        bundle="_scratch" if args.verify is not None else "dev",
        round_name=args.round,
        item_set="dev",
        settings=cfg.to_dict(),
        data_hashes=data_hashes,
        order=["p1_audit"],
        environment=collect_environment(),
    )
    log = RunLog(log_path)
    session_idx = len(meta.data["sessions"]) - 1
    for order_idx, item in enumerate(items):
        if item["qid"] in log.done_qids:
            continue
        trace = P1AuditCallback()

        def pipeline(qid: str, question: str) -> Any:
            """문항별 모델 추적 콜백을 붙여 공통 P1을 실행한다."""
            return run_p1(qid, question, settings=cfg, trace_callback=trace)

        row = run_item(
            item,
            pipeline,
            pipeline_name="p1",
            bundle="_scratch" if args.verify is not None else "dev",
            round_name=args.round,
            order_idx=order_idx,
            timeout_s=cfg.run.timeout_s,
            window=cfg.compress.window,
            is_warmup=order_idx == 0,
            session_idx=session_idx,
        )
        row.update({
            "raw_final_output": trace.raw_final_output,
            "tool_requests": trace.tool_requests,
            "trace_model_calls": trace.model_calls,
            "trace_complete": (
                not row["timeout"]
                and not trace.invalid_requests
                and trace.model_calls > 0
                and trace.model_calls == row["n_calls"]
                and (row["error"] is None or row["error"].startswith("RecursionLimitExceeded"))
            ),
        })
        log.append(row)
        print(f"[{order_idx + 1}/{len(items)}] {item['qid']}", file=sys.stderr, flush=True)
    meta.finish(vram_by_pipeline={})
    print(json.dumps(summarize(_read_rows(log_path)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
