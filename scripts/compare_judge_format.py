"""실험 C-4: 근거 판정 출력 형식 JSON vs 마지막 줄 태그 (개발 세트).

실행 (레포 루트, Ollama가 떠 있어야 한다):
    python -m scripts.compare_judge_format              # 두 형식 모두. 끊기면 같은 명령으로 이어 돈다
    python -m scripts.compare_judge_format --limit 3    # 앞 3문항만 (동작 확인용)
    python -m scripts.compare_judge_format --summarize  # 저장된 로그로 표만 다시 만든다

그래프(A-5a)가 아직 없으므로 검색 경로를 여기서 직접 잇는다:
검색 → 압축 → 근거 판정 → (RETRIEVE면) 후속 검색어로 재검색 → 압축 → 근거 판정 …
검색형 문항만 쓰므로 라우터는 부르지 않는다. 문항 유형은 문항 고르기와 집계에만 쓰고
모델 호출에는 넘기지 않는다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Sequence

from src.config.llm import create_chat_ollama
from src.config.settings import REPO_ROOT, Settings, get_settings
from src.eval.logger import RunLog, verify_frozen_hashes
from src.eval.runner import load_items
from src.eval.scorer import score_item
from src.eval.stats import mcnemar_exact
from src.meba.budget import BudgetedLLM
from src.meba.compress import compress
from src.meba.judge import OutputFormat, judge
from src.meba.loop_breaker import LoopBreaker
from src.meba.requery import Requery, search_args_key
from src.tools.bm25_search import BM25Index, bm25_search, get_default_index
from src.tools.tokenizer import tokenize

DEV_PATH = Path("data/frozen/dev.jsonl")
OUT_DIR = Path("results/runs/dev/c4_judge_format")
FORMATS: tuple[OutputFormat, ...] = ("json", "tag")
GROUPS = (
    ("전체", lambda r: True),
    ("HotpotQA", lambda r: r["type"] == "hotpotqa"),
    ("SQuAD 답 있음", lambda r: r["subtype"] == "answerable"),
    ("SQuAD 답할 수 없음", lambda r: r["subtype"] == "unanswerable"),
)


def search_items(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """근거 판정을 거치는 문항(텍스트·답할 수 없음)만 고른다. 수치 문항은 수학 경로라 뺀다.

    Args:
        items: 동결 문항

    Returns:
        `answer_kind`가 text 또는 unanswerable인 문항 (파일 순서)
    """
    return [it for it in items if it["answer_kind"] in ("text", "unanswerable")]


def run_search_path(
    item: dict[str, Any],
    *,
    output_format: OutputFormat,
    settings: Settings,
    index: BM25Index,
    llm_factory: Callable[..., Any] = create_chat_ollama,
) -> dict[str, Any]:
    """한 문항을 검색 경로로 끝까지 돌리고 로그 한 줄을 만든다.

    재검색 때는 지금까지 모은 근거에 새 근거를 이어 붙여 판정에 넘긴다.
    루프 끊기에 걸리거나 후속 검색어를 못 만들면, 지금 근거로 RETRIEVE 없는 판정을 받는다.

    Args:
        item: 동결 문항 한 줄 (`question`만 모델 쪽으로 간다)
        output_format: 근거 판정 출력 형식
        settings: 설정 (`retrieval.k`, `compress.*`, `budget.*`, `cap.*`)
        index: 코퍼스 BM25 색인
        llm_factory: ChatOllama 생성 함수 (테스트에서 가짜 모델 주입)

    Returns:
        답·판정·호출 기록·채점 결과를 담은 로그 한 줄
    """
    question = item["question"]
    llm = BudgetedLLM(settings=settings, llm_factory=llm_factory)
    requery = Requery(settings.budget.max_requery)
    breaker = LoopBreaker(normalizer=search_args_key)
    retrieved: list[list[str]] = []
    kept: list[list[str]] = []
    contexts: list[str] = []
    decisions: list[str] = []

    def add_evidence(query: str, followup: str | None) -> bool:
        """검색·압축해 근거를 더한다. 루프 끊기에 걸렸으면 False."""
        outcome = breaker.run("bm25_search", {"query": query}, lambda: bm25_search(query, settings.retrieval.k, index=index))
        if outcome.answer_directly:
            return False
        paragraphs = outcome.result if isinstance(outcome.result, list) else []
        packed = compress(
            question,
            paragraphs,
            m=settings.compress.m,
            window=settings.compress.window,
            tokenizer=tokenize,
            followup_query=followup,
        )
        retrieved.append([p["doc_id"] for p in paragraphs])
        kept.append(packed.kept_windows)
        if packed.context:
            contexts.append(packed.context)
        return True

    start = time.perf_counter()
    final = None
    error = None
    try:
        add_evidence(question, None)
        allow_retrieve = requery.allow_retrieve
        while True:
            j = judge(llm, question, "\n\n".join(contexts), allow_retrieve=allow_retrieve, output_format=output_format)
            if j is None:
                break
            decisions.append(j.decision)
            if j.decision != "RETRIEVE":
                final = j
                break
            followup = requery.next_query(j.missing)
            if followup is not None and add_evidence(followup, followup):
                allow_retrieve = requery.allow_retrieve
            else:
                allow_retrieve = False
    except Exception as exc:  # 모델 오류는 멈추지 않고 오답 한 줄로 남긴다 (러너와 같은 처리)
        error = f"{type(exc).__name__}: {exc}"

    answer = final.answer if final else ""
    calls = [asdict(c) for c in llm.calls]
    return {
        "qid": item["qid"],
        "type": item["type"],
        "subtype": item["subtype"],
        "answer_kind": item["answer_kind"],
        "format": output_format,
        "answer": answer,
        "abstained": bool(final and final.abstained),
        "decisions": decisions,
        "followups": list(requery.queries),
        "budget_exhausted": llm.budget_exhausted,
        "error": error,
        "n_calls": llm.n_calls,
        "n_retries": llm.n_retries,
        "calls": calls,
        "loop_breaks": breaker.loop_breaks,
        "retrieved_doc_ids": retrieved,
        "kept_windows": kept,
        "latency_s": round(time.perf_counter() - start, 4),
        **score_item(item, answer),
    }


def _rate(values: Sequence[bool]) -> float | None:
    """참의 비율. 값이 없으면 None."""
    return sum(values) / len(values) if values else None


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """한 형식·한 그룹의 지표를 계산한다.

    Args:
        rows: 같은 형식의 문항 로그

    Returns:
        정확도, 텍스트 F1, 과잉 기권률, 호출 단위 잘림률·파싱 실패율 등
    """
    calls = [c for r in rows for c in r["calls"]]
    checked = [c for c in calls if c["parse_ok"] is not None]
    answerable = [r for r in rows if r["answer_kind"] == "text"]
    return {
        "n": len(rows),
        "accuracy": _rate([r["correct"] for r in rows]),
        "text_f1": mean(r["f1"] for r in answerable) if answerable else None,
        "over_abstention": _rate([r["abstained"] for r in answerable]),
        "judge_calls": mean(r["n_calls"] for r in rows) if rows else None,
        "retries": sum(r["n_retries"] for r in rows),
        "truncated_rate": _rate([c["truncated"] for c in calls]),
        "truncated_after_retry": sum(c["truncated"] and c["is_retry"] for c in calls),
        "parse_fail_rate": _rate([not c["parse_ok"] for c in checked]),
        "output_tokens": mean(sum(c["output_tokens"] for c in r["calls"]) for r in rows) if rows else None,
        "first_decision": {d: sum(1 for r in rows if r["decisions"][:1] == [d]) for d in ("ANSWER", "RETRIEVE", "ABSTAIN")},
        "budget_exhausted": sum(r["budget_exhausted"] for r in rows),
        "errors": sum(r["error"] is not None for r in rows),
    }


def _fmt(value: Any, pct: bool = False) -> str:
    """표 칸 하나를 문자열로."""
    if value is None:
        return "-"
    if pct:
        return f"{value:.0%}"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def render_summary(logs: dict[str, list[dict[str, Any]]]) -> str:
    """형식별 로그로 마크다운 표를 만든다.

    Args:
        logs: 형식 이름 → 문항 로그

    Returns:
        그룹별 지표 표, 첫 판정 분포, 두 형식의 문항 대응 비교(McNemar)
    """
    lines = [
        "| 그룹 | 형식 | 문항 | 정확도 | 텍스트 F1 | 과잉 기권률 | 판정 호출/문항 | 잘림률(호출) | 재시도 후 잘림 | 파싱 실패율 | 출력 토큰/문항 | 예산 소진 | 오류 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for group, keep in GROUPS:
        for fmt, rows in logs.items():
            s = summarize_rows([r for r in rows if keep(r)])
            lines.append(
                f"| {group} | {fmt} | {s['n']} | {_fmt(s['accuracy'], True)} | {_fmt(s['text_f1'])} | "
                f"{_fmt(s['over_abstention'], True)} | {_fmt(s['judge_calls'])} | {_fmt(s['truncated_rate'], True)} | "
                f"{s['truncated_after_retry']} | {_fmt(s['parse_fail_rate'], True)} | {_fmt(s['output_tokens'])} | "
                f"{s['budget_exhausted']} | {s['errors']} |"
            )
    lines += ["", "첫 판정 분포 (전체)", "", "| 형식 | ANSWER | RETRIEVE | ABSTAIN |", "|---|---|---|---|"]
    for fmt, rows in logs.items():
        d = summarize_rows(rows)["first_decision"]
        lines.append(f"| {fmt} | {d['ANSWER']} | {d['RETRIEVE']} | {d['ABSTAIN']} |")
    if len(logs) == 2:
        (a_name, a_rows), (b_name, b_rows) = logs.items()
        a = {r["qid"]: r["correct"] for r in a_rows}
        b = {r["qid"]: r["correct"] for r in b_rows}
        common = [q for q in a if q in b]
        t = mcnemar_exact((a[q], b[q]) for q in common)
        lines += [
            "",
            f"같은 문항 비교 ({a_name} 대 {b_name}, {t['n']}문항): 둘 다 정답 {t['both_correct']}, "
            f"{a_name}만 정답 {t['m_only']}, {b_name}만 정답 {t['p1_only']}, 둘 다 오답 {t['both_wrong']}, "
            f"McNemar 정확 검정 p = {_fmt(t['p_value'])}",
        ]
    return "\n".join(lines)


def read_log(path: Path) -> list[dict[str, Any]]:
    """저장된 `.jsonl` 로그를 읽는다. 없으면 빈 리스트."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _git_commit() -> str | None:
    """레포의 현재 커밋. git이 없으면 None."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    """두 형식을 개발 세트 검색형 문항에 돌리고(이어 돌기 지원) 요약 표를 쓴다."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=list(FORMATS))
    parser.add_argument("--limit", type=int, default=None, help="앞에서 N문항만 (확인용)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--summarize", action="store_true", help="실행하지 않고 저장된 로그로 표만 만든다")
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    if not args.summarize:
        hashes = verify_frozen_hashes(REPO_ROOT / "data/frozen", REPO_ROOT / "data/README.md")
        settings = get_settings()
        items = search_items(load_items(REPO_ROOT / DEV_PATH))[: args.limit]
        index = get_default_index()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "meta.json").write_text(
            json.dumps({"git_commit": _git_commit(), "data_hashes": hashes, "settings": settings.to_dict(), "n_items": len(items)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for fmt in args.formats:
            log = RunLog(out_dir / f"{fmt}.jsonl")
            todo = [it for it in items if it["qid"] not in log.done_qids]
            print(f"[{fmt}] {len(items)}문항 중 {len(items) - len(todo)}개 완료, {len(todo)}개 실행", flush=True)
            for n, item in enumerate(todo, 1):
                row = run_search_path(item, output_format=fmt, settings=settings, index=index)
                log.append(row)
                print(f"  [{fmt}] {n}/{len(todo)} {row['qid']} {'+'.join(row['decisions']) or '-'} -> {row['answer']!r} correct={row['correct']}", flush=True)

    logs = {fmt: read_log(out_dir / f"{fmt}.jsonl") for fmt in args.formats}
    table = render_summary(logs)
    (out_dir / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
