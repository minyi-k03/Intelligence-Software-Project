"""실험 C-4: 근거 판정 출력 형식 JSON vs 마지막 줄 태그 (개발 세트).

실행 (레포 루트, Ollama가 떠 있어야 한다):
    python -m scripts.compare_judge_format              # 두 형식 모두. 끊기면 같은 명령으로 이어 돈다
    python -m scripts.compare_judge_format --limit 3    # 앞 3문항만 (동작 확인용)
    python -m scripts.compare_judge_format --summarize  # 저장된 로그로 표만 다시 만든다
    python -m scripts.compare_judge_format --formats json --out-dir results/runs/dev/x \
        --compare base=results/runs/dev/a new=results/runs/dev/b      # 두 실행을 나란히 비교

그래프(A-5a)가 아직 없으므로 검색 경로를 여기서 직접 잇는다:
검색 → 압축 → 근거 판정 → (RETRIEVE면) 후속 검색어로 재검색 → 압축 → 근거 판정 …
검색형 문항만 쓰므로 라우터는 부르지 않는다. 문항 유형은 문항 고르기와 집계에만 쓰고
모델 호출에는 넘기지 않는다.

측정 규칙은 본 측정 러너와 같다 (EVALUATION §4, INTERFACES §7): 문항은 `run_item`으로 돌려
`run.timeout_s`를 넘기면 오답으로 남기고, 형식마다 첫 문항은 워밍업으로 시간·비용 집계에서 뺀다.
이어 돌 때 설정·데이터·형식 목록이 처음과 다르면 `RunMeta`가 멈춘다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Sequence

from src.common.types import PipelineResult
from src.config.llm import create_chat_ollama
from src.config.settings import REPO_ROOT, Settings, get_settings
from src.eval.logger import RunLog, RunMeta, collect_environment, verify_frozen_hashes
from src.eval.runner import load_items, run_item
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


def make_pipeline(
    output_format: OutputFormat,
    *,
    settings: Settings,
    index: BM25Index,
    llm_factory: Callable[..., Any] = create_chat_ollama,
    followups: dict[str, list[str]] | None = None,
) -> Callable[[str, str], PipelineResult]:
    """검색 경로를 `(qid, question) -> PipelineResult` 파이프라인으로 만든다.

    재검색 때는 지금까지 모은 근거에 새 근거를 이어 붙여 판정에 넘긴다.
    루프 끊기에 걸리거나 후속 검색어를 못 만들면, 지금 근거로 RETRIEVE 없는 판정을 받는다.

    Args:
        output_format: 근거 판정 출력 형식
        settings: 설정 (`retrieval.k`, `compress.*`, `budget.*`, `cap.*`)
        index: 코퍼스 BM25 색인
        llm_factory: ChatOllama 생성 함수 (테스트에서 가짜 모델 주입)
        followups: 넘기면 문항별 후속 검색어를 `qid` 키로 채운다 (로그용)

    Returns:
        러너의 `run_item`에 넘길 파이프라인
    """

    def pipeline(qid: str, question: str) -> PipelineResult:
        """한 문항을 검색 경로로 끝까지 돌린다. 모델 오류는 러너가 오답 한 줄로 남긴다."""
        llm = BudgetedLLM(settings=settings, llm_factory=llm_factory)
        requery = Requery(settings.budget.max_requery)
        breaker = LoopBreaker(normalizer=search_args_key)
        retrieved: list[list[str]] = []
        kept: list[list[str]] = []
        contexts: list[str] = []
        decisions: list[str] = []
        if followups is not None:
            followups[qid] = requery.queries

        def add_evidence(query: str, followup: str | None) -> bool:
            """검색·압축해 근거를 더한다. 루프 끊기에 걸렸으면 False."""
            outcome = breaker.run(
                "bm25_search", {"query": query}, lambda: bm25_search(query, settings.retrieval.k, index=index)
            )
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

        final = None
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

        return PipelineResult(
            answer=final.answer if final else "",
            abstained=bool(final and final.abstained),
            budget_exhausted=llm.budget_exhausted,
            route="search",
            decisions=decisions,
            n_calls=llm.n_calls,
            n_retries=llm.n_retries,
            calls=list(llm.calls),
            retrieved_doc_ids=retrieved,
            kept_windows=kept,
            loop_breaks=breaker.loop_breaks,
        )

    return pipeline


def run_search_path(
    item: dict[str, Any],
    *,
    output_format: OutputFormat,
    settings: Settings,
    index: BM25Index,
    llm_factory: Callable[..., Any] = create_chat_ollama,
    order_idx: int = 0,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """한 문항을 본 측정과 같은 규칙(`run_item`)으로 돌리고 로그 한 줄을 만든다.

    Args:
        item: 동결 문항 한 줄 (`qid`, `question`만 파이프라인으로 간다)
        output_format: 근거 판정 출력 형식
        settings: 설정
        index: 코퍼스 BM25 색인
        llm_factory: ChatOllama 생성 함수
        order_idx: 이 형식 안에서의 문항 순번. 0이면 워밍업
        timeout_s: 문항 제한 시간. 없으면 `run.timeout_s`

    Returns:
        러너 로그 형식(INTERFACES §7.2) + `format`, `followups`
    """
    followups: dict[str, list[str]] = {}
    pipeline = make_pipeline(output_format, settings=settings, index=index, llm_factory=llm_factory, followups=followups)
    row = run_item(
        item,
        pipeline,
        pipeline_name=output_format,
        bundle="dev",
        round_name="c4",
        order_idx=order_idx,
        timeout_s=settings.run.timeout_s if timeout_s is None else timeout_s,
        window=settings.compress.window,
        is_warmup=order_idx == 0,
    )
    return {**row, "format": output_format, "followups": list(followups.get(item["qid"], []))}


def _rate(values: Sequence[bool]) -> float | None:
    """참의 비율. 값이 없으면 None."""
    return sum(values) / len(values) if values else None


def _is_warmup(row: dict[str, Any], position: int) -> bool:
    """워밍업 문항인가. `is_warmup`이 없는 예전 로그는 파일의 첫 줄을 워밍업으로 본다."""
    return row["is_warmup"] if "is_warmup" in row else position == 0


def summarize_rows(rows: Sequence[dict[str, Any]], warmup_qids: Iterable[str] = ()) -> dict[str, Any]:
    """한 형식·한 그룹의 지표를 계산한다.

    정확도·F1·부작용 지표(잘림·파싱 실패·기권)는 워밍업 문항을 포함하고,
    판정 호출 수·출력 토큰은 워밍업을 빼고 평균을 낸다 (INTERFACES §7.4).

    Args:
        rows: 같은 형식의 문항 로그
        warmup_qids: 이 형식의 워밍업 문항 ID

    Returns:
        정확도, 텍스트 F1, 과잉 기권률, 호출 단위 잘림률·파싱 실패율 등
    """
    warm = set(warmup_qids)
    cost = [r for r in rows if r["qid"] not in warm]
    calls = [c for r in rows for c in r["calls"]]
    checked = [c for c in calls if c["parse_ok"] is not None]
    answerable = [r for r in rows if r["answer_kind"] == "text"]
    return {
        "n": len(rows),
        "accuracy": _rate([r["correct"] for r in rows]),
        "text_f1": mean(r["f1"] for r in answerable) if answerable else None,
        "over_abstention": _rate([r["abstained"] for r in answerable]),
        "judge_calls": mean(r["n_calls"] for r in cost) if cost else None,
        "retries": sum(r["n_retries"] for r in rows),
        "truncated_rate": _rate([c["truncated"] for c in calls]),
        "truncated_after_retry": sum(c["truncated"] and c["is_retry"] for c in calls),
        "parse_fail_rate": _rate([not c["parse_ok"] for c in checked]),
        "output_tokens": mean(sum(c["output_tokens"] for c in r["calls"]) for r in cost) if cost else None,
        "first_decision": {d: sum(1 for r in rows if r["decisions"][:1] == [d]) for d in ("ANSWER", "RETRIEVE", "ABSTAIN")},
        "budget_exhausted": sum(r["budget_exhausted"] for r in rows),
        "timeouts": sum(bool(r.get("timeout")) for r in rows),
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
        logs: 형식 이름 → 문항 로그 (실행 순서대로)

    Returns:
        그룹별 지표 표, 첫 판정 분포, 두 형식의 문항 대응 비교(McNemar)
    """
    warmups = {fmt: {r["qid"] for i, r in enumerate(rows) if _is_warmup(r, i)} for fmt, rows in logs.items()}
    lines = [
        "판정 호출/문항과 출력 토큰/문항은 형식마다 첫 문항(워밍업)을 뺀 평균이다. 나머지 지표는 모든 문항이다.",
        "",
        "| 그룹 | 형식 | 문항 | 정확도 | 텍스트 F1 | 과잉 기권률 | 판정 호출/문항 | 잘림률(호출) | 재시도 후 잘림 | 파싱 실패율 | 출력 토큰/문항 | 예산 소진 | 시간 초과 | 오류 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for group, keep in GROUPS:
        for fmt, rows in logs.items():
            s = summarize_rows([r for r in rows if keep(r)], warmups[fmt])
            lines.append(
                f"| {group} | {fmt} | {s['n']} | {_fmt(s['accuracy'], True)} | {_fmt(s['text_f1'])} | "
                f"{_fmt(s['over_abstention'], True)} | {_fmt(s['judge_calls'])} | {_fmt(s['truncated_rate'], True)} | "
                f"{s['truncated_after_retry']} | {_fmt(s['parse_fail_rate'], True)} | {_fmt(s['output_tokens'])} | "
                f"{s['budget_exhausted']} | {s['timeouts']} | {s['errors']} |"
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


def parse_compare(specs: Sequence[str], fmt: str) -> list[tuple[str, Path]]:
    """`이름=폴더` 목록을 (이름, 그 폴더의 `<형식>.jsonl` 경로)로 바꾼다.

    Args:
        specs: `NAME=DIR` 문자열들
        fmt: 비교할 형식 (`json` 또는 `tag`)

    Returns:
        (이름, 로그 경로) 리스트. 순서는 입력 순서

    Raises:
        SystemExit: `=`가 없거나 이름이 겹칠 때
    """
    pairs: list[tuple[str, Path]] = []
    for spec in specs:
        name, sep, folder = spec.partition("=")
        if not sep or not name or not folder:
            raise SystemExit(f"--compare 값은 NAME=DIR 형식이어야 한다: {spec!r}")
        if name in {n for n, _ in pairs}:
            raise SystemExit(f"--compare 이름이 겹친다: {name!r}")
        pairs.append((name, Path(folder) / f"{fmt}.jsonl"))
    return pairs


def main() -> None:
    """두 형식을 개발 세트 검색형 문항에 돌리고(이어 돌기 지원) 요약 표를 쓴다."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=list(FORMATS))
    parser.add_argument("--limit", type=int, default=None, help="앞에서 N문항만 (확인용)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--summarize", action="store_true", help="실행하지 않고 저장된 로그로 표만 만든다")
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="NAME=DIR",
        help="실행하지 않고 여러 실행 폴더의 같은 형식 로그를 나란히 비교한다 (예: base=dir1 new=dir2)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    if not args.summarize and not args.compare:
        hashes = verify_frozen_hashes(REPO_ROOT / "data/frozen", REPO_ROOT / "data/README.md")
        settings = get_settings()
        items = search_items(load_items(REPO_ROOT / DEV_PATH))[: args.limit]
        index = get_default_index()
        meta = RunMeta(out_dir / "RUN_META.json")
        if not meta.path.exists() and any((out_dir / f"{fmt}.jsonl").exists() for fmt in args.formats):
            raise SystemExit(f"{out_dir}에 로그는 있는데 RUN_META.json이 없어 실행 조건을 확인할 수 없다. 새 --out-dir로 돌린다.")
        meta.begin(
            bundle="dev",
            round_name="c4",
            item_set="dev" if args.limit is None else f"dev[:{args.limit}]",
            settings=settings.to_dict(),
            data_hashes=hashes,
            order=list(args.formats),
            environment=collect_environment(),
        )
        for fmt in args.formats:
            log = RunLog(out_dir / f"{fmt}.jsonl")
            todo = [(i, it) for i, it in enumerate(items) if it["qid"] not in log.done_qids]
            print(f"[{fmt}] {len(items)}문항 중 {len(items) - len(todo)}개 완료, {len(todo)}개 실행", flush=True)
            for n, (order_idx, item) in enumerate(todo, 1):
                row = run_search_path(item, output_format=fmt, settings=settings, index=index, order_idx=order_idx)
                log.append(row)
                print(
                    f"  [{fmt}] {n}/{len(todo)} {row['qid']} {'+'.join(row['decisions']) or '-'} "
                    f"-> {row['answer']!r} correct={row['correct']}",
                    flush=True,
                )
        meta.finish(vram_by_pipeline={})

    if args.compare:
        fmt = args.formats[0]
        logs = {name: read_log(path if path.is_absolute() else REPO_ROOT / path) for name, path in parse_compare(args.compare, fmt)}
        table = render_summary(logs)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"compare_{fmt}.md").write_text(table + "\n", encoding="utf-8")
        print(table)
        return
    logs = {fmt: read_log(out_dir / f"{fmt}.jsonl") for fmt in args.formats}
    table = render_summary(logs)
    (out_dir / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
