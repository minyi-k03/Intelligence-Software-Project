"""실험 C-2: 근거 문장 보존율로 `compress.m`을 정한다 (개발 세트, 모델 호출 없음).

실행 (레포 루트에서):
    python -m scripts.measure_compress_m
    python -m scripts.measure_compress_m --k 3 5 8 --m-max 12

개발 세트의 HotpotQA 문항만 쓴다. `supporting_facts`가 있는 유형이 HotpotQA뿐이기 때문이다.
여기서 문항 유형을 보는 것은 측정용이며, 에이전트에는 넘기지 않는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.config.settings import REPO_ROOT, get_settings
from src.meba.compress import compress, covered_sentences, full_paragraphs
from src.tools.bm25_search import DEFAULT_CORPUS_PATH, BM25Index, get_default_index
from src.tools.tokenizer import tokenize

DEV_PATH = Path("data/frozen/dev.jsonl")
Sentence = tuple[str, int]


def load_items(path: Path) -> list[dict[str, Any]]:
    """개발 세트에서 근거 문장이 표시된 문항만 읽는다.

    Args:
        path: `dev.jsonl` 경로

    Returns:
        `supporting_facts`가 비어 있지 않은 문항 리스트
    """
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if r["supporting_facts"]]


def gold_sentences(item: dict[str, Any]) -> set[Sentence]:
    """문항의 정답 근거 문장을 (doc_id, 문장 번호) 집합으로 바꾼다.

    Args:
        item: `supporting_facts`가 `[title, 문장 번호]` 목록인 문항

    Returns:
        {(`hp::<title>`, 문장 번호)}
    """
    return {(f"hp::{title}", idx) for title, idx in item["supporting_facts"]}


def retention(
    gold: set[Sentence],
    retrieved_ids: Iterable[str],
    kept_windows: Iterable[str],
    window: int,
) -> tuple[int, int]:
    """압축 뒤 남은 근거 문장 수와, 압축 전에 있던 근거 문장 수를 센다.

    Args:
        gold: 정답 근거 문장
        retrieved_ids: 검색된 문단의 doc_id (압축의 입력)
        kept_windows: 압축이 남긴 문장 창 ID
        window: 압축에 쓴 `compress.window`

    Returns:
        (남은 수, 압축 전에 있던 수). 검색이 못 찾은 문단의 근거 문장은 세지 않는다
    """
    retrieved = set(retrieved_ids)
    present = {s for s in gold if s[0] in retrieved}
    return len(present & covered_sentences(kept_windows, window)), len(present)


def followup_title(index: BM25Index, item: dict[str, Any], n_docs: int) -> str:
    """재검색 측정에 쓸 후속 검색어를 고른다.

    모델이 없으므로 "이상적인 근거 판정이 빠진 정보로 지목했을 것"을 대신 쓴다:
    정답 문단 가운데 1차 검색 순위가 가장 낮은 문단의 제목.

    Args:
        index: 코퍼스 BM25 색인
        item: `gold_doc_ids`가 있는 문항
        n_docs: 코퍼스 문단 수 (전체 순위를 얻기 위한 k)

    Returns:
        후속 검색어로 쓸 문단 제목
    """
    rank = {d["doc_id"]: i for i, d in enumerate(index.search(item["question"], k=n_docs))}
    worst = max(item["gold_doc_ids"], key=lambda doc_id: rank.get(doc_id, n_docs))
    return worst.removeprefix("hp::")


def measure(
    index: BM25Index,
    items: Sequence[dict[str, Any]],
    *,
    k: int,
    m: int,
    window: int,
    requery: bool,
    n_docs: int,
) -> dict[str, float]:
    """한 (k, m) 조합의 근거 문장 보존율을 잰다.

    Args:
        index: 코퍼스 BM25 색인
        items: 측정할 문항
        k: 검색 문단 수
        m: 남길 문장 창 수
        window: `compress.window`
        requery: True면 재검색 회차(후속 검색어로 검색, 질문 + 후속 검색어로 압축)
        n_docs: 코퍼스 문단 수

    Returns:
        retention: 남은 근거 문장 / 압축 전에 있던 근거 문장
        end_to_end: 남은 근거 문장 / 전체 근거 문장 (검색 실패 포함, 참고용)
        chars: 문항당 평균 근거 텍스트 글자 수
        full_chars: 압축하지 않았을 때(문단 통째)의 문항당 평균 글자 수
        present, total: 압축 전에 있던 근거 문장 수, 전체 근거 문장 수
    """
    kept = present = total = chars = full_chars = 0
    for item in items:
        gold = gold_sentences(item)
        followup = followup_title(index, item, n_docs) if requery else None
        paragraphs = index.search(followup if requery else item["question"], k=k)
        out = compress(
            item["question"],
            paragraphs,
            m=m,
            window=window,
            tokenizer=tokenize,
            followup_query=followup,
        )
        if requery:
            # 재검색은 빠진 문단을 찾는 회차이므로 그 문단의 근거 문장만 본다
            gold = {s for s in gold if s[0] == f"hp::{followup}"}
        n_kept, n_present = retention(gold, (p["doc_id"] for p in paragraphs), out.kept_windows, window)
        kept += n_kept
        present += n_present
        total += len(gold)
        chars += len(out.context)
        full_chars += len(full_paragraphs(paragraphs).context)
    return {
        "retention": kept / present if present else 0.0,
        "end_to_end": kept / total if total else 0.0,
        "chars": chars / len(items),
        "full_chars": full_chars / len(items),
        "present": present,
        "total": total,
    }


def min_m(rates: Sequence[float], threshold: float) -> int | None:
    """보존율이 기준 이상인 가장 작은 m을 찾는다.

    Args:
        rates: m = 1, 2, ... 순서의 보존율
        threshold: 기준 (예: 0.9)

    Returns:
        가장 작은 m. 기준을 넘는 값이 없으면 None
    """
    return next((i + 1 for i, r in enumerate(rates) if r >= threshold), None)


def main() -> None:
    """k 후보별로 m을 늘려 가며 보존율을 재고 마크다운 표로 출력한다."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5, 8], help="검색 문단 수 후보")
    parser.add_argument("--m-max", type=int, default=10, help="시험할 m의 최댓값")
    parser.add_argument("--threshold", type=float, default=0.9, help="보존율 기준")
    args = parser.parse_args()

    window = get_settings().compress.window
    index = get_default_index()
    with (REPO_ROOT / DEFAULT_CORPUS_PATH).open(encoding="utf-8") as f:
        n_docs = sum(1 for line in f if line.strip())
    items = load_items(REPO_ROOT / DEV_PATH)
    ms = range(1, args.m_max + 1)
    header = " | ".join(f"m={m}" for m in ms)

    print(f"문항 {len(items)}개, 코퍼스 {n_docs}문단, compress.window={window}, 기준 {args.threshold:.0%}\n")
    for requery in (False, True):
        results = {
            k: [measure(index, items, k=k, m=m, window=window, requery=requery, n_docs=n_docs) for m in ms]
            for k in args.k
        }
        print(f"### {'재검색' if requery else '1차 검색'}\n")
        print("| k | 전체 근거 문장 | 검색된 문단 안의 근거 문장 | 압축 없음 글자 수 |")
        print("|---|---|---|---|")
        for k, rows in results.items():
            print(f"| {k} | {rows[0]['total']} | {rows[0]['present']} | {rows[0]['full_chars']:.0f} |")
        print()
        for title, key, fmt in (
            ("보존율 (남은 근거 문장 / 압축 전에 있던 근거 문장)", "retention", "{:.0%}"),
            ("전체 기준 (남은 근거 문장 / 전체 근거 문장, 검색 실패 포함)", "end_to_end", "{:.0%}"),
            ("문항당 평균 근거 텍스트 글자 수", "chars", "{:.0f}"),
        ):
            print(f"{title}\n")
            print(f"| k | {header} |" + (" 최소 m |" if key == "retention" else ""))
            print("|---|" + "---|" * (len(ms) + (key == "retention")))
            for k, rows in results.items():
                line = f"| {k} | " + " | ".join(fmt.format(r[key]) for r in rows) + " |"
                if key == "retention":
                    best = min_m([r[key] for r in rows], args.threshold)
                    line += f" {best if best else '없음'} |"
                print(line)
            print()


if __name__ == "__main__":
    main()
