"""실험 B-4 (1단계): 개발 세트 검색 재현율로 `retrieval.k`를 정한다 (모델 호출 없음).

실행 (레포 루트에서):
    python -m scripts.measure_retrieval_k
    python -m scripts.measure_retrieval_k --k 1 3 5 8 10

개발 세트(`data/frozen/dev.jsonl`)에서 정답 근거 문단(`gold_doc_ids`)이 있는 40문항
(HotpotQA 20 + SQuAD 2.0 20)만 쓴다. GSM8K는 검색 문단이 없어 제외한다.
문항 유형(`type`, `subtype`)은 오프라인 재현율 분석용으로만 쓰며 에이전트에 넘기지 않는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from src.config.settings import REPO_ROOT, get_settings
from src.meba.compress import compress, covered_sentences, full_paragraphs
from src.tools.bm25_search import (
    DEFAULT_CORPUS_PATH,
    BM25Index,
    format_search_results,
    get_default_index,
)
from src.tools.tokenizer import tokenize

DEV_PATH = Path("data/frozen/dev.jsonl")

# 실험 C-2(docs/experiments/2026-11-15-c-compress-m.md)에서 확정한 k별 권장 compress.m
RECOMMENDED_M_BY_K: dict[int, int] = {
    3: 6,
    5: 7,
    8: 10,
}


def load_search_items(path: Path) -> list[dict[str, Any]]:
    """개발 세트에서 검색 대상 문항(`gold_doc_ids`가 비어 있지 않은 문항)만 읽는다.

    Args:
        path: `dev.jsonl` 경로.

    Returns:
        검색 대상 문항 리스트 (HotpotQA 20 + SQuAD 2.0 20 = 40문항).
    """
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [row for row in rows if row.get("gold_doc_ids")]


def evaluate_k(
    index: BM25Index,
    items: Sequence[dict[str, Any]],
    *,
    k: int,
    m: int | None = None,
    window: int = 1,
) -> dict[str, Any]:
    """주어진 `k`에서 개발 세트 검색 재현율과 문맥 길이를 계산한다.

    Args:
        index: 코퍼스 BM25 색인.
        items: 검색 대상 문항 목록.
        k: 검색할 상위 문단 수 (`retrieval.k`).
        m: 함께 적용할 압축 문장 창 수 (`compress.m`). 없으면 `RECOMMENDED_M_BY_K.get(k)`.
        window: 문장 창 반경 (`compress.window`).

    Returns:
        재현율·문단 회수율·평균 글자 수를 담은 딕셔너리.
    """
    if not items:
        raise ValueError("items must not be empty")

    effective_m = RECOMMENDED_M_BY_K.get(k) if m is None else m

    any_hits = 0
    all_hits = 0
    found_gold_docs = 0
    total_gold_docs = 0

    squad_hits = 0
    squad_total = 0
    hp_any_hits = 0
    hp_all_hits = 0
    hp_bridge_all = 0
    hp_bridge_total = 0
    hp_comp_all = 0
    hp_comp_total = 0
    hp_total = 0

    p1_obs_chars = 0
    m_full_chars = 0
    m_comp_chars = 0
    hp_gold_sents_total = 0
    hp_gold_sents_kept = 0

    for item in items:
        hits = index.search(item["question"], k=k)
        got_ids = {doc["doc_id"] for doc in hits}
        gold_ids = set(item["gold_doc_ids"])

        matched_docs = len(got_ids & gold_ids)
        found_gold_docs += matched_docs
        total_gold_docs += len(gold_ids)

        has_any = matched_docs > 0
        has_all = gold_ids <= got_ids
        if has_any:
            any_hits += 1
        if has_all:
            all_hits += 1

        p1_obs_chars += len(format_search_results(hits))
        m_full_chars += len(full_paragraphs(hits).context)

        if effective_m is not None:
            comp = compress(
                item["question"],
                hits,
                m=effective_m,
                window=window,
                tokenizer=tokenize,
            )
            m_comp_chars += len(comp.context)
            if item.get("supporting_facts"):
                gold_sents = {(f"hp::{title}", idx) for title, idx in item["supporting_facts"]}
                hp_gold_sents_total += len(gold_sents)
                hp_gold_sents_kept += len(gold_sents & covered_sentences(comp.kept_windows, window))

        if item.get("type") == "squad2":
            squad_total += 1
            if has_all:
                squad_hits += 1
        elif item.get("type") == "hotpotqa":
            hp_total += 1
            if has_any:
                hp_any_hits += 1
            if has_all:
                hp_all_hits += 1
            if item.get("subtype") == "bridge":
                hp_bridge_total += 1
                if has_all:
                    hp_bridge_all += 1
            elif item.get("subtype") == "comparison":
                hp_comp_total += 1
                if has_all:
                    hp_comp_all += 1

    n_items = len(items)
    return {
        "k": k,
        "m": effective_m,
        "n_items": n_items,
        "any_hits": any_hits,
        "any_rate": any_hits / n_items,
        "all_hits": all_hits,
        "all_rate": all_hits / n_items,
        "found_gold_docs": found_gold_docs,
        "total_gold_docs": total_gold_docs,
        "doc_recall": found_gold_docs / total_gold_docs if total_gold_docs else 0.0,
        "squad_hits": squad_hits,
        "squad_total": squad_total,
        "squad_rate": squad_hits / squad_total if squad_total else 0.0,
        "hp_any_hits": hp_any_hits,
        "hp_all_hits": hp_all_hits,
        "hp_total": hp_total,
        "hp_any_rate": hp_any_hits / hp_total if hp_total else 0.0,
        "hp_all_rate": hp_all_hits / hp_total if hp_total else 0.0,
        "hp_bridge_all": hp_bridge_all,
        "hp_bridge_total": hp_bridge_total,
        "hp_comp_all": hp_comp_all,
        "hp_comp_total": hp_comp_total,
        "p1_obs_chars": p1_obs_chars / n_items,
        "m_full_chars": m_full_chars / n_items,
        "m_comp_chars": (m_comp_chars / n_items) if effective_m is not None else None,
        "hp_e2e_sent_kept": hp_gold_sents_kept,
        "hp_e2e_sent_total": hp_gold_sents_total,
        "hp_e2e_sent_rate": (
            hp_gold_sents_kept / hp_gold_sents_total
            if (effective_m is not None and hp_gold_sents_total)
            else None
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """CLI 진입점."""
    parser = argparse.ArgumentParser(description="개발 세트 BM25 검색 재현율(retrieval.k)을 측정한다.")
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5, 8, 10])
    args = parser.parse_args(argv)

    window = get_settings().compress.window
    index = get_default_index()
    with (REPO_ROOT / DEFAULT_CORPUS_PATH).open(encoding="utf-8") as handle:
        n_docs = sum(1 for line in handle if line.strip())
    items = load_search_items(REPO_ROOT / DEV_PATH)

    rows = [evaluate_k(index, items, k=k, window=window) for k in args.k]

    print(f"개발 세트 검색 문항 {len(items)}개 (HotpotQA 20 + SQuAD 2.0 20), 코퍼스 {n_docs}문단\n")
    print("### 1. 후보 k별 1차 검색 재현율\n")
    print(
        "| k | 전체 any_gold (40) | 전체 all_gold (40) | 정답 문단 회수율 (60) "
        "| SQuAD 2.0 (20) | HotpotQA 1개+ (20) | HotpotQA 2개 모두 (20) | bridge 2개 (10) | comparison 2개 (10) |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['k']} "
            f"| {r['any_hits']}/{r['n_items']} ({r['any_rate']:.1%}) "
            f"| {r['all_hits']}/{r['n_items']} ({r['all_rate']:.1%}) "
            f"| {r['found_gold_docs']}/{r['total_gold_docs']} ({r['doc_recall']:.1%}) "
            f"| {r['squad_hits']}/{r['squad_total']} ({r['squad_rate']:.1%}) "
            f"| {r['hp_any_hits']}/{r['hp_total']} ({r['hp_any_rate']:.1%}) "
            f"| {r['hp_all_hits']}/{r['hp_total']} ({r['hp_all_rate']:.1%}) "
            f"| {r['hp_bridge_all']}/{r['hp_bridge_total']} "
            f"| {r['hp_comp_all']}/{r['hp_comp_total']} |"
        )

    print("\n### 2. 문맥 길이 및 C-2 권장 `compress.m` 결합 효과 (40문항 평균)\n")
    print("| k | C-2 권장 m | P1 관찰문 글자 수 | M 압축 전 글자 수 | M 압축 후 글자 수 | 글자 수 감소율 | HotpotQA 끝단 문장 보존 (47) |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        m_str = str(r["m"]) if r["m"] is not None else "-"
        if r["m_comp_chars"] is not None and r["m_full_chars"] > 0:
            comp_str = f"{r['m_comp_chars']:.0f}"
            red_str = f"{1.0 - (r['m_comp_chars'] / r['m_full_chars']):.1%}"
            e2e_str = f"{r['hp_e2e_sent_kept']}/{r['hp_e2e_sent_total']} ({r['hp_e2e_sent_rate']:.1%})"
        else:
            comp_str = "-"
            red_str = "-"
            e2e_str = "-"
        print(
            f"| {r['k']} | {m_str} | {r['p1_obs_chars']:.0f} | {r['m_full_chars']:.0f} "
            f"| {comp_str} | {red_str} | {e2e_str} |"
        )


if __name__ == "__main__":
    main()
