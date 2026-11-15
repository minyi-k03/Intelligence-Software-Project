"""실험 C-2 측정 스크립트의 계산 함수 테스트. 모델·GPU 없이 돈다."""

from __future__ import annotations

from scripts.measure_compress_m import followup_title, gold_sentences, measure, min_m, retention
from src.tools.bm25_search import BM25Index

DOCS = [
    {
        "doc_id": "hp::Alpha",
        "title": "Alpha",
        "sentences": ["Alpha is a river.", "It flows north.", "It ends in a lake.", "Fish live there."],
    },
    {
        "doc_id": "hp::Beta",
        "title": "Beta",
        "sentences": ["Beta is a town.", "The town is near the Alpha river."],
    },
    {
        "doc_id": "hp::Gamma",
        "title": "Gamma",
        "sentences": ["Gamma is a mountain far away."],
    },
]
ITEM = {
    "question": "Which river is the town of Beta near?",
    "gold_doc_ids": ["hp::Alpha", "hp::Beta"],
    "supporting_facts": [["Alpha", 0], ["Beta", 1]],
}
INDEX = BM25Index([{**d, "text": " ".join(d["sentences"])} for d in DOCS])


def test_gold_sentences_maps_title_to_doc_id() -> None:
    assert gold_sentences(ITEM) == {("hp::Alpha", 0), ("hp::Beta", 1)}


def test_retention_counts_only_retrieved_paragraphs() -> None:
    gold = {("hp::Alpha", 0), ("hp::Beta", 1)}
    # Beta 문단은 검색되지 않았으므로 분모에서 빠진다
    assert retention(gold, ["hp::Alpha"], ["hp::Alpha#0"], window=1) == (1, 1)
    assert retention(gold, ["hp::Alpha"], ["hp::Alpha#3"], window=1) == (0, 1)


def test_retention_window_covers_neighbour() -> None:
    gold = {("hp::Alpha", 0)}
    assert retention(gold, ["hp::Alpha"], ["hp::Alpha#1"], window=1) == (1, 1)
    assert retention(gold, ["hp::Alpha"], ["hp::Alpha#1"], window=0) == (0, 1)


def test_min_m() -> None:
    assert min_m([0.5, 0.89, 0.9, 1.0], 0.9) == 3
    assert min_m([0.5, 0.6], 0.9) is None


def test_followup_title_is_lower_ranked_gold_paragraph() -> None:
    title = followup_title(INDEX, ITEM, n_docs=len(DOCS))
    ranked = [d["doc_id"] for d in INDEX.search(ITEM["question"], k=len(DOCS))]
    assert f"hp::{title}" == max(ITEM["gold_doc_ids"], key=ranked.index)


def test_measure_reaches_full_retention_with_large_m() -> None:
    for requery in (False, True):
        out = measure(INDEX, [ITEM], k=3, m=10, window=1, requery=requery, n_docs=len(DOCS))
        assert out["retention"] == 1.0
        assert out["chars"] <= out["full_chars"]


def test_measure_requery_counts_only_followup_paragraph() -> None:
    out = measure(INDEX, [ITEM], k=3, m=10, window=1, requery=True, n_docs=len(DOCS))
    assert out["total"] == 1
