"""검색 결과 압축 단위 테스트 (PR C-1). 모델·GPU 없이 돈다."""

from __future__ import annotations

import re

from src.meba.compress import (
    build_windows,
    bm25_scores,
    compress,
    covered_sentences,
    full_paragraphs,
)


def tok(text: str) -> list[str]:
    """테스트용 토크나이저. 실제로는 검색 도구(B)의 토크나이저를 주입한다."""
    return re.findall(r"\w+", text.lower())


ARTHUR = {
    "doc_id": "hp::Arthur's Magazine",
    "title": "Arthur's Magazine",
    "sentences": [
        "Arthur's Magazine was an American literary periodical.",
        "It was published in Philadelphia in the 19th century.",
        "Edited by T. S. Arthur, it featured work by Edgar A. Poe.",
        "In May 1846 it was merged into Godey's Lady's Book.",
    ],
}
FIRST = {
    "doc_id": "hp::First for Women",
    "title": "First for Women",
    "sentences": [
        "First for Women is a woman's magazine published by Bauer Media Group.",
        "The magazine was started in 1989.",
        "It is based in Englewood Cliffs, New Jersey.",
    ],
}
DOCS = [ARTHUR, FIRST]


def test_window_ids_and_spans() -> None:
    ws = build_windows(DOCS, window=1)
    assert [w.window_id for w in ws] == [
        "hp::Arthur's Magazine#0",
        "hp::Arthur's Magazine#1",
        "hp::Arthur's Magazine#2",
        "hp::Arthur's Magazine#3",
        "hp::First for Women#0",
        "hp::First for Women#1",
        "hp::First for Women#2",
    ]
    assert [(w.start, w.end) for w in ws[:4]] == [(0, 2), (0, 3), (1, 4), (2, 4)]
    assert ws[0].text.startswith("Arthur's Magazine: Arthur's Magazine was")


def test_window_zero_is_single_sentence() -> None:
    ws = build_windows([FIRST], window=0)
    assert ws[1].text == "First for Women: The magazine was started in 1989."


def test_paragraph_without_sentences_yields_no_window() -> None:
    empty = {"doc_id": "sq::x::0", "title": "X", "sentences": []}
    assert build_windows([empty, FIRST], window=1)[0].doc_id == "hp::First for Women"


def test_bm25_prefers_matching_text() -> None:
    scores = bm25_scores(["apple banana", "banana cherry", "dog"], "apple", tok)
    assert scores[0] > 0 and scores[1] == 0 and scores[2] == 0


def test_bm25_empty_inputs() -> None:
    assert bm25_scores([], "apple", tok) == []
    assert bm25_scores(["apple"], "", tok) == [0.0]


def test_keeps_top_m_relevant_window() -> None:
    out = compress("When was First for Women started?", DOCS, m=1, window=1, tokenizer=tok)
    assert len(out.kept_windows) == 1
    assert ("hp::First for Women", 1) in covered_sentences(out.kept_windows, window=1)
    assert "1989" in out.context
    assert "Philadelphia" not in out.context


def test_title_counts_toward_score() -> None:
    # 본문에 "First for Women"이 없는 창도 제목 덕분에 질문과 맞는다
    out = compress("Where is First for Women based?", DOCS, m=1, window=0, tokenizer=tok)
    assert out.kept_windows == ["hp::First for Women#2"]


def test_followup_query_is_added_to_question() -> None:
    q = "Which magazine was started first?"
    first = compress(q, DOCS, m=1, window=0, tokenizer=tok)
    requery = compress(q, DOCS, m=1, window=0, tokenizer=tok, followup_query="Arthur's Magazine Philadelphia 19th century")
    assert first.kept_windows[0].startswith("hp::First for Women")
    assert requery.kept_windows[0].startswith("hp::Arthur's Magazine")


def test_overlapping_windows_do_not_repeat_sentences() -> None:
    out = compress("Arthur's Magazine Philadelphia Poe", DOCS, m=3, window=1, tokenizer=tok)
    assert len(out.kept_windows) == 3
    for sent in ARTHUR["sentences"]:
        assert out.context.count(sent) <= 1


def test_separate_spans_joined_with_ellipsis() -> None:
    long_doc = {
        "doc_id": "d",
        "title": "T",
        "sentences": ["alpha one.", "filler two.", "filler three.", "filler four.", "omega five."],
    }
    out = compress("alpha omega", [long_doc], m=2, window=0, tokenizer=tok)
    assert out.context == "T: alpha one. ... omega five."


def test_paragraph_order_follows_best_score() -> None:
    out = compress("1989 Englewood Cliffs Philadelphia", DOCS, m=7, window=0, tokenizer=tok)
    assert out.context.index("First for Women:") < out.context.index("Arthur's Magazine:")


def test_m_larger_than_windows_keeps_all() -> None:
    out = compress("magazine", DOCS, m=100, window=1, tokenizer=tok)
    assert len(out.kept_windows) == 7


def test_ties_are_deterministic() -> None:
    out = compress("zzz", DOCS, m=2, window=0, tokenizer=tok)
    assert out.kept_windows == ["hp::Arthur's Magazine#0", "hp::Arthur's Magazine#1"]


def test_non_positive_m_keeps_nothing() -> None:
    for m in (0, -1):
        out = compress("magazine", DOCS, m=m, window=1, tokenizer=tok)
        assert out.context == "" and out.kept_windows == []


def test_repeated_paragraph_is_used_once() -> None:
    out = compress("1989", [FIRST, ARTHUR, FIRST], m=100, window=0, tokenizer=tok)
    assert len(out.kept_windows) == len(set(out.kept_windows)) == 7


def test_no_paragraphs() -> None:
    out = compress("anything", [], m=3, window=1, tokenizer=tok)
    assert out.context == "" and out.kept_windows == []


def test_full_paragraphs_keeps_everything() -> None:
    out = full_paragraphs(DOCS)
    assert len(out.kept_windows) == 7
    for sent in ARTHUR["sentences"] + FIRST["sentences"]:
        assert sent in out.context


def test_covered_sentences() -> None:
    covered = covered_sentences(["hp::Arthur's Magazine#0", "hp::First for Women#2"], window=1)
    assert ("hp::Arthur's Magazine", 0) in covered
    assert ("hp::Arthur's Magazine", 1) in covered
    assert ("hp::Arthur's Magazine", 2) not in covered
    assert ("hp::First for Women", 1) in covered
    assert ("hp::First for Women", 0) not in covered


def test_covered_sentences_doc_id_with_hash() -> None:
    assert ("hp::C#", 3) in covered_sentences(["hp::C##3"], window=0)
