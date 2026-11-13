"""검색 결과 압축 — 핵심 부품, 전략 ③ (SPEC §5, 담당 C).

검색된 문단을 문장 창(제목 + 가운데 문장 + 앞뒤 `window`문장)으로 나누고,
창마다 BM25 점수를 매겨 상위 m개만 남긴다. 모델을 부르지 않는다.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

Tokenizer = Callable[[str], list[str]]
Paragraph = Mapping[str, Any]

# 검색 도구(B)의 BM25와 같은 값을 써야 한다.
BM25_K1 = 1.5
BM25_B = 0.75


@dataclass(frozen=True)
class Window:
    """문장 창 하나.

    window_id: "{doc_id}#{가운데 문장 번호}" (INTERFACES §4 `kept_windows` 형식)
    start, end: 창이 덮는 문장 범위 [start, end)
    rank: 검색 결과에서 이 문단의 순위 (동점 처리용)
    text: 점수를 매기는 텍스트 "{title}: {창 문장들}"
    """

    window_id: str
    doc_id: str
    title: str
    center: int
    start: int
    end: int
    rank: int
    text: str


@dataclass(frozen=True)
class Compressed:
    """압축 결과.

    context: 근거 판정 프롬프트에 넣을 근거 텍스트
    kept_windows: 남긴 문장 창 ID (점수 내림차순)
    """

    context: str
    kept_windows: list[str]


def build_windows(paragraphs: Sequence[Paragraph], window: int) -> list[Window]:
    """문단마다 문장 하나를 가운데로 하는 문장 창을 만든다.

    Args:
        paragraphs: 검색 결과 `[{doc_id, title, sentences, ...}]` (검색 순위 순)
        window: 가운데 문장 앞뒤로 붙일 문장 수 (`compress.window`)

    Returns:
        문단 순서, 문장 순서대로 나열한 문장 창 리스트. 같은 doc_id가 다시 나오면 건너뛴다
    """
    windows: list[Window] = []
    seen: set[str] = set()
    for rank, p in enumerate(paragraphs):
        if p["doc_id"] in seen:
            continue
        seen.add(p["doc_id"])
        sents = p["sentences"]
        n = len(sents)
        for i in range(n):
            start, end = max(0, i - window), min(n, i + window + 1)
            windows.append(
                Window(
                    window_id=f"{p['doc_id']}#{i}",
                    doc_id=p["doc_id"],
                    title=p["title"],
                    center=i,
                    start=start,
                    end=end,
                    rank=rank,
                    text=f"{p['title']}: {' '.join(sents[start:end])}",
                )
            )
    return windows


def bm25_scores(texts: Sequence[str], query: str, tokenizer: Tokenizer) -> list[float]:
    """`texts`를 하나의 작은 컬렉션으로 보고 BM25 Okapi 점수를 매긴다.

    IDF는 Lucene 방식 log(1 + (N - df + 0.5) / (df + 0.5))로 계산해, 창이 적어도
    음수가 나오지 않게 한다.

    Args:
        texts: 점수를 매길 텍스트들
        query: 검색어
        tokenizer: 검색 도구와 같은 토크나이저

    Returns:
        `texts`와 같은 순서의 점수 리스트
    """
    if not texts:
        return []
    docs = [tokenizer(t) for t in texts]
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n or 1.0
    df = Counter(term for d in docs for term in set(d))
    query_terms = [t for t in tokenizer(query) if t in df]
    idf = {t: math.log(1.0 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in set(query_terms)}

    scores: list[float] = []
    for d in docs:
        tf = Counter(d)
        norm = BM25_K1 * (1.0 - BM25_B + BM25_B * len(d) / avgdl)
        scores.append(
            sum(idf[t] * tf[t] * (BM25_K1 + 1.0) / (tf[t] + norm) for t in query_terms if tf[t])
        )
    return scores


def _merge_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """겹치거나 맞닿은 [start, end) 범위를 합친다."""
    merged: list[tuple[int, int]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def render(paragraphs: Sequence[Paragraph], kept: Sequence[Window]) -> str:
    """남긴 창을 근거 텍스트로 만든다.

    같은 문단에서 뽑힌 창은 문장 범위를 합쳐 같은 문장이 두 번 들어가지 않게 하고,
    떨어진 범위는 " ... "로 잇는다. 문단은 가장 높은 창 점수 순으로 놓는다.

    Args:
        paragraphs: 검색 결과 문단 (문장 원문 조회용)
        kept: 남긴 창 (점수 내림차순)

    Returns:
        "{title}: {문장들}" 블록을 빈 줄로 이은 문자열
    """
    sentences = {p["doc_id"]: p["sentences"] for p in paragraphs}
    by_doc: dict[str, list[Window]] = {}
    for w in kept:
        by_doc.setdefault(w.doc_id, []).append(w)

    blocks = []
    for doc_id, ws in by_doc.items():
        spans = _merge_spans((w.start, w.end) for w in ws)
        body = " ... ".join(" ".join(sentences[doc_id][s:e]) for s, e in spans)
        blocks.append(f"{ws[0].title}: {body}")
    return "\n\n".join(blocks)


def compress(
    question: str,
    paragraphs: Sequence[Paragraph],
    *,
    m: int,
    window: int,
    tokenizer: Tokenizer,
    followup_query: str | None = None,
) -> Compressed:
    """검색 결과를 상위 m개 문장 창으로 줄인다.

    Args:
        question: 원래 질문
        paragraphs: 검색 결과 `[{doc_id, title, sentences, ...}]` (검색 순위 순)
        m: 남길 문장 창 수 (`compress.m`)
        window: 앞뒤로 붙일 문장 수 (`compress.window`)
        tokenizer: 검색 도구와 같은 토크나이저
        followup_query: 재검색일 때 후속 검색어. 주면 "질문 + 후속 검색어"로 점수를 매긴다

    Returns:
        근거 텍스트와 남긴 창 ID. 동점은 검색 순위, 문장 번호 순으로 정해 결과가 항상 같다
    """
    query = f"{question} {followup_query}" if followup_query else question
    windows = build_windows(paragraphs, window)
    scores = bm25_scores([w.text for w in windows], query, tokenizer)
    order = sorted(
        range(len(windows)),
        key=lambda j: (-scores[j], windows[j].rank, windows[j].center),
    )
    kept = [windows[j] for j in order[: max(m, 0)]]
    return Compressed(
        context=render(paragraphs, kept),
        kept_windows=[w.window_id for w in kept],
    )


def full_paragraphs(paragraphs: Sequence[Paragraph]) -> Compressed:
    """절제 변형 `m_minus_compress`: 상위 k개 문단을 통째로 넣는다.

    Args:
        paragraphs: 검색 결과 문단

    Returns:
        문단 전체를 담은 근거 텍스트, 모든 문장을 가운데로 하는 창 ID
    """
    return Compressed(
        context="\n\n".join(f"{p['title']}: {' '.join(p['sentences'])}" for p in paragraphs),
        kept_windows=[f"{p['doc_id']}#{i}" for p in paragraphs for i in range(len(p["sentences"]))],
    )


def covered_sentences(kept_windows: Iterable[str], window: int) -> set[tuple[str, int]]:
    """남긴 창이 덮는 문장 집합. 근거 문장 보존율 측정(C-2)에 쓴다.

    Args:
        kept_windows: "{doc_id}#{문장 번호}" 창 ID들
        window: 압축 때 쓴 `compress.window`

    Returns:
        {(doc_id, 문장 번호)}. 문단 끝을 넘는 번호가 섞일 수 있지만 근거 문장과는 겹치지 않는다
    """
    covered: set[tuple[str, int]] = set()
    for wid in kept_windows:
        doc_id, _, center = wid.rpartition("#")
        c = int(center)
        covered.update((doc_id, i) for i in range(max(0, c - window), c + window + 1))
    return covered
