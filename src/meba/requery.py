"""다중 홉 재검색 — 핵심 부품, 전략 ③ (SPEC §4·§5, 담당 C).

RETRIEVE 판정의 빠진 정보를 후속 검색어로 다듬고, 문항당 재검색 횟수(`budget.max_requery`)를
센다. 같은 검색의 반복은 루프 끊기가 막는다. 모델을 부르지 않는다.
"""

from __future__ import annotations

from typing import Any

from src.tools.tokenizer import Tokenizer, tokenize

QueryKey = tuple[str, ...]


def query_key(query: str, tokenizer: Tokenizer = tokenize) -> QueryKey:
    """BM25 검색 결과가 같아지는 검색어끼리 같은 값이 되는 키.

    BM25 점수는 토큰의 종류와 개수에만 달려 있으므로, 정렬한 토큰이 같으면
    대소문자·공백·불용어·단어 순서가 달라도 검색 결과가 같다.

    Args:
        query: 검색어
        tokenizer: 검색 도구와 같은 토크나이저

    Returns:
        정렬한 토큰 튜플
    """
    return tuple(sorted(tokenizer(query)))


def search_args_key(args: Any) -> QueryKey:
    """`LoopBreaker`의 정규화 함수로 넘길 수 있는 `bm25_search` 인자 키.

    Args:
        args: 검색어 문자열 또는 `{"query": ...}`

    Returns:
        `query_key`와 같은 키
    """
    return query_key(args["query"] if isinstance(args, dict) else args)


def clean_followup(missing: str) -> str:
    """빠진 정보 서술을 한 줄 검색어로 다듬는다.

    Args:
        missing: 근거 판정이 낸 빠진 정보

    Returns:
        공백을 정리하고 양끝 따옴표·마침표를 뗀 문자열
    """
    return " ".join(missing.split()).strip("\"'`*. ")


class Requery:
    """한 문항의 재검색 횟수와 후속 검색어. 문항마다 새로 만든다.

    같은 검색을 다시 하는지는 여기서 막지 않는다. 그건 루프 끊기(`LoopBreaker`)의 일이며,
    `search_args_key`를 정규화 함수로 넘기면 검색 결과가 같아지는 검색어끼리 중복으로 잡힌다.
    그래야 `loop_breaks`가 집계되고 `m_minus_loop_breaker`에서 반복을 허용할 수 있다.

    Args:
        max_requery: 재검색 최대 횟수 (`budget.max_requery`). 0이면 재검색 절제 변형
        tokenizer: 검색 도구와 같은 토크나이저
    """

    def __init__(self, max_requery: int, tokenizer: Tokenizer = tokenize) -> None:
        self._max = max_requery
        self._tokenizer = tokenizer
        self.queries: list[str] = []

    @property
    def n_requery(self) -> int:
        """지금까지 한 재검색 횟수."""
        return len(self.queries)

    @property
    def allow_retrieve(self) -> bool:
        """다음 근거 판정에서 RETRIEVE를 허용할지. 횟수를 다 쓰면 False (마지막 판정)."""
        return self.n_requery < self._max

    def next_query(self, missing: str) -> str | None:
        """빠진 정보로 다음 후속 검색어를 정하고 재검색 1회로 기록한다.

        Args:
            missing: RETRIEVE 판정의 빠진 정보

        Returns:
            후속 검색어. 재검색 횟수를 다 썼거나 검색어로 쓸 토큰이 없으면 None이고,
            호출한 쪽은 지금 근거로 RETRIEVE 없는 판정을 받으면 된다.
        """
        if not self.allow_retrieve:
            return None
        followup = clean_followup(missing)
        if not self._tokenizer(followup):
            return None
        self.queries.append(followup)
        return followup
