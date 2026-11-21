"""다중 홉 재검색(requery.py) 단위 테스트 (PR C-3). 모델·GPU 없이 돈다."""

from __future__ import annotations

from src.config.settings import get_settings
from src.meba.compress import compress
from src.meba.judge import parse_judgment
from src.meba.loop_breaker import LoopBreaker
from src.meba.requery import Requery, clean_followup, query_key, search_args_key
from src.tools.bm25_search import BM25Index
from src.tools.tokenizer import tokenize

QUESTION = "Who did Neleus of Scepsis study under?"
DOCS = [
    {"doc_id": "a", "title": "Neleus of Scepsis", "sentences": ["Neleus studied under Aristotle."], "text": "Neleus studied under Aristotle."},
    {"doc_id": "b", "title": "Theophrastus", "sentences": ["Theophrastus led the Peripatetic school."], "text": "Theophrastus led the Peripatetic school."},
    {"doc_id": "c", "title": "Scepsis", "sentences": ["Scepsis was a town in the Troad."], "text": "Scepsis was a town in the Troad."},
]


def test_query_key_ignores_case_spacing_order_and_stopwords() -> None:
    base = query_key("Neleus of Scepsis")
    assert base == query_key("  scepsis   NELEUS ") == query_key("the Scepsis of Neleus")
    assert base != query_key("Neleus library")


def test_equal_query_keys_give_identical_search_results() -> None:
    index = BM25Index(DOCS)
    a, b = "Neleus of Scepsis", "the scepsis  NELEUS"
    assert query_key(a) == query_key(b)
    assert index.search(a, k=3) == index.search(b, k=3)


def test_clean_followup() -> None:
    assert clean_followup('  "the birthplace\n of   Theophrastus."  ') == "the birthplace of Theophrastus"


def test_allows_retrieve_until_max_requery_is_used() -> None:
    rq = Requery(max_requery=2)
    assert rq.allow_retrieve and rq.n_requery == 0

    assert rq.next_query("teacher of Theophrastus") == "teacher of Theophrastus"
    assert rq.allow_retrieve and rq.n_requery == 1

    assert rq.next_query("Peripatetic school founder") == "Peripatetic school founder"
    assert not rq.allow_retrieve and rq.n_requery == 2

    assert rq.next_query("anything else") is None
    assert rq.queries == ["teacher of Theophrastus", "Peripatetic school founder"]


def test_default_limit_comes_from_settings() -> None:
    limit = get_settings().budget.max_requery
    rq = Requery(max_requery=limit)
    for i in range(limit):
        assert rq.next_query(f"distinct topic number{i}") is not None
    assert not rq.allow_retrieve


def test_zero_max_requery_never_allows_retrieve() -> None:
    rq = Requery(max_requery=0)
    assert not rq.allow_retrieve
    assert rq.next_query("teacher of Theophrastus") is None


def test_requery_does_not_block_repeats_itself() -> None:
    # 중복 검색은 루프 끊기의 일이다. 여기서 막으면 loop_breaks가 안 세지고 m_minus_loop_breaker를 만들 수 없다
    rq = Requery(max_requery=2)
    assert rq.next_query("who did neleus of scepsis study under") is not None
    assert rq.next_query("Theophrastus teacher") is not None
    assert rq.n_requery == 2


def run_search_path(breaker: LoopBreaker | None, followups: list[str]) -> tuple[list[str], int]:
    """질문으로 1차 검색 → 후속 검색어로 재검색. (실제로 검색한 검색어, 루프 끊기 횟수)."""
    index = BM25Index(DOCS)
    searched: list[str] = []
    rq = Requery(max_requery=len(followups))

    def search(query: str) -> bool:
        """검색하고, 루프 끊기에 막혔으면 True(바로 답하기)."""
        def run() -> list:
            searched.append(query)
            return index.search(query, k=1)
        if breaker is None:
            run()
            return False
        return breaker.run("bm25_search", {"query": query}, run).answer_directly

    search(QUESTION)
    for missing in followups:
        followup = rq.next_query(missing)
        if followup is None or search(followup):
            break
    return searched, breaker.loop_breaks if breaker else 0


def test_loop_breaker_catches_followup_equal_to_question() -> None:
    searched, breaks = run_search_path(LoopBreaker(normalizer=search_args_key), ["who did neleus of scepsis study under", "Theophrastus"])
    assert searched == [QUESTION] and breaks == 1


def test_loop_breaker_catches_repeated_followup() -> None:
    searched, breaks = run_search_path(LoopBreaker(normalizer=search_args_key), ["Theophrastus teacher", "the teacher of THEOPHRASTUS."])
    assert searched == [QUESTION, "Theophrastus teacher"] and breaks == 1


def test_without_loop_breaker_repeats_are_searched_again() -> None:
    # m_minus_loop_breaker: 반복 허용
    searched, breaks = run_search_path(None, ["Theophrastus teacher", "the teacher of THEOPHRASTUS."])
    assert searched == [QUESTION, "Theophrastus teacher", "the teacher of THEOPHRASTUS"] and breaks == 0


def test_missing_without_any_token_is_rejected() -> None:
    rq = Requery(max_requery=2)
    for missing in ("", "   ", '"."', "?!"):
        assert rq.next_query(missing) is None
    assert rq.n_requery == 0


def test_stopword_only_followup_is_searchable_like_the_search_tool() -> None:
    # 공용 토크나이저는 전부 불용어인 검색어("The Who")를 지우지 않는다
    rq = Requery(max_requery=2)
    assert rq.next_query("The Who") == "The Who"


def test_search_args_key_works_as_loop_breaker_normalizer() -> None:
    breaker = LoopBreaker(normalizer=search_args_key)
    runs: list[str] = []

    def search(query: str) -> str:
        outcome = breaker.run("bm25_search", {"query": query}, lambda: runs.append(query) or f"result {len(runs)}")
        return outcome.result

    assert search("Neleus of Scepsis") == "result 1"
    assert search("scepsis NELEUS") == "result 1"
    assert search("Neleus library") == "result 2"
    assert breaker.loop_breaks == 1 and runs == ["Neleus of Scepsis", "Neleus library"]


def test_search_path_walkthrough_with_scripted_judgments() -> None:
    """검색 → 압축 → 판정(RETRIEVE) → 재검색 → 압축 → 마지막 판정이 이어지는지 확인한다."""
    index = BM25Index(DOCS)
    rq = Requery(max_requery=2)

    first = compress(QUESTION, index.search(QUESTION, k=1), m=2, window=1, tokenizer=tokenize)
    assert "Aristotle" in first.context and "Peripatetic" not in first.context

    scripted = '{"reasoning": "r", "decision": "RETRIEVE", "missing": "Theophrastus school", "answer": ""}'
    judgment = parse_judgment(scripted, allow_retrieve=rq.allow_retrieve)
    followup = rq.next_query(judgment.missing)
    assert followup == "Theophrastus school"

    second = compress(QUESTION, index.search(followup, k=1), m=2, window=1, tokenizer=tokenize, followup_query=followup)
    assert "Peripatetic" in second.context
    assert second.kept_windows == ["b#0"]

    assert rq.next_query("Scepsis town") is not None
    assert not rq.allow_retrieve
    last = parse_judgment('{"reasoning": "r", "decision": "RETRIEVE", "missing": "x", "answer": ""}', allow_retrieve=rq.allow_retrieve)
    assert (last.decision, last.answer, last.parse_ok, last.abstained) == ("ANSWER", "", False, False)
