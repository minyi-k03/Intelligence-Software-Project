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
    rq = Requery(QUESTION, max_requery=2)
    assert rq.allow_retrieve and rq.n_requery == 0

    assert rq.next_query("teacher of Theophrastus") == "teacher of Theophrastus"
    assert rq.allow_retrieve and rq.n_requery == 1

    assert rq.next_query("Peripatetic school founder") == "Peripatetic school founder"
    assert not rq.allow_retrieve and rq.n_requery == 2

    assert rq.next_query("anything else") is None
    assert rq.queries == ["teacher of Theophrastus", "Peripatetic school founder"]


def test_default_limit_comes_from_settings() -> None:
    limit = get_settings().budget.max_requery
    rq = Requery(QUESTION, max_requery=limit)
    for i in range(limit):
        assert rq.next_query(f"distinct topic number{i}") is not None
    assert not rq.allow_retrieve


def test_zero_max_requery_never_allows_retrieve() -> None:
    rq = Requery(QUESTION, max_requery=0)
    assert not rq.allow_retrieve
    assert rq.next_query("teacher of Theophrastus") is None


def test_followup_equal_to_question_is_rejected_without_using_a_requery() -> None:
    rq = Requery(QUESTION, max_requery=2)
    assert rq.next_query("who did neleus of scepsis study under") is None
    assert rq.n_requery == 0 and rq.allow_retrieve


def test_repeated_followup_is_rejected() -> None:
    rq = Requery(QUESTION, max_requery=2)
    assert rq.next_query("Theophrastus teacher") is not None
    assert rq.next_query("the teacher of THEOPHRASTUS.") is None
    assert rq.n_requery == 1


def test_missing_without_any_token_is_rejected() -> None:
    rq = Requery(QUESTION, max_requery=2)
    for missing in ("", "   ", '"."', "?!"):
        assert rq.next_query(missing) is None
    assert rq.n_requery == 0


def test_stopword_only_followup_is_searchable_like_the_search_tool() -> None:
    # 공용 토크나이저는 전부 불용어인 검색어("The Who")를 지우지 않는다
    rq = Requery(QUESTION, max_requery=2)
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
    rq = Requery(QUESTION, max_requery=2)

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
    assert (last.decision, last.parse_ok) == ("ABSTAIN", False)
