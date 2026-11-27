"""조건부 부품(C-5) 테스트: 근거 장부 상세화, 동적 압축량. 가짜 모델로 그래프 전체를 돌린다."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

pytest.importorskip("langgraph")

from src.config.settings import RetrievalConfig, Settings, load_settings  # noqa: E402
from src.meba.conditional.dynamic_m import choose_m, proper_noun_spans  # noqa: E402
from src.meba.conditional.ledger import next_missing_item  # noqa: E402
from src.meba.graph import ConditionalHooks, MComponents, build_m  # noqa: E402
from src.tools.bm25_search import BM25Index  # noqa: E402
from tests.test_graph import CORPUS, QUESTION, FakeFactory, route  # noqa: E402


@pytest.fixture
def settings() -> Settings:
    return load_settings("configs/default.yaml")


@pytest.fixture
def index() -> BM25Index:
    return BM25Index(CORPUS)


def ledger_verdict(decision: str, ledger: list[dict[str, str]], *, missing: str = "", answer: str = "") -> dict[str, Any]:
    body = {"reasoning": "r", "ledger": ledger, "decision": decision, "missing": missing, "answer": answer}
    return {"content": json.dumps(body), "tool_calls": []}


# --- 근거 장부 상세화 --------------------------------------------------------------


def test_next_missing_item_takes_first_item() -> None:
    assert next_missing_item("q", "birthplace of Lewis Carroll; year of birth") == "birthplace of Lewis Carroll"
    assert next_missing_item("q", "single item") == "single item"
    assert next_missing_item("q", "") == ""


def test_ledger_pipeline_sends_ledger_schema_and_searches_missing_item(settings: Settings, index: BM25Index) -> None:
    one_paragraph = dataclasses.replace(settings, retrieval=RetrievalConfig(k=1))
    factory = FakeFactory([
        route("search"),
        ledger_verdict(
            "RETRIEVE",
            [{"item": "author of Wonderland", "status": "FOUND"}, {"item": "Lewis Carroll birthplace", "status": "MISSING"}],
            missing="where he was born",
        ),
        ledger_verdict("ANSWER", [{"item": "Lewis Carroll birthplace", "status": "FOUND"}], answer="Daresbury"),
    ])
    pipeline = build_m(
        one_paragraph,
        components=MComponents(ledger=True),
        hooks=ConditionalHooks(ledger=next_missing_item),
        llm_factory=factory,
        index=index,
    )

    result = pipeline("q-ledger", QUESTION)

    assert result.answer == "Daresbury" and result.decisions == ["RETRIEVE", "ANSWER"]
    assert result.retrieved_doc_ids == [["hp::Wonderland"], ["hp::Lewis Carroll"]]
    judge_calls = [entry for entry in factory.log if entry["format"]]
    assert all("ledger" in entry["format"]["properties"] for entry in judge_calls)
    assert result.n_calls == 3


def test_m_without_ledger_sends_no_ledger_field(settings: Settings, index: BM25Index) -> None:
    factory = FakeFactory([route("search"), ledger_verdict("ANSWER", [], answer="Daresbury")])
    build_m(settings, llm_factory=factory, index=index)("q", QUESTION)
    assert "ledger" not in factory.log[1]["format"]["properties"]


def test_ledger_needs_requery_and_judge() -> None:
    with pytest.raises(ValueError):
        MComponents(ledger=True, requery=False)


# --- 동적 압축량 ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "spans"),
    [
        ("What is the capital of France?", 1),
        ("Who founded the company that makes the Model S?", 1),
        ("Were Scott Derrickson and Ed Wood of the same nationality?", 2),
        ("Which magazine was started first, Arthur's Magazine or First for Women?", 3),
        ("how many legs does a spider have?", 0),
        ("Paris is the capital of which country?", 0),
    ],
)
def test_proper_noun_spans(question: str, spans: int) -> None:
    assert proper_noun_spans(question) == spans


def test_choose_m_halves_only_for_single_entity_questions() -> None:
    assert choose_m("What is the capital of France?", 7) == 3
    assert choose_m("Were Scott Derrickson and Ed Wood of the same nationality?", 7) == 7
    assert choose_m("What is the capital of France?", 1) == 1


def test_dynamic_m_pipeline_keeps_fewer_windows(settings: Settings, index: BM25Index) -> None:
    def kept(components: MComponents, hooks: ConditionalHooks) -> int:
        factory = FakeFactory([route("search"), ledger_verdict("ANSWER", [], answer="Daresbury")])
        result = build_m(settings, components=components, hooks=hooks, llm_factory=factory, index=index)("q", QUESTION)
        return len(result.kept_windows[0])

    assert proper_noun_spans(QUESTION) == 1
    base = kept(MComponents(), ConditionalHooks())
    dynamic = kept(MComponents(dynamic_m=True), ConditionalHooks(choose_m=choose_m))
    # 테스트 코퍼스에서 검색되는 문장은 6개라 기준선은 min(m, 6)개를 남긴다.
    assert dynamic == settings.compress.m // 2 < base


def test_registered_in_run_eval() -> None:
    from scripts.run_eval import PIPELINE_FACTORIES, load_factory

    for name in ("m_ledger", "m_dynamic_m"):
        assert callable(load_factory(name, PIPELINE_FACTORIES))
