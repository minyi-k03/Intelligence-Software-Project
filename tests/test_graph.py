"""Tests for the M(EBA) LangGraph skeleton (PR A-5a): routing, requery loop, ablation switches.

모델은 가짜(FakeChatOllama)로 바꿔 CPU만으로 돈다. 가짜가 진짜 B-1 콜백을 구동하므로
호출 수·상한·잘림 재시도 집계는 BudgetedLLM의 실제 동작을 거친다. 검색은 작은 가짜
코퍼스의 실제 BM25Index를 쓴다. 근거 판정 응답은 C의 실제 JSON 스키마 형식이다.
"""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("langgraph")

from src.common.types import PipelineResult  # noqa: E402
from src.config.settings import RetrievalConfig, Settings, load_settings  # noqa: E402
from src.eval.scorer import UNANSWERABLE  # noqa: E402
from src.meba.graph import (  # noqa: E402
    CONDITIONAL_PARTS,
    CORE_PARTS,
    ConditionalHooks,
    MComponents,
    build_m,
    build_variant,
    variant_components,
)
from src.tools.bm25_search import BM25Index  # noqa: E402

QUESTION = "Where was the author of the novel Wonderland born?"
FOLLOWUP = "Lewis Carroll was born in"

CORPUS = [
    {
        "doc_id": "hp::Wonderland",
        "title": "Wonderland",
        "sentences": [
            "Wonderland is a novel by Lewis Carroll.",
            "It was published in 1865.",
            "The novel follows a girl named Alice.",
        ],
    },
    {
        "doc_id": "hp::Lewis Carroll",
        "title": "Lewis Carroll",
        "sentences": [
            "Lewis Carroll was born in Daresbury.",
            "He was a mathematician and writer.",
            "He taught at Oxford.",
        ],
    },
    {
        "doc_id": "hp::Gardening",
        "title": "Gardening",
        "sentences": ["Roses need sunlight.", "Tulips bloom in spring."],
    },
]


class FakeChatOllama:
    """ChatOllama double: drives the real callback and records what it was sent."""

    def __init__(self, callback: Any, kwargs: dict[str, Any], reply: dict[str, Any], log: list):
        self.callback = callback
        self.kwargs = kwargs
        self.reply = reply
        self.log = log

    def invoke(self, model_input: Any, **kwargs: Any) -> SimpleNamespace:
        cap = self.kwargs["num_predict"]
        self.log.append(
            {
                "cap": cap,
                "format": self.kwargs.get("format_schema"),
                "input": model_input,
                "kwargs": kwargs,
            }
        )
        options = {"num_predict": cap} if cap is not None else {}
        self.callback.on_chat_model_start({}, [[]], invocation_params={"options": options})
        truncated = self.reply.get("truncated", False)
        out_tokens = self.reply.get("out_tokens", 999 if truncated else 8)
        response = SimpleNamespace(
            generations=[[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=self.reply["content"],
                        usage_metadata={"input_tokens": 10, "output_tokens": out_tokens},
                        response_metadata={
                            "done_reason": "length" if truncated else "stop",
                            "prompt_eval_count": 10,
                            "eval_count": out_tokens,
                        },
                    ),
                    text="",
                    generation_info=None,
                )
            ]],
            llm_output=None,
        )
        self.callback.on_llm_end(response)
        return SimpleNamespace(
            content=self.reply["content"], tool_calls=self.reply.get("tool_calls", [])
        )


class FakeFactory:
    """Scripted ChatOllama factory that records every request."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.log: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> FakeChatOllama:
        return FakeChatOllama(kwargs["callbacks"][0], kwargs, self.replies.pop(0), self.log)


def route(name: str) -> dict[str, Any]:
    return {"content": f"ROUTE: {name}", "tool_calls": []}


def calculator_call(expression: str, call_id: str) -> dict[str, Any]:
    return {"name": "calculator", "args": {"expression": expression}, "id": call_id}


def verdict(decision: str, *, missing: str = "", answer: str = "", truncated: bool = False) -> dict[str, Any]:
    body = {"reasoning": "because", "decision": decision, "missing": missing, "answer": answer}
    return {"content": json.dumps(body), "tool_calls": [], "truncated": truncated}


@pytest.fixture
def settings() -> Settings:
    return load_settings("configs/default.yaml")


@pytest.fixture
def index() -> BM25Index:
    return BM25Index(CORPUS)


def make(
    settings: Settings, index: BM25Index, *replies: dict[str, Any], variant: str = "m", **kwargs: Any
):
    factory = FakeFactory(list(replies))
    pipeline = build_variant(variant, settings, llm_factory=factory, index=index, **kwargs)
    return pipeline, factory


def steps(result: PipelineResult) -> list[str]:
    return [c.step for c in result.calls]


# --- 기본 경로 ----------------------------------------------------------------


def test_search_route_answers_after_one_judgment(settings: Settings, index: BM25Index) -> None:
    """라우터(search) → 검색 → 압축 → 근거 판정(ANSWER): 모델 호출 2회, 로그 필드가 규격대로 찬다."""

    pipeline, factory = make(
        settings, index, route("search"), verdict("ANSWER", answer="Daresbury")
    )

    result = pipeline("q-1", QUESTION)

    assert isinstance(result, PipelineResult)
    assert result.answer == "Daresbury" and result.abstained is False
    assert result.route == "search"
    assert result.decisions == ["ANSWER"]
    assert result.n_calls == 2 and result.n_retries == 0
    assert steps(result) == ["router", "judge"]
    assert len(result.retrieved_doc_ids) == 1 and result.retrieved_doc_ids[0]
    assert len(result.kept_windows) == 1
    assert 0 < len(result.kept_windows[0]) <= settings.compress.m
    assert all("#" in w for w in result.kept_windows[0])
    assert result.budget_exhausted is False and result.loop_breaks == 0 and result.error is None
    # 판정 호출은 cap.judge와 JSON 스키마(키 순서 reasoning → decision → missing → answer)를 쓴다.
    judge_call = factory.log[1]
    assert judge_call["cap"] == settings.cap.judge
    assert list(judge_call["format"]["properties"]) == ["reasoning", "decision", "missing", "answer"]
    assert "RETRIEVE" in judge_call["format"]["properties"]["decision"]["enum"]


def test_math_route_skips_retrieval(settings: Settings, index: BM25Index) -> None:
    """수학 경로는 라우터 호출 한 번으로 끝나고 검색·압축·판정을 하지 않는다."""

    pipeline, _ = make(
        settings,
        index,
        {"content": "ROUTE: math\n3 + 4 = 7.\nFinal answer: 7", "tool_calls": []},
    )

    result = pipeline("q-2", "Tom has 3 apples and buys 4 more. How many now?")

    assert result.answer == "7" and result.route == "math"
    assert result.n_calls == 1 and steps(result) == ["router"]
    assert result.decisions == [] and result.retrieved_doc_ids == [] and result.kept_windows == []


def test_math_route_with_calculator_uses_finalize_call(settings: Settings, index: BM25Index) -> None:
    """계산기를 쓴 수학 경로는 마무리 호출까지 2회이고 cap.finalize를 쓴다."""

    pipeline, factory = make(
        settings,
        index,
        {
            "content": "ROUTE: math",
            "tool_calls": [{"name": "calculator", "args": {"expression": "12 * 7"}, "id": "c1"}],
        },
        {"content": "Final answer: 84", "tool_calls": []},
    )

    result = pipeline("q-3", "Each box holds 12 items. How many items in 7 boxes?")

    assert result.answer == "84" and result.route == "math"
    assert steps(result) == ["router", "finalize"]
    assert factory.log[1]["cap"] == settings.cap.finalize


def test_abstain_gives_unanswerable(settings: Settings, index: BM25Index) -> None:
    """판정 ABSTAIN → 답은 UNANSWERABLE, abstained=True, 예산 소진과는 별개다."""

    pipeline, _ = make(settings, index, route("search"), verdict("ABSTAIN"))

    result = pipeline("q-4", "Who won the 2031 chess championship?")

    assert result.answer == UNANSWERABLE and result.abstained is True
    assert result.decisions == ["ABSTAIN"] and result.budget_exhausted is False


def test_answer_equal_to_unanswerable_is_abstained(settings: Settings, index: BM25Index) -> None:
    """decision=ANSWER인데 답이 UNANSWERABLE이면 abstained=True, decisions에는 모델 값 그대로 (INTERFACES §4)."""

    pipeline, _ = make(settings, index, route("search"), verdict("ANSWER", answer="UNANSWERABLE"))

    result = pipeline("q-5", QUESTION)

    assert result.abstained is True and result.answer == UNANSWERABLE
    assert result.decisions == ["ANSWER"]


# --- 재검색 루프 --------------------------------------------------------------


def test_requery_searches_again_with_followup_and_answers(settings: Settings, index: BM25Index) -> None:
    """RETRIEVE → 후속 검색어로 재검색 → 근거를 이어 붙여 다시 판정. 호출 3회."""

    one_paragraph = dataclasses.replace(settings, retrieval=RetrievalConfig(k=1))
    pipeline, factory = make(
        one_paragraph,
        index,
        route("search"),
        verdict("RETRIEVE", missing=FOLLOWUP),
        verdict("ANSWER", answer="Daresbury"),
    )

    result = pipeline("q-6", QUESTION)

    assert result.answer == "Daresbury"
    assert result.decisions == ["RETRIEVE", "ANSWER"]
    assert steps(result) == ["router", "judge", "judge"]
    assert result.retrieved_doc_ids == [["hp::Wonderland"], ["hp::Lewis Carroll"]]
    assert len(result.kept_windows) == 2
    # 두 번째 판정에는 1차 근거와 재검색 근거가 모두 들어간다.
    last_prompt = " ".join(str(part) for part in factory.log[2]["input"])
    assert "Wonderland is a novel" in last_prompt and "Daresbury" in last_prompt


def test_last_judgment_cannot_retrieve(settings: Settings, index: BM25Index) -> None:
    """재검색 2회를 쓰면 마지막 판정 스키마에서 RETRIEVE가 빠진다. 호출은 라우터 1 + 판정 3 = 4회."""

    pipeline, factory = make(
        settings,
        index,
        route("search"),
        verdict("RETRIEVE", missing="Lewis Carroll birthplace Daresbury"),
        verdict("RETRIEVE", missing="Oxford college teaching mathematician"),
        verdict("ANSWER", answer="Daresbury"),
    )

    result = pipeline("q-7", QUESTION)

    assert result.n_calls == settings.budget.max_calls == 4
    assert result.decisions == ["RETRIEVE", "RETRIEVE", "ANSWER"]
    enums = [c["format"]["properties"]["decision"]["enum"] for c in factory.log[1:]]
    assert ["RETRIEVE" in e for e in enums] == [True, True, False]
    assert result.budget_exhausted is False and result.answer == "Daresbury"


def test_duplicate_search_is_blocked_and_forces_final_judgment(
    settings: Settings, index: BM25Index
) -> None:
    """같은 검색을 다시 하려 하면 루프 끊기가 막고, 지금 근거로 RETRIEVE 없는 판정을 받는다."""

    pipeline, factory = make(
        settings,
        index,
        route("search"),
        verdict("RETRIEVE", missing=QUESTION),  # 1차 검색과 같은 검색어
        verdict("ANSWER", answer="Daresbury"),
    )

    result = pipeline("q-8", QUESTION)

    assert result.loop_breaks == 1
    assert len(result.retrieved_doc_ids) == 1  # 막힌 검색은 새 회차가 아니다
    assert result.decisions == ["RETRIEVE", "ANSWER"]
    assert "RETRIEVE" not in factory.log[2]["format"]["properties"]["decision"]["enum"]


def test_unusable_followup_forces_final_judgment(settings: Settings, index: BM25Index) -> None:
    """빠진 정보에서 검색어를 못 만들면(토큰 없음) 재검색 없이 마지막 판정으로 간다."""

    pipeline, factory = make(
        settings,
        index,
        route("search"),
        verdict("RETRIEVE", missing="..."),
        verdict("ANSWER", answer="Daresbury"),
    )

    result = pipeline("q-9", QUESTION)

    assert result.decisions == ["RETRIEVE", "ANSWER"] and len(result.retrieved_doc_ids) == 1
    assert "RETRIEVE" not in factory.log[2]["format"]["properties"]["decision"]["enum"]


# --- 예산·잘림 ----------------------------------------------------------------


def test_budget_exhausted_is_reported_separately_from_abstain(
    settings: Settings, index: BM25Index
) -> None:
    """호출 예산을 라우터가 다 쓰면 판정을 못 부르고 budget_exhausted=True, 답은 빈 문자열(기권 아님)."""

    tiny = dataclasses.replace(settings, budget=dataclasses.replace(settings.budget, max_calls=1))
    pipeline, factory = make(tiny, index, route("search"))

    result = pipeline("q-10", QUESTION)

    assert result.budget_exhausted is True
    assert result.answer == "" and result.abstained is False
    assert result.n_calls == 1 and len(factory.log) == 1


def test_budget_runs_out_mid_requery_loop(settings: Settings, index: BM25Index) -> None:
    """RETRIEVE 뒤 재검색까지 했는데 다음 판정에서 예산이 바닥나면 빈 답, 기권 아님."""

    small = dataclasses.replace(settings, budget=dataclasses.replace(settings.budget, max_calls=2))
    pipeline, factory = make(small, index, route("search"), verdict("RETRIEVE", missing=FOLLOWUP))

    result = pipeline("q-10b", QUESTION)

    assert result.budget_exhausted is True
    assert result.answer == "" and result.abstained is False
    assert result.decisions == ["RETRIEVE"]
    assert len(result.retrieved_doc_ids) == 2  # 재검색은 실행됐고 판정만 막혔다
    assert result.n_calls == 2 and len(factory.log) == 2 and result.error is None


def test_budget_runs_out_before_math_finalize(settings: Settings, index: BM25Index) -> None:
    """계산기를 부른 뒤 마무리 호출에서 예산이 바닥나면 수학 경로 그대로 빈 답, 기권 아님."""

    tiny = dataclasses.replace(settings, budget=dataclasses.replace(settings.budget, max_calls=1))
    pipeline, factory = make(
        tiny,
        index,
        {"content": "ROUTE: math", "tool_calls": [calculator_call("12 * 7", "c1")]},
    )

    result = pipeline("q-10c", "Each box holds 12 items. How many items in 7 boxes?")

    assert result.route == "math" and result.budget_exhausted is True
    assert result.answer == "" and result.abstained is False
    assert steps(result) == ["router"] and len(factory.log) == 1


def test_truncated_judgment_retries_once_without_using_budget(
    settings: Settings, index: BM25Index
) -> None:
    """판정이 잘리면 상한 2배로 한 번 재시도하고, 재시도는 n_calls가 아니라 n_retries에 센다."""

    pipeline, factory = make(
        settings,
        index,
        route("search"),
        verdict("ANSWER", answer="Dares", truncated=True),
        verdict("ANSWER", answer="Daresbury"),
    )

    result = pipeline("q-11", QUESTION)

    assert result.answer == "Daresbury"
    assert result.n_calls == 2 and result.n_retries == 1
    assert [c["cap"] for c in factory.log] == [
        settings.cap.first,
        settings.cap.judge,
        settings.cap.judge * settings.cap.retry_multiplier,
    ]


def test_model_error_is_reported_with_partial_calls(settings: Settings, index: BM25Index) -> None:
    """모델 오류는 예외로 새지 않고 error에 남는다. 그때까지의 호출 기록은 보존한다."""

    class Boom(FakeFactory):
        def __call__(self, **kwargs: Any) -> FakeChatOllama:
            if self.log:
                raise RuntimeError("ollama down")
            return super().__call__(**kwargs)

    factory = Boom([route("search")])
    pipeline = build_variant("m", settings, llm_factory=factory, index=index)

    result = pipeline("q-12", QUESTION)

    assert result.error is not None and result.error.startswith("RuntimeError")
    assert result.answer == "" and result.abstained is False
    assert steps(result) == ["router"]


def test_search_tool_error_is_an_observation_not_a_crash(
    settings: Settings, index: BM25Index, monkeypatch: pytest.MonkeyPatch
) -> None:
    """검색 도구가 오류 문자열을 돌려주면 근거 없음으로 보고 판정까지 간다 (오류로 멈추지 않는다)."""

    monkeypatch.setattr(
        "src.meba.graph.bm25_search", lambda *args, **kwargs: "ERROR: index unavailable"
    )
    pipeline, factory = make(settings, index, route("search"), verdict("ABSTAIN"))

    result = pipeline("q-12b", QUESTION)

    assert result.error is None
    assert result.retrieved_doc_ids == [[]] and result.kept_windows == [[]]
    assert result.decisions == ["ABSTAIN"] and result.abstained is True
    assert steps(result) == ["router", "judge"] and len(factory.log) == 2


# --- 에이전트에 유형을 알려 주지 않는다 ---------------------------------------


def test_only_the_question_reaches_the_model(settings: Settings, index: BM25Index) -> None:
    """qid(문항 ID에 유형이 들어 있다)는 어떤 모델 입력에도 들어가지 않는다 (AGENTS.md)."""

    pipeline, factory = make(settings, index, route("search"), verdict("ANSWER", answer="x"))

    pipeline("hotpotqa-0007", QUESTION)

    sent = json.dumps([entry["input"] for entry in factory.log], default=str)
    assert "hotpotqa" not in sent and "0007" not in sent and QUESTION in sent


# --- 절제 변형 ----------------------------------------------------------------


def test_variant_names_match_ablation_table() -> None:
    """절제 변형 이름은 `m_minus_<부품>`이고 결과표(`tables.CORE_ABLATIONS`)와 일치한다."""

    from src.eval.tables import CORE_ABLATIONS

    assert {f"m_minus_{part}" for part in CORE_PARTS} == set(CORE_ABLATIONS)
    assert variant_components("m") == MComponents()
    assert set(CORE_PARTS) == {"router", "requery", "judge", "compress", "cap", "loop_breaker"}
    assert set(CONDITIONAL_PARTS) == {"ledger", "concise", "dynamic_m"}
    with pytest.raises(ValueError, match="m_minus_bogus"):
        variant_components("m_minus_bogus")
    with pytest.raises(ValueError):
        variant_components("p1")


@pytest.mark.parametrize("part", CORE_PARTS)
def test_each_variant_turns_off_exactly_one_core_part(part: str) -> None:
    """변형마다 핵심 부품 하나만 꺼지고 나머지 다섯은 M과 같다."""

    components = variant_components(f"m_minus_{part}")

    assert getattr(components, part) is False
    assert [p for p in CORE_PARTS if getattr(components, p)] == [p for p in CORE_PARTS if p != part]


def test_turning_off_a_core_part_turns_off_conditional_parts_on_top_of_it() -> None:
    """핵심 부품을 빼면 그 위에서 동작하는 조건부 부품도 함께 빠진다 (SPEC §6, ADR-0006)."""

    everything = MComponents(ledger=True, concise=True, dynamic_m=True)

    assert everything.without("requery").ledger is False
    assert everything.without("judge").ledger is False
    assert everything.without("compress").dynamic_m is False
    assert everything.without("cap").concise is False
    # 상관없는 부품을 빼도 조건부 부품은 그대로다.
    assert everything.without("router") == dataclasses.replace(everything, router=False)


@pytest.mark.parametrize(
    ("core", "conditional"),
    [("cap", "concise"), ("judge", "ledger"), ("requery", "ledger"), ("compress", "dynamic_m")],
)
def test_conditional_part_cannot_run_on_a_switched_off_core_part(core: str, conditional: str) -> None:
    """직접 만든 구성이라도 핵심 부품을 끈 채 그 위의 조건부 부품을 켜면 만들 때 실패한다 (SPEC §6)."""

    with pytest.raises(ValueError, match=rf"{conditional}.*{core}"):
        MComponents(**{core: False, conditional: True})


def test_m_minus_router_runs_the_full_requery_loop(settings: Settings, index: BM25Index) -> None:
    """라우터를 꺼도 재검색 2회는 그대로: 판정 3회이고 마지막 판정에는 RETRIEVE가 없다."""

    pipeline, factory = make(
        settings,
        index,
        verdict("RETRIEVE", missing="Lewis Carroll birthplace Daresbury"),
        verdict("RETRIEVE", missing="Oxford college teaching mathematician"),
        verdict("ANSWER", answer="Daresbury"),
        variant="m_minus_router",
    )

    result = pipeline("q-13b", QUESTION)

    assert result.n_calls == 3 and steps(result) == ["judge", "judge", "judge"]
    assert result.decisions == ["RETRIEVE", "RETRIEVE", "ANSWER"]
    enums = [c["format"]["properties"]["decision"]["enum"] for c in factory.log]
    assert ["RETRIEVE" in e for e in enums] == [True, True, False]
    assert result.answer == "Daresbury" and result.budget_exhausted is False


def test_m_minus_router_sends_every_question_to_search(settings: Settings, index: BM25Index) -> None:
    """라우터를 끄면 라우터 호출 없이 모든 질문이 검색 경로로 간다 (수학도)."""

    pipeline, factory = make(
        settings, index, verdict("ANSWER", answer="7"), variant="m_minus_router"
    )

    result = pipeline("q-13", "Tom has 3 apples and buys 4 more. How many now?")

    assert result.route == "search"
    assert steps(result) == ["judge"] and result.n_calls == 1
    assert len(factory.log) == 1


def test_m_minus_requery_searches_once(settings: Settings, index: BM25Index) -> None:
    """재검색을 끄면 판정 스키마에 RETRIEVE가 없고 검색은 한 번뿐이다."""

    pipeline, factory = make(
        settings,
        index,
        route("search"),
        verdict("ANSWER", answer="Daresbury"),
        variant="m_minus_requery",
    )

    result = pipeline("q-14", QUESTION)

    assert len(result.retrieved_doc_ids) == 1 and result.n_calls == 2
    assert "RETRIEVE" not in factory.log[1]["format"]["properties"]["decision"]["enum"]


def test_m_minus_judge_removes_abstain_but_keeps_requery(settings: Settings, index: BM25Index) -> None:
    """근거 판정을 끄면 ABSTAIN이 스키마에서 빠지고 RETRIEVE는 남는다."""

    pipeline, factory = make(
        settings,
        index,
        route("search"),
        verdict("ANSWER", answer="Daresbury"),
        variant="m_minus_judge",
    )

    pipeline("q-15", QUESTION)

    enum = factory.log[1]["format"]["properties"]["decision"]["enum"]
    assert "ABSTAIN" not in enum and "RETRIEVE" in enum


def test_m_minus_compress_keeps_whole_paragraphs(settings: Settings, index: BM25Index) -> None:
    """압축을 끄면 상위 k개 문단을 통째로 넣고, kept_windows는 모든 문장 창이다."""

    pipeline, factory = make(
        settings,
        index,
        route("search"),
        verdict("ANSWER", answer="Daresbury"),
        variant="m_minus_compress",
    )

    result = pipeline("q-16", QUESTION)

    prompt = " ".join(str(part) for part in factory.log[1]["input"])
    assert "It was published in 1865." in prompt  # 압축이면 점수가 낮아 빠질 문장
    assert "hp::Wonderland#2" in result.kept_windows[0]


def test_m_minus_cap_calls_without_output_caps(settings: Settings, index: BM25Index) -> None:
    """출력 상한을 끄면 num_predict 없이 부르고, 잘려도 재시도하지 않는다."""

    pipeline, factory = make(
        settings,
        index,
        route("search"),
        verdict("ANSWER", answer="Dares", truncated=True),
        variant="m_minus_cap",
    )

    result = pipeline("q-17", QUESTION)

    assert [entry["cap"] for entry in factory.log] == [None, None]
    assert result.n_retries == 0 and all(c.cap is None for c in result.calls)
    assert result.n_calls == 2  # 호출 예산은 그대로 센다


def test_m_minus_cap_long_output_is_not_mistaken_for_truncation(
    settings: Settings, index: BM25Index
) -> None:
    """상한이 없으면 출력 토큰이 많아도(done_reason=stop) 잘림이 아니고 재시도도 없다."""

    long_answer = {**verdict("ANSWER", answer="Daresbury"), "out_tokens": 4096}
    pipeline, factory = make(
        settings, index, route("search"), long_answer, variant="m_minus_cap"
    )

    result = pipeline("q-17b", QUESTION)

    assert [c.truncated for c in result.calls] == [False, False]
    assert all(c.cap is None for c in result.calls)
    assert result.n_retries == 0 and len(factory.log) == 2
    assert result.answer == "Daresbury"


def test_m_minus_loop_breaker_repeats_the_same_search(settings: Settings, index: BM25Index) -> None:
    """루프 끊기를 끄면 같은 검색을 다시 실행하고 loop_breaks는 0이다."""

    pipeline, _ = make(
        settings,
        index,
        route("search"),
        verdict("RETRIEVE", missing=QUESTION),
        verdict("ANSWER", answer="Daresbury"),
        variant="m_minus_loop_breaker",
    )

    result = pipeline("q-18", QUESTION)

    assert result.loop_breaks == 0
    assert len(result.retrieved_doc_ids) == 2  # 같은 검색이 한 번 더 실행됐다
    assert result.decisions == ["RETRIEVE", "ANSWER"]


@pytest.mark.parametrize(("variant", "expected_breaks"), [("m", 1), ("m_minus_loop_breaker", 0)])
def test_loop_breaker_switch_reaches_the_calculator(
    settings: Settings, index: BM25Index, variant: str, expected_breaks: int
) -> None:
    """수학 경로에서 같은 계산기 호출 2회: M은 하나를 막고, 루프 끊기를 끈 변형은 둘 다 실행한다."""

    pipeline, factory = make(
        settings,
        index,
        {
            "content": "ROUTE: math",
            "tool_calls": [calculator_call("12 * 7", "c1"), calculator_call("12 * 7", "c2")],
        },
        {"content": "Final answer: 84", "tool_calls": []},
        variant=variant,
    )

    result = pipeline("q-18b", "Each box holds 12 items. How many items in 7 boxes?")

    assert result.loop_breaks == expected_breaks
    assert result.answer == "84" and steps(result) == ["router", "finalize"]
    tool_outputs = [m["content"] for m in factory.log[1]["input"] if m.get("role") == "tool"]
    assert len(tool_outputs) == 2 and "84" in tool_outputs[1]


# --- 조건부 부품 자리 ---------------------------------------------------------


def test_conditional_part_without_hook_fails_loudly(settings: Settings, index: BM25Index) -> None:
    """조건부 스위치를 켰는데 부품이 아직 안 붙었으면 조용히 M으로 돌지 않고 바로 실패한다."""

    for part in CONDITIONAL_PARTS:
        with pytest.raises(ValueError, match=part):
            build_m(settings, components=MComponents(**{part: True}), index=index)


def test_conditional_hooks_are_called_where_the_parts_attach(
    settings: Settings, index: BM25Index
) -> None:
    """dynamic_m은 압축 m을, ledger는 후속 검색어 재료를, concise는 모델 입력을 바꿀 수 있다."""

    seen: dict[str, list[Any]] = {"m": [], "ledger": [], "concise": []}

    def choose_m(question: str, default_m: int) -> int:
        seen["m"].append((question, default_m))
        return 2

    def ledger(question: str, missing: str) -> str:
        seen["ledger"].append((question, missing))
        return FOLLOWUP

    def concise(step: str, model_input: Any) -> Any:
        seen["concise"].append(step)
        return [*model_input, {"role": "system", "content": f"CONCISE-{step}"}]

    factory = FakeFactory(
        [route("search"), verdict("RETRIEVE", missing="something vague"), verdict("ANSWER", answer="Daresbury")]
    )
    pipeline = build_m(
        settings,
        components=MComponents(ledger=True, concise=True, dynamic_m=True),
        hooks=ConditionalHooks(choose_m=choose_m, ledger=ledger, concise=concise),
        llm_factory=factory,
        index=index,
    )

    result = pipeline("q-19", QUESTION)

    assert [len(w) for w in result.kept_windows] == [2, 2]
    assert seen["m"] == [(QUESTION, settings.compress.m)] * 2
    assert seen["ledger"] == [(QUESTION, "something vague")]
    assert seen["concise"] == ["router", "judge", "judge"]
    # 훅이 돌려준 입력이 실제로 모델에 간다 (돌려준 값을 버리면 실패한다).
    assert [entry["input"][-1]["content"] for entry in factory.log] == [
        "CONCISE-router",
        "CONCISE-judge",
        "CONCISE-judge",
    ]
    assert result.decisions == ["RETRIEVE", "ANSWER"]


# --- 러너 연결 ----------------------------------------------------------------


def test_pipeline_is_reusable_across_questions(settings: Settings, index: BM25Index) -> None:
    """그래프는 한 번 만들어 문항마다 재사용한다. 문항 사이에 예산·재검색 횟수·루프 끊기가 새지 않는다."""

    pipeline, _ = make(
        settings,
        index,
        route("search"),
        verdict("RETRIEVE", missing=FOLLOWUP),
        verdict("ANSWER", answer="Daresbury"),
        route("search"),
        verdict("RETRIEVE", missing=FOLLOWUP),
        verdict("ANSWER", answer="Daresbury"),
    )

    first = pipeline("q-20", QUESTION)
    second = pipeline("q-21", QUESTION)

    assert first.n_calls == second.n_calls == 3
    assert first.decisions == second.decisions == ["RETRIEVE", "ANSWER"]
    assert second.loop_breaks == 0


def test_runner_registry_points_at_existing_factories() -> None:
    """`scripts/run_eval.py`가 등록한 M과 절제 변형 이름이 모두 불러와진다."""

    from scripts.run_eval import PIPELINE_FACTORIES, load_factory

    names = ["m", *(f"m_minus_{part}" for part in CORE_PARTS)]
    for name in names:
        assert name in PIPELINE_FACTORIES
        assert callable(load_factory(name, PIPELINE_FACTORIES))
