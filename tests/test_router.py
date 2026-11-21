"""Tests for the M(EBA) router (PR A-4): route decision and math-path flow.

모델은 가짜(FakeChatOllama)로 바꿔 CPU만으로 돈다. 가짜가 진짜 B-1 콜백을
구동하므로 호출 수·상한·잘림 재시도 집계는 BudgetedLLM의 실제 동작을 거친다.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from src.baselines.prompts import SHARED_ANSWER_INSTRUCTION
from src.config.settings import load_settings
from src.meba.budget import BudgetedLLM
from src.meba.router import route_question


class FakeChatOllama:
    """ChatOllama double: drives the real callback and records what it was sent."""

    def __init__(self, callback: Any, cap: int, reply: dict[str, Any], log: list[dict[str, Any]]):
        self.callback = callback
        self.cap = cap
        self.reply = reply
        self.log = log

    def invoke(self, model_input: Any, **kwargs: Any) -> SimpleNamespace:
        self.log.append({"cap": self.cap, "input": model_input, "kwargs": kwargs})
        self.callback.on_chat_model_start(
            {}, [[]], invocation_params={"options": {"num_predict": self.cap}}
        )
        truncated = self.reply.get("truncated", False)
        out_tokens = 999 if truncated else 8
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
            content=self.reply["content"],
            tool_calls=self.reply.get("tool_calls", []),
        )


class FakeFactory:
    """Scripted ChatOllama factory that records every request."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.log: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> FakeChatOllama:
        return FakeChatOllama(
            kwargs["callbacks"][0], kwargs["num_predict"], self.replies.pop(0), self.log
        )


def make_budget(*replies: dict[str, Any]) -> tuple[BudgetedLLM, FakeFactory]:
    factory = FakeFactory(list(replies))
    budget = BudgetedLLM(load_settings("configs/default.yaml"), llm_factory=factory)
    return budget, factory


def say(content: str, tool_calls: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    return {"content": content, "tool_calls": tool_calls or []}


def test_search_route_uses_one_router_call_and_no_answer() -> None:
    """ROUTE: search -> 검색 경로. 첫 호출 한 번(cap.first)만 쓰고 답은 만들지 않는다."""

    budget, factory = make_budget(say("ROUTE: search"))

    outcome = route_question("Who wrote Hamlet?", budget)

    assert outcome.route == "search"
    assert outcome.answer is None
    assert budget.n_calls == 1
    assert factory.log[0]["cap"] == budget.settings.cap.first
    assert [c.step for c in budget.calls] == ["router"]


def test_math_route_without_calculator_answers_in_one_call() -> None:
    """수학 경로: 계산기를 안 쓰면 첫 호출 한 번으로 끝나고, 접두어 없는 답이 나온다."""

    budget, factory = make_budget(
        say("ROUTE: math\n3 apples and 4 apples make 7 apples.\nFinal answer: 7")
    )

    outcome = route_question("Tom has 3 apples and buys 4 more. How many now?", budget)

    assert outcome.route == "math"
    assert outcome.answer == "7"
    assert budget.n_calls == 1
    assert [c.step for c in budget.calls] == ["router"]
    # 수학 경로는 공통 답 형식 지시문 ①+②를 쓴다 (INTERFACES §6).
    sent = " ".join(m["content"] for m in factory.log[0]["input"])
    assert SHARED_ANSWER_INSTRUCTION in sent


def test_math_route_with_calculator_finishes_with_finalize_call() -> None:
    """계산기를 쓰면 마무리 호출(cap.finalize)이 이어지고 계산 결과가 거기에 전달된다."""

    budget, factory = make_budget(
        say(
            "ROUTE: math\nEach box holds 12 items and there are 7 boxes.",
            tool_calls=[{"name": "calculator", "args": {"expression": "12 * 7"}, "id": "call-1"}],
        ),
        say("Final answer: 84"),
    )

    outcome = route_question("Each box holds 12 items. How many items in 7 boxes?", budget)

    assert outcome.route == "math"
    assert outcome.answer == "84"
    assert outcome.used_calculator is True
    assert budget.n_calls == 2
    assert [c.step for c in budget.calls] == ["router", "finalize"]
    assert [entry["cap"] for entry in factory.log] == [
        budget.settings.cap.first,
        budget.settings.cap.finalize,
    ]
    # 첫 호출에는 계산기가 제공되고, 마무리 호출에는 계산 결과(도구 메시지)가 들어간다.
    assert factory.log[0]["kwargs"]["tools"]
    tool_messages = [m for m in factory.log[1]["input"] if m["role"] == "tool"]
    assert [m["content"] for m in tool_messages] == ["84"]
    assert tool_messages[0]["tool_call_id"] == "call-1"


def test_tool_call_without_route_line_goes_to_math_path() -> None:
    """도구 호출 응답은 본문이 비어 `ROUTE:` 줄이 없다. 계산기를 불렀으면 수학 경로로 본다.

    실제 ChatOllama(langchain-ollama 1.1.0)에서 도구 호출 응답의 content가 ''임을 확인했다.
    """

    budget, _ = make_budget(
        say("", tool_calls=[{"name": "calculator", "args": {"expression": "6 * 7"}, "id": "c1"}]),
        say("Final answer: 42"),
    )

    outcome = route_question("What is six times seven?", budget)

    assert outcome.route == "math"
    assert outcome.answer == "42"
    assert outcome.used_calculator is True


def test_missing_route_line_without_tool_call_goes_to_search() -> None:
    """`ROUTE:` 줄도 계산기 호출도 없으면 검색 경로. 직답 경로를 열지 않는다 (SPEC §4·§9)."""

    budget, _ = make_budget(say("William Shakespeare\nFinal answer: William Shakespeare"))

    outcome = route_question("Who wrote Hamlet?", budget)

    assert outcome.route == "search"
    assert outcome.answer is None


def test_route_line_never_leaks_into_math_answer() -> None:
    """`Final answer:` 줄이 없어도 `ROUTE: math` 줄은 답에 섞이지 않는다."""

    budget, _ = make_budget(say("ROUTE: math\n7"))

    outcome = route_question("What is 3 plus 4?", budget)

    assert outcome.answer == "7"


def test_malformed_calculator_arguments_become_error_observations() -> None:
    """계산기 인자가 깨져도 예외 없이 `ERROR:` 관찰 결과로 넘긴다 (INTERFACES §3)."""

    budget, factory = make_budget(
        say(
            "ROUTE: math",
            tool_calls=[
                {"name": "calculator", "args": "1+1", "id": "c1"},
                {"name": "calculator", "args": {"expression": ["unhashable"]}, "id": "c2"},
                {"name": "web_search", "args": {"query": "x"}, "id": "c3"},
            ],
        ),
        say("Final answer: UNANSWERABLE"),
    )

    outcome = route_question("What is one plus one?", budget)

    tool_messages = [m for m in factory.log[1]["input"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["c1", "c2", "c3"]
    assert all(m["content"].startswith("ERROR:") for m in tool_messages)
    assert outcome.used_calculator is False
    assert outcome.answer == "UNANSWERABLE"


def test_empty_finalize_content_gives_empty_answer() -> None:
    """마무리 호출 본문이 None이면 답은 빈 문자열이다 (문자열 'None'이 아니다)."""

    budget, _ = make_budget(
        say("ROUTE: math", tool_calls=[{"name": "calculator", "args": {"expression": "2+2"}, "id": "c1"}]),
        {"content": None, "tool_calls": []},
    )

    outcome = route_question("What is two plus two?", budget)

    assert outcome.answer == ""


def test_router_offers_the_same_calculator_tool_as_p1() -> None:
    """라우터가 모델에 주는 계산기 정의는 B-2b `get_calculator_tool()`과 같아야 한다 (AGENTS.md)."""

    import pytest

    function_calling = pytest.importorskip("langchain_core.utils.function_calling")
    from src.meba.router import CALCULATOR_TOOL_SPEC
    from src.tools.calculator import get_calculator_tool

    assert CALCULATOR_TOOL_SPEC == function_calling.convert_to_openai_tool(get_calculator_tool())
