"""Tests for M(EBA) output caps, retries, and call budget."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from dataclasses import asdict
import json

import pytest

from src.config.settings import load_settings
from src.meba.budget import BudgetedLLM
from src.common.types import PipelineResult
from src.meba.loop_breaker import LoopBreaker


class FakeChatOllama:
    """CPU-only ChatOllama double that drives the real B-1 callback."""

    def __init__(self, callback: Any, cap: int, reply: dict[str, Any]) -> None:
        self.callback = callback
        self.cap = cap
        self.reply = reply

    def invoke(self, model_input: Any, **kwargs: Any) -> SimpleNamespace:
        self.callback.on_chat_model_start(
            {},
            [[]],
            invocation_params={"options": {"num_predict": self.cap}},
        )
        output_tokens = self.reply["output_tokens"]
        response = SimpleNamespace(
            generations=[[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=self.reply["content"],
                        usage_metadata={"input_tokens": 10, "output_tokens": output_tokens},
                        response_metadata={
                            "done_reason": self.reply["done_reason"],
                            "prompt_eval_count": 10,
                            "eval_count": output_tokens,
                        },
                    ),
                    text="",
                    generation_info=None,
                )
            ]],
            llm_output=None,
        )
        self.callback.on_llm_end(response)
        return SimpleNamespace(content=self.reply["content"])


class FakeFactory:
    """Return scripted fake models and record every requested cap."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.caps: list[int] = []

    def __call__(self, **kwargs: Any) -> FakeChatOllama:
        cap = kwargs["num_predict"]
        self.caps.append(cap)
        return FakeChatOllama(kwargs["callbacks"][0], cap, self.replies.pop(0))


def reply(content: str = "ok", *, truncated: bool = False) -> dict[str, Any]:
    """Build a scripted response using Ollama's existing truncation signal."""

    return {
        "content": content,
        "done_reason": "length" if truncated else "stop",
        "output_tokens": 999 if truncated else 8,
    }


def test_step_caps_are_read_from_settings() -> None:
    """Router/finalize/judge must receive their configured num_predict caps."""

    factory = FakeFactory([reply(), reply(), reply()])
    budget = BudgetedLLM(load_settings("configs/default.yaml"), llm_factory=factory)

    budget.invoke("router", "question")
    budget.invoke("finalize", "calculator result")
    budget.invoke("judge", "evidence")

    assert factory.caps == [
        budget.settings.cap.first,
        budget.settings.cap.finalize,
        budget.settings.cap.judge,
    ]
    assert budget.n_calls == 3
    assert budget.n_retries == 0


def test_normal_response_is_not_retried() -> None:
    """A normal Ollama stop signal must produce exactly one base call."""

    factory = FakeFactory([reply("normal")])
    budget = BudgetedLLM(load_settings("configs/default.yaml"), llm_factory=factory)

    result = budget.invoke("router", "question")

    assert result.response.content == "normal"
    assert not result.budget_exhausted
    assert len(budget.calls) == 1
    assert not budget.calls[0].truncated
    assert not budget.calls[0].is_retry


def test_truncation_retries_once_at_double_cap_and_returns_retry() -> None:
    """Even a second truncation must not trigger a third call."""

    settings = load_settings("configs/default.yaml")
    factory = FakeFactory([
        reply("partial-1", truncated=True),
        reply("partial-2", truncated=True),
    ])
    budget = BudgetedLLM(settings, llm_factory=factory)

    result = budget.invoke("judge", "evidence")

    assert result.response.content == "partial-2"
    assert factory.caps == [settings.cap.judge, settings.cap.judge * settings.cap.retry_multiplier]
    assert budget.n_calls == 1
    assert budget.n_retries == 1
    assert [record.is_retry for record in budget.calls] == [False, True]
    assert all(record.truncated for record in budget.calls)
    assert budget.callback.n_budget_calls == 1


def test_fifth_base_call_is_blocked_and_signals_budget_exhaustion() -> None:
    """Only the configured number of base calls may reach ChatOllama."""

    settings = load_settings("configs/default.yaml")
    factory = FakeFactory([reply() for _ in range(settings.budget.max_calls)])
    budget = BudgetedLLM(settings, llm_factory=factory)

    for _ in range(settings.budget.max_calls):
        assert not budget.invoke("judge", "evidence").budget_exhausted
    blocked = budget.invoke("judge", "evidence")

    assert blocked.response is None
    assert blocked.budget_exhausted
    assert budget.budget_exhausted
    assert budget.n_calls == settings.budget.max_calls
    assert len(factory.caps) == settings.budget.max_calls


def test_fourth_base_call_may_still_receive_free_retry() -> None:
    """A truncation retry remains allowed after the fourth charged call."""

    settings = load_settings("configs/default.yaml")
    factory = FakeFactory([
        reply(),
        reply(),
        reply(),
        reply("fourth-partial", truncated=True),
        reply("fourth-complete"),
    ])
    budget = BudgetedLLM(settings, llm_factory=factory)

    for _ in range(settings.budget.max_calls - 1):
        budget.invoke("judge", "evidence")
    fourth = budget.invoke("judge", "evidence")

    assert fourth.response.content == "fourth-complete"
    assert budget.n_calls == settings.budget.max_calls
    assert budget.n_retries == 1
    assert len(factory.caps) == settings.budget.max_calls + 1
    assert budget.calls[-1].is_retry


def test_reset_starts_a_new_question_budget() -> None:
    """Reset must clear callback records and both counters."""

    factory = FakeFactory([reply(), reply()])
    budget = BudgetedLLM(load_settings("configs/default.yaml"), llm_factory=factory)
    budget.invoke("router", "first question")

    budget.reset()
    result = budget.invoke("router", "second question")

    assert not result.budget_exhausted
    assert budget.n_calls == 1
    assert budget.n_retries == 0
    assert len(budget.calls) == 1


def test_factory_failure_does_not_consume_budget() -> None:
    """Failed model construction must leave the full base-call budget available."""
    settings = load_settings()
    factory = FakeFactory([reply()])
    broken = True

    def create(**kwargs: Any) -> Any:
        """Fail before invocation until construction is repaired."""
        if broken:
            raise RuntimeError("construction failed")
        return factory(**kwargs)

    budget = BudgetedLLM(settings, llm_factory=create)
    for _ in range(settings.budget.max_calls + 1):
        with pytest.raises(RuntimeError, match="construction failed"):
            budget.invoke("judge", "test")
    assert budget.n_calls == budget.callback.n_budget_calls == 0
    assert budget.n_retries == 0
    assert not budget.budget_exhausted
    broken = False
    assert budget.invoke("judge", "test").response.content == "ok"
    assert budget.n_calls == budget.callback.n_budget_calls == 1


def test_retry_factory_failure_is_not_counted_as_retry() -> None:
    """Only the completed base call remains recorded if retry creation fails."""
    factory = FakeFactory([reply(truncated=True)])

    def create(**kwargs: Any) -> Any:
        """Fail when constructing the retry model."""
        if factory.caps:
            raise RuntimeError("retry construction failed")
        return factory(**kwargs)

    budget = BudgetedLLM(load_settings(), llm_factory=create)
    with pytest.raises(RuntimeError, match="retry construction failed"):
        budget.invoke("judge", "test")
    assert budget.n_calls == budget.callback.n_budget_calls == 1
    assert budget.n_retries == budget.callback.n_retries == 0


@pytest.mark.parametrize("missing_record", [False, True])
def test_invocation_error_ends_question_until_reset(missing_record: bool) -> None:
    """An unrecorded invocation cannot silently lead to unlimited attempts."""
    attempts = []

    class BrokenModel:
        def invoke(self, model_input: Any, **kwargs: Any) -> Any:
            """Simulate an invocation failure or missing callback wiring."""
            attempts.append(model_input)
            if not missing_record:
                raise RuntimeError("connection failed")
            return SimpleNamespace(content="unrecorded")

    def create(**kwargs: Any) -> Any:
        """Return the broken model without making any network request."""
        return BrokenModel()

    budget = BudgetedLLM(load_settings(), llm_factory=create)
    with pytest.raises(RuntimeError):
        budget.invoke("judge", "first")
    with pytest.raises(RuntimeError, match="Previous model call failed"):
        budget.invoke("judge", "second")
    assert attempts == ["first"]
    assert budget.n_calls == budget.callback.n_budget_calls == 0
    budget.reset()
    with pytest.raises(RuntimeError):
        budget.invoke("judge", "new question")
    assert attempts == ["first", "new question"]


@pytest.mark.parametrize("step", ["router", "finalize", "judge"])
@pytest.mark.parametrize("signal", ["reason", "tokens"])
def test_independent_truncation_signals_for_each_step(step: str, signal: str) -> None:
    """A length reason and reaching the cap each independently trigger retry."""
    settings = load_settings()
    caps = {"router": settings.cap.first, "finalize": settings.cap.finalize,
            "judge": settings.cap.judge}
    partial = reply("partial")
    if signal == "reason":
        partial["done_reason"] = "length"
    else:
        partial["output_tokens"] = caps[step]
    factory = FakeFactory([partial, reply("complete")])
    budget = BudgetedLLM(settings, llm_factory=factory)
    assert budget.invoke(step, "test").response.content == "complete"
    assert factory.caps == [caps[step], caps[step] * settings.cap.retry_multiplier]
    assert budget.n_calls == budget.callback.n_budget_calls == 1
    assert budget.n_retries == budget.callback.n_retries == 1


@pytest.mark.parametrize("key", ["num_predict", "options", "temperature", "callbacks"])
def test_invoke_options_cannot_override_budget_settings(key: str) -> None:
    """Invocation options must not bypass the configured cap or metrics."""
    factory = FakeFactory([])
    budget = BudgetedLLM(load_settings(), llm_factory=factory)
    with pytest.raises(ValueError, match="cannot be overridden"):
        budget.invoke("judge", "test", invoke_kwargs={key: None})
    assert factory.caps == []
    assert budget.n_calls == 0


def test_real_b1_factory_preserves_input_and_schema_on_retry() -> None:
    """Use the real B-1 factory, replacing only the ChatOllama constructor."""
    settings = load_settings()
    messages = [{"role": "user", "content": "Return a short answer."}]
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    configs = []
    received = []

    class RecordingModel(FakeChatOllama):
        def invoke(self, model_input: Any, **kwargs: Any) -> SimpleNamespace:
            """Capture inputs before driving the real metrics callback."""
            received.append((model_input, kwargs))
            return super().invoke(model_input, **kwargs)

    def construct(**kwargs: Any) -> RecordingModel:
        """Capture ChatOllama constructor options without contacting Ollama."""
        configs.append(kwargs)
        return RecordingModel(kwargs["callbacks"][0], kwargs["num_predict"],
                              reply(truncated=len(configs) == 1))

    with patch("src.config.llm.ChatOllama", side_effect=construct):
        budget = BudgetedLLM(settings)
        budget.invoke("judge", messages, format_schema=schema,
                      invoke_kwargs={"stop": ["END"]})
    assert len(configs) == len(received) == 2
    for options in configs:
        assert options["model"] == settings.model.tag
        assert options["temperature"] == settings.model.temperature
        assert options["num_ctx"] == settings.model.num_ctx
        assert options["format"] is schema
    assert all(model_input is messages for model_input, _ in received)
    assert all(kwargs == {"stop": ["END"]} for _, kwargs in received)


def test_mock_answer_path_and_pipeline_log_with_duplicate_tool() -> None:
    """Exercise the future graph handoff, not a real graph or model run."""
    factory = FakeFactory([reply("42")])
    budget = BudgetedLLM(load_settings(), llm_factory=factory)
    breaker = LoopBreaker()
    breaker.run("calculator", {"expression": "6 * 7"}, lambda: "42")
    duplicate = breaker.run("calculator", {"expression": "6 * 7"},
                            lambda: pytest.fail("duplicate executed"))
    assert duplicate.answer_directly
    answer = budget.invoke("finalize", duplicate.result)
    result = PipelineResult(
        answer=answer.response.content, abstained=False,
        budget_exhausted=answer.budget_exhausted, route="math", decisions=[],
        n_calls=budget.n_calls, n_retries=budget.n_retries,
        calls=list(budget.calls), loop_breaks=breaker.loop_breaks,
    )
    row = json.loads(json.dumps(asdict(result)))
    assert row["answer"] == "42"
    assert row["loop_breaks"] == 1
    assert row["n_calls"] == 1
    assert row["n_retries"] == 0
    assert row["calls"][0]["step"] == "finalize"
    assert not row["budget_exhausted"]
