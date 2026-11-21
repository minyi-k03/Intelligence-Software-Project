"""Unit tests for baseline QA pipelines (P0, P0-raw, P1 ReAct) and prompts.

References:
- docs/INTERFACES.md §4, §6
- docs/IMPLEMENTATION_PLAN.md §5 (PR B-3)
"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from src.baselines import (
    CONTENT_INSTRUCTION,
    FORMAT_LINE,
    SHARED_ANSWER_INSTRUCTION,
    build_p0_prompt,
    build_p0_raw_prompt,
    build_p1_system_prompt,
    run_p0,
    run_p0_raw,
    run_p1,
)
from src.common.types import PipelineResult
from src.config.settings import (
    BudgetConfig,
    CapConfig,
    CompressConfig,
    DataConfig,
    ModelConfig,
    P1Config,
    RetrievalConfig,
    RunConfig,
    Settings,
)


def _make_settings(*, retrieval_k: int = 2, max_steps: int = 7) -> Settings:
    """Create validated Settings in-memory (no PyYAML required in offline tests)."""
    return Settings(
        model=ModelConfig(
            tag="qwen2.5:7b-instruct-q4_K_M",
            temperature=0,
            num_ctx=8192,
            keep_alive=-1,
        ),
        budget=BudgetConfig(max_calls=4, max_requery=2),
        cap=CapConfig(first=512, finalize=128, judge=256, retry_multiplier=2),
        retrieval=RetrievalConfig(k=retrieval_k),
        compress=CompressConfig(m=3, window=1),
        p1=P1Config(max_steps=max_steps),
        run=RunConfig(timeout_s=120),
        data=DataConfig(seed=20261113),
    )


def _fake_llm_result(text: str, *, in_tokens: int = 25, out_tokens: int = 8) -> Any:
    """Create a minimal LangChain-like LLMResult for MetricsCallbackHandler."""
    msg = types.SimpleNamespace(
        type="ai",
        content=text,
        usage_metadata={"input_tokens": in_tokens, "output_tokens": out_tokens},
        response_metadata={"done_reason": "stop"},
    )
    gen = types.SimpleNamespace(message=msg, generation_info=None, text=text)
    return types.SimpleNamespace(generations=[[gen]], llm_output=None)


class _FakeLLM:
    """Mock ChatOllama that triggers MetricsCallbackHandler lifecycle hooks."""

    def __init__(self, response_text: str, callbacks: list[Any] | None = None) -> None:
        self.response_text = response_text
        self.callbacks = callbacks or []
        self.prompts: list[Any] = []

    def invoke(self, prompt: Any, config: dict[str, Any] | None = None) -> Any:
        self.prompts.append(prompt)
        active_callbacks = (config or {}).get("callbacks") or self.callbacks
        for cb in active_callbacks:
            cb.on_chat_model_start({}, [[prompt]], run_id="fake-run-1")
            cb.on_llm_end(_fake_llm_result(self.response_text), run_id="fake-run-1")
        return types.SimpleNamespace(type="ai", content=self.response_text)


class TestPrompts(unittest.TestCase):
    """Tests for shared English answer-format instructions and prompt builders."""

    def test_prompts_are_english_and_hide_question_types(self) -> None:
        p0_prompt = build_p0_prompt("What is 2 + 2?")
        p1_prompt = build_p1_system_prompt()
        raw_prompt = build_p0_raw_prompt("What is 2 + 2?")

        self.assertIn(CONTENT_INSTRUCTION, p0_prompt)
        self.assertIn(FORMAT_LINE, p0_prompt)
        self.assertIn(SHARED_ANSWER_INSTRUCTION, p1_prompt)
        self.assertEqual(raw_prompt, "What is 2 + 2?")

        for text in (p0_prompt, p1_prompt):
            lowered = text.lower()
            for forbidden in ("hotpotqa", "squad", "gsm8k", "multi-hop"):
                self.assertNotIn(forbidden, lowered)

    def test_empty_or_non_string_question_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_p0_prompt("   ")
        with self.assertRaises(TypeError):
            build_p0_raw_prompt(123)  # type: ignore[arg-type]


class TestP0Baselines(unittest.TestCase):
    """Tests for P0 and P0-raw single-call baseline execution."""

    def test_public_signatures_accept_only_qid_and_question_positionally(self) -> None:
        for fn in (run_p0, run_p0_raw, run_p1):
            params = [
                p
                for p in inspect.signature(fn).parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
            ]
            self.assertEqual([p.name for p in params], ["qid", "question"])

    def test_run_p0_extracts_final_answer_and_records_metrics(self) -> None:
        settings = _make_settings()
        captured: dict[str, Any] = {}

        def fake_factory(**kwargs: Any) -> _FakeLLM:
            captured["kwargs"] = kwargs
            llm = _FakeLLM(
                "Thinking briefly.\n**Final answer:**\n42.",
                callbacks=kwargs.get("callbacks"),
            )
            captured["llm"] = llm
            return llm

        result = run_p0("gsm8k-0001", "What is 6 times 7?", settings=settings, llm_factory=fake_factory)
        self.assertIsInstance(result, PipelineResult)
        self.assertEqual(result.answer, "42")
        self.assertFalse(result.abstained)
        self.assertFalse(result.budget_exhausted)
        self.assertEqual(result.n_calls, 1)
        self.assertEqual(result.n_retries, 0)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].step, "p0")
        self.assertIsNone(result.calls[0].cap)
        self.assertEqual(result.calls[0].input_tokens, 25)
        self.assertEqual(result.calls[0].output_tokens, 8)
        self.assertIsNone(result.error)
        self.assertNotIn("gsm8k-0001", captured["llm"].prompts[0])
        self.assertIn("Final answer:", captured["llm"].prompts[0])
        self.assertIsNone(captured["kwargs"]["num_predict"])

    def test_run_p0_normalizes_unanswerable_and_sets_abstained(self) -> None:
        settings = _make_settings()

        def fake_factory(**kwargs: Any) -> _FakeLLM:
            return _FakeLLM("Final answer: unanswerable.", callbacks=kwargs.get("callbacks"))

        result = run_p0("sq-0001", "Unknown question?", settings=settings, llm_factory=fake_factory)
        self.assertTrue(result.abstained)
        self.assertEqual(result.answer, "UNANSWERABLE")

    def test_run_p0_raw_sends_only_question_and_handles_errors(self) -> None:
        settings = _make_settings()
        captured: dict[str, Any] = {}

        def fake_factory(**kwargs: Any) -> _FakeLLM:
            llm = _FakeLLM("Paris", callbacks=kwargs.get("callbacks"))
            captured["llm"] = llm
            return llm

        res = run_p0_raw("hp-0001", "Capital of France?", settings=settings, llm_factory=fake_factory)
        self.assertEqual(captured["llm"].prompts[0], "Capital of France?")
        self.assertEqual(res.answer, "Paris")
        self.assertFalse(res.abstained)

        def failing_factory(**_kwargs: Any) -> Any:
            raise ConnectionError("Ollama unavailable")

        err_res = run_p0_raw("hp-0002", "Capital?", settings=settings, llm_factory=failing_factory)
        self.assertEqual(err_res.answer, "")
        self.assertIn("ConnectionError", err_res.error or "")


class TestP1ReAct(unittest.TestCase):
    """Tests for P1 ReAct pipeline with mocked LangChain tool/agent plumbing."""

    SAMPLE_CORPUS = [
        {
            "doc_id": "hp::Nikola Tesla",
            "title": "Nikola Tesla",
            "sentences": ["Nikola Tesla was born in 1856."],
            "text": "Nikola Tesla was born in 1856.",
            "origin": "hotpotqa",
        },
        {
            "doc_id": "sq::Alternating_current::0",
            "title": "Alternating current",
            "sentences": ["Alternating current reverses direction periodically."],
            "text": "Alternating current reverses direction periodically.",
            "origin": "squad2",
        },
    ]

    def _fake_langchain_core(self) -> tuple[types.ModuleType, types.ModuleType]:
        fake_lc = types.ModuleType("langchain_core")
        fake_tools = types.ModuleType("langchain_core.tools")

        def fake_tool(name: str):
            def decorator(fn):
                fn.tool_name = name
                return fn

            return decorator

        fake_tools.tool = fake_tool  # type: ignore[attr-defined]
        fake_lc.tools = fake_tools  # type: ignore[attr-defined]
        return fake_lc, fake_tools

    def test_run_p1_hides_doc_id_from_model_logs_retrieved_ids_and_enforces_steps(self) -> None:
        settings = _make_settings(retrieval_k=1, max_steps=7)
        captured: dict[str, Any] = {}

        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_path = Path(tmp_dir) / "corpus.jsonl"
            corpus_path.write_text(
                "\n".join(json.dumps(row) for row in self.SAMPLE_CORPUS) + "\n",
                encoding="utf-8",
            )

            def fake_llm_factory(**kwargs: Any) -> _FakeLLM:
                captured["llm_kwargs"] = kwargs
                return _FakeLLM("unused")

            def fake_agent_factory(*, model: Any, tools: list[Any], system_prompt: str) -> Any:
                captured["tools"] = tools
                captured["system_prompt"] = system_prompt

                class _FakeAgent:
                    def invoke(self, inputs: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
                        captured["inputs"] = inputs
                        captured["invoke_config"] = config or {}
                        bm25_tool, calc_tool = tools
                        captured["bm25_params"] = list(inspect.signature(bm25_tool).parameters)
                        captured["obs"] = bm25_tool(query="Nikola Tesla born")
                        captured["calc_obs"] = calc_tool(expression="1856 + 10")

                        for cb in (config or {}).get("callbacks", []):
                            cb.on_chat_model_start({}, [inputs["messages"]], run_id="p1-step-1")
                            cb.on_llm_end(_fake_llm_result("Tool call", in_tokens=30, out_tokens=10), run_id="p1-step-1")
                            cb.on_chat_model_start({}, [inputs["messages"]], run_id="p1-step-2")
                            cb.on_llm_end(_fake_llm_result("Final answer: 1856", in_tokens=45, out_tokens=6), run_id="p1-step-2")

                        return {
                            "messages": [
                                {"role": "user", "content": inputs["messages"][0]["content"]},
                                {"role": "ai", "content": "calling tools"},
                                {"role": "tool", "content": captured["obs"]},
                                {"role": "ai", "content": "Year verified.\nFinal answer: 1856"},
                            ]
                        }

                return _FakeAgent()

            fake_lc, fake_tools = self._fake_langchain_core()
            with mock.patch.dict(
                sys.modules,
                {"langchain_core": fake_lc, "langchain_core.tools": fake_tools},
            ):
                result = run_p1(
                    "hp-0001",
                    "When was Nikola Tesla born?",
                    settings=settings,
                    llm_factory=fake_llm_factory,
                    agent_factory=fake_agent_factory,
                    corpus_path=corpus_path,
                )

        self.assertIsNone(captured["llm_kwargs"]["callbacks"])
        self.assertIsNone(captured["llm_kwargs"]["num_predict"])
        self.assertEqual(captured["bm25_params"], ["query"])
        self.assertIn("[1] Nikola Tesla\nNikola Tesla was born in 1856.", captured["obs"])
        self.assertNotIn("hp::", captured["obs"])
        self.assertNotIn("doc_id", captured["obs"])
        self.assertEqual(captured["calc_obs"], "1866")
        self.assertEqual(captured["invoke_config"]["recursion_limit"], 15)
        self.assertEqual(captured["inputs"], {"messages": [{"role": "user", "content": "When was Nikola Tesla born?"}]})
        self.assertIn(SHARED_ANSWER_INSTRUCTION, captured["system_prompt"])

        self.assertEqual(result.answer, "1856")
        self.assertFalse(result.abstained)
        self.assertEqual(result.n_calls, 2)
        self.assertEqual([c.step for c in result.calls], ["p1_step", "p1_step"])
        self.assertEqual(result.retrieved_doc_ids, [["hp::Nikola Tesla"]])
        self.assertIsNone(result.error)

    def test_run_p1_handles_recursion_limit_error_cleanly(self) -> None:
        settings = _make_settings(retrieval_k=1, max_steps=7)

        class GraphRecursionError(RuntimeError):
            pass

        def fake_agent_factory(*, model: Any, tools: list[Any], system_prompt: str) -> Any:
            class _LoopingAgent:
                def invoke(self, inputs: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
                    raise GraphRecursionError("Recursion limit of 15 reached")

            return _LoopingAgent()

        fake_lc, fake_tools = self._fake_langchain_core()
        with mock.patch.dict(
            sys.modules,
            {"langchain_core": fake_lc, "langchain_core.tools": fake_tools},
        ):
            result = run_p1(
                "hp-0002",
                "Looping question?",
                settings=settings,
                llm_factory=lambda **_: _FakeLLM("unused"),
                agent_factory=fake_agent_factory,
            )

        self.assertEqual(result.answer, "")
        self.assertFalse(result.abstained)
        self.assertIn("RecursionLimitExceeded", result.error or "")
        self.assertIn("max_steps=7", result.error or "")


if __name__ == "__main__":
    unittest.main()
