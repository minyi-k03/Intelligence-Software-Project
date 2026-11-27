"""간결 출력 훅의 메시지 형식, 필수 추론 지시, 기본 M 연결을 확인한다."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from src.meba.conditional.concise import build_m_concise, concise_messages


class ConciseTests(unittest.TestCase):
    """모델 없이 프롬프트 변환과 조건부 스위치를 검증한다."""

    def test_router_preserves_question_and_final_answer_instruction(self) -> None:
        original = [
            {"role": "system", "content": "Solve step by step. Final answer: <answer>"},
            {"role": "user", "content": "What is 2 + 3?"},
        ]
        changed = concise_messages("router", original)
        self.assertEqual(changed[1], original[1])
        self.assertIn("each calculation and intermediate value", changed[0]["content"])
        self.assertIn("Final answer: <answer>", changed[0]["content"])
        self.assertEqual(original[0]["content"], "Solve step by step. Final answer: <answer>")

    def test_judge_keeps_json_instruction_and_evidence(self) -> None:
        original = [
            ("system", 'Use JSON keys: "reasoning", "decision", "missing", "answer".'),
            ("human", "Question: Who?\nEvidence: A founded B."),
        ]
        changed = concise_messages("judge", original)
        self.assertEqual(changed[1], original[1])
        self.assertIn("decisive evidence", changed[0][1])
        self.assertIn(original[0][1], changed[0][1])
        self.assertEqual(original[0][1], 'Use JSON keys: "reasoning", "decision", "missing", "answer".')

    def test_finalize_and_other_steps_are_unchanged(self) -> None:
        messages = [{"role": "system", "content": "Final answer only"}]
        self.assertIs(concise_messages("finalize", messages), messages)
        self.assertIs(concise_messages("other", messages), messages)

    def test_wrong_message_shape_fails_before_model_call(self) -> None:
        with self.assertRaises(TypeError):
            concise_messages("judge", [{"role": "system", "content": "x"}])
        with self.assertRaises(ValueError):
            concise_messages("router", [{"role": "user", "content": "x"}])
        with self.assertRaises(ValueError):
            concise_messages("router", [])
        with self.assertRaises(TypeError):
            concise_messages("judge", None)

    def test_unknown_question_type_gets_conditional_math_guidance_only(self) -> None:
        original = [
            {"role": "system", "content": "ROUTE: math or ROUTE: search. Final answer: <answer>"},
            {"role": "user", "content": "Where was the author born?"},
        ]
        changed = concise_messages("router", original)
        self.assertIn("For the math route", changed[0]["content"])
        self.assertEqual(changed[1], original[1])
        self.assertNotIn("HotpotQA", changed[0]["content"])
        self.assertNotIn("GSM8K", changed[0]["content"])

    def test_factory_attaches_only_concise_switch(self) -> None:
        fake_graph = types.ModuleType("src.meba.graph")

        class Parts:
            def __init__(self, concise: bool = False) -> None:
                self.concise = concise

        class Hooks:
            def __init__(self, concise: object = None) -> None:
                self.concise = concise

        seen: dict[str, object] = {}

        def build_m(settings: object, *, components: Parts, hooks: Hooks, llm_factory: object) -> object:
            seen.update(settings=settings, components=components, hooks=hooks, llm_factory=llm_factory)
            return lambda qid, question: (qid, question)

        fake_graph.MComponents = Parts
        fake_graph.ConditionalHooks = Hooks
        fake_graph.build_m = build_m
        with patch.dict(sys.modules, {"src.meba.graph": fake_graph}):
            pipeline = build_m_concise(None)
        self.assertTrue(seen["components"].concise)
        self.assertIs(seen["hooks"].concise, concise_messages)
        self.assertTrue(callable(seen["llm_factory"]))
        self.assertEqual(pipeline("id", "question"), ("id", "question"))


if __name__ == "__main__":
    unittest.main()
