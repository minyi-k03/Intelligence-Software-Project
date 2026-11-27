"""D-3 재측정 로그의 도구 요청 추적과 지표 분모를 검증한다."""

from __future__ import annotations

import types
import unittest

from scripts.audit_p1_dev import P1AuditCallback, has_final_answer_line, summarize


class TestP1Audit(unittest.TestCase):
    """모델 없이 콜백이 보존하는 이력과 집계 규칙을 확인한다."""

    def test_callback_records_model_requested_tools_in_order(self) -> None:
        callback = P1AuditCallback()
        message = types.SimpleNamespace(
            tool_calls=[
                {"name": "bm25_search", "args": {"query": "New York"}},
                {"name": "calculator", "args": {"expression": "2 + 2"}},
            ],
            invalid_tool_calls=[],
        )
        response = types.SimpleNamespace(generations=[[types.SimpleNamespace(message=message)]])
        callback.on_llm_end(response)
        self.assertEqual(callback.model_calls, 1)
        self.assertEqual([request["name"] for request in callback.tool_requests], ["bm25_search", "calculator"])
        self.assertFalse(callback.invalid_requests)

    def test_format_and_loop_denominators_exclude_incomplete_traces(self) -> None:
        rows = [
            {
                "qid": "a", "timeout": False, "error": None,
                "raw_final_output": "Reasoning\nFinal answer: 4",
                "trace_complete": True,
                "tool_requests": [
                    {"name": "bm25_search", "args": {"query": " New  YORK "}},
                    {"name": "bm25_search", "args": '{"query":"new york"}'},
                    {"name": "calculator", "args": {"expression": "2+2"}},
                ],
            },
            {
                "qid": "b", "timeout": False, "error": None,
                "raw_final_output": "4", "trace_complete": True, "tool_requests": [],
            },
            {
                "qid": "c", "timeout": True, "error": "timeout",
                "raw_final_output": None, "trace_complete": False,
                "tool_requests": [{"name": "bm25_search", "args": {"query": "x"}}],
            },
        ]
        result = summarize(rows)
        self.assertEqual(result["format_errors"], 1)
        self.assertEqual(result["format_checked"], 2)
        self.assertEqual(result["format_error_rate"], 0.5)
        self.assertEqual(result["duplicate_requests"], 1)
        self.assertEqual(result["tool_requests_checked"], 3)
        self.assertAlmostEqual(result["repeat_call_rate"], 1 / 3)
        self.assertEqual(result["loop_questions"], 1)
        self.assertEqual(result["history_checked"], 2)
        self.assertEqual(result["repeat_loop_rate"], 0.5)

    def test_final_answer_must_be_last_nonempty_line(self) -> None:
        self.assertTrue(has_final_answer_line("Work\nFinal answer: Paris\n\n"))
        self.assertFalse(has_final_answer_line("Final answer: Paris\nMore text"))
        self.assertFalse(has_final_answer_line("Final answer:   "))
        self.assertFalse(has_final_answer_line("**Final answer:** Paris"))


if __name__ == "__main__":
    unittest.main()
