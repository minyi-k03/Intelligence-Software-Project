"""D-5 쌍비교가 문항별 순정답과 출력 감소율을 정확히 계산하는지 검증한다."""

from __future__ import annotations

import unittest

from scripts.run_concise_experiment import compare


def _row(qid: str, kind: str, *, correct: bool, output: int, warmup: bool, route: str) -> dict:
    """쌍비교에 필요한 최소 문항 로그를 만든다."""
    step = "router" if route == "math" else "judge"
    return {
        "qid": qid, "type": kind, "correct": correct, "is_warmup": warmup,
        "route": route, "calls": [{"step": step, "output_tokens": output,
                                 "parse_ok": True if step == "judge" else None,
                                 "truncated": False}],
        "latency_s": 1.0, "timeout": False, "error": None,
    }


def _trace(qid: str, step: str, raw: str) -> dict:
    """완성된 모델 원문 보조 로그를 만든다."""
    return {"qid": qid, "trace_complete": True,
            "calls": [{"step": step, "done_reason": "stop", "raw_output": raw}]}


class ConciseExperimentTests(unittest.TestCase):
    """정답 변화·출력·형식을 동일 문항으로 대응시키는지 확인한다."""

    def test_paired_reduction_and_math_review(self) -> None:
        baseline = [
            _row("h", "hotpotqa", correct=True, output=10, warmup=True, route="search"),
            _row("g", "gsm8k", correct=False, output=100, warmup=False, route="math"),
            _row("s", "squad2", correct=True, output=100, warmup=False, route="search"),
        ]
        concise = [
            _row("h", "hotpotqa", correct=True, output=10, warmup=True, route="search"),
            _row("g", "gsm8k", correct=True, output=80, warmup=False, route="math"),
            _row("s", "squad2", correct=False, output=80, warmup=False, route="search"),
        ]
        baseline_traces = [
            _trace("h", "judge", "{}"),
            _trace("g", "router", "2 + 3 = 5\nFinal answer: 5"),
            _trace("s", "judge", "{}"),
        ]
        concise_traces = [
            _trace("h", "judge", "{}"),
            _trace("g", "router", "2 + 3 = 5\nFinal answer: 5"),
            _trace("s", "judge", "{}"),
        ]
        result = compare(baseline, concise, baseline_traces, concise_traces)
        self.assertEqual(result["output_reduction_pct"], 20)
        self.assertEqual(result["by_type"]["gsm8k"]["net_correct"], 1)
        self.assertEqual(result["by_type"]["squad2"]["net_correct"], -1)
        self.assertEqual(result["math_raw_review_qids"], ["g"])
        self.assertEqual(result["concise_format"], {"checked": 3, "errors": 0, "unknown": 0})
        self.assertEqual(result["math_step_preservation"], "manual_review_required")

    def test_mismatched_question_set_is_rejected(self) -> None:
        left = [_row("a", "gsm8k", correct=True, output=10, warmup=False, route="math")]
        right = [_row("b", "gsm8k", correct=True, output=10, warmup=False, route="math")]
        with self.assertRaises(ValueError):
            compare(left, right, [], [])


if __name__ == "__main__":
    unittest.main()
