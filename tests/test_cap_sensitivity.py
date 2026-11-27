"""D-4 실험 설정과 종료 사유 기반 잘림 집계를 모델 없이 확인한다."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.run_cap_sensitivity import (
    _done_reason, compare_conditions, load_conditions, settings_for_cap, summarize,
)
from src.config.settings import get_settings
from src.meba.budget import BudgetedLLM


def _call(*, step: str = "judge", output: int = 10, parse_ok: object = True, truncated: bool = False) -> dict:
    """집계에 필요한 호출 로그를 만든다."""
    return {"step": step, "cap": 96, "output_tokens": output, "parse_ok": parse_ok,
            "truncated": truncated, "is_retry": False}


def _row(qid: str, *, correct: bool, warmup: bool, calls: list[dict]) -> dict:
    """집계에 필요한 문항 로그를 만든다."""
    return {
        "qid": qid, "type": "hotpotqa", "correct": correct, "is_warmup": warmup,
        "calls": calls, "latency_s": 2.0, "timeout": False, "error": None,
    }


class CapSensitivityTests(unittest.TestCase):
    """상한 값만으로 잘림을 확정하지 않는지 검증한다."""

    def test_stop_at_cap_is_not_confirmed_truncation(self) -> None:
        rows = [
            _row("warmup", correct=True, warmup=True, calls=[_call(output=96)]),
            _row("a", correct=True, warmup=False, calls=[_call(output=96, truncated=True)]),
            _row("b", correct=False, warmup=False, calls=[_call(output=12, parse_ok=False)]),
        ]
        traces = [
            {"qid": "warmup", "trace_complete": True, "calls": [{"step": "judge", "done_reason": "stop"}]},
            {"qid": "a", "trace_complete": True, "calls": [{"step": "judge", "done_reason": "stop"}]},
            {"qid": "b", "trace_complete": True, "calls": [{"step": "judge", "done_reason": "length"}]},
        ]
        result = summarize(rows, traces)
        self.assertEqual((result["correct"], result["items"]), (2, 3))
        self.assertEqual((result["output_tokens"], result["mean_output_tokens"]), (108, 54))
        self.assertEqual((result["confirmed_truncations"], result["judge_reasons_checked"]), (1, 3))
        self.assertEqual(result["cap_only_truncated_flags"], 1)
        self.assertEqual((result["format_errors"], result["format_checked"]), (1, 3))
        self.assertEqual(result["format_error_rate"], 1 / 3)
        self.assertEqual(result["confirmed_truncation_rate"], 1 / 3)
        self.assertEqual(result["by_type"]["hotpotqa"]["accuracy"], 2 / 3)
        self.assertEqual((result["total_latency_s"], result["mean_latency_s"]), (4, 2))
        self.assertEqual((result["judge_output_tokens"], result["judge_output_calls"]), (108, 2))
        self.assertEqual(result["judge_mean_output_tokens"], 54)
        self.assertEqual((result["judge_cap_reached"], result["judge_cap_checked"]), (2, 3))
        self.assertEqual(result["judge_cap_reached_rate"], 2 / 3)
        self.assertEqual(result["confirmed_truncation_cases"], [
            {"qid": "b", "cap": 96, "output_tokens": 12, "is_retry": False}
        ])

    def test_missing_or_incomplete_trace_is_unknown(self) -> None:
        rows = [_row("a", correct=False, warmup=False, calls=[_call()])]
        traces = [{"qid": "a", "trace_complete": False, "calls": [{"step": "judge", "done_reason": "length"}]}]
        result = summarize(rows, traces)
        self.assertEqual(result["confirmed_truncations"], 0)
        self.assertEqual(result["judge_reasons_checked"], 0)
        self.assertEqual(result["judge_reasons_unknown"], 1)

    def test_done_reason_requires_explicit_metadata(self) -> None:
        self.assertEqual(_done_reason(SimpleNamespace(response_metadata={"done_reason": "stop"})), "stop")
        self.assertIsNone(_done_reason(SimpleNamespace(response_metadata={})))

    def test_duplicate_conditions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            path.write_text(json.dumps({
                "base_config": "configs/default.yaml",
                "conditions": [{"judge_cap": 96, "round": "r20"}, {"judge_cap": 96, "round": "r21"}],
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_conditions(path)

    def test_only_judge_cap_changes(self) -> None:
        base = get_settings("configs/default.yaml")
        adjusted = settings_for_cap(base, 96)
        before, after = base.to_dict(), adjusted.to_dict()
        self.assertEqual(before["cap"]["judge"], 256)
        self.assertEqual(after["cap"]["judge"], 96)
        after["cap"]["judge"] = before["cap"]["judge"]
        self.assertEqual(before, after)
        budget = BudgetedLLM(adjusted)
        self.assertEqual(budget.cap_for_step("router"), base.cap.first)
        self.assertEqual(budget.cap_for_step("finalize"), base.cap.finalize)
        self.assertEqual(budget.cap_for_step("judge"), 96)

    def test_candidate_requires_preserved_accuracy_and_known_stop_reason(self) -> None:
        baseline = [_row("a", correct=True, warmup=True, calls=[_call()]),
                    _row("b", correct=False, warmup=False, calls=[_call()])]
        lower = [_row("a", correct=True, warmup=True, calls=[_call()]),
                 _row("b", correct=True, warmup=False, calls=[_call()])]
        def stats(correct: int, tokens: int, truncated: int = 0, unknown: int = 0) -> dict:
            """후보 선정에 필요한 최소 집계를 만든다."""
            return {"correct": correct, "by_type": {"hotpotqa": {"correct": correct}},
                    "mean_output_tokens": tokens, "judge_reasons_checked": 2,
                    "judge_reasons_unknown": unknown, "confirmed_truncations": truncated}
        result = compare_conditions({
            256: (baseline, stats(1, 20)),
            160: (lower, stats(2, 15, unknown=1)),
            96: (lower, stats(2, 10)),
        })
        self.assertEqual(result["recommended_cap"], 96)
        self.assertFalse(result["conditions"][160]["eligible"])
        self.assertEqual(result["conditions"][96]["improved_qids"], ["b"])


if __name__ == "__main__":
    unittest.main()
