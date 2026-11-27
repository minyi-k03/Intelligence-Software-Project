"""실험 C-4 스크립트 테스트. 가짜 모델을 실제 콜백·BudgetedLLM에 연결하며 Ollama 없이 돈다."""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import scripts.compare_judge_format as c4
from scripts.compare_judge_format import render_summary, run_search_path, search_items, summarize_rows
from src.config.settings import get_settings
from src.eval.logger import RunMetaMismatch
from src.tools.bm25_search import BM25Index

DOCS = [
    {"doc_id": "hp::Neleus of Scepsis", "title": "Neleus of Scepsis", "sentences": ["Neleus of Scepsis was a disciple of Theophrastus."]},
    {"doc_id": "hp::Theophrastus", "title": "Theophrastus", "sentences": ["Theophrastus was taught by Aristotle in the Peripatetic school."]},
    {"doc_id": "hp::Scepsis", "title": "Scepsis", "sentences": ["Scepsis was an ancient town in the Troad."]},
]
INDEX = BM25Index([{**d, "text": " ".join(d["sentences"])} for d in DOCS])
ITEM = {
    "qid": "hotpotqa-9999",
    "type": "hotpotqa",
    "subtype": "bridge",
    "answer_kind": "text",
    "question": "Who taught the teacher of Neleus of Scepsis?",
    "answers": ["Aristotle"],
    "gold_doc_ids": ["hp::Neleus of Scepsis", "hp::Theophrastus"],
    "supporting_facts": [["Neleus of Scepsis", 0], ["Theophrastus", 0]],
}
UNANSWERABLE_ITEM = {
    **ITEM,
    "qid": "squad2-9999",
    "type": "squad2",
    "subtype": "unanswerable",
    "answer_kind": "unanswerable",
    "answers": ["UNANSWERABLE"],
    "gold_doc_ids": [],
    "supporting_facts": [],
}


def j(decision: str, answer: str = "", missing: str = "") -> str:
    return json.dumps({"reasoning": "r", "decision": decision, "missing": missing, "answer": answer})


class FakeChat(BaseChatModel):
    """대본대로 답하는 채팅 모델. 받은 메시지와 스키마를 기록한다."""

    cap: Any = None
    fmt: Any = None
    script: Any = None
    seen: Any = None
    delay_s: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        time.sleep(self.delay_s)
        text = self.script.pop(0)
        self.seen.append({"messages": messages, "schema": self.fmt})
        message = AIMessage(
            content=text,
            response_metadata={"done_reason": "stop"},
            usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def run(
    script: list[str],
    item: dict[str, Any] = ITEM,
    fmt: str = "json",
    *,
    order_idx: int = 1,
    delay_s: float = 0.0,
    timeout_s: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    def factory(settings: Any = None, num_predict: Any = None, format_schema: Any = None, callbacks: Any = None) -> FakeChat:
        return FakeChat(cap=num_predict, fmt=format_schema, script=script, seen=seen, delay_s=delay_s, callbacks=callbacks)

    row = run_search_path(
        item, output_format=fmt, settings=get_settings(), index=INDEX, llm_factory=factory, order_idx=order_idx, timeout_s=timeout_s
    )
    return row, seen


def test_search_items_drops_numeric_questions() -> None:
    items = [ITEM, UNANSWERABLE_ITEM, {**ITEM, "qid": "gsm8k-1", "answer_kind": "numeric"}]
    assert [it["qid"] for it in search_items(items)] == ["hotpotqa-9999", "squad2-9999"]


def test_answer_on_first_judgment() -> None:
    row, seen = run([j("ANSWER", "Aristotle")])
    assert (row["answer"], row["correct"], row["decisions"], row["n_calls"]) == ("Aristotle", True, ["ANSWER"], 1)
    assert len(row["retrieved_doc_ids"]) == 1 and row["error"] is None
    assert "Question: Who taught the teacher of Neleus of Scepsis?" in seen[0]["messages"][1].content


def test_retrieve_then_answer_accumulates_evidence() -> None:
    row, seen = run([j("RETRIEVE", missing="Theophrastus teacher"), j("ANSWER", "Aristotle")])
    assert row["decisions"] == ["RETRIEVE", "ANSWER"] and row["followups"] == ["Theophrastus teacher"]
    assert row["correct"] and row["n_calls"] == 2 and len(row["retrieved_doc_ids"]) == 2
    second_evidence = seen[1]["messages"][1].content
    assert "Neleus of Scepsis:" in second_evidence and "Theophrastus:" in second_evidence


def test_last_judgment_has_no_retrieve_option() -> None:
    limit = get_settings().budget.max_requery
    script = [j("RETRIEVE", missing=f"topic number{i} Scepsis") for i in range(limit)] + [j("ABSTAIN")]
    row, seen = run(script)
    assert row["decisions"] == ["RETRIEVE"] * limit + ["ABSTAIN"]
    assert seen[-1]["schema"]["properties"]["decision"]["enum"] == ["ANSWER", "ABSTAIN"]
    assert row["answer"] == "UNANSWERABLE" and row["abstained"]


def test_followup_repeating_the_question_is_caught_by_loop_breaker() -> None:
    row, seen = run([j("RETRIEVE", missing="who taught the teacher of neleus of scepsis"), j("ANSWER", "Aristotle")])
    assert row["loop_breaks"] == 1 and len(row["retrieved_doc_ids"]) == 1
    assert seen[1]["schema"]["properties"]["decision"]["enum"] == ["ANSWER", "ABSTAIN"]


def test_broken_output_is_wrong_not_abstained_on_unanswerable_item() -> None:
    row, _ = run(['{"reasoning": "cut'], item=UNANSWERABLE_ITEM)
    assert (row["answer"], row["abstained"], row["correct"]) == ("", False, False)
    assert row["calls"][0]["parse_ok"] is False


def test_tag_format_sends_no_schema() -> None:
    row, seen = run(["Found it.\nDECISION: ANSWER\nANSWER: Aristotle"], fmt="tag")
    assert row["answer"] == "Aristotle" and row["format"] == "tag"
    assert seen[0]["schema"] is None


def test_model_error_is_recorded_as_wrong_answer() -> None:
    row, _ = run([])  # 대본이 비면 가짜 모델이 예외를 던진다
    assert row["answer"] == "" and not row["correct"] and row["error"].startswith("IndexError")


def test_summary_counts_and_paired_comparison() -> None:
    json_rows = [run([j("ANSWER", "Aristotle")])[0], run([j("ABSTAIN")], item=UNANSWERABLE_ITEM)[0]]
    tag_rows = [run(["x\nDECISION: ANSWER\nANSWER: Plato"], fmt="tag")[0], run(["broken"], item=UNANSWERABLE_ITEM, fmt="tag")[0]]
    s = summarize_rows(json_rows)
    assert (s["n"], s["accuracy"], s["parse_fail_rate"], s["over_abstention"]) == (2, 1.0, 0.0, 0.0)
    assert summarize_rows(tag_rows)["parse_fail_rate"] == 0.5
    table = render_summary({"json": json_rows, "tag": tag_rows})
    assert "| 전체 | json | 2 | 100% |" in table
    assert "json만 정답 2" in table


def test_row_uses_runner_log_format() -> None:
    row, _ = run([j("ANSWER", "Aristotle")], order_idx=0)
    for key in ("timeout", "is_warmup", "order_idx", "latency_s", "evidence_kept", "max_input_tokens", "format", "followups"):
        assert key in row
    assert row["is_warmup"] and not row["timeout"] and row["pipeline"] == "json"


def test_timeout_is_recorded_as_wrong_answer() -> None:
    row, _ = run([j("ANSWER", "Aristotle")], delay_s=1.0, timeout_s=0.2)
    assert row["timeout"] and row["answer"] == "" and not row["correct"] and row["error"] == "timeout"
    assert row["latency_s"] >= 0.2


def test_cost_averages_exclude_warmup_but_accuracy_does_not() -> None:
    warm, _ = run([j("RETRIEVE", missing="Theophrastus teacher"), j("ANSWER", "Aristotle")], order_idx=0)
    rest, _ = run([j("ANSWER", "Plato")], item={**ITEM, "qid": "hotpotqa-9998"}, order_idx=1)
    s = summarize_rows([warm, rest], {warm["qid"]})
    assert s["judge_calls"] == 1 and s["output_tokens"] == 20
    assert s["accuracy"] == 0.5 and s["n"] == 2
    table = render_summary({"json": [warm, rest]})
    assert "| 전체 | json | 2 | 50% |" in table and "| 1 |" in table


def test_old_logs_without_warmup_flag_treat_first_line_as_warmup() -> None:
    a, _ = run([j("RETRIEVE", missing="Theophrastus teacher"), j("ANSWER", "Aristotle")])
    b, _ = run([j("ANSWER", "Plato")], item={**ITEM, "qid": "hotpotqa-9998"})
    for row in (a, b):
        del row["is_warmup"]
    assert "| 전체 | json | 2 | 50% | 0.50 | 0% | 1 |" in render_summary({"json": [a, b]})


def _run_main(monkeypatch: pytest.MonkeyPatch, out_dir: Path, settings: Any, *extra: str) -> None:
    """main()을 가짜 실행으로 돌린다 (Ollama·코퍼스 없이)."""
    def fake_run(item: dict[str, Any], *, output_format: str, settings: Any, index: Any, order_idx: int) -> dict[str, Any]:
        return run([j("ANSWER", "Aristotle")], item={**ITEM, "qid": item["qid"]}, fmt=output_format, order_idx=order_idx)[0]

    monkeypatch.setattr(c4, "run_search_path", fake_run)
    monkeypatch.setattr(c4, "get_default_index", lambda: INDEX)
    monkeypatch.setattr(c4, "get_settings", lambda: settings)
    monkeypatch.setattr(c4, "collect_environment", lambda: {"git_commit": None, "git_dirty": None, "gpu": None, "ollama_version": None})
    monkeypatch.setattr(sys, "argv", ["compare_judge_format", "--limit", "2", "--out-dir", str(out_dir), *extra])
    c4.main()


def test_resume_with_changed_settings_stops(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = get_settings()
    _run_main(monkeypatch, tmp_path, settings)
    assert (tmp_path / "RUN_META.json").exists() and len(c4.read_log(tmp_path / "json.jsonl")) == 2
    _run_main(monkeypatch, tmp_path, settings)  # 같은 조건이면 이어 돌기(새로 돌 문항 없음)
    assert len(c4.read_log(tmp_path / "json.jsonl")) == 2

    changed = dataclasses.replace(settings, compress=dataclasses.replace(settings.compress, m=settings.compress.m + 1))
    with pytest.raises(RunMetaMismatch):
        _run_main(monkeypatch, tmp_path, changed)


def test_logs_without_run_meta_refuse_to_resume(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "json.jsonl").write_text(json.dumps({"qid": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, tmp_path, get_settings())


def test_parse_compare() -> None:
    assert c4.parse_compare(["base=a/b", "new=c"], "json") == [("base", Path("a/b/json.jsonl")), ("new", Path("c/json.jsonl"))]
    for bad in (["base"], ["=a"], ["base="], ["x=a", "x=b"]):
        with pytest.raises(SystemExit):
            c4.parse_compare(bad, "json")


def test_compare_mode_pairs_two_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base, new = tmp_path / "base", tmp_path / "new"
    for folder, answers in ((base, ["Plato", "Plato"]), (new, ["Aristotle", "Plato"])):
        folder.mkdir()
        rows = [
            run([j("ANSWER", a)], item={**ITEM, "qid": f"hotpotqa-99{i}"}, order_idx=i)[0]
            for i, a in enumerate(answers)
        ]
        (folder / "json.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["x", "--formats", "json", "--out-dir", str(tmp_path / "out"), "--compare", f"base={base}", f"new={new}"])
    c4.main()
    table = (tmp_path / "out" / "compare_json.md").read_text(encoding="utf-8")
    assert "| 전체 | base | 2 | 0% |" in table and "| 전체 | new | 2 | 50% |" in table
    assert "new만 정답 1" in table
