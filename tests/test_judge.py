"""근거 판정(judge.py) 단위 테스트 (PR C-3). 가짜 모델을 쓰며 Ollama·GPU 없이 돈다."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.baselines.prompts import CONTENT_INSTRUCTION
from src.config.settings import get_settings
from src.eval.scorer import extract_final_answer, is_abstained
from src.meba.budget import BudgetedLLM
from src.meba.judge import (
    _response_text,
    build_judge_messages,
    judge,
    judge_schema,
    parse_judgment,
)


def reply(decision: str, answer: str = "", missing: str = "", reasoning: str = "Because.") -> str:
    """키 순서를 지킨 JSON 출력."""
    return json.dumps({"reasoning": reasoning, "decision": decision, "missing": missing, "answer": answer})


class FakeChat(BaseChatModel):
    """대본대로 답하는 채팅 모델. 대본 항목은 (출력 텍스트, 잘림 여부)."""

    cap: Any = None
    script: Any = None
    seen: Any = None

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        text, truncated = self.script.pop(0)
        self.seen.append(messages)
        n_out = self.cap if truncated else 20
        message = AIMessage(
            content=text,
            response_metadata={"done_reason": "length" if truncated else "stop"},
            usage_metadata={"input_tokens": 100, "output_tokens": n_out, "total_tokens": 100 + n_out},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def make_llm(script: list[tuple[str, bool]]) -> tuple[BudgetedLLM, list[dict[str, Any]], list[Any]]:
    """가짜 모델을 실제 콜백에 연결한 BudgetedLLM, 모델 생성 기록, 받은 메시지 기록."""
    created: list[dict[str, Any]] = []
    seen: list[Any] = []

    def factory(settings: Any = None, num_predict: Any = None, format_schema: Any = None, callbacks: Any = None) -> FakeChat:
        created.append({"num_predict": num_predict, "format_schema": format_schema})
        return FakeChat(cap=num_predict, script=script, seen=seen, callbacks=callbacks)

    return BudgetedLLM(settings=get_settings(), llm_factory=factory), created, seen


# --- 스키마 ---------------------------------------------------------------


def test_schema_key_order_puts_reasoning_first_and_answer_last() -> None:
    schema = judge_schema(allow_retrieve=True)
    assert list(schema["properties"]) == ["reasoning", "decision", "missing", "answer"]
    assert schema["required"] == ["reasoning", "decision", "missing", "answer"]
    assert schema["properties"]["decision"]["enum"] == ["ANSWER", "RETRIEVE", "ABSTAIN"]


def test_last_judgment_schema_has_no_retrieve() -> None:
    assert judge_schema(allow_retrieve=False)["properties"]["decision"]["enum"] == ["ANSWER", "ABSTAIN"]


def test_ablation_schema_without_abstain() -> None:
    schema = judge_schema(allow_retrieve=True, allow_abstain=False)
    assert schema["properties"]["decision"]["enum"] == ["ANSWER", "RETRIEVE"]


# --- 프롬프트 -------------------------------------------------------------


def test_prompt_uses_shared_content_instruction_without_final_answer_line() -> None:
    for fmt in ("json", "tag"):
        (_, system), (_, human) = build_judge_messages("Who wrote it?", "T: Evidence.", allow_retrieve=True, output_format=fmt)
        assert CONTENT_INSTRUCTION in system
        assert "Final answer:" not in system
        assert human == "Question: Who wrote it?\n\nEvidence:\nT: Evidence."


def test_prompt_is_english_and_does_not_name_question_types() -> None:
    for fmt in ("json", "tag"):
        for allow_retrieve in (True, False):
            (_, system), _ = build_judge_messages("q", "e", allow_retrieve=allow_retrieve, output_format=fmt)
            assert system.isascii()
            lowered = system.lower()
            for word in ("hotpot", "squad", "gsm8k", "multi-hop", "math"):
                assert word not in lowered


def test_last_judgment_prompt_does_not_offer_retrieve() -> None:
    for fmt in ("json", "tag"):
        (_, system), _ = build_judge_messages("q", "e", allow_retrieve=False, output_format=fmt)
        assert "RETRIEVE" not in system


def test_empty_evidence_is_stated() -> None:
    _, (_, human) = build_judge_messages("q", "  ", allow_retrieve=True)
    assert human.endswith("Evidence:\n(no evidence was found)")


# --- JSON 파싱 ------------------------------------------------------------


def test_parse_answer() -> None:
    j = parse_judgment(reply("ANSWER", "Aristotle"), allow_retrieve=True)
    assert (j.decision, j.answer, j.missing, j.parse_ok, j.abstained) == ("ANSWER", "Aristotle", "", True, False)
    assert j.reasoning == "Because."


def test_parse_retrieve_keeps_missing_and_drops_answer() -> None:
    j = parse_judgment(reply("RETRIEVE", "guess", "birthplace of Neleus"), allow_retrieve=True)
    assert (j.decision, j.answer, j.missing, j.parse_ok) == ("RETRIEVE", "", "birthplace of Neleus", True)


def test_parse_abstain_sets_unanswerable() -> None:
    j = parse_judgment(reply("ABSTAIN"), allow_retrieve=True)
    assert (j.decision, j.answer, j.abstained, j.parse_ok) == ("ABSTAIN", "UNANSWERABLE", True, True)


def test_answer_decision_with_unanswerable_text_counts_as_abstained() -> None:
    j = parse_judgment(reply("ANSWER", "Unanswerable."), allow_retrieve=True)
    assert (j.decision, j.answer, j.abstained, j.parse_ok) == ("ANSWER", "UNANSWERABLE", True, True)


def test_abstained_matches_shared_scorer_rule() -> None:
    for answer in ("UNANSWERABLE", "unanswerable.", "**Unanswerable**", "UNANSWERABLE because no evidence", "", "Paris"):
        j = parse_judgment(reply("ANSWER", answer), allow_retrieve=True)
        assert j.abstained == is_abstained(answer)
        assert j.abstained == is_abstained(j.answer)


def test_empty_answer_is_not_turned_into_abstain() -> None:
    j = parse_judgment(reply("ANSWER", ""), allow_retrieve=True)
    assert (j.decision, j.answer, j.abstained) == ("ANSWER", "", False)


def test_retrieve_in_last_judgment_is_a_format_violation() -> None:
    with_answer = parse_judgment(reply("RETRIEVE", "1989", "more"), allow_retrieve=False)
    assert (with_answer.decision, with_answer.answer, with_answer.missing, with_answer.parse_ok) == ("ANSWER", "1989", "", False)
    without = parse_judgment(reply("RETRIEVE", "", "more"), allow_retrieve=False)
    assert (without.decision, without.answer, without.parse_ok, without.abstained) == ("ANSWER", "", False, False)


def test_broken_json_becomes_empty_answer_not_abstain() -> None:
    # 깨진 출력을 기권으로 바꾸면 답할 수 없음 문항에서 정답이 된다 (EVALUATION §3.3)
    for raw in ('{"reasoning": "The evidence says', "", "not json", "[1, 2]"):
        for allow_abstain in (True, False):
            j = parse_judgment(raw, allow_retrieve=True, allow_abstain=allow_abstain, truncated=True)
            assert (j.decision, j.answer, j.parse_ok, j.truncated, j.abstained, j.raw) == ("ANSWER", "", False, True, False, raw)


def test_invalid_decision_keeps_extracted_answer() -> None:
    j = parse_judgment(reply("MAYBE", "Aristotle"), allow_retrieve=True)
    assert (j.decision, j.answer, j.parse_ok) == ("ANSWER", "Aristotle", False)


def test_answer_goes_through_shared_final_answer_extraction() -> None:
    j = parse_judgment(reply("ANSWER", '"Aristotle."'), allow_retrieve=True)
    assert j.answer == extract_final_answer('"Aristotle."') == "Aristotle"


def test_decision_with_trailing_period_is_accepted() -> None:
    assert parse_judgment(reply("ANSWER.", "7"), allow_retrieve=True).parse_ok
    tag = parse_judgment("Fine.\nDECISION: ANSWER.\nANSWER: 7.", allow_retrieve=True, output_format="tag")
    assert (tag.decision, tag.answer, tag.parse_ok) == ("ANSWER", "7", True)


def test_json_wrapped_in_code_fence_and_lowercase_decision() -> None:
    raw = "```json\n" + reply("answer", "  New   York ") + "\n```"
    j = parse_judgment(raw, allow_retrieve=True)
    assert (j.decision, j.answer, j.parse_ok) == ("ANSWER", "New York", True)


# --- 태그(대조군) 파싱 ----------------------------------------------------


def test_tag_answer() -> None:
    raw = "The evidence names the teacher.\nDECISION: ANSWER\nMISSING: NONE\nANSWER: Aristotle"
    j = parse_judgment(raw, allow_retrieve=True, output_format="tag")
    assert (j.decision, j.answer, j.missing, j.parse_ok) == ("ANSWER", "Aristotle", "", True)
    assert j.reasoning == "The evidence names the teacher."


def test_tag_retrieve() -> None:
    raw = "Related but incomplete.\nDECISION: RETRIEVE\nMISSING: founding year of First for Women\nANSWER: NONE"
    j = parse_judgment(raw, allow_retrieve=True, output_format="tag")
    assert (j.decision, j.missing, j.answer) == ("RETRIEVE", "founding year of First for Women", "")


def test_tag_tolerates_bold_and_takes_last_occurrence() -> None:
    raw = "DECISION: RETRIEVE\nOn reflection.\n**Decision:** ABSTAIN\n**Answer:** UNANSWERABLE"
    j = parse_judgment(raw, allow_retrieve=True, output_format="tag")
    assert (j.decision, j.answer, j.parse_ok) == ("ABSTAIN", "UNANSWERABLE", True)


def test_tag_without_decision_line_is_parse_failure() -> None:
    j = parse_judgment("I think the answer is Paris.", allow_retrieve=True, output_format="tag")
    assert (j.decision, j.answer, j.parse_ok, j.abstained) == ("ANSWER", "", False, False)


def test_tag_answer_line_is_not_confused_with_decision_value() -> None:
    raw = "Fine.\nDECISION: ANSWER\nANSWER: 42"
    j = parse_judgment(raw, allow_retrieve=False, output_format="tag")
    assert (j.decision, j.answer) == ("ANSWER", "42")


# --- 호출 -----------------------------------------------------------------


def test_judge_calls_model_with_judge_cap_and_schema() -> None:
    llm, created, seen = make_llm([(reply("ANSWER", "Aristotle"), False)])
    j = judge(llm, "Who taught Neleus?", "Neleus: He studied under Aristotle.", allow_retrieve=True)
    settings = get_settings()

    assert j is not None and (j.decision, j.answer, j.truncated) == ("ANSWER", "Aristotle", False)
    assert created == [{"num_predict": settings.cap.judge, "format_schema": judge_schema(allow_retrieve=True)}]
    assert [m.type for m in seen[0]] == ["system", "human"]
    assert "Who taught Neleus?" in seen[0][1].content
    record = llm.calls[0]
    assert (record.step, record.cap, record.parse_ok, record.is_retry) == ("judge", settings.cap.judge, True, False)
    assert llm.n_calls == 1


def test_last_judgment_passes_schema_without_retrieve() -> None:
    llm, created, _ = make_llm([(reply("ABSTAIN"), False)])
    j = judge(llm, "q", "e", allow_retrieve=False)
    assert j is not None and j.abstained
    assert created[0]["format_schema"]["properties"]["decision"]["enum"] == ["ANSWER", "ABSTAIN"]


def test_tag_format_sends_no_schema() -> None:
    llm, created, _ = make_llm([("Ok.\nDECISION: ANSWER\nANSWER: 7", False)])
    j = judge(llm, "q", "e", allow_retrieve=True, output_format="tag")
    assert j is not None and j.answer == "7"
    assert created[0]["format_schema"] is None


def test_truncated_call_is_retried_once_and_retry_result_is_used() -> None:
    settings = get_settings()
    llm, created, _ = make_llm([('{"reasoning": "cut', True), (reply("ANSWER", "1989"), False)])
    j = judge(llm, "q", "e", allow_retrieve=True)

    assert j is not None and (j.answer, j.parse_ok, j.truncated) == ("1989", True, False)
    assert [c["num_predict"] for c in created] == [settings.cap.judge, settings.cap.judge * settings.cap.retry_multiplier]
    assert (llm.n_calls, llm.n_retries) == (1, 1)
    assert llm.calls[-1].parse_ok is True


def test_truncation_after_retry_is_reported_separately_from_parse_failure() -> None:
    llm, _, _ = make_llm([('{"reasoning": "cut', True), ('{"reasoning": "cut again', True)])
    j = judge(llm, "q", "e", allow_retrieve=True)

    assert j is not None and (j.decision, j.answer, j.parse_ok, j.truncated, j.abstained) == ("ANSWER", "", False, True, False)
    assert llm.calls[-1].parse_ok is False and llm.calls[-1].truncated is True


def test_returns_none_when_budget_is_exhausted() -> None:
    max_calls = get_settings().budget.max_calls
    llm, created, _ = make_llm([(reply("RETRIEVE", missing="x"), False)] * max_calls)
    for _ in range(max_calls):
        assert judge(llm, "q", "e", allow_retrieve=True) is not None

    assert judge(llm, "q", "e", allow_retrieve=False) is None
    assert llm.budget_exhausted and len(created) == max_calls


def test_response_text_handles_none_and_content_blocks() -> None:
    assert _response_text(SimpleNamespace(content=None)) == ""
    assert _response_text(SimpleNamespace(content=["a", {"text": "b"}, {"text": None}, {"type": "x"}])) == "ab"
    assert _response_text(object()) == ""


def test_broken_output_is_not_scored_correct_on_unanswerable_items() -> None:
    from src.eval.scorer import score_unanswerable

    for raw in ("", '{"reasoning": "cut', "not json"):
        j = parse_judgment(raw, allow_retrieve=False, truncated=True)
        assert not score_unanswerable(j.answer)
    explicit = parse_judgment(reply("ABSTAIN"), allow_retrieve=False)
    assert score_unanswerable(explicit.answer)


def test_first_judgment_tells_model_to_answer_instead_of_searching_to_confirm() -> None:
    (_, system), _ = build_judge_messages("q", "e", allow_retrieve=True)
    assert "Do not search again only to confirm" in system
    assert "Do not put a guessed answer in it" in system
    assert "No more searches are possible" not in system


def test_last_judgment_says_no_more_searches() -> None:
    for fmt in ("json", "tag"):
        (_, system), _ = build_judge_messages("q", "e", allow_retrieve=False, output_format=fmt)
        assert "No more searches are possible" in system and "otherwise choose ABSTAIN" in system
        assert "Do not search again" not in system


def test_last_judgment_without_abstain_does_not_mention_abstain() -> None:
    (_, system), _ = build_judge_messages("q", "e", allow_retrieve=False, allow_abstain=False)
    assert "ABSTAIN" not in system
