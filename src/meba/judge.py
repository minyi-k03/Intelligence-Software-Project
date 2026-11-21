"""근거 판정 + 답 — 핵심 부품, 전략 ③ (SPEC §4·§7, INTERFACES §5·§6, 담당 C).

압축된 근거와 질문을 한 번의 모델 호출에 넣어 판정(ANSWER / RETRIEVE / ABSTAIN)과 답을
함께 받는다. 기본 형식은 서술을 맨 앞에 둔 JSON(ADR-0004)이고, 대조군은 자유 서술 뒤
마지막 줄 태그다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from src.baselines.prompts import CONTENT_INSTRUCTION
from src.eval.scorer import UNANSWERABLE, is_abstained
from src.meba.budget import BudgetedLLM

Decision = Literal["ANSWER", "RETRIEVE", "ABSTAIN"]
OutputFormat = Literal["json", "tag"]

JUDGE_STEP = "judge"

_DECISION_HELP = {
    "ANSWER": "the evidence states the answer.",
    "RETRIEVE": (
        "the evidence is related to the question, but one specific fact needed "
        "for the answer is not in it and could be found by another search."
    ),
    "ABSTAIN": "the evidence does not contain the answer and another search would not help.",
}
_MISSING_HELP = (
    "the missing fact, naming the people, places, works or terms involved "
    "(used as the next search query)"
)
_NO_EVIDENCE = "(no evidence was found)"


@dataclass(frozen=True)
class Judgment:
    """근거 판정 한 번의 결과.

    decision: 허용된 판정 값으로 정리한 판정
    answer: ANSWER면 답, ABSTAIN이면 "UNANSWERABLE", RETRIEVE면 ""
    missing: RETRIEVE일 때의 빠진 정보, 그 외에는 ""
    reasoning: 모델이 쓴 서술
    parse_ok: 출력이 약속한 형식대로였는가 (허용되지 않은 판정 값도 False)
    truncated: 재시도 뒤에도 출력 상한에 걸렸는가
    raw: 모델 출력 원문
    """

    decision: Decision
    answer: str
    missing: str
    reasoning: str
    parse_ok: bool
    truncated: bool
    raw: str

    @property
    def abstained(self) -> bool:
        """최종 답이 UNANSWERABLE인가. ANSWER로 판정하고 답을 UNANSWERABLE로 쓴 경우도 포함한다."""
        return is_abstained(self.answer)


def allowed_decisions(*, allow_retrieve: bool, allow_abstain: bool = True) -> list[str]:
    """이번 호출에서 낼 수 있는 판정 값.

    Args:
        allow_retrieve: False면 RETRIEVE를 뺀다 (마지막 판정, 재검색 절제 변형)
        allow_abstain: False면 ABSTAIN을 뺀다 (근거 판정 절제 변형)

    Returns:
        ANSWER를 맨 앞에 둔 판정 값 리스트
    """
    decisions = ["ANSWER"]
    if allow_retrieve:
        decisions.append("RETRIEVE")
    if allow_abstain:
        decisions.append("ABSTAIN")
    return decisions


def judge_schema(*, allow_retrieve: bool, allow_abstain: bool = True) -> dict[str, Any]:
    """Ollama 구조화 출력에 넘길 JSON 스키마.

    Args:
        allow_retrieve: RETRIEVE를 허용할지
        allow_abstain: ABSTAIN을 허용할지

    Returns:
        키 순서가 reasoning → decision → missing → answer인 스키마 (ADR-0004)
    """
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "decision": {
                "type": "string",
                "enum": allowed_decisions(allow_retrieve=allow_retrieve, allow_abstain=allow_abstain),
            },
            "missing": {"type": "string"},
            "answer": {"type": "string"},
        },
        "required": ["reasoning", "decision", "missing", "answer"],
    }


def build_judge_messages(
    question: str,
    evidence: str,
    *,
    allow_retrieve: bool,
    allow_abstain: bool = True,
    output_format: OutputFormat = "json",
) -> list[tuple[str, str]]:
    """근거 판정 호출의 메시지를 만든다. 문항 유형은 받지도 쓰지도 않는다.

    Args:
        question: 질문 원문
        evidence: 압축된 근거 텍스트 (`Compressed.context`)
        allow_retrieve: RETRIEVE를 허용할지
        allow_abstain: ABSTAIN을 허용할지
        output_format: "json"(기본) 또는 "tag"(대조군)

    Returns:
        `ChatOllama.invoke`에 넘길 [("system", ...), ("human", ...)]
    """
    decisions = allowed_decisions(allow_retrieve=allow_retrieve, allow_abstain=allow_abstain)
    options = "\n".join(f"- {d}: {_DECISION_HELP[d]}" for d in decisions)
    names = ", ".join(decisions)
    missing = f"if the decision is RETRIEVE, {_MISSING_HELP}" if allow_retrieve else None

    if output_format == "json":
        fields = [
            '"reasoning": one or two sentences on whether the evidence answers the question.',
            f'"decision": one of {names}.',
            f'"missing": {missing}; otherwise "".' if missing else '"missing": "".',
            '"answer": the answer; "" if the decision is RETRIEVE.' if allow_retrieve else '"answer": the answer.',
        ]
        reply = "Reply with a JSON object with exactly these keys, in this order:\n" + "\n".join(fields)
    else:
        lines = [f"DECISION: <one of {names}>"]
        if missing:
            lines.append(f"MISSING: <{missing}; otherwise NONE>")
        lines.append("ANSWER: <the answer; NONE if the decision is RETRIEVE>" if allow_retrieve else "ANSWER: <the answer>")
        reply = (
            "First write one or two sentences on whether the evidence answers the question.\n"
            "Then end your reply with these lines:\n" + "\n".join(lines)
        )

    system = (
        "You answer a question using only the evidence passages given to you. "
        "Do not use outside knowledge.\n\n"
        f"Choose one decision:\n{options}\n\n"
        f"For the answer:\n{CONTENT_INSTRUCTION}\n\n"
        f"{reply}"
    )
    human = f"Question: {question}\n\nEvidence:\n{evidence.strip() or _NO_EVIDENCE}"
    return [("system", system), ("human", human)]


def _clean(value: Any) -> str:
    """필드 값을 한 줄 문자열로 정리한다."""
    return " ".join(str(value).split()) if value is not None else ""


def _parse_json_fields(raw: str) -> dict[str, str] | None:
    """JSON 출력에서 네 필드를 꺼낸다. 객체가 아니거나 깨졌으면 None."""
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return {key: _clean(data.get(key)) for key in ("reasoning", "decision", "missing", "answer")}


def _blank_if_none(value: str) -> str:
    """태그 값의 "NONE" 표기를 빈 문자열로 바꾼다."""
    return "" if value.casefold() in {"none", "n/a", "-"} else value


_TAG_LINE = re.compile(r"^[ \t>*_-]*(DECISION|MISSING|ANSWER)[ \t*_]*:[ \t*_]*(.*)$", re.IGNORECASE | re.MULTILINE)


def _parse_tag_fields(raw: str) -> dict[str, str] | None:
    """대조군 출력에서 태그 줄을 꺼낸다. 같은 태그가 여러 번이면 마지막 것. DECISION이 없으면 None."""
    found: dict[str, str] = {}
    first_tag = len(raw)
    for match in _TAG_LINE.finditer(raw):
        found[match.group(1).lower()] = _clean(match.group(2)).strip("*_ ")
        first_tag = min(first_tag, match.start())
    if "decision" not in found:
        return None
    return {
        "reasoning": _clean(raw[:first_tag]),
        "decision": found["decision"],
        "missing": _blank_if_none(found.get("missing", "")),
        "answer": _blank_if_none(found.get("answer", "")),
    }


def parse_judgment(
    raw: str,
    *,
    allow_retrieve: bool,
    allow_abstain: bool = True,
    output_format: OutputFormat = "json",
    truncated: bool = False,
) -> Judgment:
    """모델 출력을 `Judgment`로 바꾼다. 예외를 던지지 않는다.

    형식이 깨졌거나 허용되지 않은 판정 값이면 `parse_ok=False`로 두고 다음과 같이 정리한다.
    - 답이 있으면 ANSWER
    - 답이 없으면 ABSTAIN (ABSTAIN이 허용되지 않으면 빈 답의 ANSWER)

    Args:
        raw: 모델 출력 원문
        allow_retrieve: 이 호출에서 RETRIEVE를 허용했는지
        allow_abstain: 이 호출에서 ABSTAIN을 허용했는지
        output_format: 호출에 쓴 형식
        truncated: 출력이 상한에 걸렸는지 (그대로 결과에 싣는다)

    Returns:
        정리된 판정. ABSTAIN이면 `answer`는 "UNANSWERABLE"
    """
    fields = _parse_json_fields(raw) if output_format == "json" else _parse_tag_fields(raw)
    allowed = allowed_decisions(allow_retrieve=allow_retrieve, allow_abstain=allow_abstain)
    fields = fields or {"reasoning": "", "decision": "", "missing": "", "answer": ""}
    decision = fields["decision"].upper()
    answer = fields["answer"]
    parse_ok = decision in allowed

    if not parse_ok:
        has_answer = bool(answer) and not is_abstained(answer)
        decision = "ANSWER" if has_answer or not allow_abstain else "ABSTAIN"

    if decision == "ABSTAIN":
        answer = UNANSWERABLE
    elif decision == "RETRIEVE":
        answer = ""
    elif is_abstained(answer):
        answer = UNANSWERABLE

    return Judgment(
        decision=decision,  # type: ignore[arg-type]
        answer=answer,
        missing=fields["missing"] if decision == "RETRIEVE" else "",
        reasoning=fields["reasoning"],
        parse_ok=parse_ok,
        truncated=truncated,
        raw=raw,
    )


def _response_text(response: Any) -> str:
    """모델 응답에서 텍스트를 꺼낸다 (내용이 조각 리스트로 오는 경우 포함)."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        return "".join(part if isinstance(part, str) else str(part.get("text", "")) for part in content)
    return str(content)


def judge(
    llm: BudgetedLLM,
    question: str,
    evidence: str,
    *,
    allow_retrieve: bool,
    allow_abstain: bool = True,
    output_format: OutputFormat = "json",
) -> Judgment | None:
    """근거 판정 + 답 호출을 한 번 한다.

    출력 상한(`cap.judge`), 잘림 재시도, 호출 예산은 `llm`이 처리한다. 파싱 결과는
    이 호출의 `CallRecord.parse_ok`에 기록한다.

    Args:
        llm: 이 문항의 호출 예산을 관리하는 래퍼
        question: 질문 원문
        evidence: 압축된 근거 텍스트
        allow_retrieve: RETRIEVE를 허용할지. 마지막 판정에서는 False
        allow_abstain: ABSTAIN을 허용할지
        output_format: "json"(기본) 또는 "tag"(대조군)

    Returns:
        판정 결과. 호출 예산이 이미 소진되어 모델을 부르지 못했으면 None
    """
    messages = build_judge_messages(
        question,
        evidence,
        allow_retrieve=allow_retrieve,
        allow_abstain=allow_abstain,
        output_format=output_format,
    )
    schema = judge_schema(allow_retrieve=allow_retrieve, allow_abstain=allow_abstain) if output_format == "json" else None
    result = llm.invoke(JUDGE_STEP, messages, format_schema=schema)
    if result.budget_exhausted:
        return None

    judgment = parse_judgment(
        _response_text(result.response),
        allow_retrieve=allow_retrieve,
        allow_abstain=allow_abstain,
        output_format=output_format,
        truncated=llm.calls[-1].truncated,
    )
    llm.callback.record_parse_result(judgment.parse_ok)
    return judgment
