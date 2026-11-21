"""Router: M's first call. Chooses the search path or solves math in place.

References:
- docs/SPEC.md §4, §5, §7 (router = first call, cap.first; math path -> optional
  finalize call, cap.finalize; no direct-answer path)
- docs/INTERFACES.md §4, §6, §8
- docs/adr/0004-m-architecture-call-budget.md
- docs/experiments/2026-11-20-a-p1-search-probe.md (the shared answer
  instruction suppresses searching, so the route is forced onto the first line)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

from src.baselines.prompts import SHARED_ANSWER_INSTRUCTION
from src.eval.scorer import extract_final_answer
from src.meba.budget import BudgetedLLM
from src.meba.loop_breaker import LoopBreaker
from src.tools.calculator import calculate

Route = Literal["math", "search"]

# 문항 유형(데이터셋 이름 등)을 알려 주지 않는 일반 표현만 쓴다 (AGENTS.md).
ROUTER_SYSTEM_PROMPT: str = (
    "You are the first step of a question-answering system.\n"
    "Decide how the question below should be handled, and put your decision on the "
    "very first line of your reply, in exactly one of these two forms:\n"
    "ROUTE: math\n"
    "ROUTE: search\n"
    "\n"
    "Choose `ROUTE: math` only when the question is an arithmetic or quantitative "
    "problem that can be solved by reasoning and calculation alone. After that line, "
    "solve it step by step.\n"
    "Choose `ROUTE: search` for every other question, such as a question about "
    "people, places, events, dates, or any other knowledge. A document search will "
    "be run for it, so reply with the single line `ROUTE: search` and nothing else. "
    "Do not answer from memory.\n"
    "\n"
    "When you choose `ROUTE: math`, work through the steps first, then give the final "
    "answer following these rules:\n"
    f"{SHARED_ANSWER_INSTRUCTION}"
)

FINALIZE_SYSTEM_PROMPT: str = (
    "Use the calculator results in the conversation to answer the question. "
    "Reply with the final answer only, following these rules:\n"
    f"{SHARED_ANSWER_INSTRUCTION}"
)

CALCULATOR_TOOL_NAME: str = "calculator"

# `convert_to_openai_tool(get_calculator_tool())`(B-2b)과 같은 값이다. langchain 없이도 테스트가
# 돌도록 사전으로 두고, 어긋나면 tests/test_router.py의 계약 테스트가 실패한다 (AGENTS.md: 같은 도구).
CALCULATOR_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": CALCULATOR_TOOL_NAME,
        "description": "Evaluate an arithmetic expression (e.g. '(12 + 8) * 3 / 4' or 'sqrt(144)').",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}

_ROUTE_RE = re.compile(r"^\s*ROUTE\s*:\s*(math|search)\b", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class RouterOutcome:
    """What the router decided and, on the math path, the answer it produced.

    Attributes:
        route: ``"math"`` or ``"search"``. ``None`` only when the budget was
            already exhausted before the router call, so no route was chosen.
        answer: Math-path answer without prefix, or ``None`` on the search path.
        used_calculator: Whether at least one calculator call was executed.
        loop_breaks: Duplicate calculator calls that the loop breaker blocked.
        budget_exhausted: Whether a requested call was blocked by the budget.
    """

    route: Optional[Route]
    answer: Optional[str]
    used_calculator: bool = False
    loop_breaks: int = 0
    budget_exhausted: bool = False


def _response_text(response: Any) -> str:
    """Return the plain text of a ChatOllama response.

    Args:
        response: Model response; ``content`` may be ``str``, ``None``, or a
            list of content blocks (strings or ``{"text": ...}`` dicts).

    Returns:
        The text, or ``""`` when there is none. Never the string ``"None"``.
    """

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, (str, dict))
        ]
        return "\n".join(part for part in parts if isinstance(part, str))
    return ""


def _strip_route_line(text: str) -> str:
    """Remove the ``ROUTE:`` line so it cannot leak into answer extraction."""

    return _ROUTE_RE.sub("", text, count=1)


def parse_route(text: str, *, has_tool_call: bool = False) -> Route:
    """Read the route from the router's reply.

    The first line that starts with ``ROUTE:`` (case-insensitive) wins; the
    prompt asks for it on the first line, but leading text is tolerated.
    Without such a line, a calculator request means the math path (tool-call
    replies arrive with empty text, so the line is usually lost). Anything
    else goes to search, so no direct-answer path opens (SPEC §4, §9).

    Args:
        text: Raw model output of the first call.
        has_tool_call: Whether the reply requested a tool.

    Returns:
        ``"math"`` or ``"search"``.
    """

    match = _ROUTE_RE.search(text)
    if match:
        return "math" if match.group(1).lower() == "math" else "search"
    return "math" if has_tool_call else "search"


def route_question(question: str, budget: BudgetedLLM) -> RouterOutcome:
    """Run the router call for one question.

    Args:
        question: Question text only. No dataset type or source is ever passed.
        budget: Per-question budgeted model wrapper (caps, retry, call budget).

    Returns:
        The chosen route and, for the math path, the answer.
    """

    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    result = budget.invoke("router", messages, invoke_kwargs={"tools": [CALCULATOR_TOOL_SPEC]})
    if result.response is None:  # 예산 소진: 호출하지 않았으므로 경로를 정하지 않는다.
        return RouterOutcome(route=None, answer=None, budget_exhausted=True)

    text = _response_text(result.response)
    tool_calls = list(getattr(result.response, "tool_calls", None) or [])
    route = parse_route(text, has_tool_call=bool(tool_calls))
    if route == "search":
        return RouterOutcome(route=route, answer=None)

    if not tool_calls:
        return RouterOutcome(route=route, answer=extract_final_answer(_strip_route_line(text)))

    return _finalize_with_calculator(question, text, tool_calls, budget)


def _finalize_with_calculator(
    question: str,
    first_text: str,
    tool_calls: list[dict[str, Any]],
    budget: BudgetedLLM,
) -> RouterOutcome:
    """Run requested calculator calls, then make the capped finalize call.

    Args:
        question: Original question text.
        first_text: Raw text of the first call (the ``ROUTE:`` line is dropped).
        tool_calls: Calculator calls requested by the first call.
        budget: Per-question budgeted model wrapper.

    Returns:
        Math-path outcome whose answer comes from the finalize call.
    """

    breaker = LoopBreaker()
    used_calculator = False
    calls: list[dict[str, Any]] = []
    tool_messages: list[dict[str, Any]] = []
    for index, call in enumerate(tool_calls):
        call_id = call.get("id") or f"call-{index}"
        name = call.get("name")
        args = call.get("args")
        # LangChain이 AIMessage로 바꿀 수 있도록 assistant 메시지에는 사전 인자만 남긴다.
        calls.append({"id": call_id, "name": name, "args": args if isinstance(args, dict) else {}})
        expression = args.get("expression") if isinstance(args, dict) else None
        if name != CALCULATOR_TOOL_NAME:
            output = f"ERROR: Unknown tool '{name}'"
        elif not isinstance(expression, str):
            output = "ERROR: calculator requires a string 'expression' argument"
        else:
            # 식 문자열만 키로 쓴다. 인자 전체를 넘기면 정규화할 수 없는 값에서 TypeError가 난다.
            outcome = breaker.run(name, expression, lambda e=expression: calculate(e))
            output = outcome.result
            used_calculator = True
        tool_messages.append({"role": "tool", "tool_call_id": call_id, "content": str(output)})

    messages = [
        {"role": "system", "content": FINALIZE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": _strip_route_line(first_text).strip(), "tool_calls": calls},
        *tool_messages,
    ]
    result = budget.invoke("finalize", messages)
    answer = "" if result.response is None else extract_final_answer(_response_text(result.response))
    return RouterOutcome(
        route="math",
        answer=answer,
        used_calculator=used_calculator,
        loop_breaks=breaker.loop_breaks,
        budget_exhausted=result.budget_exhausted,
    )
