"""Shared English answer-format instructions and baseline prompt builders.

References:
- AGENTS.md (prompts must be in English; never leak question type/source)
- docs/INTERFACES.md §6 (two-part shared answer instruction)
- docs/IMPLEMENTATION_PLAN.md §5 (PR B-3)
"""

from __future__ import annotations

# Part 1: Content instruction shared across P0, P1, and M evidence judgment.
CONTENT_INSTRUCTION: str = (
    "Give a short final answer only: a few words, a name, a date, or a number.\n"
    "Do not include explanations in the final answer. If the answer is a number, write only the number.\n"
    "If the question cannot be answered, reply exactly: UNANSWERABLE"
)

# Part 2: Output-line tag appended only to free-text calls (P0, P1, M math finalize).
FORMAT_LINE: str = (
    "End your reply with a last line of the form:\n"
    "Final answer: <short answer or UNANSWERABLE>"
)

SHARED_ANSWER_INSTRUCTION: str = f"{CONTENT_INSTRUCTION}\n\n{FORMAT_LINE}"

P1_TOOL_GUIDANCE: str = (
    "You have two tools: bm25_search and calculator.\n"
    "- For factual or entity questions, you MUST call bm25_search first to retrieve evidence before answering. "
    "Never output UNANSWERABLE before checking bm25_search.\n"
    "- For arithmetic or word problems, solve step by step and use calculator to verify numbers."
)


def validate_question(question: str) -> str:
    """Validate that the input question is a non-empty string."""
    if not isinstance(question, str):
        raise TypeError(f"question must be a str, got {type(question).__name__}")
    stripped = question.strip()
    if not stripped:
        raise ValueError("question must not be empty")
    return stripped


def build_p0_prompt(question: str) -> str:
    """Build the minimal instruction + question prompt for P0."""
    clean_question = validate_question(question)
    return f"{SHARED_ANSWER_INSTRUCTION}\n\nQuestion: {clean_question}"


def build_p0_raw_prompt(question: str) -> str:
    """Build the raw question-only prompt for P0-raw (no instructions)."""
    return validate_question(question)


def build_p1_system_prompt() -> str:
    """Build the English ReAct system prompt for P1 with shared instructions."""
    return f"{P1_TOOL_GUIDANCE}\n\n{SHARED_ANSWER_INSTRUCTION}"
