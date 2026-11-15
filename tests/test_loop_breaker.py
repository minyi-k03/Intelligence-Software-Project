"""Tests for duplicate tool-call loop breaking."""

from __future__ import annotations

import pytest

from src.meba.loop_breaker import LoopBreaker, normalize_args


def test_first_tool_call_is_executed() -> None:
    """The first occurrence of a tool/input pair must run normally."""

    calls: list[str] = []
    breaker = LoopBreaker()

    outcome = breaker.run(
        "search",
        {"query": "Ada Lovelace", "k": 3},
        lambda: calls.append("ran") or "search result",
    )

    assert calls == ["ran"]
    assert outcome.result == "search result"
    assert outcome.executed
    assert not outcome.blocked
    assert not outcome.answer_directly
    assert breaker.loop_breaks == 0


def test_duplicate_returns_previous_result_without_reexecution() -> None:
    """A duplicate must route directly with the cached prior result."""

    execution_count = 0
    breaker = LoopBreaker()

    def execute() -> str:
        nonlocal execution_count
        execution_count += 1
        return "previous result"

    first = breaker.run("search", {"query": "same", "k": 3}, execute)
    duplicate = breaker.run("search", {"k": 3, "query": "same"}, execute)

    assert first.executed
    assert execution_count == 1
    assert duplicate.result == "previous result"
    assert duplicate.blocked
    assert duplicate.answer_directly
    assert breaker.loop_breaks == 1


def test_same_tool_with_different_normalized_args_is_allowed() -> None:
    """Different arguments must not be treated as a loop."""

    calls: list[str] = []
    breaker = LoopBreaker()

    first = breaker.run("search", {"query": "first"}, lambda: calls.append("first") or "r1")
    second = breaker.run("search", {"query": "second"}, lambda: calls.append("second") or "r2")

    assert first.executed
    assert second.executed
    assert calls == ["first", "second"]
    assert breaker.loop_breaks == 0


def test_normalization_handles_safe_representation_differences() -> None:
    """Mapping order is ignored without altering the values themselves."""

    left = {"query": "Ada", "options": [1, 2]}
    right = {"options": [1, 2], "query": "Ada"}

    assert normalize_args(left) == normalize_args(right)


def test_reset_clears_per_question_history_and_counter() -> None:
    """The same call must execute again after moving to a new question."""

    execution_count = 0
    breaker = LoopBreaker()

    def execute() -> str:
        nonlocal execution_count
        execution_count += 1
        return str(execution_count)

    breaker.run("calculator", {"expression": "6 * 7"}, execute)
    breaker.run("calculator", {"expression": "6 * 7"}, execute)
    assert breaker.loop_breaks == 1

    breaker.reset()
    after_reset = breaker.run("calculator", {"expression": "6 * 7"}, execute)

    assert after_reset.executed
    assert after_reset.result == "2"
    assert execution_count == 2
    assert breaker.loop_breaks == 0


@pytest.mark.parametrize("left,right", [
    ("2³", "23"), ("Ａda", "Ada"), (" Ada ", "Ada"),
    ("Ada", "ada"), ("1 + 2", "1+2"),
])
def test_distinct_text_arguments_are_not_merged(left: str, right: str) -> None:
    """Unspecified text equivalences must never suppress a distinct call."""
    breaker = LoopBreaker()
    calls = []
    breaker.run("calculator", {"expression": left}, lambda: calls.append(left) or left)
    result = breaker.run("calculator", {"expression": right}, lambda: calls.append(right) or right)
    assert result.executed
    assert result.result == right
    assert calls == [left, right]
    assert breaker.loop_breaks == 0


def test_nonconsecutive_duplicate_uses_matching_result() -> None:
    """Repeating A after B returns A's result, even when it is None."""
    breaker = LoopBreaker()
    breaker.run("search", {"query": "A"}, lambda: None)
    breaker.run("search", {"query": "B"}, lambda: "B result")
    for count in (1, 2):
        result = breaker.run("search", {"query": "A"}, lambda: pytest.fail("duplicate executed"))
        assert result.result is None
        assert result.answer_directly
        assert breaker.loop_breaks == count


def test_tool_name_is_part_of_identity() -> None:
    """Equal arguments to distinct tools must both execute."""
    breaker = LoopBreaker()
    breaker.run("tool_a", "input", lambda: "A")
    result = breaker.run("tool_b", "input", lambda: "B")
    assert result.executed
    assert result.result == "B"


def test_tool_exception_is_not_cached() -> None:
    """An exception must propagate without storing a nonexistent result."""
    breaker = LoopBreaker()

    def fail() -> None:
        """Simulate an unexpected tool failure."""
        raise RuntimeError("tool failed")

    with pytest.raises(RuntimeError, match="tool failed"):
        breaker.run("search", "input", fail)
    assert breaker.run("search", "input", lambda: "recovered").executed
    assert breaker.loop_breaks == 0
