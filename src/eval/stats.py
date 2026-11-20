"""Paired exact tests for question-level correctness."""

from __future__ import annotations

from collections.abc import Iterable
from math import comb


def mcnemar_exact(pairs: Iterable[tuple[bool, bool]]) -> dict[str, int | float | None]:
    """Return the four M/P1 correctness counts and two-sided exact p-value."""
    counts = {
        "both_correct": 0,
        "m_only": 0,
        "p1_only": 0,
        "both_wrong": 0,
    }
    cells = {
        (True, True): "both_correct",
        (True, False): "m_only",
        (False, True): "p1_only",
        (False, False): "both_wrong",
    }
    for pair in pairs:
        counts[cells[pair]] += 1

    if not any(counts.values()):
        return {"n": 0, **counts, "p_value": None}

    discordant = counts["m_only"] + counts["p1_only"]
    tail = min(counts["m_only"], counts["p1_only"])
    p_value = min(
        1.0, 2 * sum(comb(discordant, k) for k in range(tail + 1)) / 2**discordant
    )
    return {"n": sum(counts.values()), **counts, "p_value": p_value}
