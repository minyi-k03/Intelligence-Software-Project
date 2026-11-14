"""Common dataclasses and types for M(EBA) pipelines and evaluations.

Reference:
- docs/INTERFACES.md §4
- docs/IMPLEMENTATION_PLAN.md §3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class CallRecord:
    """Record of an individual LLM call.

    Attributes:
        step: Step name, e.g., 'p0', 'p1_step', 'router', 'finalize', 'judge', 'ledger'.
        cap: Applied output token cap (num_predict), or None if unbounded.
        input_tokens: Number of prompt/input tokens measured via callback.
        output_tokens: Number of completion/output tokens measured via callback.
        truncated: Whether generation was truncated by the output cap.
        is_retry: Whether this call is a truncation retry.
        parse_ok: Structured output parsing success, or None if not applicable.
        latency_s: Call latency in seconds.
    """

    step: str
    cap: Optional[int]
    input_tokens: int
    output_tokens: int
    truncated: bool
    is_retry: bool
    parse_ok: Optional[bool]
    latency_s: float


@dataclass
class PipelineResult:
    """Standardized output returned by every pipeline (P0, P0-raw, P1, M).

    Attributes:
        answer: Final answer string. "UNANSWERABLE" if abstained.
        abstained: Whether the pipeline explicitly abstained.
        budget_exhausted: Whether execution terminated due to exhausted call budget.
        route: M route selection: 'math' | 'search' | None.
        decisions: Sequence of evidence decisions, e.g., ['RETRIEVE', 'ANSWER'].
        n_calls: Number of LLM calls charged against the call budget (excluding retries).
        n_retries: Number of truncation retry calls.
        calls: Detailed call records.
        retrieved_doc_ids: List of retrieved doc_ids per retrieval round.
        kept_windows: List of window IDs kept per compression round (e.g. 'doc_id#sentence_idx').
        loop_breaks: Number of times loop breaker was triggered.
        error: Unhandled exception message, or None if successful.
    """

    answer: str
    abstained: bool
    budget_exhausted: bool
    route: Optional[Literal["math", "search"]]
    decisions: list[str]
    n_calls: int
    n_retries: int
    calls: list[CallRecord] = field(default_factory=list)
    retrieved_doc_ids: list[list[str]] = field(default_factory=list)
    kept_windows: list[list[str]] = field(default_factory=list)
    loop_breaks: int = 0
    error: Optional[str] = None
