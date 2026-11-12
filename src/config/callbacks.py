"""LangChain metrics callback handler for call and token accounting.

References:
- AGENTS.md (all LLM calls must pass through LangChain ChatOllama, counted via callback)
- docs/INTERFACES.md §4, §7
- docs/IMPLEMENTATION_PLAN.md §4
"""

from __future__ import annotations

import time
from typing import Any, Optional

from src.common.types import CallRecord

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
except ImportError:
    # Graceful fallback when langchain is not yet installed
    class BaseCallbackHandler:  # type: ignore
        pass

    class LLMResult:  # type: ignore
        pass


def _extract_tokens_and_reason(
    mapping: dict[str, Any],
) -> tuple[Optional[int], Optional[int], Optional[str]]:
    """Extract (input_tokens, output_tokens, done_reason) from a metadata dict."""
    in_tok: Optional[int] = None
    out_tok: Optional[int] = None

    for key in ("input_tokens", "prompt_eval_count", "prompt_tokens"):
        val = mapping.get(key)
        if isinstance(val, int) and not isinstance(val, bool):
            in_tok = val
            break

    for key in ("output_tokens", "eval_count", "completion_tokens"):
        val = mapping.get(key)
        if isinstance(val, int) and not isinstance(val, bool):
            out_tok = val
            break

    # Also check nested token_usage dict if present
    token_usage = mapping.get("token_usage")
    if isinstance(token_usage, dict):
        if in_tok is None:
            for key in ("input_tokens", "prompt_eval_count", "prompt_tokens"):
                val = token_usage.get(key)
                if isinstance(val, int) and not isinstance(val, bool):
                    in_tok = val
                    break
        if out_tok is None:
            for key in ("output_tokens", "eval_count", "completion_tokens"):
                val = token_usage.get(key)
                if isinstance(val, int) and not isinstance(val, bool):
                    out_tok = val
                    break

    reason = mapping.get("done_reason") or mapping.get("finish_reason")
    return in_tok, out_tok, str(reason) if reason is not None else None


class MetricsCallbackHandler(BaseCallbackHandler):
    """Tracks LLM invocations, input/output tokens, latencies, and creates CallRecords."""

    def __init__(
        self,
        current_step: str = "unknown",
        cap: Optional[int] = None,
        is_retry: bool = False,
    ) -> None:
        super().__init__()
        self.current_step = current_step
        self.cap = cap
        self.is_retry = is_retry
        self.parse_ok: Optional[bool] = None

        self._start_time: float = 0.0
        self._start_times: dict[Any, float] = {}
        self.calls: list[CallRecord] = []
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    def set_call_context(
        self,
        step: str,
        cap: Optional[int] = None,
        is_retry: bool = False,
        parse_ok: Optional[bool] = None,
    ) -> None:
        """Update metadata context for the upcoming call."""
        self.current_step = step
        self.cap = cap
        self.is_retry = is_retry
        self.parse_ok = parse_ok

    def record_parse_result(self, parse_ok: bool) -> None:
        """Update parse_ok on the most recently recorded CallRecord (post-invocation)."""
        self.parse_ok = parse_ok
        if self.calls:
            self.calls[-1].parse_ok = parse_ok

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """Fires when an LLM / Chat model begins generation."""
        now = time.perf_counter()
        self._start_time = now
        run_id = kwargs.get("run_id")
        if run_id is not None:
            self._start_times[run_id] = now

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        **kwargs: Any,
    ) -> None:
        """Fires when a Chat model begins generation."""
        now = time.perf_counter()
        self._start_time = now
        run_id = kwargs.get("run_id")
        if run_id is not None:
            self._start_times[run_id] = now

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Clean up timer state if an LLM call raises an exception."""
        run_id = kwargs.get("run_id")
        if run_id is not None:
            self._start_times.pop(run_id, None)
        self._start_time = 0.0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Fires when an LLM / Chat model completes generation."""
        run_id = kwargs.get("run_id")
        start = self._start_times.pop(run_id, None) if run_id is not None else None
        if start is None:
            start = self._start_time
        latency_s = time.perf_counter() - start if start > 0 else 0.0
        self._start_time = 0.0

        in_tok: Optional[int] = None
        out_tok: Optional[int] = None
        done_reason: Optional[str] = None
        text = ""

        # 1. Inspect generation object (ChatGeneration.message and generation_info)
        generations = getattr(response, "generations", [])
        if generations and generations[0]:
            gen = generations[0][0]
            message = getattr(gen, "message", None)

            if message is not None:
                # langchain-ollama attaches usage_metadata and response_metadata on AIMessage
                usage_meta = getattr(message, "usage_metadata", None)
                if isinstance(usage_meta, dict):
                    i, o, r = _extract_tokens_and_reason(usage_meta)
                    if in_tok is None and i is not None:
                        in_tok = i
                    if out_tok is None and o is not None:
                        out_tok = o
                    if done_reason is None and r is not None:
                        done_reason = r

                resp_meta = getattr(message, "response_metadata", None)
                if isinstance(resp_meta, dict):
                    i, o, r = _extract_tokens_and_reason(resp_meta)
                    if in_tok is None and i is not None:
                        in_tok = i
                    if out_tok is None and o is not None:
                        out_tok = o
                    if done_reason is None and r is not None:
                        done_reason = r

                msg_content = getattr(message, "content", "")
                if isinstance(msg_content, str):
                    text = msg_content

            gen_info = getattr(gen, "generation_info", None)
            if isinstance(gen_info, dict):
                i, o, r = _extract_tokens_and_reason(gen_info)
                if in_tok is None and i is not None:
                    in_tok = i
                if out_tok is None and o is not None:
                    out_tok = o
                if done_reason is None and r is not None:
                    done_reason = r

            if not text:
                gen_text = getattr(gen, "text", "")
                if isinstance(gen_text, str):
                    text = gen_text

        # 2. Inspect top-level LLMResult.llm_output
        llm_output = getattr(response, "llm_output", None)
        if isinstance(llm_output, dict):
            i, o, r = _extract_tokens_and_reason(llm_output)
            if in_tok is None and i is not None:
                in_tok = i
            if out_tok is None and o is not None:
                out_tok = o
            if done_reason is None and r is not None:
                done_reason = r

        input_tokens = in_tok if in_tok is not None else 0
        if out_tok is not None:
            output_tokens = out_tok
        elif text:
            output_tokens = max(1, len(text.split()))
        else:
            output_tokens = 0

        # 3. Determine truncation from Ollama done_reason or cap threshold
        is_truncated = False
        if done_reason in ("length", "max_tokens"):
            is_truncated = True
        if self.cap is not None and output_tokens >= self.cap:
            is_truncated = True

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        record = CallRecord(
            step=self.current_step,
            cap=self.cap,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            truncated=is_truncated,
            is_retry=self.is_retry,
            parse_ok=self.parse_ok,
            latency_s=round(latency_s, 4),
        )
        self.calls.append(record)

    @property
    def n_calls(self) -> int:
        """Total calls recorded (including retries)."""
        return len(self.calls)

    @property
    def n_budget_calls(self) -> int:
        """Calls charged against the budget (excluding retries per SPEC §7)."""
        return sum(1 for c in self.calls if not c.is_retry)

    @property
    def n_retries(self) -> int:
        """Number of truncation retries."""
        return sum(1 for c in self.calls if c.is_retry)

    @property
    def max_input_tokens(self) -> int:
        """Maximum input tokens across recorded calls (INTERFACES.md §7.2)."""
        return max((c.input_tokens for c in self.calls), default=0)

    def reset(self) -> None:
        """Reset all tracked metrics and records."""
        self.calls.clear()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._start_time = 0.0
        self._start_times.clear()
        self.parse_ok = None
