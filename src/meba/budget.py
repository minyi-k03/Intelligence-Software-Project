"""Output caps, truncation retries, and per-question LLM call budget.

References:
- docs/SPEC.md §4, §7
- docs/INTERFACES.md §2, §4
- docs/adr/0004-m-architecture-call-budget.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.common.types import CallRecord
from src.config.callbacks import MetricsCallbackHandler
from src.config.llm import create_chat_ollama
from src.config.settings import Settings, get_settings


_CAP_CONFIG_BY_STEP = {
    "router": "first",
    "first": "first",
    "finalize": "finalize",
    "judge": "judge",
}


@dataclass(frozen=True)
class BudgetCallResult:
    """Result of one requested base call.

    ``response`` is ``None`` only when the base-call budget was already
    exhausted. A successful truncation retry replaces the truncated base
    response; if the retry is also truncated, that retry response is returned
    without another attempt.
    """

    response: Any | None
    budget_exhausted: bool


class BudgetedLLM:
    """Invoke ChatOllama with configured caps and a per-question call budget.

    Create one instance per question, or call :meth:`reset` before starting a
    new question. Base calls consume ``budget.max_calls``; the single retry
    allowed after a truncated base call does not consume that budget.

    Args:
        settings: Validated project settings. Defaults to the shared settings.
        callback: Metrics callback used to produce ``CallRecord`` entries.
        llm_factory: Injectable ChatOllama factory, primarily for CPU-only tests.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        callback: Optional[MetricsCallbackHandler] = None,
        llm_factory: Callable[..., Any] = create_chat_ollama,
    ) -> None:
        self.settings = settings or get_settings()
        self.callback = callback or MetricsCallbackHandler()
        if self.callback.calls:
            raise ValueError("Use a fresh metrics callback for each question")
        self._llm_factory = llm_factory
        self._n_calls = 0
        self._n_retries = 0
        self._budget_exhausted = False
        self._failed = False

    def cap_for_step(self, step: str) -> int:
        """Return the configured ``num_predict`` cap for a model step.

        Args:
            step: ``router``/``first``, ``finalize``, or ``judge``.

        Returns:
            Configured positive output-token cap.

        Raises:
            ValueError: If the step has no cap defined by the team design.
        """

        config_name = _CAP_CONFIG_BY_STEP.get(step)
        if config_name is None:
            supported = ", ".join(sorted(_CAP_CONFIG_BY_STEP))
            raise ValueError(f"Unknown capped model step {step!r}; expected one of: {supported}")
        return int(getattr(self.settings.cap, config_name))

    def invoke(
        self,
        step: str,
        model_input: Any,
        *,
        format_schema: Optional[dict[str, Any] | str] = None,
        invoke_kwargs: Optional[dict[str, Any]] = None,
    ) -> BudgetCallResult:
        """Request a capped base call and, when truncated, one free retry.

        Args:
            step: Pipeline model step used to choose the configured cap.
            model_input: Messages or prompt passed to ``ChatOllama.invoke``.
            format_schema: Optional Ollama structured-output schema.
            invoke_kwargs: Optional keyword arguments for ``invoke``.

        Returns:
            Response plus the budget-exhaustion signal for the graph state.

        Raises:
            ValueError: An unsupported step or protected invocation option.
            Exception: Model construction/invocation errors are propagated to
                the caller's shared error handler. Construction failures use
                no budget. Invocation/metrics failures end the question; only
                reset for a new question permits further calls. B-1 has no
                failed-call record, so failed usage is not fabricated here.
        """

        if self._failed:
            raise RuntimeError("Previous model call failed; end this question before reset()")
        # These options could override settings or invalidate callback accounting.
        protected = {
            "num_predict", "options", "model", "temperature", "num_ctx",
            "callbacks", "format", "keep_alive", "base_url",
        }
        if protected.intersection(invoke_kwargs or {}):
            raise ValueError("Model settings and callbacks cannot be overridden via invoke_kwargs")
        cap = self.cap_for_step(step)
        if self._n_calls >= self.settings.budget.max_calls:
            self._budget_exhausted = True
            return BudgetCallResult(response=None, budget_exhausted=True)

        response, record = self._invoke_once(
            step=step,
            model_input=model_input,
            cap=cap,
            is_retry=False,
            format_schema=format_schema,
            invoke_kwargs=invoke_kwargs,
        )

        if record.truncated:
            retry_cap = cap * self.settings.cap.retry_multiplier
            response, _ = self._invoke_once(
                step=step,
                model_input=model_input,
                cap=retry_cap,
                is_retry=True,
                format_schema=format_schema,
                invoke_kwargs=invoke_kwargs,
            )

        return BudgetCallResult(response=response, budget_exhausted=False)

    def _invoke_once(
        self,
        *,
        step: str,
        model_input: Any,
        cap: int,
        is_retry: bool,
        format_schema: Optional[dict[str, Any] | str],
        invoke_kwargs: Optional[dict[str, Any]],
    ) -> tuple[Any, CallRecord]:
        """Perform one callback-instrumented ChatOllama invocation."""

        calls_before = len(self.callback.calls)
        self.callback.set_call_context(step=step, cap=cap, is_retry=is_retry)
        llm = self._llm_factory(
            settings=self.settings,
            num_predict=cap,
            format_schema=format_schema,
            callbacks=[self.callback],
        )
        # Construction errors above have not invoked the model and consume no
        # budget. Once invocation begins, errors terminate this question: B-1
        # does not emit CallRecords on errors, so silently continuing could
        # allow unmeasured attempts past the budget. Do not fabricate usage.
        try:
            response = llm.invoke(model_input, **(invoke_kwargs or {}))
            if len(self.callback.calls) != calls_before + 1:
                raise RuntimeError(
                    "ChatOllama invocation did not produce exactly one CallRecord; "
                    "ensure the B-1 MetricsCallbackHandler is attached"
                )
        except Exception:
            self._failed = True
            raise
        finally:
            self._n_calls = self.callback.n_budget_calls
            self._n_retries = self.callback.n_retries
        return response, self.callback.calls[-1]

    @property
    def n_calls(self) -> int:
        """Callback-recorded base calls charged to this question's budget."""

        return self._n_calls

    @property
    def n_retries(self) -> int:
        """Number of truncation retries, excluded from the base-call budget."""

        return self._n_retries

    @property
    def calls(self) -> list[CallRecord]:
        """Callback-produced records for base calls and retries."""

        return self.callback.calls

    @property
    def budget_exhausted(self) -> bool:
        """Whether a requested base call was blocked after budget exhaustion."""

        return self._budget_exhausted

    def reset(self) -> None:
        """Clear all per-question budget and callback state."""

        self._n_calls = 0
        self._n_retries = 0
        self._budget_exhausted = False
        self._failed = False
        self.callback.reset()
