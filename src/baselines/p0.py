"""P0 and P0-raw single-call baseline pipelines.

References:
- AGENTS.md (ChatOllama only, config-driven, (qid, question) -> PipelineResult)
- docs/INTERFACES.md §4, §6
- docs/SPEC.md §3
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from src.baselines.prompts import build_p0_prompt, build_p0_raw_prompt
from src.common.types import PipelineResult
from src.config.callbacks import MetricsCallbackHandler
from src.config.llm import create_chat_ollama
from src.config.settings import Settings, get_settings
from src.eval.scorer import UNANSWERABLE, extract_final_answer, is_abstained


def extract_message_text(message_or_text: Any) -> str:
    """Extract plain text from a LangChain AIMessage, dict, list of blocks, or str."""
    if isinstance(message_or_text, str):
        return message_or_text

    content = (
        message_or_text.get("content")
        if isinstance(message_or_text, dict)
        else getattr(message_or_text, "content", None)
    )
    if content is None:
        return str(message_or_text)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def build_baseline_result(
    raw_output: str,
    callback: MetricsCallbackHandler,
    *,
    retrieved_doc_ids: Optional[list[list[str]]] = None,
    error: Optional[str] = None,
) -> PipelineResult:
    """Convert raw baseline output and callback metrics into a ``PipelineResult``."""
    if raw_output:
        abstained = is_abstained(raw_output)
        answer = UNANSWERABLE if abstained else extract_final_answer(raw_output)
    else:
        abstained = False
        answer = ""

    return PipelineResult(
        answer=answer,
        abstained=abstained,
        budget_exhausted=False,
        route=None,
        decisions=[],
        n_calls=callback.n_budget_calls,
        n_retries=callback.n_retries,
        calls=list(callback.calls),
        retrieved_doc_ids=retrieved_doc_ids if retrieved_doc_ids is not None else [],
        kept_windows=[],
        loop_breaks=0,
        error=error,
    )


def _run_single_call(
    qid: str,
    question: str,
    *,
    prompt_builder: Callable[[str], str],
    settings: Optional[Settings] = None,
    llm_factory: Callable[..., Any] = create_chat_ollama,
    base_url: Optional[str] = None,
) -> PipelineResult:
    """Execute a single uncapped ChatOllama baseline call and standardize output."""
    _ = qid  # Logged by external runner; not passed to the model.
    callback = MetricsCallbackHandler(current_step="p0", cap=None, is_retry=False)

    try:
        cfg = settings or get_settings()
        prompt = prompt_builder(question)
        callback.set_call_context(step="p0", cap=None, is_retry=False, parse_ok=None)
        llm = llm_factory(
            settings=cfg,
            num_predict=None,
            callbacks=[callback],
            base_url=base_url,
        )
        response = llm.invoke(prompt)
        raw_text = extract_message_text(response)
        return build_baseline_result(raw_text, callback, retrieved_doc_ids=[], error=None)
    except Exception as exc:
        return build_baseline_result(
            "",
            callback,
            retrieved_doc_ids=[],
            error=f"{type(exc).__name__}: {exc}",
        )


def run_p0(
    qid: str,
    question: str,
    *,
    settings: Optional[Settings] = None,
    llm_factory: Callable[..., Any] = create_chat_ollama,
    base_url: Optional[str] = None,
) -> PipelineResult:
    """Run the P0 baseline (minimal shared instructions + question).

    Args:
        qid: Question identifier (not exposed to the model).
        question: Raw English question text.
        settings: Optional Settings instance; defaults to ``get_settings()``.
        llm_factory: ChatOllama factory (override in tests).
        base_url: Optional Ollama endpoint URL override.

    Returns:
        Standardized ``PipelineResult``.
    """
    return _run_single_call(
        qid=qid,
        question=question,
        prompt_builder=build_p0_prompt,
        settings=settings,
        llm_factory=llm_factory,
        base_url=base_url,
    )


def run_p0_raw(
    qid: str,
    question: str,
    *,
    settings: Optional[Settings] = None,
    llm_factory: Callable[..., Any] = create_chat_ollama,
    base_url: Optional[str] = None,
) -> PipelineResult:
    """Run the P0-raw baseline (raw question only, no format instructions).

    Args:
        qid: Question identifier (not exposed to the model).
        question: Raw English question text.
        settings: Optional Settings instance; defaults to ``get_settings()``.
        llm_factory: ChatOllama factory (override in tests).
        base_url: Optional Ollama endpoint URL override.

    Returns:
        Standardized ``PipelineResult``.
    """
    return _run_single_call(
        qid=qid,
        question=question,
        prompt_builder=build_p0_raw_prompt,
        settings=settings,
        llm_factory=llm_factory,
        base_url=base_url,
    )
