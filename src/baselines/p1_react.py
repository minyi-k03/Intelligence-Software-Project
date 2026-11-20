"""P1 ReAct baseline built with LangChain ``create_agent``.

References:
- AGENTS.md (ChatOllama only, shared tools/error handling, (qid, question) -> PipelineResult)
- docs/SPEC.md §3 (create_agent, 2 tools, max 7 steps, shared instruction)
- docs/INTERFACES.md §3, §4, §6
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from src.baselines.p0 import build_baseline_result, extract_message_text
from src.baselines.prompts import build_p1_system_prompt, validate_question
from src.common.types import PipelineResult
from src.config.callbacks import MetricsCallbackHandler
from src.config.llm import create_chat_ollama
from src.config.settings import Settings, get_settings
from src.tools.bm25_search import bm25_search, format_search_results
from src.tools.calculator import get_calculator_tool


def _default_agent_factory(
    *,
    model: Any,
    tools: list[Any],
    system_prompt: str,
) -> Any:
    """Create a LangChain v1.0+ ReAct agent backed by LangGraph."""
    try:
        from langchain.agents import create_agent
    except ImportError as exc:
        raise RuntimeError(
            "langchain>=1.0.0 is required for langchain.agents.create_agent. "
            "Install via: pip install -r requirements.txt"
        ) from exc

    return create_agent(model=model, tools=tools, system_prompt=system_prompt)


def _make_tracked_bm25_tool(
    retrieval_k: int,
    retrieved_rounds: list[list[str]],
    corpus_path: Optional[str | Path] = None,
) -> Any:
    """Build a ``bm25_search`` LangChain tool that logs ``doc_id``s out-of-band.

    The tool schema exposes only ``query`` to the model (hiding ``k``), and
    returns title + text only (hiding ``doc_id`` prefixes and scores), while
    recording retrieved ``doc_id`` lists per search round into ``retrieved_rounds``.
    """
    try:
        from langchain_core.tools import tool
    except ImportError as exc:
        raise RuntimeError(
            "langchain-core is required to create the LangChain bm25_search tool."
        ) from exc

    @tool("bm25_search")
    def bm25_search_tool(query: str) -> str:
        """Search the offline Wikipedia paragraph corpus using BM25 keyword retrieval."""
        results = bm25_search(query=query, k=retrieval_k, corpus_path=corpus_path)
        if isinstance(results, list):
            retrieved_rounds.append([doc["doc_id"] for doc in results])
        return format_search_results(results)

    return bm25_search_tool


def _extract_last_ai_text(agent_output: Any) -> str:
    """Extract text from the final AI/assistant message returned by an agent."""
    if isinstance(agent_output, str):
        return agent_output

    messages = (
        agent_output.get("messages")
        if isinstance(agent_output, dict)
        else getattr(agent_output, "messages", None)
    )
    if not isinstance(messages, list) or not messages:
        return extract_message_text(agent_output)

    for msg in reversed(messages):
        msg_type = (
            msg.get("type") or msg.get("role")
            if isinstance(msg, dict)
            else getattr(msg, "type", None)
        )
        if msg_type in ("ai", "assistant"):
            return extract_message_text(msg)

    return extract_message_text(messages[-1])


def _is_recursion_limit_error(exc: BaseException) -> bool:
    """Return True if an exception indicates LangGraph recursion_limit exhaustion."""
    cls_names = {cls.__name__ for cls in type(exc).__mro__}
    if "GraphRecursionError" in cls_names:
        return True
    msg = str(exc).lower()
    return "recursion limit" in msg or "recursion_limit" in msg


def run_p1(
    qid: str,
    question: str,
    *,
    settings: Optional[Settings] = None,
    llm_factory: Callable[..., Any] = create_chat_ollama,
    agent_factory: Optional[Callable[..., Any]] = None,
    corpus_path: Optional[str | Path] = None,
    base_url: Optional[str] = None,
) -> PipelineResult:
    """Run the P1 ReAct baseline using ``create_agent`` and two shared tools.

    Args:
        qid: Question identifier (not exposed to the model).
        question: Raw English question text.
        settings: Optional Settings instance; defaults to ``get_settings()``.
        llm_factory: ChatOllama factory (override in tests).
        agent_factory: Optional ``create_agent``-compatible factory (override in tests).
        corpus_path: Optional path to ``corpus.jsonl``.
        base_url: Optional Ollama endpoint URL override.

    Returns:
        Standardized ``PipelineResult``.
    """
    _ = qid  # Logged by external runner; not passed to the model.
    callback = MetricsCallbackHandler(current_step="p1_step", cap=None, is_retry=False)
    retrieved_rounds: list[list[str]] = []
    cfg: Optional[Settings] = settings

    try:
        cfg = settings or get_settings()
        clean_question = validate_question(question)

        callback.set_call_context(step="p1_step", cap=None, is_retry=False, parse_ok=None)
        # Attach callback via invoke config below to avoid double-counting inside create_agent.
        llm = llm_factory(
            settings=cfg,
            num_predict=None,
            callbacks=None,
            base_url=base_url,
        )

        bm25_tool = _make_tracked_bm25_tool(
            retrieval_k=cfg.retrieval.k,
            retrieved_rounds=retrieved_rounds,
            corpus_path=corpus_path,
        )
        calc_tool = get_calculator_tool()

        factory = agent_factory or _default_agent_factory
        agent = factory(
            model=llm,
            tools=[bm25_tool, calc_tool],
            system_prompt=build_p1_system_prompt(),
        )

        # In LangGraph, each ReAct step uses 2 graph steps (model -> tools) + 1 final model step.
        recursion_limit = 2 * cfg.p1.max_steps + 1
        agent_output = agent.invoke(
            {"messages": [{"role": "user", "content": clean_question}]},
            config={"callbacks": [callback], "recursion_limit": recursion_limit},
        )
        raw_text = _extract_last_ai_text(agent_output)
        return build_baseline_result(
            raw_text,
            callback,
            retrieved_doc_ids=retrieved_rounds,
            error=None,
        )
    except Exception as exc:
        if _is_recursion_limit_error(exc):
            max_steps = getattr(getattr(cfg, "p1", None), "max_steps", None)
            err_msg = (
                f"RecursionLimitExceeded: reached max_steps={max_steps}"
                if max_steps is not None
                else f"RecursionLimitExceeded: {exc}"
            )
        else:
            err_msg = f"{type(exc).__name__}: {exc}"
        return build_baseline_result(
            "",
            callback,
            retrieved_doc_ids=retrieved_rounds,
            error=err_msg,
        )
