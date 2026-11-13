"""Factory for creating ChatOllama instances strictly bound to configurations.

References:
- AGENTS.md (no hardcoded model tag, temperature=0, num_ctx from configs; use langchain-ollama)
- docs/INTERFACES.md §2
- docs/IMPLEMENTATION_PLAN.md §5 (PR B-1)
"""

from __future__ import annotations

import os
from typing import Any, Optional

from src.config.settings import Settings, get_settings

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None  # type: ignore


def get_llm_kwargs(
    settings: Optional[Settings] = None,
    num_predict: Optional[int] = None,
    format_schema: Optional[dict[str, Any] | str] = None,
    callbacks: Optional[list[Any]] = None,
    base_url: Optional[str] = None,
) -> dict[str, Any]:
    """Assemble validated parameters for ChatOllama creation."""
    cfg = settings or get_settings()

    # Enforce strict temperature rule from AGENTS.md
    if isinstance(cfg.model.temperature, bool) or cfg.model.temperature != 0:
        raise ValueError(
            f"model.temperature must strictly be 0 (AGENTS.md). Found: {cfg.model.temperature!r}"
        )

    if num_predict is not None:
        if isinstance(num_predict, bool) or not isinstance(num_predict, int) or num_predict <= 0:
            raise ValueError(f"num_predict must be a positive int when set. Got: {num_predict!r}")

    url = base_url or os.getenv("OLLAMA_HOST") or "http://localhost:11434"

    kwargs: dict[str, Any] = {
        "model": cfg.model.tag,
        "temperature": cfg.model.temperature,
        "num_ctx": cfg.model.num_ctx,
        "keep_alive": cfg.model.keep_alive,
        "base_url": url,
    }

    if num_predict is not None:
        kwargs["num_predict"] = num_predict

    if format_schema is not None:
        kwargs["format"] = format_schema

    if callbacks:
        kwargs["callbacks"] = callbacks

    return kwargs


def create_chat_ollama(
    settings: Optional[Settings] = None,
    num_predict: Optional[int] = None,
    format_schema: Optional[dict[str, Any] | str] = None,
    callbacks: Optional[list[Any]] = None,
    base_url: Optional[str] = None,
) -> Any:
    """Create a LangChain ChatOllama instance strictly configured via settings.

    Args:
        settings: Settings instance. Defaults to cached global settings.
        num_predict: Output token cap limit for this call (None for uncapped ablation).
        format_schema: JSON schema dict or 'json' for Ollama structured output.
        callbacks: List of LangChain callback handlers (e.g., MetricsCallbackHandler).
        base_url: Ollama base endpoint URL.

    Returns:
        ChatOllama: Configured LangChain chat model from langchain-ollama.

    Raises:
        RuntimeError: If langchain-ollama is not installed.
        ValueError: If temperature is not 0 or num_predict is invalid.
    """
    kwargs = get_llm_kwargs(
        settings=settings,
        num_predict=num_predict,
        format_schema=format_schema,
        callbacks=callbacks,
        base_url=base_url,
    )

    if ChatOllama is None:
        raise RuntimeError(
            "langchain-ollama is required (fallback to langchain-community is disabled "
            "to keep token/truncation metadata consistent). Install via: pip install -r requirements.txt"
        )

    return ChatOllama(**kwargs)
