"""Offline BM25 Okapi corpus search shared across P1 and M (EBA) pipelines."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.config.settings import REPO_ROOT, get_settings
from src.tools.tokenizer import BM25_B, BM25_K1, Tokenizer, tokenize

DEFAULT_CORPUS_PATH = Path("data/frozen/corpus.jsonl")

_INDEX_CACHE: dict[Path, "BM25Index"] = {}


def _resolve_corpus_path(corpus_path: str | Path | None) -> Path:
    """Resolve the corpus path relative to the workspace or repository root."""
    path = Path(corpus_path) if corpus_path is not None else DEFAULT_CORPUS_PATH
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _resolve_default_k(k: int | None) -> int:
    """Resolve retrieval top-k from settings when omitted and validate it."""
    if k is None:
        resolved = get_settings().retrieval.k
    elif isinstance(k, bool) or not isinstance(k, int):
        raise ValueError("k must be a positive integer.")
    else:
        resolved = k
    if resolved < 1:
        raise ValueError("k must be at least 1.")
    return resolved


def _normalize_paragraph(record: Mapping[str, Any], line_number: int) -> dict[str, Any]:
    """Validate and normalize a corpus JSONL paragraph record."""
    doc_id = record.get("doc_id")
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError(f"Corpus line {line_number}: 'doc_id' must be a non-empty string.")

    title = record.get("title")
    if not isinstance(title, str):
        raise ValueError(f"Corpus line {line_number}: 'title' must be a string.")

    raw_sentences = record.get("sentences")
    raw_text = record.get("text")

    if raw_sentences is None:
        if not isinstance(raw_text, str):
            raise ValueError(
                f"Corpus line {line_number}: paragraph must provide 'sentences' or 'text'."
            )
        sentences = [raw_text] if raw_text else []
    elif isinstance(raw_sentences, list) and all(isinstance(item, str) for item in raw_sentences):
        sentences = list(raw_sentences)
    else:
        raise ValueError(
            f"Corpus line {line_number}: 'sentences' must be a list of strings."
        )

    if raw_text is None:
        text = " ".join(sentences)
    elif isinstance(raw_text, str):
        text = raw_text
    else:
        raise ValueError(f"Corpus line {line_number}: 'text' must be a string.")

    return {
        "doc_id": doc_id,
        "title": title,
        "sentences": sentences,
        "text": text,
    }


def _load_corpus(corpus_path: Path) -> list[dict[str, Any]]:
    """Load and validate paragraphs from a JSONL corpus file."""
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    paragraphs: list[dict[str, Any]] = []
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Corpus line {line_number}: invalid JSON ({exc.msg})."
                ) from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"Corpus line {line_number}: expected a JSON object.")
            paragraphs.append(_normalize_paragraph(payload, line_number))

    return paragraphs


class BM25Index:
    """Pure-Python inverted-index Okapi BM25 retriever over corpus paragraphs."""

    def __init__(
        self,
        paragraphs: Sequence[Mapping[str, Any]],
        tokenizer: Tokenizer = tokenize,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        """Build an Okapi BM25 index over validated corpus paragraphs.

        Args:
            paragraphs: Sequence of paragraph mappings with ``doc_id``, ``title``,
                ``sentences``, and ``text`` fields.
            tokenizer: Callable turning text into normalized tokens.
            k1: BM25 term-frequency saturation parameter.
            b: BM25 document-length normalization parameter.
        """
        self.tokenizer = tokenizer
        self.k1 = float(k1)
        self.b = float(b)

        self._paragraphs: list[dict[str, Any]] = []
        self._doc_lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        total_length = 0
        for idx, item in enumerate(paragraphs, start=1):
            normalized = _normalize_paragraph(item, idx)
            self._paragraphs.append(normalized)

            searchable = f"{normalized['title']} {normalized['text']}".strip()
            tokens = self.tokenizer(searchable)
            doc_len = len(tokens)
            self._doc_lengths.append(doc_len)
            total_length += doc_len

            doc_idx = len(self._paragraphs) - 1
            for term, freq in Counter(tokens).items():
                self._postings[term].append((doc_idx, freq))

        self._num_docs = len(self._paragraphs)
        self._avgdl = (total_length / self._num_docs) if self._num_docs > 0 else 0.0
        self._idf: dict[str, float] = {
            term: math.log(1.0 + (self._num_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self._postings.items()
        }

    @classmethod
    def from_jsonl(
        cls,
        corpus_path: str | Path | None = None,
        tokenizer: Tokenizer = tokenize,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> "BM25Index":
        """Load a JSONL corpus file and build a ``BM25Index`` instance.

        Args:
            corpus_path: Path to ``corpus.jsonl``. Uses ``data/frozen/corpus.jsonl``
                relative to repository root when omitted.
            tokenizer: Callable turning text into normalized tokens.
            k1: BM25 term-frequency saturation parameter.
            b: BM25 document-length normalization parameter.

        Returns:
            Initialized ``BM25Index`` ready for queries.
        """
        resolved_path = _resolve_corpus_path(corpus_path)
        paragraphs = _load_corpus(resolved_path)
        return cls(paragraphs=paragraphs, tokenizer=tokenizer, k1=k1, b=b)

    def search(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        """Rank indexed paragraphs by Okapi BM25 score descending.

        Args:
            query: Natural-language search query.
            k: Maximum number of results to return. Defaults to ``settings.retrieval.k``.

        Returns:
            List of dicts sorted by descending ``score``, each containing
            ``doc_id``, ``title``, ``sentences``, ``text``, and ``score``.
        """
        if not isinstance(query, str):
            raise ValueError("query must be a string.")
        top_k = _resolve_default_k(k)
        if not query.strip() or self._num_docs == 0 or self._avgdl <= 0.0:
            return []

        query_tokens = self.tokenizer(query)
        if not query_tokens:
            return []

        scores: dict[int, float] = defaultdict(float)
        query_counts = Counter(query_tokens)

        for term, q_freq in query_counts.items():
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for doc_idx, tf in postings:
                doc_len = self._doc_lengths[doc_idx]
                norm = self.k1 * (1.0 - self.b + self.b * (doc_len / self._avgdl))
                tf_score = (tf * (self.k1 + 1.0)) / (tf + norm)
                scores[doc_idx] += q_freq * idf * tf_score

        if not scores:
            return []

        ranked = sorted(
            ((doc_idx, score) for doc_idx, score in scores.items() if score > 0.0),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]

        results: list[dict[str, Any]] = []
        for doc_idx, score in ranked:
            paragraph = self._paragraphs[doc_idx]
            results.append(
                {
                    "doc_id": paragraph["doc_id"],
                    "title": paragraph["title"],
                    "sentences": list(paragraph["sentences"]),
                    "text": paragraph["text"],
                    "score": float(score),
                }
            )
        return results


def get_default_index(
    corpus_path: str | Path | None = None,
    reload: bool = False,
) -> BM25Index:
    """Return a cached ``BM25Index`` keyed by resolved corpus path.

    Args:
        corpus_path: Optional path to ``corpus.jsonl``.
        reload: When True, rebuild the cached index from disk.

    Returns:
        Cached or freshly constructed ``BM25Index``.
    """
    resolved_path = _resolve_corpus_path(corpus_path)
    if reload or resolved_path not in _INDEX_CACHE:
        _INDEX_CACHE[resolved_path] = BM25Index.from_jsonl(resolved_path)
    return _INDEX_CACHE[resolved_path]


def bm25_search(
    query: str,
    k: int | None = None,
    *,
    index: BM25Index | None = None,
    corpus_path: str | Path | None = None,
) -> list[dict[str, Any]] | str:
    """Search the offline corpus safely, returning ``ERROR: <reason>`` on failure.

    Args:
        query: Search query string.
        k: Top-k paragraph count. Defaults to ``settings.retrieval.k``.
        index: Optional prebuilt ``BM25Index`` (useful for testing or shared state).
        corpus_path: Optional path to ``corpus.jsonl`` when loading the default index.

    Returns:
        Ranked list of ``{doc_id, title, sentences, text, score}`` dicts, or
        an ``ERROR: <reason>`` string on invalid input or missing/corrupt corpus.
    """
    if not isinstance(query, str):
        return "ERROR: query must be a string."
    if not query.strip():
        return "ERROR: query must not be empty."

    try:
        top_k = _resolve_default_k(k)
        active_index = index if index is not None else get_default_index(corpus_path)
        return active_index.search(query=query, k=top_k)
    except Exception as exc:
        return f"ERROR: {exc}"


def format_search_results(results: list[dict[str, Any]] | str) -> str:
    """Format ``bm25_search`` output into an English observation string for ReAct.

    Args:
        results: Return value from ``bm25_search``.

    Returns:
        Pass-through error string, empty-result notice, or numbered paragraphs.
    """
    if isinstance(results, str):
        return results
    if not results:
        return "No relevant documents found."

    formatted: list[str] = []
    for idx, doc in enumerate(results, start=1):
        formatted.append(
            f"[{idx}] {doc['title']} (doc_id: {doc['doc_id']}, score: {doc['score']:.4f})\n"
            f"{doc['text']}"
        )
    return "\n\n".join(formatted)


def get_bm25_search_tool(
    corpus_path: str | Path | None = None,
    default_k: int | None = None,
) -> Any:
    """Build a LangChain ``@tool`` wrapping ``bm25_search`` for ReAct/M pipelines.

    Args:
        corpus_path: Optional corpus JSONL path bound to the tool instance.
        default_k: Optional default ``k`` override; falls back to config settings.

    Returns:
        LangChain tool object wrapping ``bm25_search``.

    Raises:
        RuntimeError: If ``langchain-core`` is not installed in the current environment.
    """
    try:
        from langchain_core.tools import tool
    except ImportError as exc:
        raise RuntimeError(
            "langchain-core is required to create the LangChain bm25_search tool."
        ) from exc

    @tool("bm25_search")
    def bm25_search_tool(query: str, k: int | None = None) -> str:
        """Search the offline Wikipedia paragraph corpus using BM25 keyword retrieval."""
        effective_k = default_k if k is None else k
        results = bm25_search(query=query, k=effective_k, corpus_path=corpus_path)
        return format_search_results(results)

    return bm25_search_tool
