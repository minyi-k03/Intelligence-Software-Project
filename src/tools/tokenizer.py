"""Shared BM25 tokenizer and hyperparameters for retrieval and compression.

References:
- docs/INTERFACES.md §3 (BM25 search and compression must share the same tokenizer)
- docs/ADR/0002-shared-corpus-bm25.md
- docs/IMPLEMENTATION_PLAN.md §5 (PR B-2a)
"""

from __future__ import annotations

import re
from typing import Callable
import unicodedata

Tokenizer = Callable[[str], list[str]]

# Shared BM25 Okapi hyperparameters across bm25_search (B-2b) and compress (C-1).
BM25_K1: float = 1.5
BM25_B: float = 0.75

# Standard English stopwords + apostrophe contraction/possessive fragments (offline, zero dependencies).
ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "couldn",
        "d",
        "did",
        "didn",
        "do",
        "does",
        "doesn",
        "doing",
        "don",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn",
        "has",
        "hasn",
        "have",
        "haven",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "isn",
        "it",
        "its",
        "itself",
        "just",
        "ll",
        "m",
        "ma",
        "me",
        "more",
        "most",
        "mustn",
        "my",
        "myself",
        "needn",
        "no",
        "nor",
        "not",
        "now",
        "o",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "re",
        "s",
        "same",
        "shan",
        "she",
        "should",
        "shouldn",
        "so",
        "some",
        "such",
        "t",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "ve",
        "very",
        "was",
        "wasn",
        "we",
        "were",
        "weren",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "won",
        "wouldn",
        "y",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)

_DIGIT_COMMA_PATTERN = re.compile(r"(?<=\d),(?=\d)")
_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    r"""Tokenize English text deterministically for BM25 indexing and scoring.

    Steps:
    1. Validate string input and return `[]` on empty strings.
    2. Normalize Unicode via NFKC and convert to lowercase.
    3. Strip thousands-separator commas inside numbers (e.g. `100,000` -> `100000`).
    4. Extract alphanumeric word tokens (`[^\W_]+`), splitting punctuation,
       hyphens, underscores, and possessives (`Arthur's` -> `['arthur', 's']`).
    5. Filter out `ENGLISH_STOPWORDS` when `remove_stopwords=True`. If every
       extracted token in a non-empty phrase is a stopword (e.g. `"The Who"`),
       fall back to the unfiltered tokens so the query/span remains searchable.

    Args:
        text: Input string (query, paragraph, or sentence window).
        remove_stopwords: Whether to filter out common English stopwords. Defaults to True.

    Returns:
        list[str]: Normalized list of tokens in order of appearance.

    Raises:
        TypeError: If `text` is not a `str`.
    """
    if not isinstance(text, str):
        raise TypeError(f"tokenize() expects a str, got {type(text).__name__}")
    if not text:
        return []

    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = _DIGIT_COMMA_PATTERN.sub("", normalized)
    tokens = _TOKEN_PATTERN.findall(normalized)
    if not tokens:
        return []

    if remove_stopwords:
        filtered = [tok for tok in tokens if tok not in ENGLISH_STOPWORDS]
        return filtered if filtered else tokens
    return tokens
