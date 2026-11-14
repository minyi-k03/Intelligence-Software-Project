"""Unit tests for shared tools (PR B-2a: tokenizer, PR B-2b: calculator & BM25).

References:
- docs/INTERFACES.md §3
- docs/IMPLEMENTATION_PLAN.md §5 (PR B-2a, PR B-2b)
"""

from pathlib import Path
from typing import Any
import unittest

from src.tools import BM25_B, BM25_K1, ENGLISH_STOPWORDS, Tokenizer, tokenize


class TestTokenizer(unittest.TestCase):
    def test_empty_and_punctuation_inputs(self):
        """Empty, whitespace, or punctuation-only strings should return an empty list."""
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize("   \n\t  "), [])
        self.assertEqual(tokenize("... --- !!! ??? ___"), [])

    def test_non_string_raises_type_error(self):
        """Passing non-string input should raise TypeError rather than failing silently."""
        with self.assertRaises(TypeError):
            tokenize(None)  # type: ignore
        with self.assertRaises(TypeError):
            tokenize(123)  # type: ignore

    def test_lowercasing_possessives_and_stopword_removal(self):
        """Verify standard English QA sentence tokenization, curly apostrophes, and stopwords."""
        text_ascii = "Arthur's Magazine was an American literary periodical."
        text_curly = "Arthur’s Magazine was an American literary periodical."
        expected = ["arthur", "magazine", "american", "literary", "periodical"]
        self.assertEqual(tokenize(text_ascii), expected)
        self.assertEqual(tokenize(text_curly), expected)

    def test_keep_stopwords_option(self):
        """Verify remove_stopwords=False preserves stopwords."""
        text = "It was published in Philadelphia."
        tokens = tokenize(text, remove_stopwords=False)
        self.assertEqual(tokens, ["it", "was", "published", "in", "philadelphia"])

    def test_all_stopword_fallback(self):
        """Phrases consisting entirely of stopwords (e.g., 'The Who') must not collapse to []."""
        self.assertEqual(tokenize("The Who"), ["the", "who"])
        self.assertEqual(
            tokenize("To be, or not to be?"),
            ["to", "be", "or", "not", "to", "be"],
        )

    def test_numbers_commas_hyphens_and_unicode(self):
        """Verify formatted numbers (100,000 -> 100000), dates, hyphens, and Unicode letters."""
        text = "In May 1, 1846, Zürich-based 19th-century authors sold 100,000 copies."
        tokens = tokenize(text)
        self.assertEqual(
            tokens,
            [
                "may",
                "1",
                "1846",
                "zürich",
                "based",
                "19th",
                "century",
                "authors",
                "sold",
                "100000",
                "copies",
            ],
        )

    def test_nfkc_normalization(self):
        """Full-width / compatibility Unicode characters should normalize cleanly."""
        text = "Ａｌａｎ Ｔｕｒｉｎｇ １９５０"
        tokens = tokenize(text)
        self.assertEqual(tokens, ["alan", "turing", "1950"])

    def test_callable_signature_constants_and_determinism(self):
        """Tokenizer must match Tokenizer = Callable[[str], list[str]] and expose BM25 constants."""
        fn: Tokenizer = tokenize
        sample = "Who edited Arthur's Magazine in the 19th century?"
        first_run = fn(sample)
        second_run = fn(sample)
        self.assertEqual(first_run, second_run)
        self.assertEqual(first_run, ["edited", "arthur", "magazine", "19th", "century"])
        self.assertIn("the", ENGLISH_STOPWORDS)
        self.assertEqual(BM25_K1, 1.5)
        self.assertEqual(BM25_B, 0.75)


class TestCalculator(unittest.TestCase):
    """Tests for the AST-based safe arithmetic calculator."""

    def test_basic_arithmetic_and_precedence(self) -> None:
        from src.tools import calculate

        self.assertEqual(calculate("2 + 3 * 4"), "14")
        self.assertEqual(calculate("(2 + 3) * 4"), "20")
        self.assertEqual(calculate("15 / 4"), "3.75")
        self.assertEqual(calculate("15 // 4"), "3")
        self.assertEqual(calculate("15 % 4"), "3")
        self.assertEqual(calculate("2 ** 5"), "32")
        self.assertEqual(calculate("-7 + +3"), "-4")

    def test_thousands_commas_and_unicode_operators(self) -> None:
        from src.tools import calculate

        self.assertEqual(calculate("1,200 + 3,450"), "4650")
        self.assertEqual(calculate("12 × 3 − 6 ÷ 2"), "33")
        self.assertEqual(calculate("1,000,000 / 250,000"), "4")

    def test_allowed_functions_and_constants(self) -> None:
        from src.tools import calculate

        self.assertEqual(calculate("sqrt(144)"), "12")
        self.assertEqual(calculate("abs(-42)"), "42")
        self.assertEqual(calculate("round(3.14159, 2)"), "3.14")
        self.assertEqual(calculate("max(3, 9, 5) + min(4, 2)"), "11")
        self.assertEqual(calculate("floor(3.9) + ceil(3.1)"), "7")
        self.assertEqual(calculate("0.1 + 0.2"), "0.3")

    def test_errors_return_error_string_without_raising(self) -> None:
        from src.tools import calculate

        for expr in [
            "",
            "   ",
            "1 / 0",
            "1 % 0",
            "2 + * 3",
            "sqrt(-1)",
            "unknown_var + 1",
            "__import__('os').system('echo hi')",
            "[1, 2, 3]",
            "2 ** 10000",
        ]:
            result = calculate(expr)
            self.assertTrue(
                result.startswith("ERROR: "),
                f"Expected ERROR prefix for {expr!r}, got {result!r}",
            )

    def test_non_string_input_returns_error(self) -> None:
        from src.tools import calculate

        result = calculate(123)  # type: ignore[arg-type]
        self.assertTrue(result.startswith("ERROR: "))

    def test_langchain_tool_wrapper_or_clean_missing_dependency(self) -> None:
        from src.tools import get_calculator_tool

        try:
            tool = get_calculator_tool()
        except RuntimeError as exc:
            self.assertIn("langchain-core", str(exc))
        else:
            self.assertEqual(tool.invoke({"expression": "6 * 7"}), "42")


class TestBM25Search(unittest.TestCase):
    """Tests for pure-Python Okapi BM25 retrieval and error handling."""

    SAMPLE_PARAGRAPHS = [
        {
            "doc_id": "hotpot_1",
            "title": "First for Women",
            "sentences": [
                "First for Women is an American women's magazine.",
                "It was founded in 1989 and covers health and lifestyle.",
            ],
            "text": (
                "First for Women is an American women's magazine. "
                "It was founded in 1989 and covers health and lifestyle."
            ),
            "origin": "hotpotqa",
        },
        {
            "doc_id": "hotpot_2",
            "title": "Arthur's Magazine",
            "sentences": [
                "Arthur's Magazine was an American literary periodical.",
                "It was published in Philadelphia in the 19th century.",
            ],
            "text": (
                "Arthur's Magazine was an American literary periodical. "
                "It was published in Philadelphia in the 19th century."
            ),
            "origin": "hotpotqa",
        },
        {
            "doc_id": "squad_1",
            "title": "Nikola Tesla",
            "sentences": [
                "Nikola Tesla was a Serbian-American inventor and electrical engineer.",
                "He contributed to the modern alternating current electricity supply system.",
            ],
            "text": (
                "Nikola Tesla was a Serbian-American inventor and electrical engineer. "
                "He contributed to the modern alternating current electricity supply system."
            ),
            "origin": "squad",
        },
        {
            "doc_id": "squad_2",
            "title": "Alternating Current",
            "sentences": [
                "Alternating current periodically reverses direction.",
                "Tesla designed motors powered by alternating current.",
            ],
            "text": (
                "Alternating current periodically reverses direction. "
                "Tesla designed motors powered by alternating current."
            ),
            "origin": "squad",
        },
    ]

    def _write_temp_corpus(self, lines: list[str]) -> tuple[Path, Any]:
        import tempfile

        tmp_dir = tempfile.TemporaryDirectory()
        corpus_path = Path(tmp_dir.name) / "corpus.jsonl"
        corpus_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return corpus_path, tmp_dir

    def test_from_jsonl_search_schema_and_ranking(self) -> None:
        import json
        from src.tools import BM25Index, bm25_search

        lines = [json.dumps(item) for item in self.SAMPLE_PARAGRAPHS]
        corpus_path, tmp_dir = self._write_temp_corpus(lines)
        try:
            index = BM25Index.from_jsonl(corpus_path)
            results = bm25_search("First for Women magazine 1989", k=2, index=index)
        finally:
            tmp_dir.cleanup()

        self.assertIsInstance(results, list)
        assert isinstance(results, list)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["doc_id"], "hotpot_1")
        self.assertGreaterEqual(results[0]["score"], results[1]["score"])
        self.assertEqual(
            set(results[0].keys()),
            {"doc_id", "title", "sentences", "text", "score"},
        )
        self.assertEqual(results[0]["sentences"], self.SAMPLE_PARAGRAPHS[0]["sentences"])

    def test_default_k_comes_from_config(self) -> None:
        from src.config.settings import get_settings
        from src.tools import BM25Index, bm25_search

        settings_k = get_settings().retrieval.k
        paragraphs = [
            {
                "doc_id": f"doc_{i}",
                "title": f"Tesla Note {i}",
                "sentences": [f"Tesla alternating current experiment {i}."],
                "text": f"Tesla alternating current experiment {i}.",
            }
            for i in range(settings_k + 3)
        ]
        index = BM25Index(paragraphs)
        results = bm25_search("Tesla alternating current", index=index)
        self.assertIsInstance(results, list)
        assert isinstance(results, list)
        self.assertEqual(len(results), settings_k)

    def test_synthesizes_text_from_sentences_when_text_missing(self) -> None:
        from src.tools import BM25Index

        index = BM25Index(
            [
                {
                    "doc_id": "d1",
                    "title": "Bauer Publishing",
                    "sentences": ["Bauer Media Group publishes magazines.", "Headquartered in Germany."],
                }
            ]
        )
        results = index.search("Bauer Germany", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0]["text"],
            "Bauer Media Group publishes magazines. Headquartered in Germany.",
        )

    def test_stopword_only_or_unmatched_query_returns_empty_list(self) -> None:
        from src.tools import BM25Index, bm25_search

        index = BM25Index(self.SAMPLE_PARAGRAPHS)
        self.assertEqual(bm25_search("the and or", index=index), [])
        self.assertEqual(bm25_search("quantum entanglement", index=index), [])

    def test_invalid_inputs_missing_or_corrupt_corpus_return_error(self) -> None:
        from src.tools import BM25Index, bm25_search

        index = BM25Index(self.SAMPLE_PARAGRAPHS)
        self.assertTrue(bm25_search("", index=index).startswith("ERROR: "))  # type: ignore[union-attr]
        self.assertTrue(bm25_search("   ", index=index).startswith("ERROR: "))  # type: ignore[union-attr]
        self.assertTrue(bm25_search("Tesla", k=0, index=index).startswith("ERROR: "))  # type: ignore[union-attr]
        self.assertTrue(bm25_search("Tesla", k=-2, index=index).startswith("ERROR: "))  # type: ignore[union-attr]
        self.assertTrue(
            bm25_search("Tesla", corpus_path="/nonexistent/corpus.jsonl").startswith("ERROR: ")  # type: ignore[union-attr]
        )

        corrupt_path, tmp_dir = self._write_temp_corpus(["{not valid json"])
        try:
            res = bm25_search("Tesla", corpus_path=corrupt_path)
            self.assertIsInstance(res, str)
            assert isinstance(res, str)
            self.assertTrue(res.startswith("ERROR: "))
        finally:
            tmp_dir.cleanup()

    def test_get_default_index_caches_by_path(self) -> None:
        import json
        from src.tools import get_default_index

        lines = [json.dumps(item) for item in self.SAMPLE_PARAGRAPHS]
        corpus_path, tmp_dir = self._write_temp_corpus(lines)
        try:
            idx1 = get_default_index(corpus_path)
            idx2 = get_default_index(corpus_path)
            self.assertIs(idx1, idx2)
            idx3 = get_default_index(corpus_path, reload=True)
            self.assertIsNot(idx1, idx3)
        finally:
            tmp_dir.cleanup()

    def test_format_search_results(self) -> None:
        from src.tools import format_search_results

        self.assertEqual(format_search_results("ERROR: bad query"), "ERROR: bad query")
        self.assertEqual(format_search_results([]), "No relevant documents found.")
        rendered = format_search_results(
            [
                {
                    "doc_id": "d1",
                    "title": "Sample Title",
                    "sentences": ["Hello world."],
                    "text": "Hello world.",
                    "score": 1.25,
                }
            ]
        )
        self.assertIn("[1] Sample Title (doc_id: d1, score: 1.2500)", rendered)
        self.assertIn("Hello world.", rendered)

    def test_langchain_bm25_tool_wrapper_or_clean_missing_dependency(self) -> None:
        import json
        from src.tools import get_bm25_search_tool

        lines = [json.dumps(item) for item in self.SAMPLE_PARAGRAPHS]
        corpus_path, tmp_dir = self._write_temp_corpus(lines)
        try:
            try:
                tool = get_bm25_search_tool(corpus_path=corpus_path, default_k=1)
            except RuntimeError as exc:
                self.assertIn("langchain-core", str(exc))
            else:
                out = tool.invoke({"query": "Nikola Tesla"})
                self.assertIn("Nikola Tesla", out)
        finally:
            tmp_dir.cleanup()

    def test_frozen_corpus_search_when_available(self) -> None:
        from src.config.settings import REPO_ROOT
        from src.tools import bm25_search

        frozen_corpus = REPO_ROOT / "data" / "frozen" / "corpus.jsonl"
        if not frozen_corpus.exists():
            self.skipTest("data/frozen/corpus.jsonl is not present in this checkout")

        results = bm25_search("First for Women Arthur's Magazine")
        self.assertIsInstance(results, list)
        assert isinstance(results, list)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("doc_id", results[0])
        self.assertIn("score", results[0])
        self.assertGreater(results[0]["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
