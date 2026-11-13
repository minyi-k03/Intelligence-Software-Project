"""Unit tests for shared tools (PR B-2a: tokenizer).

References:
- docs/INTERFACES.md §3
- docs/IMPLEMENTATION_PLAN.md §5 (PR B-2a)
"""

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


if __name__ == "__main__":
    unittest.main()
