"""실험 B-4 (1단계) `scripts/measure_retrieval_k.py` 단위 테스트."""

from __future__ import annotations

import unittest

from scripts.measure_retrieval_k import RECOMMENDED_M_BY_K, evaluate_k
from src.tools.bm25_search import BM25Index

SAMPLE_DOCS = [
    {
        "doc_id": "hp::Alpha",
        "title": "Alpha",
        "sentences": ["Alpha is a historic river.", "It flows north into Beta lake."],
        "text": "Alpha is a historic river. It flows north into Beta lake.",
    },
    {
        "doc_id": "hp::Beta",
        "title": "Beta",
        "sentences": ["Beta lake is located in Gamma province."],
        "text": "Beta lake is located in Gamma province.",
    },
    {
        "doc_id": "sq::Delta::0",
        "title": "Delta",
        "sentences": ["Delta city was founded in 1901."],
        "text": "Delta city was founded in 1901.",
    },
]

SAMPLE_ITEMS = [
    {
        "qid": "hotpotqa-0040",
        "type": "hotpotqa",
        "subtype": "bridge",
        "question": "Which province is the lake that Alpha river flows into located in?",
        "gold_doc_ids": ["hp::Alpha", "hp::Beta"],
        "supporting_facts": [["Alpha", 1], ["Beta", 0]],
    },
    {
        "qid": "squad2-0040",
        "type": "squad2",
        "subtype": "answerable",
        "question": "When was Delta city founded?",
        "gold_doc_ids": ["sq::Delta::0"],
        "supporting_facts": [],
    },
]


class TestMeasureRetrievalK(unittest.TestCase):
    """Unit tests for B-4 retrieval.k recall evaluation helper."""

    def test_evaluate_k_computes_recall_and_char_stats(self) -> None:
        index = BM25Index(SAMPLE_DOCS)
        res = evaluate_k(index, SAMPLE_ITEMS, k=3, m=6, window=1)

        self.assertEqual(res["k"], 3)
        self.assertEqual(res["m"], 6)
        self.assertEqual(res["n_items"], 2)
        self.assertEqual(res["any_hits"], 2)
        self.assertEqual(res["all_hits"], 2)
        self.assertEqual(res["found_gold_docs"], 3)
        self.assertEqual(res["total_gold_docs"], 3)
        self.assertEqual(res["squad_hits"], 1)
        self.assertEqual(res["hp_all_hits"], 1)
        self.assertGreater(res["p1_obs_chars"], 0)
        self.assertLessEqual(res["m_comp_chars"], res["m_full_chars"])

    def test_recommended_m_mapping_matches_c2(self) -> None:
        self.assertEqual(RECOMMENDED_M_BY_K, {3: 6, 5: 7, 8: 10})

    def test_empty_items_raises_value_error(self) -> None:
        index = BM25Index(SAMPLE_DOCS)
        with self.assertRaises(ValueError):
            evaluate_k(index, [], k=3)


if __name__ == "__main__":
    unittest.main()
