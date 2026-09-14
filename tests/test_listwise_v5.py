from __future__ import annotations

import unittest
from typing import Any

from thai_listwise_v5 import (
    annotate_listwise_scores,
    build_listwise_document,
    build_listwise_query,
    rank_v5_candidates,
)


class _FakeListwiseRanker:
    def __init__(self, order: list[int], scores: list[float]) -> None:
        self.order = order
        self.scores = scores
        self.last_query: str | None = None
        self.last_documents: list[str] | None = None

    def rank(self, query_text: str, documents: list[str]) -> list[dict[str, Any]]:
        self.last_query = query_text
        self.last_documents = documents
        return [
            {"index": index, "rank": rank, "score": self.scores[index]}
            for rank, index in enumerate(self.order, start=1)
        ]


def _candidate(
    word: str,
    *,
    score: float,
    definition: str,
    query_sense: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "word": word,
        "score": score,
        "definition": definition,
        "matched_candidate_sense": {
            "sense": 1,
            "definition": definition,
        },
        "query_sense": query_sense,
    }


class V5ListwiseTests(unittest.TestCase):
    def setUp(self) -> None:
        sense = {"sense": 1, "definition": "ที่อยู่อาศัย"}
        self.candidates = [
            _candidate("วาสะ", score=0.03, definition="ที่อยู่", query_sense=sense),
            _candidate("เรือน", score=0.029, definition="บ้าน ที่อยู่อาศัย", query_sense=sense),
            _candidate(
                "บ้านเรือน",
                score=0.028,
                definition="บ้านและเรือน ที่อยู่อาศัย",
                query_sense=sense,
            ),
        ]

    def test_query_contains_global_ranking_instruction_and_category(self) -> None:
        text = build_listwise_query(
            "บ้าน",
            self.candidates[0]["query_sense"],
            category="noun:place",
        )
        self.assertIn("against one another", text)
        self.assertIn("common contemporary Thai", text)
        self.assertIn("noun:place", text)
        self.assertIn("ที่อยู่อาศัย", text)

    def test_candidate_document_uses_word_and_definition(self) -> None:
        text = build_listwise_document(self.candidates[1])
        self.assertIn("Candidate: เรือน", text)
        self.assertIn("Dictionary meaning:", text)

    def test_listwise_annotation_maps_global_order_back_to_candidates(self) -> None:
        ranker = _FakeListwiseRanker([1, 2, 0], [0.2, 0.9, 0.8])
        annotated = annotate_listwise_scores(
            "บ้าน",
            self.candidates,
            ranker,
            category="noun:place",
        )

        by_word = {item["word"]: item for item in annotated}
        self.assertEqual(by_word["เรือน"]["listwise_rank"], 1)
        self.assertEqual(by_word["บ้านเรือน"]["listwise_rank"], 2)
        self.assertEqual(by_word["วาสะ"]["listwise_rank"], 3)
        self.assertIn("noun:place", ranker.last_query or "")
        self.assertEqual(len(ranker.last_documents or []), 3)

    def test_pure_listwise_uses_joint_order(self) -> None:
        ranker = _FakeListwiseRanker([1, 2, 0], [0.2, 0.9, 0.8])
        annotated = annotate_listwise_scores("บ้าน", self.candidates, ranker)
        ranked = rank_v5_candidates(
            annotated,
            mode="listwise",
            top_k=3,
        )
        self.assertEqual(
            [item["word"] for item in ranked],
            ["เรือน", "บ้านเรือน", "วาสะ"],
        )

    def test_light_v25_fusion_keeps_listwise_as_primary_signal(self) -> None:
        ranker = _FakeListwiseRanker([1, 2, 0], [0.2, 0.9, 0.8])
        annotated = annotate_listwise_scores("บ้าน", self.candidates, ranker)
        ranked = rank_v5_candidates(
            annotated,
            mode="fusion",
            top_k=3,
            v25_weight=0.2,
            listwise_weight=1.0,
            rrf_k=20,
        )
        self.assertEqual(ranked[0]["word"], "เรือน")
        self.assertEqual(ranked[0]["v5_mode"], "fusion")

    def test_invalid_mode_fails_closed(self) -> None:
        ranker = _FakeListwiseRanker([0, 1, 2], [0.9, 0.8, 0.7])
        annotated = annotate_listwise_scores("บ้าน", self.candidates, ranker)
        with self.assertRaises(ValueError):
            rank_v5_candidates(annotated, mode="unknown")


if __name__ == "__main__":
    unittest.main()
