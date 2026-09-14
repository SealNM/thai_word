from __future__ import annotations

import unittest
from typing import Any

from thai_reranker_v4 import (
    annotate_commonness,
    annotate_reranker_scores,
    build_candidate_document,
    build_writer_query,
    is_high_precision_lexical,
    rank_v4_candidates,
    resolve_reranker_profile,
)


class _FakeScorer:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.last_query: str | None = None
        self.last_documents: list[str] | None = None

    def score(self, query_text: str, documents: list[str]) -> list[float]:
        self.last_query = query_text
        self.last_documents = documents
        return list(self.scores)


def _candidate(
    word: str,
    *,
    score: float,
    definition: str,
    relation_tier: int = 0,
    relation_hint: str = "dense_semantic_similarity",
    lexical_form: str = "standalone",
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
        "relation_tier": relation_tier,
        "relation_hint": relation_hint,
        "lexical_form": lexical_form,
        "query_sense": query_sense,
    }


class V4RerankerTests(unittest.TestCase):
    def setUp(self) -> None:
        query_sense = {"sense": 1, "definition": "น้ําที่ตกลงมาจากเมฆเป็นเม็ด ๆ"}
        self.candidates = [
            _candidate(
                "พิรุณ",
                score=0.031,
                definition="ฝน",
                relation_tier=5,
                relation_hint="direct_gloss_or_synonym",
                query_sense=query_sense,
            ),
            _candidate(
                "พลาหก",
                score=0.029,
                definition="ฝน เมฆ",
                query_sense=query_sense,
            ),
            _candidate(
                "เมฆ",
                score=0.028,
                definition="ละอองน้ําที่รวมตัวกันอยู่ในอากาศ",
                query_sense=query_sense,
            ),
        ]

    def test_writer_query_includes_selected_sense(self) -> None:
        text = build_writer_query("ฝน", self.candidates[0]["query_sense"])
        self.assertIn("Thai query word: ฝน", text)
        self.assertIn("Intended dictionary sense:", text)

    def test_candidate_document_uses_matched_definition(self) -> None:
        text = build_candidate_document(self.candidates[0])
        self.assertIn("Thai candidate word: พิรุณ", text)
        self.assertIn("Dictionary meaning: ฝน", text)

    def test_v41_profile_prefers_common_substitutes(self) -> None:
        profile = resolve_reranker_profile("qwen3-0.6b-v4.1")
        instruction = str(profile["instruction"])
        self.assertIn("lexical substitutability", instruction)
        self.assertIn("commonly used Thai words", instruction)
        self.assertIn("grammatical role", instruction)

    def test_only_narrow_tier_five_evidence_is_protected(self) -> None:
        self.assertTrue(is_high_precision_lexical(self.candidates[0]))

        lower = dict(self.candidates[0])
        lower["relation_tier"] = 4
        self.assertFalse(is_high_precision_lexical(lower))

        unsafe_hint = dict(self.candidates[0])
        unsafe_hint["relation_hint"] = "query_definition_component"
        self.assertFalse(is_high_precision_lexical(unsafe_hint))

    def test_rerank_mode_follows_model_scores(self) -> None:
        scorer = _FakeScorer([0.4, 0.9, 0.8])
        scored = annotate_reranker_scores("ฝน", self.candidates, scorer)
        results = rank_v4_candidates(scored, mode="rerank", top_k=3)

        self.assertEqual([item["word"] for item in results], ["พลาหก", "เมฆ", "พิรุณ"])
        self.assertEqual(results[0]["reranker_rank"], 1)
        self.assertIn("ฝน", scorer.last_query or "")
        self.assertEqual(len(scorer.last_documents or []), 3)

    def test_protected_mode_keeps_safe_direct_synonym_first(self) -> None:
        scorer = _FakeScorer([0.4, 0.9, 0.8])
        scored = annotate_reranker_scores("ฝน", self.candidates, scorer)
        results = rank_v4_candidates(scored, mode="protected", top_k=3)

        self.assertEqual([item["word"] for item in results], ["พิรุณ", "พลาหก", "เมฆ"])
        self.assertTrue(results[0]["protected_lexical"])

    def test_fusion_is_rank_based_and_keeps_metadata(self) -> None:
        scorer = _FakeScorer([0.4, 0.9, 0.8])
        scored = annotate_reranker_scores("ฝน", self.candidates, scorer)
        results = rank_v4_candidates(
            scored,
            mode="fusion",
            top_k=3,
            v25_weight=0.35,
            reranker_weight=1.0,
            rrf_k=20,
        )

        self.assertEqual(results[0]["word"], "พลาหก")
        self.assertEqual(results[0]["v4_mode"], "fusion")
        self.assertIn("v25_score", results[0])
        self.assertIn("reranker_score", results[0])

    def test_commonness_rank_uses_frequency_but_missing_words_get_no_boost(self) -> None:
        scorer = _FakeScorer([0.8, 0.79, 0.78])
        scored = annotate_reranker_scores("ฝน", self.candidates, scorer)
        scored = annotate_commonness(
            scored,
            {"พลาหก": 2, "เมฆ": 1000},
            source="fake",
        )

        by_word = {item["word"]: item for item in scored}
        self.assertEqual(by_word["เมฆ"]["commonness_rank"], 1)
        self.assertEqual(by_word["พลาหก"]["commonness_rank"], 2)
        self.assertIsNone(by_word["พิรุณ"]["commonness_rank"])

        results = rank_v4_candidates(
            scored,
            mode="commonness",
            top_k=3,
            v25_weight=0.0,
            reranker_weight=1.0,
            commonness_weight=1.0,
            rrf_k=20,
        )
        self.assertEqual(results[0]["word"], "เมฆ")

    def test_zero_commonness_weight_matches_fusion_order(self) -> None:
        scorer = _FakeScorer([0.4, 0.9, 0.8])
        scored = annotate_reranker_scores("ฝน", self.candidates, scorer)
        scored = annotate_commonness(scored, {"พิรุณ": 999, "เมฆ": 1})

        fusion = rank_v4_candidates(scored, mode="fusion", top_k=3)
        common = rank_v4_candidates(
            scored,
            mode="commonness",
            top_k=3,
            commonness_weight=0.0,
        )
        self.assertEqual(
            [item["word"] for item in common],
            [item["word"] for item in fusion],
        )

    def test_mismatched_score_count_fails_closed(self) -> None:
        scorer = _FakeScorer([0.5])
        with self.assertRaises(ValueError):
            annotate_reranker_scores("ฝน", self.candidates, scorer)


if __name__ == "__main__":
    unittest.main()
