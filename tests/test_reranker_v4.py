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
    semantic_commonness_eligible,
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

    def test_writer_query_includes_category_context(self) -> None:
        text = build_writer_query(
            "บ้าน",
            {"sense": 1, "definition": "ที่อยู่อาศัย"},
            category="noun:place",
        )
        self.assertIn("Query category / grammatical role: noun:place", text)

    def test_candidate_document_uses_matched_definition(self) -> None:
        text = build_candidate_document(self.candidates[0])
        self.assertIn("Thai candidate word: พิรุณ", text)
        self.assertIn("Dictionary meaning: ฝน", text)

    def test_v43_profiles_keep_strict_prompt_and_expected_sizes(self) -> None:
        small = resolve_reranker_profile("qwen3-0.6b-v4.3")
        large = resolve_reranker_profile("qwen3-4b-v4.3")

        instruction = str(small["instruction"])
        self.assertIn("lexical substitutability", instruction)
        self.assertIn("commonly used Thai words", instruction)
        self.assertIn("grammatical role", instruction)
        self.assertEqual(small["model_id"], "Qwen/Qwen3-Reranker-0.6B")
        self.assertEqual(small["default_batch_size"], 16)
        self.assertEqual(large["model_id"], "Qwen/Qwen3-Reranker-4B")
        self.assertEqual(large["default_batch_size"], 4)
        self.assertEqual(large["model_kwargs"]["torch_dtype"], "float16")

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
        scored = annotate_reranker_scores(
            "ฝน",
            self.candidates,
            scorer,
            category="noun:nature",
        )
        results = rank_v4_candidates(scored, mode="rerank", top_k=3)

        self.assertEqual([item["word"] for item in results], ["พลาหก", "เมฆ", "พิรุณ"])
        self.assertEqual(results[0]["reranker_rank"], 1)
        self.assertIn("ฝน", scorer.last_query or "")
        self.assertIn("noun:nature", scorer.last_query or "")
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

    def test_token_aware_commonness_helps_natural_compound(self) -> None:
        candidates = [
            _candidate("บ้านเรือน", score=0.4, definition="บ้านและเรือน"),
            _candidate("วาสะ", score=0.3, definition="ที่อยู่"),
        ]
        annotated = annotate_commonness(
            candidates,
            {
                "บ้าน": 5000,
                "เรือน": 3000,
                "วาสะ": 2,
            },
            source="fake",
            tokenizer=lambda word: {
                "บ้านเรือน": ["บ้าน", "เรือน"],
                "วาสะ": ["วาสะ"],
            }[word],
        )
        by_word = {item["word"]: item for item in annotated}
        self.assertGreater(
            by_word["บ้านเรือน"]["commonness_score"],
            by_word["วาสะ"]["commonness_score"],
        )
        self.assertEqual(by_word["บ้านเรือน"]["commonness_count"], 0)

    def test_semantic_gate_rejects_frequent_but_weak_candidate(self) -> None:
        weak = {
            "relation_tier": 0,
            "lexical_form": "standalone",
            "reranker_rank": 7,
            "v25_rank": 35,
        }
        self.assertFalse(semantic_commonness_eligible(weak))

        consensus = dict(weak)
        consensus["reranker_rank"] = 8
        consensus["v25_rank"] = 15
        self.assertTrue(semantic_commonness_eligible(consensus))

        lexical = dict(weak)
        lexical["relation_tier"] = 4
        self.assertTrue(semantic_commonness_eligible(lexical))

    def test_gated_commonness_caps_frequency_promotion(self) -> None:
        candidates = []
        for index in range(1, 8):
            candidates.append(
                {
                    "word": f"w{index}",
                    "reranker_score": 1.0 - index / 100,
                    "reranker_rank": index,
                    "v25_rank": index,
                    "relation_tier": 4 if index == 7 else 0,
                    "lexical_form": "standalone",
                    "commonness_score": float(index),
                    "commonness_rank": 8 - index,
                }
            )

        results = rank_v4_candidates(
            candidates,
            mode="gated-commonness",
            top_k=7,
            v25_weight=1.0,
            reranker_weight=1.0,
            commonness_weight=1.0,
            commonness_promotion_cap=3,
            rrf_k=20,
        )
        by_word = {item["word"]: item for item in results}
        self.assertLessEqual(by_word["w7"]["commonness_promotion"], 3)
        self.assertGreaterEqual(
            by_word["w7"]["adjusted_fusion_rank"],
            by_word["w7"]["fusion_rank"] - 3,
        )

    def test_zero_gated_commonness_weight_matches_fusion_order(self) -> None:
        scorer = _FakeScorer([0.4, 0.9, 0.8])
        scored = annotate_reranker_scores("ฝน", self.candidates, scorer)
        scored = annotate_commonness(
            scored,
            {"พิรุณ": 999, "พลาหก": 2, "เมฆ": 1000},
            tokenizer=lambda word: [word],
        )

        fusion = rank_v4_candidates(scored, mode="fusion", top_k=3)
        gated = rank_v4_candidates(
            scored,
            mode="gated-commonness",
            top_k=3,
            commonness_weight=0.0,
        )
        self.assertEqual(
            [item["word"] for item in gated],
            [item["word"] for item in fusion],
        )

    def test_mismatched_score_count_fails_closed(self) -> None:
        scorer = _FakeScorer([0.5])
        with self.assertRaises(ValueError):
            annotate_reranker_scores("ฝน", self.candidates, scorer)


if __name__ == "__main__":
    unittest.main()
