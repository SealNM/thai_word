from __future__ import annotations

import unittest

import numpy as np

from thai_pairwise_v27 import (
    FEATURE_NAMES,
    PairwiseRanker,
    TNCFamiliarity,
    extract_features,
    feature_vector,
    rerank_v27,
)


class FakePipeline:
    def __init__(self, feature_name: str) -> None:
        self.index = FEATURE_NAMES.index(feature_name)

    def decision_function(self, matrix):
        matrix = np.asarray(matrix)
        return matrix[:, self.index]


def item(
    word: str,
    *,
    score: float = 0.01,
    relation: int = 0,
    protected: int = 0,
    lexical_rank=None,
    dense_rank=1,
    dense_similarity=0.8,
    lexical_form="standalone",
    sense_resolution="dense_only",
):
    return {
        "word": word,
        "score": score,
        "protected_relation_tier": protected,
        "relation_tier": relation,
        "relation_hint": "definition_similar",
        "lexical_form": lexical_form,
        "sense_resolution": sense_resolution,
        "lexical_score": 0.0,
        "lexical_rank": lexical_rank,
        "dense_similarity": dense_similarity,
        "dense_rank": dense_rank,
        "matched_candidate_sense": {
            "sense": 1,
            "definition": "คำจำลองสำหรับทดสอบ",
        },
        "definition": "คำจำลองสำหรับทดสอบ",
        "sense_count": 1,
    }


class V27PairwiseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frequency = TNCFamiliarity(
            frequencies={
                "ทั่วไป": 1000,
                "หายาก": 1,
                "ตรง": 2,
                "รอง": 100,
            },
            tokenizer=lambda text: [text],
        )

    def test_feature_schema_is_complete_and_finite(self) -> None:
        features = extract_features(
            item(
                "ทั่วไป",
                lexical_rank=3,
                sense_resolution=(
                    "selected_sense_supported"
                ),
            ),
            v25_rank=2,
            familiarity=self.frequency,
        )
        self.assertEqual(
            tuple(features),
            FEATURE_NAMES,
        )
        vector = feature_vector(features)
        self.assertEqual(
            vector.shape,
            (len(FEATURE_NAMES),),
        )
        self.assertTrue(
            np.isfinite(vector).all()
        )

    def test_frequency_is_a_feature_not_a_hard_rule(self) -> None:
        rare = extract_features(
            item("หายาก"),
            v25_rank=1,
            familiarity=self.frequency,
        )
        common = extract_features(
            item("ทั่วไป"),
            v25_rank=2,
            familiarity=self.frequency,
        )
        self.assertGreater(
            common["tnc_log1p_count"],
            rare["tnc_log1p_count"],
        )
        self.assertEqual(
            rare["tnc_missing"],
            0.0,
        )

    def test_learned_score_can_reorder_within_same_protected_bucket(
        self,
    ) -> None:
        ranker = PairwiseRanker(
            pipeline=FakePipeline(
                "tnc_log1p_count"
            ),
            feature_names=FEATURE_NAMES,
            metadata={},
        )
        results = [
            item(
                "หายาก",
                protected=0,
            ),
            item(
                "ทั่วไป",
                protected=0,
            ),
        ]
        reranked = rerank_v27(
            results,
            ranker=ranker,
            familiarity=self.frequency,
        )
        self.assertEqual(
            [
                row["word"]
                for row in reranked
            ],
            ["ทั่วไป", "หายาก"],
        )
        self.assertEqual(
            reranked[0]["v27"]["movement"],
            1,
        )

    def test_protected_direct_relation_bucket_is_preserved(
        self,
    ) -> None:
        ranker = PairwiseRanker(
            pipeline=FakePipeline(
                "tnc_log1p_count"
            ),
            feature_names=FEATURE_NAMES,
            metadata={},
        )
        results = [
            item(
                "ตรง",
                relation=4,
                protected=4,
            ),
            item(
                "ทั่วไป",
                relation=0,
                protected=0,
            ),
        ]
        reranked = rerank_v27(
            results,
            ranker=ranker,
            familiarity=self.frequency,
        )
        self.assertEqual(
            [
                row["word"]
                for row in reranked
            ],
            ["ตรง", "ทั่วไป"],
        )

    def test_runtime_matrix_shape_is_checked(self) -> None:
        ranker = PairwiseRanker(
            pipeline=FakePipeline(
                "v25_reciprocal_rank"
            ),
            feature_names=FEATURE_NAMES,
            metadata={},
        )
        with self.assertRaises(ValueError):
            ranker.score_matrix(
                np.zeros((1, 3))
            )


if __name__ == "__main__":
    unittest.main()
