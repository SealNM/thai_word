from __future__ import annotations

import unittest

from thai_reranker_v26 import (
    FrequencyEvidence,
    V26Config,
    rarity_penalty,
    rerank_v26,
    structural_penalties,
)


class FakeFrequency:
    def __init__(self, percentiles):
        self.percentiles = percentiles

    def measure(self, word: str) -> FrequencyEvidence:
        percentile = self.percentiles[word]
        return FrequencyEvidence(
            exact_count=1 if percentile < 0.2 else 100,
            token_proxy_count=0.0,
            familiarity_count=1.0 if percentile < 0.2 else 100.0,
            percentile=percentile,
            tokens=(word,),
            source="test",
        )


def item(
    word,
    *,
    dense,
    relation=0,
    protected=0,
    lexical_form="standalone",
    definition="คำจำลอง",
):
    return {
        "word": word,
        "score": 0.01,
        "protected_relation_tier": protected,
        "relation_tier": relation,
        "relation_hint": "definition_similar",
        "lexical_form": lexical_form,
        "dense_similarity": dense,
        "definition": definition,
    }


class V26RerankerTests(unittest.TestCase):
    def test_common_candidate_can_pass_rare_neighbor_without_bonus(self):
        results = [
            item("rare-a", dense=0.80),
            item("rare-b", dense=0.795),
            item("common", dense=0.79),
        ]
        reranked = rerank_v26(
            results,
            frequency_model=FakeFrequency(
                {"rare-a": 0.05, "rare-b": 0.08, "common": 0.90}
            ),
            config=V26Config(
                rerank_window=3,
                max_promotion=2,
                dense_similarity_tolerance=0.03,
            ),
        )
        self.assertEqual(
            [x["word"] for x in reranked],
            ["common", "rare-a", "rare-b"],
        )
        self.assertEqual(reranked[0]["v26"]["rarity_penalty"], 0)
        self.assertEqual(reranked[0]["v26"]["movement"], 2)

    def test_dense_guard_blocks_semantically_weaker_common_candidate(self):
        results = [
            item("rare", dense=0.80),
            item("common-but-far", dense=0.70),
        ]
        reranked = rerank_v26(
            results,
            frequency_model=FakeFrequency(
                {"rare": 0.05, "common-but-far": 0.95}
            ),
            config=V26Config(
                rerank_window=2,
                max_promotion=4,
                dense_similarity_tolerance=0.03,
            ),
        )
        self.assertEqual(
            [x["word"] for x in reranked],
            ["rare", "common-but-far"],
        )

    def test_frequency_cannot_cross_stronger_relation(self):
        results = [
            item("direct-rare", dense=0.78, relation=4, protected=4),
            item("dense-common", dense=0.80, relation=0, protected=0),
        ]
        reranked = rerank_v26(
            results,
            frequency_model=FakeFrequency(
                {"direct-rare": 0.01, "dense-common": 0.99}
            ),
            config=V26Config(rerank_window=2, max_promotion=4),
        )
        self.assertEqual(
            [x["word"] for x in reranked],
            ["direct-rare", "dense-common"],
        )
        self.assertLessEqual(
            reranked[0]["v26"]["rarity_penalty"],
            2,
        )

    def test_bound_form_gets_structural_penalty(self):
        penalties = structural_penalties(
            item("วัส-", dense=0.7, lexical_form="bound_form")
        )
        self.assertEqual(penalties["bound_form"], 3)

    def test_direct_relation_caps_rarity_penalty(self):
        evidence = FrequencyEvidence(
            0,
            0.0,
            0.0,
            0.0,
            ("x",),
            "test",
        )
        self.assertEqual(
            rarity_penalty(evidence, relation_tier=0),
            4,
        )
        self.assertEqual(
            rarity_penalty(evidence, relation_tier=4),
            2,
        )


if __name__ == "__main__":
    unittest.main()
