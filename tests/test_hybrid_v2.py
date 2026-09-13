from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from thai_dense_v2 import DenseArtifacts, resolve_device
from thai_hybrid_v2 import HybridSearcher, lexical_relation_weight, weighted_rrf
from thai_lexical_v1 import SearchArtifacts


class _FakeEncoder:
    def encode_query(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


class HybridV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.lexical = SearchArtifacts(
            entries=[
                {"word": "พูด", "sense_count": 1, "source_ids": [1]},
                {"word": "พูดจา", "sense_count": 1, "source_ids": [2]},
                {"word": "เสียง", "sense_count": 1, "source_ids": [3]},
                {"word": "เอ่ย", "sense_count": 1, "source_ids": [4]},
            ],
            senses=[
                {
                    "entry_index": 0,
                    "sense_index": 1,
                    "definition": "เปล่งเสียงออกเป็นถ้อยคำ, พูดจา ก็ว่า.",
                },
                {"entry_index": 1, "sense_index": 1, "definition": "พูด."},
                {"entry_index": 2, "sense_index": 1, "definition": "สิ่งที่ได้ยิน."},
                {"entry_index": 3, "sense_index": 1, "definition": "พูดออกมา."},
            ],
            word_to_index={"พูด": 0, "พูดจา": 1, "เสียง": 2, "เอ่ย": 3},
            vectorizer=None,
            matrix=None,
            references=[[], [], [], []],
            reverse_references=[[], [], [], []],
            token_sets=[set(), set(), set(), set()],
            entry_to_senses=[[0], [1], [2], [3]],
            tokenize=lambda text: text.split(),
        )
        dense = DenseArtifacts(
            embeddings=np.array(
                [
                    [1.0, 0.0],
                    [0.80, 0.60],
                    [0.20, 0.98],
                    [0.99, 0.10],
                ],
                dtype=np.float32,
            ),
            metadata={"model_id": "fake"},
        )
        self.searcher = HybridSearcher(
            lexical=self.lexical,
            dense=dense,
            encoder=_FakeEncoder(),
        )

    def test_relation_weight_ignores_weak_lexical_rank(self) -> None:
        self.assertEqual(lexical_relation_weight(0), 0.0)
        self.assertEqual(lexical_relation_weight(1), 0.0)
        self.assertEqual(lexical_relation_weight(2), 0.25)
        self.assertEqual(lexical_relation_weight(3), 0.5)
        self.assertEqual(lexical_relation_weight(4), 1.0)

    def test_weighted_rrf_is_scale_independent(self) -> None:
        score = weighted_rrf(lexical_rank=1, dense_rank=1, k=60)
        self.assertAlmostEqual(score, 2 / 61)

    @patch("torch.cuda.is_available", return_value=False)
    def test_cuda_request_falls_back_to_cpu(self, mocked_cuda) -> None:
        with self.assertWarns(RuntimeWarning):
            device = resolve_device("cuda")
        self.assertEqual(device, "cpu")
        mocked_cuda.assert_called_once()

    @patch("thai_hybrid_v2.lexical_search")
    def test_dense_can_promote_semantic_candidate_but_not_beat_direct_alias(
        self,
        mocked_search,
    ) -> None:
        mocked_search.return_value = [
            {
                "word": "พูดจา",
                "score": 0.70,
                "relation_tier": 4,
                "relation_hint": "query_gloss_or_alias",
                "lexical_form": "standalone",
                "sense_resolution": "selected_sense_supported",
                "definition": "พูด.",
                "matched_candidate_sense": {"sense": 1, "definition": "พูด."},
            },
            {
                "word": "เสียง",
                "score": 0.18,
                "relation_tier": 1,
                "relation_hint": "query_definition_component",
                "lexical_form": "standalone",
                "sense_resolution": "selected_sense_supported",
                "definition": "สิ่งที่ได้ยิน.",
                "matched_candidate_sense": {"sense": 1, "definition": "สิ่งที่ได้ยิน."},
            },
        ]

        results = self.searcher.search(
            "พูด",
            top_k=3,
            lexical_pool=3,
            dense_pool=3,
        )
        words = [item["word"] for item in results]

        self.assertEqual(words[0], "พูดจา")
        self.assertLess(words.index("เอ่ย"), words.index("เสียง"))
        eui = next(item for item in results if item["word"] == "เอ่ย")
        self.assertEqual(eui["relation_hint"], "dense_semantic_similarity")
        self.assertEqual(eui["lexical_fusion_weight"], 0.0)


if __name__ == "__main__":
    unittest.main()
