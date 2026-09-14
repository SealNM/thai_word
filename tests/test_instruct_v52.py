from __future__ import annotations

import unittest

from thai_instruct_v52 import (
    THAI_WORDS_INSTRUCT_PROMPT,
    rank_ctxl_candidates,
)


class V52InstructionRerankerTests(unittest.TestCase):
    def test_prompt_prioritizes_semantics_before_commonness(self) -> None:
        self.assertIn("Preserve the intended dictionary sense", THAI_WORDS_INSTRUCT_PROMPT)
        self.assertIn("grammatical role", THAI_WORDS_INSTRUCT_PROMPT)
        self.assertIn("common contemporary Thai", THAI_WORDS_INSTRUCT_PROMPT)
        self.assertIn("equally valid substitutes", THAI_WORDS_INSTRUCT_PROMPT)

    def test_instruct_mode_uses_reranker_rank(self) -> None:
        candidates = [
            {
                "word": "วาสะ",
                "v25_rank": 1,
                "reranker_rank": 3,
                "reranker_score": 0.1,
            },
            {
                "word": "เรือน",
                "v25_rank": 20,
                "reranker_rank": 1,
                "reranker_score": 0.9,
            },
            {
                "word": "บ้านเรือน",
                "v25_rank": 10,
                "reranker_rank": 2,
                "reranker_score": 0.8,
            },
        ]

        ranked = rank_ctxl_candidates(
            candidates,
            mode="instruct",
            top_k=3,
        )
        self.assertEqual(
            [item["word"] for item in ranked],
            ["เรือน", "บ้านเรือน", "วาสะ"],
        )

    def test_light_fusion_keeps_instruction_signal_primary(self) -> None:
        candidates = [
            {
                "word": "วาสะ",
                "v25_rank": 1,
                "reranker_rank": 3,
                "reranker_score": 0.1,
            },
            {
                "word": "เรือน",
                "v25_rank": 20,
                "reranker_rank": 1,
                "reranker_score": 0.9,
            },
            {
                "word": "บ้านเรือน",
                "v25_rank": 10,
                "reranker_rank": 2,
                "reranker_score": 0.8,
            },
        ]

        ranked = rank_ctxl_candidates(
            candidates,
            mode="fusion",
            top_k=3,
            v25_weight=0.1,
            instruct_weight=1.0,
            rrf_k=20,
        )
        self.assertEqual(ranked[0]["word"], "เรือน")
        self.assertEqual(ranked[0]["v52_mode"], "fusion")

    def test_invalid_mode_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            rank_ctxl_candidates([], mode="unknown")


if __name__ == "__main__":
    unittest.main()
