from __future__ import annotations

import unittest

from thai_gemma_v53 import (
    GEMMA4_SYSTEM_PROMPT,
    build_gemma_ranking_prompt,
    parse_gemma_ranking,
)


class V53GemmaTests(unittest.TestCase):
    def test_system_prompt_prioritizes_semantics_before_commonness(self) -> None:
        self.assertIn("Preserve the intended dictionary sense", GEMMA4_SYSTEM_PROMPT)
        self.assertIn("grammatical role", GEMMA4_SYSTEM_PROMPT)
        self.assertIn("common contemporary Thai", GEMMA4_SYSTEM_PROMPT)
        self.assertIn("equally valid substitutes", GEMMA4_SYSTEM_PROMPT)\n        self.assertIn("HARD lexical-validity gate", GEMMA4_SYSTEM_PROMPT)\n        self.assertIn("merely associated", GEMMA4_SYSTEM_PROMPT)\n        self.assertIn("different parts of speech", GEMMA4_SYSTEM_PROMPT)

    def test_prompt_requests_only_top_k_ids(self) -> None:
        candidates = [
            {
                "word": "วาสะ",
                "definition": "ที่อยู่",
                "matched_candidate_sense": {"definition": "ที่อยู่"},
            },
            {
                "word": "เรือน",
                "definition": "บ้าน",
                "matched_candidate_sense": {"definition": "บ้าน"},
            },
            {
                "word": "บ้านเรือน",
                "definition": "บ้านและเรือน",
                "matched_candidate_sense": {"definition": "บ้านและเรือน"},
            },
        ]
        prompt = build_gemma_ranking_prompt(
            "Target Thai word: บ้าน",
            candidates,
            top_k=2,
        )
        self.assertIn("[0] Candidate: วาสะ", prompt)
        self.assertIn("[1] Candidate: เรือน", prompt)
        self.assertIn("[2] Candidate: บ้านเรือน", prompt)
        self.assertIn("best 2 candidate IDs", prompt)

    def test_parser_accepts_exact_json_top_k(self) -> None:
        ids, complete, strategy = parse_gemma_ranking(
            "[2, 1]",
            3,
            expected_count=2,
        )
        self.assertEqual(ids, [2, 1])
        self.assertTrue(complete)
        self.assertEqual(strategy, "json")

    def test_parser_does_not_silently_fill_missing_ids(self) -> None:
        ids, complete, strategy = parse_gemma_ranking(
            "[2]",
            3,
            expected_count=2,
        )
        self.assertEqual(ids, [2])
        self.assertFalse(complete)
        self.assertEqual(strategy, "json")

    def test_parser_prefers_largest_json_array(self) -> None:
        ids, complete, strategy = parse_gemma_ranking(
            "[0] note [2, 1]",
            3,
            expected_count=2,
        )
        self.assertEqual(ids, [2, 1])
        self.assertTrue(complete)
        self.assertEqual(strategy, "json")

    def test_parser_recovers_ordered_singleton_labels(self) -> None:
        ids, complete, strategy = parse_gemma_ranking(
            "[2] first\n[1] second",
            3,
            expected_count=2,
        )
        self.assertEqual(ids, [2, 1])
        self.assertTrue(complete)
        self.assertEqual(strategy, "bracket-ids")


if __name__ == "__main__":
    unittest.main()
