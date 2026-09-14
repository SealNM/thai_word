from __future__ import annotations

import unittest

from thai_generative_v51 import (
    QWEN35_RANKING_POLICY,
    build_generative_listwise_prompt,
    parse_ranked_candidate_ids,
)


class V51GenerativeListwiseTests(unittest.TestCase):
    def test_policy_encodes_semantics_before_commonness(self) -> None:
        self.assertIn("Preserve the intended dictionary sense", QWEN35_RANKING_POLICY)
        self.assertIn("Preserve the target grammatical role", QWEN35_RANKING_POLICY)
        self.assertIn("common contemporary Thai", QWEN35_RANKING_POLICY)
        self.assertIn("equally valid substitutes", QWEN35_RANKING_POLICY)

    def test_prompt_contains_all_ids_and_strict_output_contract(self) -> None:
        prompt = build_generative_listwise_prompt(
            "Target Thai word: บ้าน\nTarget category / grammatical role: noun:place",
            [
                "Candidate: วาสะ\nDictionary meaning: ที่อยู่",
                "Candidate: เรือน\nDictionary meaning: บ้าน ที่อยู่อาศัย",
                "Candidate: บ้านเรือน\nDictionary meaning: บ้านและเรือน",
            ],
        )
        self.assertIn("[0] Candidate: วาสะ", prompt)
        self.assertIn("[1] Candidate: เรือน", prompt)
        self.assertIn("[2] Candidate: บ้านเรือน", prompt)
        self.assertIn("exactly 3 candidates", prompt)
        self.assertIn("Output only the JSON array", prompt)

    def test_parser_accepts_complete_json_order(self) -> None:
        order, complete, strategy = parse_ranked_candidate_ids(
            "[1, 2, 0]",
            3,
        )
        self.assertEqual(order, [1, 2, 0])
        self.assertTrue(complete)
        self.assertEqual(strategy, "json")

    def test_parser_recovers_wrapped_json_array(self) -> None:
        order, complete, strategy = parse_ranked_candidate_ids(
            "Ranking follows: [2, 0, 1]",
            3,
        )
        self.assertEqual(order, [2, 0, 1])
        self.assertTrue(complete)
        self.assertEqual(strategy, "json")

    def test_parser_appends_missing_ids_in_original_order(self) -> None:
        order, complete, strategy = parse_ranked_candidate_ids(
            "[2, 0]",
            4,
        )
        self.assertEqual(order, [2, 0, 1, 3])
        self.assertFalse(complete)
        self.assertEqual(strategy, "json")

    def test_parser_ignores_duplicates_and_out_of_range_ids(self) -> None:
        order, complete, _ = parse_ranked_candidate_ids(
            "[1, 99, 1, 0]",
            3,
        )
        self.assertEqual(order, [1, 0, 2])
        self.assertFalse(complete)


if __name__ == "__main__":
    unittest.main()
