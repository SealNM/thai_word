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
            output_count=2,
        )
        self.assertIn("[0] Candidate: วาสะ", prompt)
        self.assertIn("[1] Candidate: เรือน", prompt)
        self.assertIn("[2] Candidate: บ้านเรือน", prompt)
        self.assertIn("exactly 3 candidates", prompt)
        self.assertIn("best 2 candidate IDs", prompt)
        self.assertIn("Output only the JSON array", prompt)

    def test_parser_accepts_complete_json_order(self) -> None:
        order, complete, strategy, explicit_count = parse_ranked_candidate_ids(
            "[1, 2, 0]",
            3,
            expected_count=3,
        )
        self.assertEqual(order, [1, 2, 0])
        self.assertTrue(complete)
        self.assertEqual(strategy, "json")
        self.assertEqual(explicit_count, 3)

    def test_parser_recovers_wrapped_json_array(self) -> None:
        order, complete, strategy, explicit_count = parse_ranked_candidate_ids(
            "Ranking follows: [2, 0, 1]",
            3,
            expected_count=3,
        )
        self.assertEqual(order, [2, 0, 1])
        self.assertTrue(complete)
        self.assertEqual(strategy, "json")
        self.assertEqual(explicit_count, 3)

    def test_parser_appends_missing_ids_in_original_order(self) -> None:
        order, complete, strategy, explicit_count = parse_ranked_candidate_ids(
            "[2, 0]",
            4,
            expected_count=3,
        )
        self.assertEqual(order, [2, 0, 1, 3])
        self.assertFalse(complete)
        self.assertEqual(strategy, "json")
        self.assertEqual(explicit_count, 2)

    def test_parser_ignores_duplicates_and_out_of_range_ids(self) -> None:
        order, complete, _, explicit_count = parse_ranked_candidate_ids(
            "[1, 99, 1, 0]",
            3,
            expected_count=3,
        )
        self.assertEqual(order, [1, 0, 2])
        self.assertFalse(complete)
        self.assertEqual(explicit_count, 2)

    def test_parser_prefers_largest_json_array_over_singleton_labels(self) -> None:
        order, complete, strategy, explicit_count = parse_ranked_candidate_ids(
            "[0] note [2, 1, 0]",
            3,
            expected_count=3,
        )
        self.assertEqual(order, [2, 1, 0])
        self.assertTrue(complete)
        self.assertEqual(strategy, "json")
        self.assertEqual(explicit_count, 3)

    def test_parser_collects_bracketed_singleton_ids_in_order(self) -> None:
        order, complete, strategy, explicit_count = parse_ranked_candidate_ids(
            "[2] first\n[0] second\n[1] third",
            3,
            expected_count=3,
        )
        self.assertEqual(order, [2, 0, 1])
        self.assertTrue(complete)
        self.assertEqual(strategy, "bracket-ids")
        self.assertEqual(explicit_count, 3)


if __name__ == "__main__":
    unittest.main()
