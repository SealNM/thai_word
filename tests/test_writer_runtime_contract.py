from __future__ import annotations

import unittest

from thai_writer_runtime import (
    runtime_row_from_v25_result,
    writer_feature_dict,
    writer_text_pair,
)


def _candidate() -> dict:
    return {
        "word": "พิรุณ",
        "score": 0.03125,
        "relation_tier": 5,
        "relation_hint": "candidate_defined_via_query",
        "lexical_form": "standalone",
        "sense_resolution": "selected_sense_supported",
        "lexical_score": 0.91,
        "lexical_rank": 1,
        "dense_similarity": 0.72,
        "dense_rank": 3,
        "definition": "ฝน.",
        "matched_candidate_sense": {
            "sense": 1,
            "definition": "ฝน.",
        },
    }


class WriterRuntimeContractTests(unittest.TestCase):
    def test_runtime_adapter_contains_no_annotation_fields(self) -> None:
        row = runtime_row_from_v25_result(
            query="ฝน",
            query_definition="น้ำที่ตกลงมาจากเมฆ",
            candidate=_candidate(),
            v25_rank=1,
            query_sense=1,
        )

        self.assertNotIn("annotation", row)
        self.assertNotIn("utility", str(row))
        self.assertEqual(row["retrieval"]["v25_rank"], 1)
        self.assertEqual(row["candidate"]["word"], "พิรุณ")
        self.assertEqual(row["candidate"]["definition"], "ฝน.")

    def test_runtime_adapter_preserves_v25_evidence(self) -> None:
        row = runtime_row_from_v25_result(
            query="ฝน",
            query_definition="น้ำที่ตกลงมาจากเมฆ",
            candidate=_candidate(),
            v25_rank=7,
        )
        retrieval = row["retrieval"]

        self.assertEqual(retrieval["v25_rank"], 7)
        self.assertEqual(retrieval["relation_tier"], 5)
        self.assertEqual(retrieval["lexical_rank"], 1)
        self.assertEqual(retrieval["dense_rank"], 3)
        self.assertAlmostEqual(retrieval["dense_similarity"], 0.72)

    def test_feature_contract_supports_category_modes(self) -> None:
        row = runtime_row_from_v25_result(
            query="ฝน",
            query_definition="น้ำที่ตกลงมาจากเมฆ",
            candidate=_candidate(),
            v25_rank=1,
            query_category="noun:nature",
        )

        included = writer_feature_dict(row, category_mode="include")
        none = writer_feature_dict(row, category_mode="none")
        omitted = writer_feature_dict(row, category_mode="omit")

        self.assertEqual(included["query_category"], "noun:nature")
        self.assertEqual(none["query_category"], "<none>")
        self.assertNotIn("query_category", omitted)

    def test_text_pair_can_remove_category_without_changing_evidence(self) -> None:
        row = runtime_row_from_v25_result(
            query="ฝน",
            query_definition="น้ำที่ตกลงมาจากเมฆ",
            candidate=_candidate(),
            v25_rank=1,
            query_category="noun:nature",
        )

        with_category = writer_text_pair(row, category_mode="include")
        no_category = writer_text_pair(row, category_mode="omit")

        self.assertIn("หมวด: noun:nature", with_category[0])
        self.assertNotIn("หมวด:", no_category[0])
        self.assertIn("v25_rank=1", with_category[1])
        self.assertIn("v25_rank=1", no_category[1])
        self.assertEqual(with_category[1], no_category[1])

    def test_invalid_category_mode_is_rejected(self) -> None:
        row = runtime_row_from_v25_result(
            query="ฝน",
            query_definition="",
            candidate=_candidate(),
            v25_rank=1,
        )
        with self.assertRaises(ValueError):
            writer_feature_dict(row, category_mode="auto")
        with self.assertRaises(ValueError):
            writer_text_pair(row, category_mode="auto")

    def test_invalid_v25_rank_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            runtime_row_from_v25_result(
                query="ฝน",
                query_definition="",
                candidate=_candidate(),
                v25_rank=0,
            )


if __name__ == "__main__":
    unittest.main()
