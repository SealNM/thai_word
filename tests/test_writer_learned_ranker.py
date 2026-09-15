from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from thai_writer_learned import (
    LEARNED_RANKER_METADATA_FILENAME,
    LearnedWriterRanker,
)
from thai_writer_runtime import runtime_row_from_v25_result


def _train_row(index: int, utility: int, relation: str) -> dict:
    return {
        "query_id": f"q{index // 2}#1",
        "split": "train",
        "query": {
            "word": f"คำ{index}",
            "definition": f"ความหมายเป้าหมาย {index}",
            "category": "legacy-category",
        },
        "candidate": {
            "word": f"ผล{index}",
            "definition": f"ความหมายผล {index}",
        },
        "retrieval": {
            "v25_rank": index + 1,
            "score": 1.0 / (index + 2),
            "relation_tier": 5 if utility >= 2 else 1,
            "lexical_score": 0.8 if utility >= 2 else 0.2,
            "lexical_rank": index + 1,
            "dense_similarity": 0.7 if utility >= 2 else 0.3,
            "dense_rank": index + 2,
            "relation_hint": "direct" if utility >= 2 else "definition_similar",
            "lexical_form": "standalone",
            "sense_resolution": "selected_sense_supported",
        },
        "annotation": {
            "utility": utility,
            "semantic_relation": relation,
            "style_tags": [],
        },
    }


def _training_rows() -> list[dict]:
    return [
        _train_row(0, 3, "direct"),
        _train_row(1, 2, "subtype"),
        _train_row(2, 1, "weak_related"),
        _train_row(3, 0, "unrelated"),
        _train_row(4, 3, "direct"),
        _train_row(5, 0, "sense_mismatch"),
    ]


def _runtime_row() -> dict:
    candidate = {
        "word": "พิรุณ",
        "score": 0.03,
        "relation_tier": 5,
        "relation_hint": "candidate_defined_via_query",
        "lexical_form": "standalone",
        "sense_resolution": "selected_sense_supported",
        "lexical_score": 0.9,
        "lexical_rank": 1,
        "dense_similarity": 0.75,
        "dense_rank": 2,
        "matched_candidate_sense": {
            "sense": 1,
            "definition": "ฝน.",
        },
    }
    return runtime_row_from_v25_result(
        query="ฝน",
        query_definition="น้ำที่ตกจากเมฆ",
        candidate=candidate,
        v25_rank=1,
        query_sense=1,
    )


class LearnedWriterRankerTests(unittest.TestCase):
    def test_fit_score_save_load_round_trip(self) -> None:
        ranker = LearnedWriterRanker.fit(_training_rows())
        inference = [_runtime_row()]
        before = ranker.score_rows(inference)

        with tempfile.TemporaryDirectory() as directory:
            metadata = ranker.metadata(
                dataset_sha256="abc123",
                train_pair_count=len(_training_rows()),
                train_query_count=3,
            )
            ranker.save(directory, metadata=metadata)
            loaded = LearnedWriterRanker.load(directory)
            after = loaded.score_rows(inference)

        self.assertEqual(before, after)
        self.assertEqual(len(after), 1)
        self.assertIn("expected_utility", after[0])
        self.assertIn("severe_probability", after[0])
        self.assertIn("safe_score", after[0])

    def test_inference_does_not_require_annotation(self) -> None:
        ranker = LearnedWriterRanker.fit(_training_rows())
        row = _runtime_row()
        self.assertNotIn("annotation", row)

        scores = ranker.score_rows([row])

        self.assertEqual(len(scores), 1)

    def test_category_is_not_part_of_production_feature_schema(self) -> None:
        ranker = LearnedWriterRanker.fit(_training_rows())
        metadata = ranker.metadata(
            dataset_sha256="abc123",
            train_pair_count=6,
            train_query_count=3,
        )

        self.assertEqual(metadata["category_mode"], "omit")
        self.assertFalse(
            any(name.startswith("query_category") for name in metadata["feature_names"])
        )

    def test_metadata_schema_mismatch_is_rejected(self) -> None:
        ranker = LearnedWriterRanker.fit(_training_rows())

        with tempfile.TemporaryDirectory() as directory:
            metadata = ranker.metadata(
                dataset_sha256="abc123",
                train_pair_count=6,
                train_query_count=3,
            )
            ranker.save(directory, metadata=metadata)

            path = Path(directory) / LEARNED_RANKER_METADATA_FILENAME
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["runtime_schema_version"] = 999
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(ValueError):
                LearnedWriterRanker.load(directory)

    def test_same_seed_produces_same_scores(self) -> None:
        rows = _training_rows()
        inference = [_runtime_row()]

        first = LearnedWriterRanker.fit(rows, seed=42).score_rows(inference)
        second = LearnedWriterRanker.fit(rows, seed=42).score_rows(inference)

        self.assertEqual(first, second)

    def test_fit_rejects_category_aware_mode(self) -> None:
        with self.assertRaises(ValueError):
            LearnedWriterRanker.fit(_training_rows(), category_mode="include")


if __name__ == "__main__":
    unittest.main()
