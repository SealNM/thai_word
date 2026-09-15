from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.writer_relevance_phase3_baseline import (
    _rerank_for_metrics,
    _validate_split_integrity,
    feature_dict,
    run,
)


def _row(
    query_id: str,
    rank: int,
    utility: int,
    relation: str,
    split: str,
    *,
    lexical_score: float = 0.5,
    dense_similarity: float = 0.5,
) -> dict:
    word = query_id.split("#")[0]
    return {
        "schema_version": 3,
        "pair_id": f"{query_id}-{rank}-{split}",
        "query_id": query_id,
        "query": {
            "word": word,
            "sense": 1,
            "definition": "target",
            "category": "test",
        },
        "candidate": {
            "word": f"c{rank}",
            "sense": 1,
            "definition": "candidate",
        },
        "retrieval": {
            "system": "v2.5",
            "v25_rank": rank,
            "score": 1.0 / rank,
            "relation_tier": 1,
            "relation_hint": "test_hint",
            "lexical_form": "standalone",
            "sense_resolution": "selected_sense_supported",
            "lexical_score": lexical_score,
            "lexical_rank": rank,
            "dense_similarity": dense_similarity,
            "dense_rank": rank,
        },
        "annotation": {
            "utility": utility,
            "semantic_relation": relation,
            "style_tags": [],
            "legacy_relation": None,
            "notes": "",
        },
        "split": split,
    }


class WriterRelevancePhase3BaselineTests(unittest.TestCase):
    def test_feature_dict_uses_retrieval_evidence_not_labels(self) -> None:
        row = _row("ฝน#1", 2, 3, "direct", "train")
        features = feature_dict(row)

        self.assertIn("v25_rr", features)
        self.assertIn("dense_similarity", features)
        self.assertIn("relation_hint", features)
        self.assertNotIn("utility", features)
        self.assertNotIn("semantic_relation", features)

    def test_split_integrity_rejects_target_leakage(self) -> None:
        rows = [
            _row("ฝน#1", 1, 3, "direct", "train"),
            _row("ฝน#1", 2, 2, "direct", "validation"),
        ]
        with self.assertRaisesRegex(ValueError, "leakage"):
            _validate_split_integrity(rows)

    def test_rerank_preserves_original_rank_and_assigns_model_rank(self) -> None:
        rows = [
            _row("ฝน#1", 1, 1, "weak_related", "validation"),
            _row("ฝน#1", 2, 3, "direct", "validation"),
        ]
        reranked = _rerank_for_metrics(rows, np.array([0.1, 0.9]))

        by_candidate = {row["candidate"]["word"]: row for row in reranked}
        self.assertEqual(
            by_candidate["c2"]["retrieval"]["v25_rank"],
            1,
        )
        self.assertEqual(
            by_candidate["c2"]["retrieval"]["original_v25_rank"],
            2,
        )

    def test_run_is_validation_only_and_never_evaluates_benchmark(self) -> None:
        rows = [
            _row("train-a#1", 1, 3, "direct", "train", lexical_score=0.9),
            _row("train-a#1", 2, 0, "unrelated", "train", lexical_score=0.1),
            _row("train-b#1", 1, 2, "direct", "train", dense_similarity=0.8),
            _row("train-b#1", 2, 1, "weak_related", "train", dense_similarity=0.2),
            _row("valid#1", 1, 0, "unrelated", "validation", lexical_score=0.1),
            _row("valid#1", 2, 3, "direct", "validation", lexical_score=0.9),
            _row("benchmark#1", 1, 3, "direct", "benchmark"),
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            args = SimpleNamespace(
                input=str(path),
                output=None,
                k=2,
                seed=42,
                severe_penalty=1.0,
            )
            report = run(args)

        self.assertEqual(report["train_pair_count"], 4)
        self.assertEqual(report["validation_pair_count"], 2)
        self.assertFalse(report["benchmark_split_evaluated"])


if __name__ == "__main__":
    unittest.main()
