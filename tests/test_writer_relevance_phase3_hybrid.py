from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.writer_relevance_phase3_hybrid import (
    _load_neural_scores,
    _parse_alphas,
    _query_rank_normalize,
    run,
)


def _row(
    query_id: str,
    rank: int,
    utility: int,
    relation: str,
    split: str,
) -> dict:
    word = query_id.split("#")[0]
    return {
        "schema_version": 3,
        "pair_id": f"{query_id}-{rank}-{split}",
        "query_id": query_id,
        "query": {
            "word": word,
            "sense": 1,
            "definition": f"target definition {word}",
            "category": "test-category",
        },
        "candidate": {
            "word": f"c{rank}-{word}",
            "sense": 1,
            "definition": f"candidate definition {rank}",
        },
        "retrieval": {
            "system": "v2.5",
            "v25_rank": rank,
            "score": 1.0 / rank,
            "relation_tier": 1,
            "relation_hint": "retrieval_hint",
            "lexical_form": "standalone",
            "sense_resolution": "selected_sense_supported",
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


class WriterRelevancePhase3HybridTests(unittest.TestCase):
    def test_query_rank_normalize_is_per_query_and_preserves_ties(self) -> None:
        rows = [
            _row("a#1", 1, 3, "direct", "validation"),
            _row("a#1", 2, 2, "direct", "validation"),
            _row("a#1", 3, 1, "weak_related", "validation"),
            _row("b#1", 1, 3, "direct", "validation"),
            _row("b#1", 2, 2, "direct", "validation"),
        ]
        scores = np.asarray([3.0, 1.0, 2.0, 5.0, 5.0])
        normalized = _query_rank_normalize(rows, scores)

        np.testing.assert_allclose(normalized[:3], [1.0, 0.0, 0.5])
        np.testing.assert_allclose(normalized[3:], [0.5, 0.5])

    def test_neural_score_loader_rejects_non_validation_or_unknown_pairs(self) -> None:
        validation = [_row("valid#1", 1, 3, "direct", "validation")]
        with tempfile.TemporaryDirectory() as directory:
            score_path = Path(directory) / "scores.jsonl"
            score_path.write_text(
                json.dumps(
                    {
                        "pair_id": "benchmark-secret",
                        "query_id": "benchmark#1",
                        "split": "benchmark",
                        "score": 0.9,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _load_neural_scores(score_path, validation)

    def test_neural_score_loader_requires_every_validation_pair(self) -> None:
        validation = [
            _row("valid#1", 1, 3, "direct", "validation"),
            _row("valid#1", 2, 2, "direct", "validation"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            score_path = Path(directory) / "scores.jsonl"
            score_path.write_text(
                json.dumps(
                    {
                        "pair_id": validation[0]["pair_id"],
                        "query_id": validation[0]["query_id"],
                        "split": "validation",
                        "score": 0.9,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _load_neural_scores(score_path, validation)

    def test_run_blends_validation_only_and_never_selects_on_benchmark(self) -> None:
        rows = [
            _row("train-a#1", 1, 3, "direct", "train"),
            _row("train-a#1", 2, 0, "unrelated", "train"),
            _row("train-b#1", 1, 2, "direct", "train"),
            _row("train-b#1", 2, 1, "weak_related", "train"),
            _row("valid#1", 1, 0, "unrelated", "validation"),
            _row("valid#1", 2, 2, "direct", "validation"),
            _row("valid#1", 3, 3, "direct", "validation"),
            _row("benchmark-secret#1", 1, 3, "direct", "benchmark"),
        ]
        validation = [row for row in rows if row["split"] == "validation"]
        neural = [0.1, 0.8, 0.9]

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "rows.jsonl"
            score_path = directory_path / "scores.jsonl"
            with input_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            with score_path.open("w", encoding="utf-8") as handle:
                for row, score in zip(validation, neural):
                    handle.write(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "pair_id": row["pair_id"],
                                "query_id": row["query_id"],
                                "split": "validation",
                                "score": score,
                                "model": "fake/model",
                            }
                        )
                        + "\n"
                    )

            args = SimpleNamespace(
                input=str(input_path),
                neural_scores=str(score_path),
                output=None,
                k=3,
                seed=42,
                severe_penalty=1.0,
                alphas=_parse_alphas("0,0.25,0.5"),
                include_per_query=False,
            )
            report = run(args)

        self.assertFalse(report["benchmark_split_evaluated"])
        self.assertEqual(report["benchmark_pairs_used_for_training"], 0)
        self.assertEqual(report["benchmark_pairs_used_for_selection"], 0)
        self.assertEqual(report["neural_scores"]["score_count"], 3)
        self.assertEqual(len(report["hybrid_grid"]), 3)
        self.assertIn(report["selected_validation_alpha"], {0.0, 0.25, 0.5})

    def test_alpha_parser_rejects_out_of_range_weights(self) -> None:
        self.assertEqual(_parse_alphas("0,0.2,0.5"), [0.0, 0.2, 0.5])
        with self.assertRaises(ValueError):
            _parse_alphas("0,1.1")


if __name__ == "__main__":
    unittest.main()
