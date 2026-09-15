from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.writer_relevance_phase3_frozen_benchmark import (
    _fixed_hybrid_scores,
    _validate_dataset_against_freeze,
    _validate_locked_manifest,
    run,
)


def _row(query_id: str, split: str, rank: int = 1) -> dict:
    return {
        "pair_id": f"{query_id}-{rank}-{split}",
        "query_id": query_id,
        "split": split,
        "query": {"word": query_id, "category": "test"},
        "candidate": {"word": f"candidate-{rank}"},
        "retrieval": {"v25_rank": rank},
        "annotation": {"utility": 3, "semantic_relation": "direct"},
    }


class WriterRelevancePhase3FrozenBenchmarkTests(unittest.TestCase):
    def test_locked_manifest_requires_no_reselection_policy(self) -> None:
        manifest = {
            "status": "locked_before_frozen_benchmark",
            "hybrid": {
                "alpha": 0.5,
                "alpha_must_not_be_reselected_on_benchmark": True,
            },
            "benchmark_policy": {
                "no_hyperparameter_changes_after_opening": True,
                "no_alpha_search": True,
                "no_model_selection": True,
            },
        }
        _validate_locked_manifest(manifest)

        manifest["hybrid"]["alpha_must_not_be_reselected_on_benchmark"] = False
        with self.assertRaises(ValueError):
            _validate_locked_manifest(manifest)

    def test_fixed_hybrid_uses_supplied_alpha_only(self) -> None:
        rows = [
            _row("a#1", "benchmark", 1),
            _row("a#1", "benchmark", 2),
            _row("a#1", "benchmark", 3),
        ]
        learned = np.asarray([3.0, 2.0, 1.0])
        neural = np.asarray([1.0, 2.0, 3.0])

        scores = _fixed_hybrid_scores(
            rows,
            learned,
            neural,
            alpha=0.5,
        )
        np.testing.assert_allclose(scores, [0.5, 0.5, 0.5])

    def test_dataset_validation_requires_frozen_hash_and_queries(self) -> None:
        rows = [
            _row("train#1", "train"),
            _row("valid#1", "validation"),
            _row("bench#1", "benchmark"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = {"dataset": {"sha256": digest}}
            split_manifest = {
                "splits": {
                    "train": {
                        "pair_count": 1,
                        "queries": ["train#1"],
                    },
                    "validation": {
                        "pair_count": 1,
                        "queries": ["valid#1"],
                    },
                    "benchmark": {
                        "pair_count": 1,
                        "queries": ["bench#1"],
                    },
                }
            }
            _validate_dataset_against_freeze(
                rows,
                input_path=path,
                manifest=manifest,
                split_manifest=split_manifest,
            )

            manifest["dataset"]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                _validate_dataset_against_freeze(
                    rows,
                    input_path=path,
                    manifest=manifest,
                    split_manifest=split_manifest,
                )

    def test_dataset_validation_rejects_changed_benchmark_query(self) -> None:
        rows = [
            _row("train#1", "train"),
            _row("valid#1", "validation"),
            _row("bench#1", "benchmark"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = {"dataset": {"sha256": digest}}
            split_manifest = {
                "splits": {
                    "train": {"pair_count": 1, "queries": ["train#1"]},
                    "validation": {"pair_count": 1, "queries": ["valid#1"]},
                    "benchmark": {"pair_count": 1, "queries": ["other#1"]},
                }
            }
            with self.assertRaises(ValueError):
                _validate_dataset_against_freeze(
                    rows,
                    input_path=path,
                    manifest=manifest,
                    split_manifest=split_manifest,
                )

    def test_run_refuses_to_open_benchmark_without_confirmation(self) -> None:
        args = SimpleNamespace(confirm_frozen_benchmark=False)
        with self.assertRaises(ValueError):
            run(args)


if __name__ == "__main__":
    unittest.main()
