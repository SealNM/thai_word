from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.writer_relevance_phase3_crossencoder import (
    _configure_cuda_visibility,
    _delta_summary,
    _save_model_checkpoint,
    _write_score_output,
    text_pair,
    utility_target,
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
            "style_tags": ["literary"] if utility == 3 else [],
            "legacy_relation": None,
            "notes": "human-only note",
        },
        "split": split,
    }


class FakeModel:
    def __init__(self) -> None:
        self.seen_pairs: list[tuple[str, str]] = []
        self.device = "cpu"

    def predict(self, pairs, **kwargs):
        self.seen_pairs = list(pairs)
        return np.linspace(0.1, 0.9, num=len(self.seen_pairs))


class WriterRelevancePhase3CrossEncoderTests(unittest.TestCase):
    def test_explicit_checkpoint_save_writes_files(self) -> None:
        class SaveModel:
            def save_pretrained(self, path: str) -> None:
                target = Path(path)
                target.mkdir(parents=True, exist_ok=True)
                (target / "config.json").write_text("{}", encoding="utf-8")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "saved-model"
            saved = _save_model_checkpoint(SaveModel(), output)
            self.assertEqual(saved, str(output))
            self.assertTrue((output / "config.json").exists())

    def test_cuda_visibility_can_isolate_single_gpu_before_torch_import(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            self.assertEqual(_configure_cuda_visibility("0"), "0")
            import os
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")

    def test_cuda_visibility_rejects_invalid_indices(self) -> None:
        with self.assertRaises(ValueError):
            _configure_cuda_visibility("gpu0")
        with self.assertRaises(ValueError):
            _configure_cuda_visibility("0,0")

    def test_text_pair_contains_bounded_retrieval_evidence_not_human_labels(self) -> None:
        row = _row("ฝน#1", 2, 3, "direct", "train")
        left, right = text_pair(row)
        combined = left + "\n" + right

        self.assertIn("ฝน", combined)
        self.assertIn("v25_rank=2", combined)
        self.assertIn("retrieval_hint", combined)
        self.assertNotIn("human-only note", combined)
        self.assertNotIn("literary", combined)
        self.assertNotIn("semantic_relation", combined)
        self.assertNotIn("utility", combined)

    def test_utility_target_maps_ordinal_labels_to_zero_one_range(self) -> None:
        self.assertEqual(utility_target(_row("a#1", 1, 0, "unrelated", "train")), 0.0)
        self.assertEqual(utility_target(_row("a#1", 1, 3, "direct", "train")), 1.0)
        self.assertAlmostEqual(
            utility_target(_row("a#1", 1, 2, "direct", "train")),
            2.0 / 3.0,
        )

    def test_delta_summary_preserves_metric_direction(self) -> None:
        candidate = {
            "ndcg_at_k_mean": 0.9,
            "severe_error_rate_at_k_mean": 0.02,
        }
        reference = {
            "ndcg_at_k_mean": 0.8,
            "severe_error_rate_at_k_mean": 0.05,
        }
        delta = _delta_summary(candidate, reference)
        self.assertAlmostEqual(delta["ndcg_at_k_mean"], 0.1)
        self.assertAlmostEqual(delta["severe_error_rate_at_k_mean"], -0.03)

    def test_validation_score_artifact_contains_no_human_labels(self) -> None:
        rows = [_row("valid#1", 1, 3, "direct", "validation")]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scores.jsonl"
            _write_score_output(output, rows, np.asarray([0.75]), model="fake/model")
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["pair_id"], rows[0]["pair_id"])
        self.assertEqual(payload["split"], "validation")
        self.assertEqual(payload["model"], "fake/model")
        self.assertNotIn("annotation", payload)
        self.assertNotIn("utility", payload)
        self.assertNotIn("semantic_relation", payload)

    def test_run_never_trains_or_predicts_on_benchmark_rows(self) -> None:
        rows = [
            _row("train-a#1", 1, 3, "direct", "train"),
            _row("train-a#1", 2, 0, "unrelated", "train"),
            _row("train-b#1", 1, 2, "direct", "train"),
            _row("train-b#1", 2, 1, "weak_related", "train"),
            _row("valid#1", 1, 0, "unrelated", "validation"),
            _row("valid#1", 2, 3, "direct", "validation"),
            _row("benchmark-secret#1", 1, 3, "direct", "benchmark"),
        ]

        fake_model = FakeModel()
        captured_train: list[dict] = []

        def fake_fit(train, args):
            captured_train.extend(train)
            return fake_model, 1

        floor = {
            "ordinal_minus_severe": {
                "useful_rate_at_k_mean": 1.0,
                "high_utility_rate_at_k_mean": 0.5,
                "noise_rate_at_k_mean": 0.0,
                "severe_error_rate_at_k_mean": 0.0,
                "relation_diversity_at_k_mean": 1.0,
                "ndcg_at_k_mean": 0.8,
                "mrr_high_utility_mean": 1.0,
            }
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "rows.jsonl"
            floor_path = directory_path / "floor.json"
            with input_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            floor_path.write_text(json.dumps(floor), encoding="utf-8")

            args = SimpleNamespace(
                input=str(input_path),
                output=None,
                model="fake/model",
                model_output=None,
                learned_floor_report=str(floor_path),
                k=2,
                epochs=1,
                batch_size=2,
                eval_batch_size=2,
                learning_rate=2e-5,
                warmup_ratio=0.1,
                max_length=128,
                seed=42,
                include_per_query=False,
                no_progress=True,
            )
            with patch(
                "scripts.writer_relevance_phase3_crossencoder._fit_model",
                side_effect=fake_fit,
            ):
                report = run(args)

        self.assertEqual({row["split"] for row in captured_train}, {"train"})
        predicted_text = "\n".join(left + right for left, right in fake_model.seen_pairs)
        self.assertIn("valid", predicted_text)
        self.assertNotIn("benchmark-secret", predicted_text)
        self.assertEqual(report["benchmark_pairs_used_for_training"], 0)
        self.assertFalse(report["benchmark_split_evaluated"])
        self.assertIn("delta_vs_learned_floor", report)

    def test_no_finetune_scores_validation_without_fitting_train_labels(self) -> None:
        rows = [
            _row("train#1", 1, 3, "direct", "train"),
            _row("train#1", 2, 0, "unrelated", "train"),
            _row("valid#1", 1, 3, "direct", "validation"),
            _row("valid#1", 2, 0, "unrelated", "validation"),
            _row("benchmark-secret#1", 1, 3, "direct", "benchmark"),
        ]
        fake_model = FakeModel()
        floor = {
            "ordinal_minus_severe": {
                "useful_rate_at_k_mean": 0.5,
                "high_utility_rate_at_k_mean": 0.5,
                "noise_rate_at_k_mean": 0.5,
                "severe_error_rate_at_k_mean": 0.5,
                "relation_diversity_at_k_mean": 1.0,
                "ndcg_at_k_mean": 0.5,
                "mrr_high_utility_mean": 1.0,
            }
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "rows.jsonl"
            floor_path = directory_path / "floor.json"
            score_path = directory_path / "scores.jsonl"
            with input_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            floor_path.write_text(json.dumps(floor), encoding="utf-8")

            args = SimpleNamespace(
                input=str(input_path),
                output=None,
                score_output=str(score_path),
                model="fake/model",
                model_output=None,
                learned_floor_report=str(floor_path),
                k=2,
                epochs=2,
                batch_size=2,
                eval_batch_size=2,
                learning_rate=2e-5,
                warmup_ratio=0.1,
                max_length=128,
                seed=42,
                device=None,
                trust_remote_code=False,
                use_amp=False,
                no_finetune=True,
                include_per_query=False,
                no_progress=True,
            )
            with patch(
                "scripts.writer_relevance_phase3_crossencoder._load_model",
                return_value=fake_model,
            ), patch(
                "scripts.writer_relevance_phase3_crossencoder._fit_model"
            ) as fit_mock:
                report = run(args)

            score_rows = [
                json.loads(line)
                for line in score_path.read_text(encoding="utf-8").splitlines()
            ]

        fit_mock.assert_not_called()
        self.assertFalse(report["finetuned"])
        self.assertEqual(report["model_fit_pair_count"], 0)
        self.assertEqual({row["split"] for row in score_rows}, {"validation"})
        self.assertEqual(len(score_rows), 2)
        self.assertNotIn("benchmark-secret", json.dumps(score_rows, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
