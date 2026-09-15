from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from thai_writer_neural import NeuralWriterRanker, WRITER_RERANKER_MODEL_ENV


class FakeCrossEncoder:
    def __init__(self) -> None:
        self.device = "cpu"
        self.calls = 0

    def predict(self, pairs, *, batch_size, show_progress_bar):
        self.calls += 1
        return [0.25 + (index * 0.1) for index, _ in enumerate(pairs)]


def _runtime_row() -> dict:
    return {
        "query": {"word": "ฝน", "definition": "น้ำที่ตกจากเมฆ"},
        "candidate": {"word": "พิรุณ", "definition": "ฝน."},
        "retrieval": {
            "v25_rank": 1,
            "score": 0.03,
            "relation_tier": 5,
            "lexical_score": 0.9,
            "lexical_rank": 1,
            "dense_similarity": 0.75,
            "dense_rank": 2,
            "relation_hint": "direct_gloss_or_synonym",
            "lexical_form": "standalone",
            "sense_resolution": "selected_sense_supported",
        },
    }


class NeuralWriterRankerTests(unittest.TestCase):
    def test_model_is_lazy_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ranker = NeuralWriterRanker(directory)
            self.assertFalse(ranker.loaded)

            fake = FakeCrossEncoder()
            with patch("thai_writer_neural._create_crossencoder", return_value=fake) as create:
                scores = ranker.score_rows([_runtime_row()])

            self.assertTrue(ranker.loaded)
            create.assert_called_once()
            self.assertEqual(scores, [0.25])

    def test_environment_variable_can_supply_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {WRITER_RERANKER_MODEL_ENV: directory}):
                ranker = NeuralWriterRanker()
                fake = FakeCrossEncoder()
                with patch("thai_writer_neural._create_crossencoder", return_value=fake) as create:
                    ranker.score_rows([_runtime_row()])

            self.assertEqual(create.call_args.args[0], directory)

    def test_missing_local_checkpoint_does_not_download_by_default(self) -> None:
        ranker = NeuralWriterRanker("/definitely/missing/model")
        with patch("thai_writer_neural._create_crossencoder") as create:
            with self.assertRaises(FileNotFoundError):
                ranker.score_rows([_runtime_row()])
        create.assert_not_called()

    def test_explicit_allow_download_accepts_nonlocal_source(self) -> None:
        ranker = NeuralWriterRanker(
            "BAAI/bge-reranker-v2-m3",
            allow_download=True,
        )
        fake = FakeCrossEncoder()
        with patch("thai_writer_neural._create_crossencoder", return_value=fake) as create:
            ranker.score_rows([_runtime_row()])
        self.assertEqual(create.call_args.args[0], "BAAI/bge-reranker-v2-m3")

    def test_category_is_omitted_from_neural_runtime_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            captured = {}

            class CaptureModel(FakeCrossEncoder):
                def predict(self, pairs, *, batch_size, show_progress_bar):
                    captured["pairs"] = pairs
                    return [0.5]

            ranker = NeuralWriterRanker(directory)
            with patch("thai_writer_neural._create_crossencoder", return_value=CaptureModel()):
                ranker.score_rows([_runtime_row()])

            query_text, candidate_text = captured["pairs"][0]
            self.assertNotIn("หมวด:", query_text)
            self.assertIn("v25_rank=1", candidate_text)

    def test_warmup_loads_and_scores_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake = FakeCrossEncoder()
            ranker = NeuralWriterRanker(directory)
            with patch("thai_writer_neural._create_crossencoder", return_value=fake):
                info = ranker.warmup()

            self.assertTrue(info["loaded"])
            self.assertEqual(info["category_mode"], "omit")
            self.assertEqual(fake.calls, 1)

    def test_empty_input_does_not_load_model(self) -> None:
        ranker = NeuralWriterRanker("/missing")
        self.assertEqual(ranker.score_rows([]), [])
        self.assertFalse(ranker.loaded)


if __name__ == "__main__":
    unittest.main()
