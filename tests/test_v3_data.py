from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from thai_dense_v2 import build_dense_index
from thai_lexical_v1 import SearchArtifacts
from thai_v3_data import (
    compile_training_data,
    load_holdout_words,
    validate_teacher_record,
)


class _FakeNativeModel:
    device = "cpu"

    def encode_document(self, texts, **kwargs):
        rows = []
        for index, _ in enumerate(texts):
            vector = np.zeros(4, dtype=np.float32)
            vector[index % 4] = 1.0
            rows.append(vector)
        return np.stack(rows)


def _teacher_row() -> dict:
    return {
        "seed_id": "demo",
        "anchor": {
            "word": "เอ่ย",
            "sense": 1,
            "definition": "พูดออกมา",
        },
        "candidates": [
            {
                "word": "กล่าว",
                "definition": "พูด",
                "evidence": {"dense_rank": 1, "dense_similarity": 0.90},
                "judgment": {
                    "relation": "synonym",
                    "semantic_relatedness": 4,
                    "replaceability": 3,
                    "confidence": 0.95,
                    "register": "neutral",
                    "reason": "ใช้แทนกันได้โดยตรง",
                },
            },
            {
                "word": "เสียง",
                "definition": "สิ่งที่ได้ยิน",
                "evidence": {"dense_rank": 2, "dense_similarity": 0.82},
                "judgment": {
                    "relation": "associated",
                    "semantic_relatedness": 2,
                    "replaceability": 0,
                    "confidence": 0.92,
                    "register": "neutral",
                    "reason": "เกี่ยวข้องกับการพูดแต่ไม่ใช่คำแทน",
                },
            },
            {
                "word": "กระซิบ",
                "definition": "พูดเสียงเบา",
                "evidence": {"dense_rank": 3, "dense_similarity": 0.79},
                "judgment": {
                    "relation": "manner",
                    "semantic_relatedness": 3,
                    "replaceability": 1,
                    "confidence": 0.90,
                    "register": "neutral",
                    "reason": "เป็นลักษณะเฉพาะของการพูด",
                },
            },
        ],
    }


class V3DataTests(unittest.TestCase):
    def test_holdout_loader_and_leakage_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "holdout.json"
            path.write_text(
                json.dumps({"words": ["พูด", "รัก"]}, ensure_ascii=False),
                encoding="utf-8",
            )
            holdout = load_holdout_words(path)

        self.assertEqual(holdout, {"พูด", "รัก"})

        row = _teacher_row()
        row["candidates"][0]["word"] = "รัก"
        errors = validate_teacher_record(
            row,
            dictionary_words={"เอ่ย", "รัก", "เสียง", "กระซิบ"},
            holdout_words=holdout,
        )
        self.assertTrue(any("holdout leakage" in error for error in errors))

    def test_compile_uses_synonyms_and_hard_negatives_but_not_manner(self) -> None:
        row = _teacher_row()
        triplets, graded, report = compile_training_data(
            [row],
            holdout_words={"พูด", "รัก"},
            dictionary_words={"เอ่ย", "กล่าว", "เสียง", "กระซิบ"},
            negatives_per_positive=2,
        )

        self.assertEqual(
            triplets,
            [
                {
                    "anchor": "เอ่ย: พูดออกมา",
                    "positive": "กล่าว: พูด",
                    "negative": "เสียง: สิ่งที่ได้ยิน",
                }
            ],
        )
        self.assertEqual(len(graded), 3)
        self.assertEqual(report["relation_counts"]["manner"], 1)

    def test_custom_v3_profile_preserves_native_retrieval_metadata(self) -> None:
        lexical = SearchArtifacts(
            entries=[
                {"word": "เอ่ย", "source_ids": [1], "sense_count": 1},
                {"word": "กล่าว", "source_ids": [2], "sense_count": 1},
            ],
            senses=[
                {"entry_index": 0, "sense_index": 1, "definition": "พูดออกมา"},
                {"entry_index": 1, "sense_index": 1, "definition": "พูด"},
            ],
            word_to_index={"เอ่ย": 0, "กล่าว": 1},
            vectorizer=None,
            matrix=None,
            references=[[], []],
            reverse_references=[[], []],
            token_sets=[set(), set()],
            entry_to_senses=[[0], [1]],
            tokenize=lambda text: text.split(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            with patch("thai_dense_v2.load_model", return_value=_FakeNativeModel()):
                metadata = build_dense_index(
                    lexical,
                    tmp,
                    model="models/thai-words-embeddinggemma-v3/final",
                    batch_size=2,
                    device="cpu",
                    profile_overrides={
                        "key": "thai-words-v3-256",
                        "query_method": "encode_query",
                        "document_method": "encode_document",
                        "truncate_dim": 256,
                    },
                )

        self.assertEqual(metadata["model_key"], "thai-words-v3-256")
        self.assertEqual(metadata["query_method"], "encode_query")
        self.assertEqual(metadata["document_method"], "encode_document")
        self.assertEqual(metadata["truncate_dim"], 256)


if __name__ == "__main__":
    unittest.main()
