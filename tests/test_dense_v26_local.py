from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from thai_dense_v2 import (
    DenseEncoder,
    QWEN3_THAI_LEXICAL_TASK,
    _encode_documents,
    resolve_model_profile,
)


class _FakeEncodeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict]] = []

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            values = [texts]
        else:
            values = list(texts)
        self.calls.append((values, dict(kwargs)))
        if len(values) == 1:
            return np.array([[1.0, 0.0]], dtype=np.float32)
        return np.array(
            [[1.0, 0.0] for _ in values],
            dtype=np.float32,
        )


class DenseV26LocalEmbeddingTests(unittest.TestCase):
    def test_qwen3_profile_is_256d_and_instruction_aware(self) -> None:
        profile = resolve_model_profile("qwen3-embedding-0.6b-256")

        self.assertEqual(profile["model_id"], "Qwen/Qwen3-Embedding-0.6B")
        self.assertFalse(profile["trust_remote_code"])
        self.assertEqual(profile["truncate_dim"], 256)
        self.assertIn("Instruct:", profile["query_prefix"])
        self.assertIn(QWEN3_THAI_LEXICAL_TASK, profile["query_prefix"])
        self.assertTrue(profile["query_prefix"].endswith("\nQuery:"))
        self.assertEqual(profile["document_prefix"], "")

    def test_arctic_profile_matches_official_query_prefix(self) -> None:
        profile = resolve_model_profile("arctic-embed-l-v2-256")

        self.assertEqual(
            profile["model_id"],
            "Snowflake/snowflake-arctic-embed-l-v2.0",
        )
        self.assertFalse(profile["trust_remote_code"])
        self.assertEqual(profile["truncate_dim"], 256)
        self.assertEqual(profile["max_seq_length"], 512)
        self.assertEqual(profile["query_prefix"], "query: ")
        self.assertEqual(profile["document_prefix"], "")

    def test_local_challenger_documents_have_no_query_prefix(self) -> None:
        model = _FakeEncodeModel()
        profile = resolve_model_profile("qwen3-embedding-0.6b-256")

        vectors = _encode_documents(
            model,
            profile,
            ["บ้าน: ที่อยู่อาศัย", "เรือน: สิ่งปลูกสร้างสำหรับอยู่อาศัย"],
            batch_size=2,
        )

        self.assertEqual(
            model.calls[0][0],
            [
                "บ้าน: ที่อยู่อาศัย",
                "เรือน: สิ่งปลูกสร้างสำหรับอยู่อาศัย",
            ],
        )
        self.assertEqual(vectors.shape, (2, 2))

    def test_dense_encoder_applies_qwen_instruction_only_to_query(self) -> None:
        model = _FakeEncodeModel()
        profile = resolve_model_profile("qwen3-embedding-0.6b-256")
        metadata = {
            "model_id": profile["model_id"],
            "trust_remote_code": profile["trust_remote_code"],
            "query_prefix": profile["query_prefix"],
            "document_prefix": profile["document_prefix"],
            "query_method": "encode",
            "document_method": "encode",
            "truncate_dim": profile["truncate_dim"],
        }

        with patch("thai_dense_v2.load_model", return_value=model):
            encoder = DenseEncoder(metadata)
            vector = encoder.encode_query("  บ้าน: ที่อยู่อาศัย  ")

        encoded_text = model.calls[0][0][0]
        self.assertTrue(encoded_text.startswith("Instruct:"))
        self.assertTrue(encoded_text.endswith("Query:บ้าน: ที่อยู่อาศัย"))
        np.testing.assert_array_equal(
            vector,
            np.array([1.0, 0.0], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
