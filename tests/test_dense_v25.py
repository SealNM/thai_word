from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from thai_dense_v2 import (
    DenseEncoder,
    _encode_documents,
    resolve_model_profile,
)


class _FakeRetrievalModel:
    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def encode_document(self, texts, **kwargs):
        self.document_calls.append(list(texts))
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    def encode_query(self, text, **kwargs):
        self.query_calls.append(text)
        return np.array([0.5, 0.5], dtype=np.float32)


class DenseV25Tests(unittest.TestCase):
    def test_embeddinggemma_profiles_use_native_retrieval_methods(self) -> None:
        full = resolve_model_profile("embeddinggemma-300m")
        compact = resolve_model_profile("embeddinggemma-300m-256")

        self.assertEqual(full["model_id"], "google/embeddinggemma-300m")
        self.assertEqual(full["query_method"], "encode_query")
        self.assertEqual(full["document_method"], "encode_document")
        self.assertIsNone(full["truncate_dim"])
        self.assertEqual(compact["truncate_dim"], 256)

    def test_embeddinggemma_documents_use_encode_document_without_manual_prefix(self) -> None:
        model = _FakeRetrievalModel()
        profile = resolve_model_profile("embeddinggemma-300m")

        vectors = _encode_documents(
            model,
            profile,
            ["พูด: เปล่งเสียงออกเป็นถ้อยคำ", "ฝน: น้ำที่ตกจากฟ้า"],
            batch_size=2,
        )

        self.assertEqual(model.document_calls, [[
            "พูด: เปล่งเสียงออกเป็นถ้อยคำ",
            "ฝน: น้ำที่ตกจากฟ้า",
        ]])
        self.assertEqual(vectors.shape, (2, 2))
        self.assertEqual(vectors.dtype, np.float32)

    def test_dense_encoder_uses_native_query_method_from_metadata(self) -> None:
        model = _FakeRetrievalModel()
        metadata = {
            "model_id": "google/embeddinggemma-300m",
            "query_method": "encode_query",
            "document_method": "encode_document",
            "query_prefix": "",
            "document_prefix": "",
            "truncate_dim": 256,
        }

        with patch("thai_dense_v2.load_model", return_value=model):
            encoder = DenseEncoder(metadata)
            vector = encoder.encode_query("  พูด  ")

        self.assertEqual(model.query_calls, ["พูด"])
        np.testing.assert_array_equal(
            vector,
            np.array([0.5, 0.5], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
