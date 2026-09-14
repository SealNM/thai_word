from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from thai_dense_v26 import (
    _response_matrix,
    prepare_document_text,
    prepare_query_text,
    validate_dimension,
)


class DenseV26Tests(unittest.TestCase):
    def test_query_uses_search_result_task_prefix(self) -> None:
        self.assertEqual(
            prepare_query_text("บ้าน: ที่อยู่อาศัย"),
            "task: search result | query: บ้าน: ที่อยู่อาศัย",
        )

    def test_document_uses_title_and_definition(self) -> None:
        self.assertEqual(
            prepare_document_text("เรือน", "บ้าน ที่อยู่อาศัย"),
            "title: เรือน | text: บ้าน ที่อยู่อาศัย",
        )

    def test_document_uses_none_for_missing_title(self) -> None:
        self.assertEqual(
            prepare_document_text("", "ที่อยู่อาศัย"),
            "title: none | text: ที่อยู่อาศัย",
        )

    def test_dimension_accepts_256_and_768(self) -> None:
        self.assertEqual(validate_dimension(256), 256)
        self.assertEqual(validate_dimension(768), 768)

    def test_dimension_rejects_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            validate_dimension(64)
        with self.assertRaises(ValueError):
            validate_dimension(4096)

    def test_response_matrix_normalizes_rows(self) -> None:
        response = SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[3.0, 4.0]),
                SimpleNamespace(values=[0.0, 2.0]),
            ]
        )
        matrix = _response_matrix(response, expected=2, dimension=2)
        self.assertEqual(matrix.shape, (2, 2))
        np.testing.assert_allclose(
            np.linalg.norm(matrix, axis=1),
            np.ones(2),
            atol=1e-6,
        )

    def test_response_matrix_checks_embedding_count(self) -> None:
        response = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[1.0, 0.0])]
        )
        with self.assertRaises(ValueError):
            _response_matrix(response, expected=2, dimension=2)


if __name__ == "__main__":
    unittest.main()
