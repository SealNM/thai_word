from __future__ import annotations

import unittest

import numpy as np

from thai_pythainlp_wv import mean_token_vector


class FakeWordVectors:
    vector_size = 2

    def __init__(self):
        self.key_to_index = {
            "ดี": 0,
            "งาม": 1,
        }
        self.vectors = {
            "ดี": np.asarray([1.0, 0.0], dtype=np.float32),
            "งาม": np.asarray([0.0, 1.0], dtype=np.float32),
        }

    def get_vector(self, word):
        return self.vectors[word]


class PyThaiNLPWordVectorTests(unittest.TestCase):
    def test_mean_token_vector_normalizes_mean(self):
        vector, hits, total = mean_token_vector(
            "ดี งาม",
            model=FakeWordVectors(),
            tokenizer=lambda text: text.split(),
        )
        self.assertEqual(hits, 2)
        self.assertEqual(total, 2)
        self.assertIsNotNone(vector)
        np.testing.assert_allclose(
            vector,
            np.asarray([2 ** -0.5, 2 ** -0.5], dtype=np.float32),
            rtol=1e-5,
        )

    def test_oov_tokens_are_ignored(self):
        vector, hits, total = mean_token_vector(
            "ดี ไม่มี",
            model=FakeWordVectors(),
            tokenizer=lambda text: text.split(),
        )
        self.assertEqual(hits, 1)
        self.assertEqual(total, 2)
        np.testing.assert_allclose(
            vector,
            np.asarray([1.0, 0.0], dtype=np.float32),
        )

    def test_all_oov_returns_none(self):
        vector, hits, total = mean_token_vector(
            "ไม่มี",
            model=FakeWordVectors(),
            tokenizer=lambda text: text.split(),
        )
        self.assertIsNone(vector)
        self.assertEqual(hits, 0)
        self.assertEqual(total, 1)


if __name__ == "__main__":
    unittest.main()
