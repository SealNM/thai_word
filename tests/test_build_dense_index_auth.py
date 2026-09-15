from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from scripts.build_dense_index import _gated_model_help, _looks_like_gated_hf_error


class FakeGatedRepoError(Exception):
    pass


class BuildDenseIndexAuthTests(unittest.TestCase):
    def test_detects_gated_repo_error_by_name(self) -> None:
        FakeGatedRepoError.__name__ = "GatedRepoError"
        self.assertTrue(_looks_like_gated_hf_error(FakeGatedRepoError("restricted")))

    def test_detects_nested_401_embeddinggemma_error(self) -> None:
        try:
            try:
                raise RuntimeError("401 Unauthorized: EmbeddingGemma restricted gated repo")
            except RuntimeError as inner:
                raise ValueError("model load failed") from inner
        except ValueError as outer:
            self.assertTrue(_looks_like_gated_hf_error(outer))

    def test_unrelated_error_is_not_classified_as_gated(self) -> None:
        self.assertFalse(_looks_like_gated_hf_error(FileNotFoundError("missing index")))

    def test_help_preserves_locked_embedding_profile(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            message = _gated_model_help("embeddinggemma-300m-256")

        self.assertIn("google/embeddinggemma-300m", message)
        self.assertIn("HF_TOKEN: not set", message)
        self.assertIn("hf auth login", message)
        self.assertIn("embeddinggemma-300m-256", message)

    def test_help_reports_token_presence_without_exposing_token(self) -> None:
        secret = "hf_do_not_print_me"
        with patch.dict(os.environ, {"HF_TOKEN": secret}):
            message = _gated_model_help("embeddinggemma-300m-256")

        self.assertIn("HF_TOKEN: set", message)
        self.assertNotIn(secret, message)


if __name__ == "__main__":
    unittest.main()
