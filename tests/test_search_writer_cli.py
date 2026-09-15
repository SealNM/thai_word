from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.search_writer import _base_artifact_error, _missing_base_artifacts


class SearchWriterCliArtifactTests(unittest.TestCase):
    def test_reports_missing_v1_and_dense_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "v1"
            dense = root / "dense"

            missing = _missing_base_artifacts(index, dense)
            message = _base_artifact_error(
                index=index,
                dense_index=dense,
                dense_device="cuda",
            )

            self.assertEqual(len(missing), 5)
            self.assertIn("entries.json", message)
            self.assertIn("dense_embeddings.npy", message)
            self.assertIn("python scripts/build_index.py", message)
            self.assertIn("--model embeddinggemma-300m-256", message)
            self.assertIn("--device cuda", message)

    def test_no_error_when_required_base_artifacts_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "v1"
            dense = root / "dense"
            index.mkdir()
            dense.mkdir()

            for name in ("entries.json", "senses.json", "metadata.json"):
                (index / name).write_text("{}", encoding="utf-8")
            (dense / "dense_metadata.json").write_text("{}", encoding="utf-8")
            (dense / "dense_embeddings.npy").write_bytes(b"placeholder")

            self.assertEqual(_missing_base_artifacts(index, dense), [])
            self.assertEqual(
                _base_artifact_error(
                    index=index,
                    dense_index=dense,
                    dense_device=None,
                ),
                "",
            )


if __name__ == "__main__":
    unittest.main()
