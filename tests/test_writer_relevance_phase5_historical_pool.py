from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.writer_relevance_phase5_historical_pool import (
    build_historical_pool,
    sha256_file,
)
from thai_substitutability import read_jsonl, write_jsonl


def _row(
    query_id: str,
    pair_id: str,
    *,
    utility: int = 3,
    relation: str = "direct",
) -> dict:
    word = query_id.split("#", 1)[0]
    return {
        "schema_version": 3,
        "pair_id": pair_id,
        "query_id": query_id,
        "query": {
            "word": word,
            "sense": 1,
            "definition": f"definition for {word}",
            "category": "test",
        },
        "candidate": {
            "word": f"{word}-candidate",
            "sense": 1,
            "definition": "candidate definition",
        },
        "retrieval": {
            "system": "v2.5",
            "v25_rank": 1,
            "score": 0.5,
        },
        "annotation": {
            "utility": utility,
            "semantic_relation": relation,
            "style_tags": [],
            "legacy_relation": None,
            "notes": "",
        },
        "split": "train",
    }


class Phase5HistoricalPoolTests(unittest.TestCase):
    def _write_sources(
        self,
        root: Path,
        rows_50: list[dict],
        rows_phase4: list[dict],
    ) -> tuple[Path, Path]:
        source_50 = root / "source_50.jsonl"
        source_phase4 = root / "source_phase4.jsonl"
        write_jsonl(source_50, rows_50)
        write_jsonl(source_phase4, rows_phase4)
        return source_50, source_phase4

    def _build(
        self,
        root: Path,
        rows_50: list[dict],
        rows_phase4: list[dict],
    ) -> tuple[dict, Path, Path, Path, Path]:
        source_50, source_phase4 = self._write_sources(
            root, rows_50, rows_phase4
        )
        output = root / "combined.jsonl"
        manifest = root / "manifest.json"
        result = build_historical_pool(
            source_50,
            source_phase4,
            output,
            manifest,
            source_50_sha256=sha256_file(source_50),
            source_phase4_sha256=sha256_file(source_phase4),
            source_50_query_count=len({row["query_id"] for row in rows_50}),
            source_phase4_query_count=len(
                {row["query_id"] for row in rows_phase4}
            ),
            source_50_pair_count=len(rows_50),
            source_phase4_pair_count=len(rows_phase4),
        )
        return result, source_50, source_phase4, output, manifest

    def test_build_preserves_rows_and_adds_source_provenance(self) -> None:
        rows_50 = [
            _row("ฝน#1", "pair_a"),
            _row("รัก#1", "pair_b"),
        ]
        rows_phase4 = [_row("ทะเล#1", "pair_c")]
        originals = copy.deepcopy(rows_50 + rows_phase4)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, _, _, output, manifest_path = self._build(
                root, rows_50, rows_phase4
            )
            combined = read_jsonl(output)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(result["query_count"], 3)
        self.assertEqual(result["pair_count"], 3)
        self.assertEqual(manifest["output_sha256"], result["output_sha256"])
        self.assertEqual(
            [row["phase5_provenance"]["source"] for row in combined],
            [
                "writer_relevance_50",
                "writer_relevance_50",
                "phase4_consumed_holdout",
            ],
        )
        for original, enriched in zip(originals, combined):
            stripped = dict(enriched)
            stripped.pop("phase5_provenance")
            self.assertEqual(stripped, original)

    def test_duplicate_query_id_across_sources_is_rejected(self) -> None:
        rows_50 = [_row("ฝน#1", "pair_a")]
        rows_phase4 = [_row("ฝน#1", "pair_b")]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_50, source_phase4 = self._write_sources(
                root, rows_50, rows_phase4
            )
            output = root / "combined.jsonl"
            manifest = root / "manifest.json"

            with self.assertRaisesRegex(ValueError, "Duplicate query_id"):
                build_historical_pool(
                    source_50,
                    source_phase4,
                    output,
                    manifest,
                    source_50_sha256=sha256_file(source_50),
                    source_phase4_sha256=sha256_file(source_phase4),
                    source_50_query_count=1,
                    source_phase4_query_count=1,
                    source_50_pair_count=1,
                    source_phase4_pair_count=1,
                )

            self.assertFalse(output.exists())
            self.assertFalse(manifest.exists())

    def test_source_hash_mismatch_is_rejected_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_50, source_phase4 = self._write_sources(
                root,
                [_row("ฝน#1", "pair_a")],
                [_row("ทะเล#1", "pair_b")],
            )
            output = root / "combined.jsonl"
            manifest = root / "manifest.json"

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                build_historical_pool(
                    source_50,
                    source_phase4,
                    output,
                    manifest,
                    source_50_sha256="0" * 64,
                    source_phase4_sha256=sha256_file(source_phase4),
                    source_50_query_count=1,
                    source_phase4_query_count=1,
                    source_50_pair_count=1,
                    source_phase4_pair_count=1,
                )

            self.assertFalse(output.exists())
            self.assertFalse(manifest.exists())

    def test_severe_error_relation_requires_zero_utility(self) -> None:
        invalid = _row(
            "ฝน#1",
            "pair_a",
            utility=2,
            relation="unrelated",
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_50, source_phase4 = self._write_sources(
                root,
                [invalid],
                [_row("ทะเล#1", "pair_b")],
            )
            output = root / "combined.jsonl"
            manifest = root / "manifest.json"

            with self.assertRaisesRegex(ValueError, "requires utility 0"):
                build_historical_pool(
                    source_50,
                    source_phase4,
                    output,
                    manifest,
                    source_50_sha256=sha256_file(source_50),
                    source_phase4_sha256=sha256_file(source_phase4),
                    source_50_query_count=1,
                    source_phase4_query_count=1,
                    source_50_pair_count=1,
                    source_phase4_pair_count=1,
                )

            self.assertFalse(output.exists())
            self.assertFalse(manifest.exists())

    def test_manifest_marks_phase4_as_consumed_historical_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, _, _, _, manifest_path = self._build(
                root,
                [_row("ฝน#1", "pair_a")],
                [_row("ทะเล#1", "pair_b")],
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "phase5_historical_pool_frozen")
        self.assertTrue(manifest["validation"]["passed"])
        self.assertTrue(
            manifest["validation"]["severe_error_utility_constraint_passed"]
        )
        self.assertTrue(manifest["policy"]["phase4_holdout_is_consumed"])
        self.assertFalse(
            manifest["policy"]["phase4_holdout_is_unbiased_acceptance_set"]
        )
        self.assertFalse(
            manifest["policy"]["fresh_phase5_acceptance_holdout_opened"]
        )


if __name__ == "__main__":
    unittest.main()
