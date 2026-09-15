from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.substitutability_benchmark import annotate_interactively, export_candidates
from thai_substitutability import (
    SCHEMA_VERSION,
    benchmark_metrics,
    stable_pair_id,
    validate_annotation_row,
)


def _row(
    *,
    query_id: str,
    rank: int,
    utility: int,
    relation: str,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "pair_id": f"{query_id}-{rank}",
        "query_id": query_id,
        "query": {"word": query_id, "sense": 1, "definition": "target"},
        "candidate": {"word": f"c{rank}", "sense": 1, "definition": "candidate"},
        "retrieval": {"v25_rank": rank},
        "annotation": {
            "utility": utility,
            "relation": relation,
            "notes": "",
        },
        "split": "benchmark",
    }


class _FakeSearcher:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.results


class SubstitutabilityBenchmarkTests(unittest.TestCase):
    def test_pair_id_is_deterministic_and_sense_sensitive(self) -> None:
        one = stable_pair_id("รัก", 3, "ชอบ", 1)
        two = stable_pair_id("รัก", 3, "ชอบ", 1)
        other = stable_pair_id("รัก", 2, "ชอบ", 1)

        self.assertEqual(one, two)
        self.assertNotEqual(one, other)

    def test_cross_role_writer_word_can_have_high_utility(self) -> None:
        row = _row(
            query_id="ฝน#1",
            rank=1,
            utility=2,
            relation="manner_action",
        )
        self.assertEqual(validate_annotation_row(row), [])

    def test_scene_context_can_be_high_utility(self) -> None:
        row = _row(
            query_id="ฝน#1",
            rank=1,
            utility=2,
            relation="scene_context",
        )
        self.assertEqual(validate_annotation_row(row), [])

    def test_severe_error_requires_zero_utility(self) -> None:
        for relation in ("opposite_misleading", "sense_mismatch", "unrelated"):
            with self.subTest(relation=relation):
                row = _row(
                    query_id="เร็ว#1",
                    rank=1,
                    utility=2,
                    relation=relation,
                )
                errors = validate_annotation_row(row)
                self.assertTrue(
                    any("requires utility 0" in error for error in errors)
                )

    def test_utility_zero_is_allowed_for_weak_relation(self) -> None:
        row = _row(
            query_id="ฝน#1",
            rank=1,
            utility=0,
            relation="weak_related",
        )
        self.assertEqual(validate_annotation_row(row), [])

    def test_export_builds_writer_relevance_annotation_rows(self) -> None:
        result = {
            "word": "พิรุณ",
            "score": 0.03,
            "relation_tier": 4,
            "relation_hint": "direct_gloss_or_synonym",
            "lexical_form": "standalone",
            "sense_resolution": "selected_sense_supported",
            "lexical_score": 0.9,
            "lexical_rank": 1,
            "dense_similarity": 0.8,
            "dense_rank": 2,
            "query_sense": {"sense": 1, "definition": "น้ำที่ตกจากฟ้า"},
            "matched_candidate_sense": {"sense": 1, "definition": "ฝน"},
        }
        searcher = _FakeSearcher([result])
        lexical = SimpleNamespace(
            word_to_index={"ฝน": 0},
            entry_to_senses=[[0]],
        )

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "queries.json"
            config_path.write_text(
                json.dumps(
                    {
                        "queries": [
                            {
                                "query": "ฝน",
                                "sense": 1,
                                "category": "noun:nature",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                config=str(config_path),
                index="artifacts/v1",
                dense_index="artifacts/v2/embeddinggemma-300m-256",
                candidates=30,
                device=None,
                output=str(Path(directory) / "annotations.jsonl"),
            )

            captured = {}
            with (
                patch(
                    "scripts.substitutability_benchmark.load_artifacts",
                    return_value=lexical,
                ),
                patch(
                    "scripts.substitutability_benchmark.HybridSearcher.from_paths",
                    return_value=searcher,
                ),
                patch(
                    "scripts.substitutability_benchmark.write_jsonl",
                    side_effect=lambda path, rows: captured.update(
                        {"path": path, "rows": rows}
                    ),
                ),
            ):
                export_candidates(args)

        rows = captured["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["schema_version"], SCHEMA_VERSION)
        self.assertEqual(rows[0]["query_id"], "ฝน#1")
        self.assertEqual(rows[0]["candidate"]["word"], "พิรุณ")
        self.assertEqual(rows[0]["retrieval"]["v25_rank"], 1)
        self.assertIsNone(rows[0]["annotation"]["utility"])
        self.assertIsNone(rows[0]["annotation"]["relation"])

    def test_export_rejects_unpinned_ambiguous_target(self) -> None:
        searcher = _FakeSearcher([])
        lexical = SimpleNamespace(
            word_to_index={"รัก": 0},
            entry_to_senses=[[0, 1, 2]],
        )

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "queries.json"
            config_path.write_text(
                json.dumps(
                    {"queries": [{"query": "รัก", "sense": None}]},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                config=str(config_path),
                index="artifacts/v1",
                dense_index="artifacts/v2/embeddinggemma-300m-256",
                candidates=30,
                device=None,
                output=str(Path(directory) / "annotations.jsonl"),
            )

            with (
                patch(
                    "scripts.substitutability_benchmark.load_artifacts",
                    return_value=lexical,
                ),
                patch(
                    "scripts.substitutability_benchmark.HybridSearcher.from_paths",
                    return_value=searcher,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "set an explicit sense"):
                    export_candidates(args)

        self.assertEqual(searcher.calls, [])

    def test_annotation_cli_autosaves_and_resumes(self) -> None:
        row = _row(
            query_id="ฝน#1",
            rank=1,
            utility=0,
            relation="unrelated",
        )
        row["annotation"]["utility"] = None
        row["annotation"]["relation"] = None

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.jsonl"
            path.write_text(
                json.dumps(row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                path=str(path),
                query="ฝน#1",
                review=False,
                limit=None,
            )

            with patch("builtins.input", side_effect=["2", "4"]):
                annotate_interactively(args)

            saved = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(saved["annotation"]["utility"], 2)
            self.assertEqual(saved["annotation"]["relation"], "manner_action")

            with patch("builtins.input") as mocked_input:
                annotate_interactively(args)
            mocked_input.assert_not_called()

    def test_metrics_measure_writer_utility_and_severe_errors(self) -> None:
        rows = [
            _row(query_id="ฝน#1", rank=1, utility=2, relation="manner_action"),
            _row(query_id="ฝน#1", rank=2, utility=3, relation="direct"),
            _row(query_id="ฝน#1", rank=3, utility=2, relation="scene_context"),
            _row(query_id="ฝน#1", rank=4, utility=0, relation="unrelated"),
        ]

        report = benchmark_metrics(rows, k=4)
        query = report["queries"]["ฝน#1"]

        self.assertEqual(query["noise_at_k"], 1)
        self.assertAlmostEqual(query["noise_rate_at_k"], 1 / 4)
        self.assertEqual(query["useful_at_k"], 3)
        self.assertAlmostEqual(query["useful_rate_at_k"], 3 / 4)
        self.assertEqual(query["high_utility_at_k"], 3)
        self.assertEqual(query["severe_error_at_k"], 1)
        self.assertEqual(query["relation_diversity_at_k"], 3)
        self.assertEqual(query["mrr_high_utility"], 1.0)
        self.assertLess(query["ndcg_at_k"], 1.0)


if __name__ == "__main__":
    unittest.main()
