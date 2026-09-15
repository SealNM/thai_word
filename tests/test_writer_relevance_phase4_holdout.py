from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.writer_relevance_phase4_holdout import (
    EXPECTED_GROUPS,
    EXPECTED_PAIR_COUNT,
    TARGET_COUNT,
    apply_sense_decisions,
    freeze_targets,
    validate_target_config,
)


TARGET_FILE = Path("evaluation/writer_relevance_phase4_holdout_targets.json")
OLD_TARGET_FILE = Path("evaluation/writer_relevance_50_targets.json")
DECISIONS_FILE = Path("evaluation/writer_relevance_phase4_holdout_sense_decisions.json")
FROZEN_FILE = Path("evaluation/writer_relevance_phase4_holdout_frozen_queries.json")
MANIFEST_FILE = Path("evaluation/writer_relevance_phase4_holdout_manifest.json")


class Phase4HoldoutTests(unittest.TestCase):
    def test_committed_headword_set_is_balanced_and_disjoint(self) -> None:
        config = json.loads(TARGET_FILE.read_text(encoding="utf-8"))
        old_targets = json.loads(OLD_TARGET_FILE.read_text(encoding="utf-8"))

        errors = validate_target_config(config, old_targets=old_targets)

        self.assertEqual(errors, [])
        self.assertEqual(len(config["queries"]), TARGET_COUNT)
        self.assertEqual(EXPECTED_PAIR_COUNT, 600)

        counts = {}
        for item in config["queries"]:
            counts[item["group"]] = counts.get(item["group"], 0) + 1
        self.assertEqual(counts, EXPECTED_GROUPS)

        old_words = {item["query"] for item in old_targets["queries"]}
        new_words = {item["query"] for item in config["queries"]}
        self.assertTrue(old_words.isdisjoint(new_words))

    def test_overlap_with_old_target_is_rejected(self) -> None:
        config = json.loads(TARGET_FILE.read_text(encoding="utf-8"))
        old_targets = {"queries": [{"query": config["queries"][0]["query"]}]}

        errors = validate_target_config(config, old_targets=old_targets)

        self.assertTrue(any("overlap" in error.lower() for error in errors))

    def test_locked_sense_decisions_cover_all_targets(self) -> None:
        config = json.loads(TARGET_FILE.read_text(encoding="utf-8"))
        decisions = json.loads(DECISIONS_FILE.read_text(encoding="utf-8"))

        self.assertEqual(len(decisions["decisions"]), TARGET_COUNT)
        self.assertEqual(
            {item["query"] for item in decisions["decisions"]},
            {item["query"] for item in config["queries"]},
        )
        self.assertTrue(all(isinstance(item["sense"], int) for item in decisions["decisions"]))

    def test_committed_frozen_queries_match_locked_decisions(self) -> None:
        decisions = json.loads(DECISIONS_FILE.read_text(encoding="utf-8"))
        frozen = json.loads(FROZEN_FILE.read_text(encoding="utf-8"))
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))

        expected = {item["query"]: item["sense"] for item in decisions["decisions"]}
        actual = {item["query"]: item["sense"] for item in frozen["queries"]}

        self.assertEqual(actual, expected)
        self.assertEqual(frozen["target_count"], TARGET_COUNT)
        self.assertEqual(frozen["expected_pair_count"], EXPECTED_PAIR_COUNT)
        self.assertEqual(manifest["target_count"], TARGET_COUNT)
        self.assertEqual(manifest["expected_pair_count"], EXPECTED_PAIR_COUNT)
        self.assertFalse(manifest["candidate_exported"])
        self.assertFalse(manifest["writer_reranker_used_for_target_selection"])
        self.assertFalse(manifest["writer_reranker_used_for_candidate_export"])
        self.assertFalse(manifest["quality_selection_performed"])

    def test_apply_decisions_rejects_sense_not_in_inspection(self) -> None:
        decisions = json.loads(DECISIONS_FILE.read_text(encoding="utf-8"))
        report = {
            "targets": [
                {
                    "query": item["query"],
                    "recommended_sense": None,
                    "senses": [{"sense": 1, "definition": "test"}],
                }
                for item in decisions["decisions"]
            ]
        }
        broken = json.loads(json.dumps(decisions, ensure_ascii=False))
        broken["decisions"][0]["sense"] = 999

        with self.assertRaisesRegex(ValueError, "not present in inspected senses"):
            apply_sense_decisions(
                report,
                broken,
                expected_config_sha256=decisions["config_sha256"],
            )

    def test_freeze_rejects_unreviewed_senses(self) -> None:
        config = json.loads(TARGET_FILE.read_text(encoding="utf-8"))
        old_targets = json.loads(OLD_TARGET_FILE.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "targets.json"
            old_path = root / "old.json"
            report_path = root / "report.json"
            output_path = root / "frozen.json"
            manifest_path = root / "manifest.json"

            config_path.write_text(
                json.dumps(config, ensure_ascii=False),
                encoding="utf-8",
            )
            old_path.write_text(
                json.dumps(old_targets, ensure_ascii=False),
                encoding="utf-8",
            )

            import hashlib

            config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
            report = {
                "config_sha256": config_hash,
                "targets": [
                    {
                        "query": item["query"],
                        "recommended_sense": None,
                        "senses": [{"sense": 1, "definition": "test"}],
                    }
                    for item in config["queries"]
                ],
            }
            report_path.write_text(
                json.dumps(report, ensure_ascii=False),
                encoding="utf-8",
            )

            args = SimpleNamespace(
                config=str(config_path),
                report=str(report_path),
                old_targets=str(old_path),
                output=str(output_path),
                manifest=str(manifest_path),
            )
            with self.assertRaisesRegex(ValueError, "recommended_sense"):
                freeze_targets(args)

            self.assertFalse(output_path.exists())
            self.assertFalse(manifest_path.exists())


if __name__ == "__main__":
    unittest.main()
