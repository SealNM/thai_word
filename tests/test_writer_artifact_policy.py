from __future__ import annotations

import json
import unittest
from pathlib import Path

from thai_writer_artifact_policy import (
    WriterArtifactPolicyError,
    load_writer_artifact_registry,
    production_artifact_name,
    validate_production_artifact_name,
    validate_writer_artifact_registry,
)


REGISTRY = Path(__file__).resolve().parents[1] / "evaluation/writer_relevance_artifact_registry.json"


def _registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


class WriterArtifactPolicyTests(unittest.TestCase):
    def test_committed_registry_is_valid(self) -> None:
        payload = load_writer_artifact_registry(REGISTRY)
        self.assertEqual(payload["status"], "phase4_wave_i_policy_complete")
        self.assertEqual(payload["production_policy"]["status"], "not_created")

    def test_reproduction_cannot_claim_phase3_metrics(self) -> None:
        payload = _registry()
        payload["phase4_reproduction"]["may_claim_phase3_frozen_metrics"] = True
        with self.assertRaises(WriterArtifactPolicyError):
            validate_writer_artifact_registry(payload)

    def test_reproduction_needs_distinct_canonical_name(self) -> None:
        payload = _registry()
        payload["phase4_reproduction"]["canonical_dir_name"] = "bge-reranker-v2-m3-locked"
        with self.assertRaises(WriterArtifactPolicyError):
            validate_writer_artifact_registry(payload)

    def test_production_policy_requires_fresh_holdout(self) -> None:
        payload = _registry()
        payload["production_policy"]["requires_fresh_holdout_for_quality_claim"] = False
        with self.assertRaises(WriterArtifactPolicyError):
            validate_writer_artifact_registry(payload)

    def test_production_artifact_name_is_versioned_and_distinct(self) -> None:
        self.assertEqual(production_artifact_name(1), "bge-reranker-v2-m3-production-v1")
        self.assertEqual(validate_production_artifact_name("bge-reranker-v2-m3-production-v12"), 12)
        with self.assertRaises(WriterArtifactPolicyError):
            validate_production_artifact_name("bge-reranker-v2-m3-locked")

    def test_created_production_policy_requires_versioned_active_name(self) -> None:
        payload = _registry()
        payload["production_policy"]["status"] = "created"
        payload["production_policy"]["active_artifact_name"] = "bge-reranker-v2-m3-production-v1"
        validated = validate_writer_artifact_registry(payload)
        self.assertEqual(
            validated["production_policy"]["active_artifact_name"],
            "bge-reranker-v2-m3-production-v1",
        )


if __name__ == "__main__":
    unittest.main()
