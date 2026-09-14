from __future__ import annotations

import unittest

from thai_v31_router import (
    AUTO_LABEL_SOURCE,
    GEMINI_LABEL_SOURCE,
    LOCAL_LABEL_SOURCE,
    auto_judgment,
    build_audit_seed,
    merge_audit_labels,
    route_seed,
)


def _candidate(
    word: str,
    *,
    tier: int,
    dense_rank: int,
    lexical_rank: int | None,
    lexical_form: str = "standalone",
    relation_hint: str | None = None,
) -> dict:
    if relation_hint is None:
        relation_hint = "direct_gloss_or_synonym" if tier >= 5 else "definition_similar"
    return {
        "word": word,
        "definition": f"นิยามของ {word}",
        "evidence": {
            "relation_tier": tier,
            "relation_hint": relation_hint,
            "lexical_form": lexical_form,
            "dense_rank": dense_rank,
            "lexical_rank": lexical_rank,
            "dense_similarity": 1.0 - (dense_rank / 100.0),
        },
    }


def _seed() -> dict:
    candidates = [
        _candidate("ตรง", tier=5, dense_rank=1, lexical_rank=1),
        _candidate("ใกล้1", tier=3, dense_rank=2, lexical_rank=2),
        _candidate("ใกล้2", tier=2, dense_rank=3, lexical_rank=4),
        _candidate("ใกล้3", tier=1, dense_rank=4, lexical_rank=7),
        _candidate("เกี่ยว1", tier=0, dense_rank=5, lexical_rank=None),
        _candidate("เกี่ยว2", tier=0, dense_rank=6, lexical_rank=None),
        _candidate("เกี่ยว3", tier=0, dense_rank=7, lexical_rank=None),
        _candidate("เกี่ยว4", tier=0, dense_rank=8, lexical_rank=None),
        _candidate("ไกล1", tier=0, dense_rank=18, lexical_rank=None),
        _candidate("ไกล2", tier=0, dense_rank=22, lexical_rank=None),
    ]
    return {
        "schema_version": 1,
        "seed_id": "seed-1",
        "anchor": {
            "word": "คำหลัก",
            "sense": 1,
            "definition": "นิยามคำหลัก",
        },
        "candidates": candidates,
    }


class V31RouterTests(unittest.TestCase):
    def test_only_tier5_standalone_is_auto_labeled(self) -> None:
        strong = _candidate("ตรง", tier=5, dense_rank=1, lexical_rank=1)
        judgment = auto_judgment(strong)
        self.assertIsNotNone(judgment)
        self.assertEqual(judgment["relation"], "synonym")
        self.assertEqual(judgment["label_source"], AUTO_LABEL_SOURCE)

        weaker = _candidate("ใกล้", tier=4, dense_rank=2, lexical_rank=1)
        self.assertIsNone(auto_judgment(weaker))

        bound = _candidate(
            "รูป-",
            tier=5,
            dense_rank=2,
            lexical_rank=1,
            lexical_form="bound_form",
        )
        self.assertIsNone(auto_judgment(bound))

    def test_router_reduces_local_candidates(self) -> None:
        routed = route_seed(_seed(), max_local_candidates=4)

        routing = routed["routing"]
        self.assertEqual(routing["source_candidate_count"], 10)
        self.assertEqual(routing["auto_labeled"], 1)
        self.assertEqual(routing["local_candidates"], 4)
        self.assertEqual(routing["dropped_candidates"], 5)
        self.assertEqual(len(routed["candidates"]), 5)

        auto = next(
            item for item in routed["candidates"] if item["word"] == "ตรง"
        )
        self.assertEqual(auto["judgment"]["label_source"], AUTO_LABEL_SOURCE)

        local = [
            item
            for item in routed["candidates"]
            if item.get("route") == LOCAL_LABEL_SOURCE
        ]
        self.assertEqual(len(local), 4)

    def test_low_confidence_local_label_becomes_audit_seed(self) -> None:
        routed = route_seed(_seed(), max_local_candidates=2)
        for candidate in routed["candidates"]:
            if candidate.get("route") == LOCAL_LABEL_SOURCE:
                candidate.pop("route", None)
                candidate["judgment"] = {
                    "relation": "associated",
                    "semantic_relatedness": 2,
                    "replaceability": 0,
                    "confidence": 0.60,
                    "register": "neutral",
                    "reason": "เกี่ยวข้องแต่ไม่ใช่คำแทน",
                    "label_source": LOCAL_LABEL_SOURCE,
                }

        audit = build_audit_seed(
            routed,
            confidence_threshold=0.75,
            random_audit_fraction=0.0,
        )
        self.assertIsNotNone(audit)
        self.assertEqual(len(audit["candidates"]), 2)
        self.assertTrue(
            all("judgment" not in item for item in audit["candidates"])
        )

    def test_merge_replaces_only_audited_candidates(self) -> None:
        routed = route_seed(_seed(), max_local_candidates=1)
        local_candidate = next(
            item
            for item in routed["candidates"]
            if item.get("route") == LOCAL_LABEL_SOURCE
        )
        local_candidate.pop("route", None)
        local_candidate["judgment"] = {
            "relation": "associated",
            "semantic_relatedness": 2,
            "replaceability": 0,
            "confidence": 0.60,
            "register": "neutral",
            "reason": "local",
            "label_source": LOCAL_LABEL_SOURCE,
        }

        audit_row = {
            "seed_id": routed["seed_id"],
            "anchor": routed["anchor"],
            "candidates": [
                {
                    "word": local_candidate["word"],
                    "definition": local_candidate["definition"],
                    "judgment": {
                        "relation": "near_synonym",
                        "semantic_relatedness": 4,
                        "replaceability": 2,
                        "confidence": 0.95,
                        "register": "neutral",
                        "reason": "gemini",
                    },
                }
            ],
        }

        merged = merge_audit_labels(routed, audit_row)
        replaced = next(
            item
            for item in merged["candidates"]
            if item["word"] == local_candidate["word"]
        )
        self.assertEqual(replaced["judgment"]["relation"], "near_synonym")
        self.assertEqual(
            replaced["judgment"]["label_source"],
            GEMINI_LABEL_SOURCE,
        )

        auto = next(
            item for item in merged["candidates"] if item["word"] == "ตรง"
        )
        self.assertEqual(auto["judgment"]["label_source"], AUTO_LABEL_SOURCE)


if __name__ == "__main__":
    unittest.main()
