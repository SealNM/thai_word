from __future__ import annotations

import json
import unittest

from scripts.pathumma_writer_enrichment import (
    SYSTEM_PROMPT,
    build_prompt,
    definition_head,
    definition_status,
    normalize_definition_text,
    ensure_historical_input,
    evaluate,
    extract_json_object,
    parse_predictions,
    split_reasoning_and_final,
)


def make_row(
    pair_id: str,
    *,
    utility: int = 3,
    relation: str = "direct",
) -> dict:
    return {
        "schema_version": 3,
        "query_id": "สวย#1",
        "pair_id": pair_id,
        "query": {
            "word": "สวย",
            "sense": 1,
            "definition": "มีลักษณะงามน่าพึงพอใจ",
            "intended": "ความงามทางรูปลักษณ์",
            "category": "adjective:appearance",
        },
        "candidate": {
            "word": "งดงาม" if pair_id == "pair_a" else "น่าเกลียด",
            "sense": 1,
            "definition": "มีความงามมาก" if pair_id == "pair_a" else "ไม่น่าดู",
        },
        "retrieval": {
            "v25_rank": 1 if pair_id == "pair_a" else 2,
            "score": 0.99,
            "relation_hint": "candidate_defined_via_query",
        },
        "annotation": {
            "utility": utility,
            "semantic_relation": relation,
            "style_tags": [],
        },
    }


class PathummaWriterEnrichmentTest(unittest.TestCase):
    def test_prompt_does_not_leak_gold_or_v25_signals(self) -> None:
        prompt = build_prompt([make_row("pair_a")])
        self.assertIn("pair_a", prompt)
        self.assertIn("งดงาม", prompt)
        self.assertNotIn("0.99", prompt)
        self.assertNotIn("candidate_defined_via_query", prompt)
        self.assertNotIn('"v25_rank"', prompt)
        self.assertNotIn('"utility": 3', prompt)
        self.assertNotIn("คำอธิบายสั้น ๆ", prompt)
        self.assertIn("definition_status", prompt)
        self.assertIn("definition_head", prompt)
        self.assertIn("cross_reference_only", SYSTEM_PROMPT)
        self.assertIn("evidence_quote", prompt)

    def test_reasoning_is_split_before_json_parse(self) -> None:
        raw = '<think>internal</think>\n{"predictions": []}'
        reasoning, final = split_reasoning_and_final(raw)
        self.assertEqual(reasoning, "internal")
        self.assertEqual(extract_json_object(final), {"predictions": []})

    def test_schema_violation_is_not_silently_fixed(self) -> None:
        final = json.dumps({
            "predictions": [{
                "pair_id": "pair_a",
                "utility": 2,
                "semantic_relation": "unrelated",
                "evidence_status": "sufficient",
                "evidence_quote": "มีความงามมาก",
                "confidence": 0.8,
                "reason": "x",
            }]
        })
        parsed, errors = parse_predictions(
            final,
            {"pair_a"},
            {"pair_a": make_row("pair_a")},
        )
        self.assertEqual(parsed, [])
        self.assertTrue(any("requires utility 0" in error for error in errors))

    def test_dictionary_definition_helpers(self) -> None:
        definition = "แห้ง, พร่อง, ลดลง, เช่น แลมันทำชู้แล้วมันทอดหญิงนั้นเสีย"
        self.assertEqual(definition_head(definition), "แห้ง, พร่อง, ลดลง")
        self.assertEqual(definition_status("<i>ดูใน หัว ๑</i>."), "cross_reference_only")
        self.assertEqual(
            normalize_definition_text("<i>ดู เทียน ๒, เทียนบ้าน</i>."),
            "ดู เทียน ๒, เทียนบ้าน.",
        )

    def test_cross_reference_only_cannot_be_invented(self) -> None:
        source = make_row("pair_a")
        source["candidate"]["definition"] = "<i>ดูใน หัว ๑</i>."
        final = json.dumps({
            "predictions": [{
                "pair_id": "pair_a",
                "utility": 3,
                "semantic_relation": "direct",
                "evidence_status": "sufficient",
                "evidence_quote": "ดูใน หัว ๑.",
                "confidence": 0.9,
                "reason": "เดาความหมายจากคำอ้างอิง",
            }]
        }, ensure_ascii=False)
        parsed, errors = parse_predictions(
            final,
            {"pair_a"},
            {"pair_a": source},
        )
        self.assertEqual(parsed, [])
        self.assertTrue(any("requires semantic_relation unclear" in error for error in errors))
        self.assertTrue(any("requires utility 0" in error for error in errors))

    def test_cross_reference_only_unclear_is_valid(self) -> None:
        source = make_row("pair_a")
        source["candidate"]["definition"] = "<i>ดูใน หัว ๑</i>."
        final = json.dumps({
            "predictions": [{
                "pair_id": "pair_a",
                "utility": 0,
                "semantic_relation": "unclear",
                "evidence_status": "cross_reference_only",
                "evidence_quote": "ดูใน หัว ๑.",
                "confidence": 0.95,
                "reason": "definition มีเพียงคำอ้างอิง จึงยังยืนยันความหมายไม่ได้",
            }]
        }, ensure_ascii=False)
        parsed, errors = parse_predictions(
            final,
            {"pair_a"},
            {"pair_a": source},
        )
        self.assertEqual(errors, [])
        self.assertEqual(len(parsed), 1)

    def test_evidence_quote_must_come_from_definition(self) -> None:
        source = make_row("pair_a")
        final = json.dumps({
            "predictions": [{
                "pair_id": "pair_a",
                "utility": 3,
                "semantic_relation": "direct",
                "evidence_status": "sufficient",
                "evidence_quote": "ข้อความที่ไม่มีอยู่จริง",
                "confidence": 0.9,
                "reason": "x",
            }]
        }, ensure_ascii=False)
        parsed, errors = parse_predictions(
            final,
            {"pair_a"},
            {"pair_a": source},
        )
        self.assertEqual(parsed, [])
        self.assertTrue(any("copied from candidate definition" in error for error in errors))

    def test_relation_confusion_matrix_is_reported(self) -> None:
        gold = [
            make_row("pair_a", utility=3, relation="direct"),
            make_row("pair_b", utility=0, relation="weak_related"),
        ]
        predictions = [
            {
                "pair_id": "pair_a",
                "utility": 3,
                "semantic_relation": "direct",
                "style_tags": [],
                "confidence": 0.9,
            },
            {
                "pair_id": "pair_b",
                "utility": 0,
                "semantic_relation": "unrelated",
                "style_tags": [],
                "confidence": 0.9,
            },
        ]
        report = evaluate(gold, predictions)
        self.assertEqual(report["relation_confusion_matrix"]["direct"]["direct"], 1)
        self.assertEqual(
            report["relation_confusion_matrix"]["weak_related"]["unrelated"],
            1,
        )
        self.assertIn("utility_confusion_matrix", report)
        self.assertNotIn("style_jaccard_mean", report)

    def test_fresh_phase5_holdout_is_blocked(self) -> None:
        with self.assertRaisesRegex(ValueError, "Refusing"):
            ensure_historical_input(
                "evaluation/writer_relevance_phase5_holdout_annotations.approved.jsonl"
            )

    def test_basic_agreement_metrics(self) -> None:
        gold = [
            make_row("pair_a", utility=3, relation="direct"),
            make_row("pair_b", utility=0, relation="opposite_misleading"),
        ]
        predictions = [
            {
                "pair_id": "pair_a",
                "utility": 3,
                "semantic_relation": "direct",
                "style_tags": [],
                "confidence": 0.9,
            },
            {
                "pair_id": "pair_b",
                "utility": 0,
                "semantic_relation": "opposite_misleading",
                "style_tags": [],
                "confidence": 0.9,
            },
        ]
        report = evaluate(gold, predictions)
        self.assertEqual(report["valid_prediction_coverage"], 1.0)
        self.assertEqual(report["utility_accuracy"], 1.0)
        self.assertEqual(report["relation_accuracy"], 1.0)
        self.assertEqual(report["severe_error_binary"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
