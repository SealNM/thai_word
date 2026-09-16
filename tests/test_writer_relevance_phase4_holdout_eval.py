from __future__ import annotations

import argparse
import unittest

import numpy as np

from scripts.writer_relevance_phase4_holdout_eval import (
    _query_rank_normalize,
    _rerank_for_metrics,
    _runtime_row,
    _validate_gate,
    run,
)


def _gate() -> dict:
    return {
        "status": "phase4_post_freeze_evaluation_open",
        "holdout_opened_for_model_evaluation": True,
        "policy": {
            "no_retraining": True,
            "no_threshold_tuning": True,
            "no_alpha_search": True,
            "no_candidate_reselection": True,
            "no_architecture_change": True,
            "single_controlled_evaluation_cycle": True,
        },
        "runtime_contract": {"alpha": 0.5, "rerank_pool": 30, "category_mode": "omit"},
    }


def _row(query_id: str, rank: int, word: str) -> dict:
    return {
        "query_id": query_id,
        "query": {"word": query_id.split("#")[0], "sense": 1, "definition": "เป้าหมาย", "category": "legacy"},
        "candidate": {"word": word, "sense": 1, "definition": f"ความหมาย {word}"},
        "retrieval": {
            "v25_rank": rank, "score": 0.1, "relation_tier": 1,
            "relation_hint": "definition_similar", "lexical_form": "standalone",
            "sense_resolution": "selected_sense_supported", "lexical_score": 0.2,
            "lexical_rank": rank, "dense_similarity": 0.3, "dense_rank": rank,
        },
        "annotation": {"utility": 2, "semantic_relation": "direct", "style_tags": []},
        "split": "phase4_holdout",
    }


class Phase4HoldoutEvalTests(unittest.TestCase):
    def test_gate_rejects_alpha_change(self) -> None:
        gate = _gate()
        gate["runtime_contract"]["alpha"] = 0.4
        with self.assertRaises(ValueError):
            _validate_gate(gate)

    def test_gate_rejects_tuning_policy_change(self) -> None:
        gate = _gate()
        gate["policy"]["no_alpha_search"] = False
        with self.assertRaises(ValueError):
            _validate_gate(gate)

    def test_rank_normalization_is_per_query(self) -> None:
        rows = [_row("ก#1", 1, "ก1"), _row("ก#1", 2, "ก2"), _row("ข#1", 1, "ข1"), _row("ข#1", 2, "ข2")]
        values = _query_rank_normalize(rows, np.asarray([1.0, 2.0, 100.0, 200.0]))
        self.assertEqual(values.tolist(), [0.0, 1.0, 0.0, 1.0])

    def test_tie_break_keeps_original_v25_order(self) -> None:
        rows = [_row("ก#1", 1, "ก1"), _row("ก#1", 2, "ก2")]
        reranked = _rerank_for_metrics(rows, np.asarray([0.5, 0.5]))
        self.assertEqual([row["candidate"]["word"] for row in reranked], ["ก1", "ก2"])

    def test_runtime_row_is_annotation_and_category_free(self) -> None:
        runtime = _runtime_row(_row("ก#1", 1, "ผล"))
        self.assertNotIn("annotation", runtime)
        self.assertNotIn("category", runtime["query"])
        self.assertEqual(runtime["retrieval"]["v25_rank"], 1)

    def test_run_requires_explicit_confirmation_first(self) -> None:
        with self.assertRaises(ValueError):
            run(argparse.Namespace(confirm_phase4_holdout=False))


if __name__ == "__main__":
    unittest.main()
