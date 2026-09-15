from __future__ import annotations

import unittest

from scripts import writer_relevance_phase3_hybrid_stability as stability


METRICS = {
    "useful_rate_at_k": 1.0,
    "high_utility_rate_at_k": 1.0,
    "noise_rate_at_k": 0.0,
    "severe_error_rate_at_k": 0.0,
    "relation_diversity_at_k": 2.0,
    "ndcg_at_k": 0.8,
    "mrr_high_utility": 1.0,
}


def _q(
    ndcg: float,
    *,
    noise: float = 0.0,
    severe: float = 0.0,
    diversity: float = 2.0,
) -> dict:
    row = dict(METRICS)
    row["ndcg_at_k"] = ndcg
    row["noise_rate_at_k"] = noise
    row["severe_error_rate_at_k"] = severe
    row["relation_diversity_at_k"] = diversity
    return row


def _report() -> dict:
    base = {
        "a#1": _q(0.80),
        "b#1": _q(0.82),
        "c#1": _q(0.84),
    }
    hybrid = {
        "a#1": _q(0.85),
        "b#1": _q(0.86),
        "c#1": _q(0.88),
    }
    return {
        "status": "validation_only_phase3_hybrid",
        "benchmark_split_evaluated": False,
        "benchmark_pairs_used_for_selection": 0,
        "selected_validation_alpha": 0.4,
        "selected_validation_metrics": {},
        "neural_scores": {"model": "fake/model"},
        "hybrid_grid": [{"alpha": 0.0}, {"alpha": 0.4}],
        "hybrid_validation_queries": {"0.0": base, "0.4": hybrid},
    }


class WriterRelevancePhase3HybridStabilityTests(unittest.TestCase):
    def test_run_selects_positive_alpha_when_it_generalizes(self) -> None:
        result = stability.run(_report())

        self.assertEqual(result["alpha_selection_counts"], {"0.4": 3})
        self.assertEqual(
            result["stability_summary"]["held_out_ndcg_wins"],
            3,
        )
        self.assertTrue(result["stability_summary"]["stability_pass"])
        self.assertFalse(result["benchmark_split_evaluated"])

    def test_rejects_report_that_evaluated_benchmark(self) -> None:
        report = _report()
        report["benchmark_split_evaluated"] = True

        with self.assertRaises(ValueError):
            stability.run(report)

    def test_rejects_missing_per_query_alpha(self) -> None:
        report = _report()
        del report["hybrid_validation_queries"]["0.4"]

        with self.assertRaises(ValueError):
            stability.run(report)

    def test_noise_regression_prevents_alpha_from_selection(self) -> None:
        report = _report()
        for query in report["hybrid_validation_queries"]["0.4"].values():
            query["noise_rate_at_k"] = 0.1
            query["severe_error_rate_at_k"] = 0.1

        result = stability.run(report)

        self.assertEqual(result["alpha_selection_counts"], {"0.0": 3})
        self.assertFalse(result["stability_summary"]["stability_pass"])

    def test_per_query_full_alpha_delta_is_reported(self) -> None:
        result = stability.run(_report())
        delta = result["per_query_delta_at_full_alpha"]["a#1"]["delta"]

        self.assertAlmostEqual(delta["ndcg_at_k"], 0.05)


if __name__ == "__main__":
    unittest.main()
