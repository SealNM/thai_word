#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

METRIC_KEYS = (
    "useful_rate_at_k",
    "high_utility_rate_at_k",
    "noise_rate_at_k",
    "severe_error_rate_at_k",
    "relation_diversity_at_k",
    "ndcg_at_k",
    "mrr_high_utility",
)


def _mean_metrics(
    per_query: dict[str, dict[str, Any]],
    query_ids: list[str],
) -> dict[str, float]:
    if not query_ids:
        raise ValueError("At least one query is required for metric aggregation.")
    return {
        f"{key}_mean": sum(float(per_query[q][key]) for q in query_ids) / len(query_ids)
        for key in METRIC_KEYS
    }


def _alpha_key(alpha: float) -> str:
    return str(float(alpha))


def _select_alpha(
    per_alpha: dict[str, dict[str, dict[str, Any]]],
    alphas: list[float],
    query_ids: list[str],
) -> tuple[float, dict[str, float]]:
    baseline = _mean_metrics(per_alpha[_alpha_key(0.0)], query_ids)
    eligible: list[tuple[float, dict[str, float]]] = []
    all_candidates: list[tuple[float, dict[str, float]]] = []

    for alpha in alphas:
        metrics = _mean_metrics(per_alpha[_alpha_key(alpha)], query_ids)
        item = (alpha, metrics)
        all_candidates.append(item)
        if (
            metrics["noise_rate_at_k_mean"] <= baseline["noise_rate_at_k_mean"] + 1e-12
            and metrics["severe_error_rate_at_k_mean"]
            <= baseline["severe_error_rate_at_k_mean"] + 1e-12
        ):
            eligible.append(item)

    pool = eligible or all_candidates
    return max(
        pool,
        key=lambda item: (
            item[1]["ndcg_at_k_mean"],
            item[1]["high_utility_rate_at_k_mean"],
            item[1]["mrr_high_utility_mean"],
            item[1]["relation_diversity_at_k_mean"],
            -item[0],
        ),
    )


def _validate_report(
    report: dict[str, Any],
) -> tuple[list[float], dict[str, dict[str, dict[str, Any]]]]:
    if report.get("status") != "validation_only_phase3_hybrid":
        raise ValueError("Input must be a validation-only Phase 3 hybrid report.")
    if report.get("benchmark_split_evaluated") is not False:
        raise ValueError(
            "Refusing stability analysis for a report that evaluated benchmark rows."
        )
    if int(report.get("benchmark_pairs_used_for_selection", -1)) != 0:
        raise ValueError(
            "Refusing stability analysis when benchmark rows influenced selection."
        )

    per_alpha = report.get("hybrid_validation_queries")
    if not isinstance(per_alpha, dict) or not per_alpha:
        raise ValueError(
            "Hybrid report must include per-query metrics (--include-per-query)."
        )

    alphas = [float(item["alpha"]) for item in report.get("hybrid_grid", [])]
    if 0.0 not in alphas:
        raise ValueError("Hybrid report must include alpha=0 baseline.")

    alpha_keys = {_alpha_key(alpha) for alpha in alphas}
    if not alpha_keys.issubset(per_alpha):
        missing = sorted(alpha_keys - set(per_alpha))
        raise ValueError(f"Missing per-query metrics for alpha values: {missing}")

    reference_queries = set(per_alpha[_alpha_key(0.0)])
    if len(reference_queries) < 3:
        raise ValueError(
            "Need at least three validation queries for leave-one-query-out stability."
        )
    for key in alpha_keys:
        if set(per_alpha[key]) != reference_queries:
            raise ValueError(f"Per-query coverage differs for alpha={key}.")

    return sorted(alphas), per_alpha


def run(report: dict[str, Any]) -> dict[str, Any]:
    alphas, per_alpha = _validate_report(report)
    query_ids = sorted(per_alpha[_alpha_key(0.0)])
    full_alpha = float(report["selected_validation_alpha"])
    full_key = _alpha_key(full_alpha)
    if full_key not in per_alpha:
        raise ValueError("selected_validation_alpha has no per-query metrics.")

    per_query_delta: dict[str, Any] = {}
    for query_id in query_ids:
        baseline = per_alpha[_alpha_key(0.0)][query_id]
        selected = per_alpha[full_key][query_id]
        per_query_delta[query_id] = {
            "baseline": {key: baseline[key] for key in METRIC_KEYS},
            "selected": {key: selected[key] for key in METRIC_KEYS},
            "delta": {
                key: float(selected[key]) - float(baseline[key])
                for key in METRIC_KEYS
            },
        }

    folds: list[dict[str, Any]] = []
    selection_counts: Counter[str] = Counter()

    for held_out in query_ids:
        fit_queries = [query_id for query_id in query_ids if query_id != held_out]
        selected_alpha, fit_metrics = _select_alpha(per_alpha, alphas, fit_queries)
        selection_counts[_alpha_key(selected_alpha)] += 1

        held_baseline = per_alpha[_alpha_key(0.0)][held_out]
        held_selected = per_alpha[_alpha_key(selected_alpha)][held_out]
        held_delta = {
            key: float(held_selected[key]) - float(held_baseline[key])
            for key in METRIC_KEYS
        }

        folds.append(
            {
                "held_out_query": held_out,
                "selected_alpha_from_other_queries": selected_alpha,
                "fit_query_count": len(fit_queries),
                "fit_metrics": fit_metrics,
                "held_out_baseline": {
                    key: held_baseline[key] for key in METRIC_KEYS
                },
                "held_out_selected": {
                    key: held_selected[key] for key in METRIC_KEYS
                },
                "held_out_delta": held_delta,
                "held_out_ndcg_result": (
                    "win"
                    if held_delta["ndcg_at_k"] > 1e-12
                    else "loss"
                    if held_delta["ndcg_at_k"] < -1e-12
                    else "tie"
                ),
                "held_out_noise_regression": (
                    held_delta["noise_rate_at_k"] > 1e-12
                ),
                "held_out_severe_regression": (
                    held_delta["severe_error_rate_at_k"] > 1e-12
                ),
                "held_out_diversity_regression": (
                    held_delta["relation_diversity_at_k"] < -1e-12
                ),
            }
        )

    ndcg_deltas = [fold["held_out_delta"]["ndcg_at_k"] for fold in folds]
    diversity_deltas = [
        fold["held_out_delta"]["relation_diversity_at_k"] for fold in folds
    ]
    wins = sum(fold["held_out_ndcg_result"] == "win" for fold in folds)
    losses = sum(fold["held_out_ndcg_result"] == "loss" for fold in folds)
    ties = len(folds) - wins - losses
    positive_alpha_folds = sum(
        float(fold["selected_alpha_from_other_queries"]) > 0 for fold in folds
    )
    noise_regressions = sum(
        bool(fold["held_out_noise_regression"]) for fold in folds
    )
    severe_regressions = sum(
        bool(fold["held_out_severe_regression"]) for fold in folds
    )
    diversity_regressions = sum(
        bool(fold["held_out_diversity_regression"]) for fold in folds
    )

    # Predeclared stability gate:
    # - neural blending selected in at least two-thirds of LOO folds;
    # - mean held-out NDCG delta is positive;
    # - NDCG wins are not outnumbered by losses;
    # - no held-out noise/severe-error regressions.
    #
    # Diversity remains a diagnostic metric and does not gate the pass yet.
    required_positive_alpha_folds = (2 * len(folds) + 2) // 3
    stability_pass = (
        positive_alpha_folds >= required_positive_alpha_folds
        and (sum(ndcg_deltas) / len(ndcg_deltas)) > 0
        and wins >= losses
        and noise_regressions == 0
        and severe_regressions == 0
    )

    return {
        "status": "validation_only_phase3_hybrid_stability",
        "source_model": report.get("neural_scores", {}).get("model"),
        "validation_query_count": len(query_ids),
        "alpha_grid": alphas,
        "full_validation_selected_alpha": full_alpha,
        "full_validation_selected_metrics": report.get(
            "selected_validation_metrics"
        ),
        "per_query_delta_at_full_alpha": per_query_delta,
        "leave_one_query_out": folds,
        "alpha_selection_counts": dict(
            sorted(selection_counts.items(), key=lambda item: float(item[0]))
        ),
        "stability_summary": {
            "positive_alpha_folds": positive_alpha_folds,
            "required_positive_alpha_folds": required_positive_alpha_folds,
            "held_out_ndcg_wins": wins,
            "held_out_ndcg_ties": ties,
            "held_out_ndcg_losses": losses,
            "mean_held_out_ndcg_delta": sum(ndcg_deltas) / len(ndcg_deltas),
            "min_held_out_ndcg_delta": min(ndcg_deltas),
            "max_held_out_ndcg_delta": max(ndcg_deltas),
            "held_out_noise_regression_folds": noise_regressions,
            "held_out_severe_regression_folds": severe_regressions,
            "held_out_diversity_regression_folds": diversity_regressions,
            "mean_held_out_diversity_delta": (
                sum(diversity_deltas) / len(diversity_deltas)
            ),
            "stability_pass": stability_pass,
        },
        "stability_rule": (
            "positive alpha selected in at least two-thirds of leave-one-query-out "
            "folds; mean held-out NDCG delta > 0; NDCG wins >= losses; zero "
            "held-out noise or severe-error regressions. Diversity is diagnostic "
            "and does not gate the pass."
        ),
        "benchmark_split_evaluated": False,
        "benchmark_pairs_used_for_selection": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Leave-one-query-out stability analysis for a Phase-3 hybrid report."
        )
    )
    parser.add_argument("--hybrid-report", required=True)
    parser.add_argument(
        "--output",
        default="evaluation/writer_relevance_phase3_hybrid_stability_report.json",
    )
    args = parser.parse_args()

    report = json.loads(Path(args.hybrid_report).read_text(encoding="utf-8"))
    result = run(report)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
