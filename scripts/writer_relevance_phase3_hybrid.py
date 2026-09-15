#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn.feature_extraction import DictVectorizer

from scripts.writer_relevance_phase3_baseline import (
    BinaryProbabilityModel,
    _metric_summary,
    _rerank_for_metrics,
    _validate_split_integrity,
    feature_dict,
)
from scripts.writer_relevance_phase3_crossencoder import _delta_summary
from thai_substitutability import SEVERE_ERROR_RELATIONS, benchmark_metrics, read_jsonl


def _learned_safe_scores(
    train: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    seed: int,
    severe_penalty: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform([feature_dict(row) for row in train])
    x_rows = vectorizer.transform([feature_dict(row) for row in rows])

    utilities = np.asarray(
        [int(row["annotation"]["utility"]) for row in train],
        dtype=np.int64,
    )
    severe = np.asarray(
        [
            int(row["annotation"]["semantic_relation"] in SEVERE_ERROR_RELATIONS)
            for row in train
        ],
        dtype=np.int64,
    )

    cumulative = []
    for threshold in (1, 2, 3):
        target = (utilities >= threshold).astype(np.int64)
        model = BinaryProbabilityModel(
            seed=seed + threshold,
            balanced=True,
        ).fit(x_train, target)
        cumulative.append(model.predict_positive(x_rows))

    expected_utility = sum(cumulative)
    severe_model = BinaryProbabilityModel(
        seed=seed + 10,
        balanced=True,
    ).fit(x_train, severe)
    severe_probability = severe_model.predict_positive(x_rows)
    safe_score = expected_utility - (severe_penalty * severe_probability)

    return safe_score, {
        "feature_count": len(vectorizer.feature_names_),
        "severe_penalty": severe_penalty,
    }


def _load_neural_scores(
    path: str | Path,
    validation: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    expected = {str(row["pair_id"]): str(row["query_id"]) for row in validation}
    observed: dict[str, float] = {}
    models: set[str] = set()

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            pair_id = str(payload.get("pair_id") or "")
            query_id = str(payload.get("query_id") or "")
            split = str(payload.get("split") or "")
            if not pair_id or pair_id not in expected:
                raise ValueError(
                    f"Neural score row {line_number} is not a frozen validation pair: {pair_id!r}"
                )
            if split != "validation":
                raise ValueError(
                    f"Neural score row {line_number} must declare split='validation', got {split!r}."
                )
            if expected[pair_id] != query_id:
                raise ValueError(
                    f"Neural score row {line_number} query mismatch for {pair_id}: "
                    f"expected {expected[pair_id]!r}, got {query_id!r}."
                )
            if pair_id in observed:
                raise ValueError(f"Duplicate neural score for pair_id={pair_id!r}.")
            score = float(payload["score"])
            if not np.isfinite(score):
                raise ValueError(f"Non-finite neural score for pair_id={pair_id!r}.")
            observed[pair_id] = score
            if payload.get("model"):
                models.add(str(payload["model"]))

    missing = sorted(set(expected) - set(observed))
    if missing:
        raise ValueError(
            f"Neural score file is missing {len(missing)} validation pairs; "
            f"first missing={missing[0]!r}."
        )

    ordered = np.asarray(
        [observed[str(row["pair_id"])] for row in validation],
        dtype=np.float64,
    )
    return ordered, {
        "score_file": str(path),
        "model": sorted(models)[0] if len(models) == 1 else None,
        "models_seen": sorted(models),
        "score_count": len(observed),
    }


def _query_rank_normalize(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
) -> np.ndarray:
    if len(rows) != len(scores):
        raise ValueError("rows and scores must have the same length.")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["query_id"])].append(index)

    normalized = np.zeros(len(rows), dtype=np.float64)
    for indices in grouped.values():
        values = np.asarray([scores[index] for index in indices], dtype=np.float64)
        if len(indices) == 1:
            normalized[indices[0]] = 1.0
            continue
        ranks = rankdata(values, method="average")
        scaled = (ranks - 1.0) / (len(indices) - 1.0)
        for index, value in zip(indices, scaled):
            normalized[index] = float(value)
    return normalized


def _parse_alphas(value: str) -> list[float]:
    alphas: list[float] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        alpha = float(token)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Hybrid alpha must be in [0, 1], got {alpha}.")
        if alpha not in alphas:
            alphas.append(alpha)
    if not alphas:
        raise ValueError("At least one hybrid alpha is required.")
    return alphas


def _select_validation_candidate(
    grid: list[dict[str, Any]],
    learned: dict[str, Any],
) -> dict[str, Any]:
    epsilon = 1e-12
    eligible = [
        item
        for item in grid
        if item["metrics"]["noise_rate_at_k_mean"]
        <= learned["noise_rate_at_k_mean"] + epsilon
        and item["metrics"]["severe_error_rate_at_k_mean"]
        <= learned["severe_error_rate_at_k_mean"] + epsilon
    ]
    pool = eligible or grid
    return max(
        pool,
        key=lambda item: (
            item["metrics"]["ndcg_at_k_mean"],
            item["metrics"]["high_utility_rate_at_k_mean"],
            item["metrics"]["mrr_high_utility_mean"],
            item["metrics"]["relation_diversity_at_k_mean"],
            -item["alpha"],
        ),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.input)
    _validate_split_integrity(rows)

    train = [row for row in rows if row.get("split") == "train"]
    validation = [row for row in rows if row.get("split") == "validation"]

    learned_scores, learned_meta = _learned_safe_scores(
        train,
        validation,
        seed=args.seed,
        severe_penalty=args.severe_penalty,
    )
    neural_scores, neural_meta = _load_neural_scores(args.neural_scores, validation)

    learned_rows = _rerank_for_metrics(validation, learned_scores)
    neural_rows = _rerank_for_metrics(validation, neural_scores)
    learned_report = benchmark_metrics(learned_rows, k=args.k)
    neural_report = benchmark_metrics(neural_rows, k=args.k)
    learned_summary = _metric_summary(learned_report)
    neural_summary = _metric_summary(neural_report)

    learned_rank = _query_rank_normalize(validation, learned_scores)
    neural_rank = _query_rank_normalize(validation, neural_scores)

    grid: list[dict[str, Any]] = []
    per_query: dict[str, Any] = {}
    for alpha in args.alphas:
        blended = ((1.0 - alpha) * learned_rank) + (alpha * neural_rank)
        hybrid_rows = _rerank_for_metrics(validation, blended)
        hybrid_report = benchmark_metrics(hybrid_rows, k=args.k)
        summary = _metric_summary(hybrid_report)
        grid.append(
            {
                "alpha": alpha,
                "baseline_weight": 1.0 - alpha,
                "neural_weight": alpha,
                "metrics": summary,
                "delta_vs_learned": _delta_summary(summary, learned_summary),
            }
        )
        if args.include_per_query:
            per_query[str(alpha)] = hybrid_report.get("queries", {})

    selected = _select_validation_candidate(grid, learned_summary)
    selected_summary = selected["metrics"]
    selected_alpha = float(selected["alpha"])
    selected_is_improvement = (
        selected_summary["ndcg_at_k_mean"] > learned_summary["ndcg_at_k_mean"] + 1e-12
        and selected_summary["noise_rate_at_k_mean"]
        <= learned_summary["noise_rate_at_k_mean"] + 1e-12
        and selected_summary["severe_error_rate_at_k_mean"]
        <= learned_summary["severe_error_rate_at_k_mean"] + 1e-12
    )

    report: dict[str, Any] = {
        "status": "validation_only_phase3_hybrid",
        "input": str(args.input),
        "neural_scores": neural_meta,
        "train_pair_count": len(train),
        "validation_pair_count": len(validation),
        "train_query_count": len({row["query_id"] for row in train}),
        "validation_query_count": len({row["query_id"] for row in validation}),
        "normalization": "within_query_average_rank_0_to_1",
        "selection_rule": (
            "maximize validation NDCG among alpha grid points that do not worsen "
            "learned-baseline noise or severe-error rate"
        ),
        "learned_baseline": learned_summary,
        "learned_baseline_meta": learned_meta,
        "neural_only": neural_summary,
        "delta_neural_vs_learned": _delta_summary(neural_summary, learned_summary),
        "hybrid_grid": grid,
        "selected_validation_alpha": selected_alpha,
        "selected_validation_metrics": selected_summary,
        "selected_delta_vs_learned": _delta_summary(selected_summary, learned_summary),
        "selected_is_improvement": selected_is_improvement,
        "benchmark_split_evaluated": False,
        "benchmark_pairs_used_for_training": 0,
        "benchmark_pairs_used_for_selection": 0,
    }
    if args.include_per_query:
        report["hybrid_validation_queries"] = per_query

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Blend the Phase-3 leakage-safe learned baseline with validation-only "
            "neural reranker scores without touching the frozen benchmark split."
        )
    )
    parser.add_argument(
        "--input",
        default="evaluation/writer_relevance_50_annotations.approved.jsonl",
    )
    parser.add_argument(
        "--neural-scores",
        required=True,
        help="Validation-only JSONL score artifact emitted by the cross-encoder harness.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/writer_relevance_phase3_hybrid_report.json",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--severe-penalty", type=float, default=1.0)
    parser.add_argument(
        "--alphas",
        default="0,0.1,0.2,0.3,0.4,0.5",
        help="Comma-separated neural weights; alpha=0 is the learned baseline.",
    )
    parser.add_argument("--include-per-query", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.k < 1:
        raise ValueError("--k must be at least 1.")
    if args.severe_penalty < 0:
        raise ValueError("--severe-penalty must be non-negative.")
    args.alphas = _parse_alphas(args.alphas)
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
