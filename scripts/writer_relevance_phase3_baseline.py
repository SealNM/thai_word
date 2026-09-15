#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from thai_substitutability import SEVERE_ERROR_RELATIONS, benchmark_metrics, read_jsonl


def _num(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return float(value)
    return default


def _reciprocal(value: Any) -> float:
    number = _num(value, 0.0)
    return 1.0 / number if number > 0 else 0.0


def feature_dict(row: dict[str, Any]) -> dict[str, Any]:
    retrieval = row.get("retrieval") or {}
    query = row.get("query") or {}

    return {
        "v25_score": _num(retrieval.get("score")),
        "v25_rr": _reciprocal(retrieval.get("v25_rank")),
        "relation_tier": _num(retrieval.get("relation_tier")),
        "lexical_score": _num(retrieval.get("lexical_score")),
        "lexical_rr": _reciprocal(retrieval.get("lexical_rank")),
        "dense_similarity": _num(retrieval.get("dense_similarity")),
        "dense_rr": _reciprocal(retrieval.get("dense_rank")),
        "has_lexical_rank": float(retrieval.get("lexical_rank") is not None),
        "has_dense_rank": float(retrieval.get("dense_rank") is not None),
        "relation_hint": str(retrieval.get("relation_hint") or "<none>"),
        "lexical_form": str(retrieval.get("lexical_form") or "<none>"),
        "sense_resolution": str(retrieval.get("sense_resolution") or "<none>"),
        "query_category": str(query.get("category") or "<none>"),
    }


class BinaryProbabilityModel:
    def __init__(self, *, seed: int, balanced: bool) -> None:
        self.seed = seed
        self.balanced = balanced
        self.constant: float | None = None
        self.model: LogisticRegression | None = None

    def fit(self, x: Any, y: np.ndarray) -> "BinaryProbabilityModel":
        classes = np.unique(y)
        if len(classes) == 1:
            self.constant = float(classes[0])
            return self

        self.model = LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced" if self.balanced else None,
            random_state=self.seed,
        )
        self.model.fit(x, y)
        return self

    def predict_positive(self, x: Any) -> np.ndarray:
        if self.constant is not None:
            return np.full(x.shape[0], self.constant, dtype=np.float64)
        assert self.model is not None
        return self.model.predict_proba(x)[:, 1]


def _validate_split_integrity(rows: list[dict[str, Any]]) -> None:
    splits_by_query: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = str(row.get("split") or "")
        splits_by_query[str(row.get("query_id"))].add(split)

    leaking = {
        query_id: sorted(splits)
        for query_id, splits in splits_by_query.items()
        if len(splits) != 1
    }
    if leaking:
        raise ValueError(f"Target-sense leakage across splits: {leaking}")

    counts = Counter(str(row.get("split") or "") for row in rows)
    if not counts.get("train") or not counts.get("validation"):
        raise ValueError(
            "Input must contain frozen train and validation rows. "
            f"Observed split counts: {dict(counts)}"
        )


def _rerank_for_metrics(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[int, dict[str, Any], float]]] = defaultdict(list)
    for index, (row, score) in enumerate(zip(rows, scores)):
        grouped[str(row["query_id"])].append((index, row, float(score)))

    copied: list[dict[str, Any]] = []
    by_index: dict[int, dict[str, Any]] = {}

    for group in grouped.values():
        ordered = sorted(
            group,
            key=lambda item: (
                -item[2],
                int(item[1]["retrieval"]["v25_rank"]),
            ),
        )
        for new_rank, (index, row, score) in enumerate(ordered, start=1):
            clone = dict(row)
            retrieval = dict(row["retrieval"])
            retrieval["original_v25_rank"] = retrieval["v25_rank"]
            retrieval["v25_rank"] = new_rank
            retrieval["phase3_score"] = score
            clone["retrieval"] = retrieval
            by_index[index] = clone

    for index in range(len(rows)):
        copied.append(by_index[index])
    return copied


def _metric_summary(report: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "useful_rate_at_k_mean",
        "high_utility_rate_at_k_mean",
        "noise_rate_at_k_mean",
        "severe_error_rate_at_k_mean",
        "relation_diversity_at_k_mean",
        "ndcg_at_k_mean",
        "mrr_high_utility_mean",
    )
    return {key: report[key] for key in keys}


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.input)
    _validate_split_integrity(rows)

    train = [row for row in rows if row.get("split") == "train"]
    validation = [row for row in rows if row.get("split") == "validation"]

    # Intentionally never read benchmark labels into model fitting/evaluation.
    train_features = [feature_dict(row) for row in train]
    validation_features = [feature_dict(row) for row in validation]

    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(train_features)
    x_validation = vectorizer.transform(validation_features)

    utilities = np.array(
        [int(row["annotation"]["utility"]) for row in train],
        dtype=np.int64,
    )
    severe = np.array(
        [
            int(row["annotation"]["semantic_relation"] in SEVERE_ERROR_RELATIONS)
            for row in train
        ],
        dtype=np.int64,
    )

    cumulative_models: list[BinaryProbabilityModel] = []
    cumulative_validation: list[np.ndarray] = []
    for threshold in (1, 2, 3):
        target = (utilities >= threshold).astype(np.int64)
        model = BinaryProbabilityModel(
            seed=args.seed + threshold,
            balanced=True,
        ).fit(x_train, target)
        cumulative_models.append(model)
        cumulative_validation.append(model.predict_positive(x_validation))

    expected_utility = sum(cumulative_validation)

    severe_model = BinaryProbabilityModel(
        seed=args.seed + 10,
        balanced=True,
    ).fit(x_train, severe)
    severe_probability = severe_model.predict_positive(x_validation)

    safe_score = expected_utility - (args.severe_penalty * severe_probability)

    baseline = benchmark_metrics(validation, k=args.k)
    ordinal_rows = _rerank_for_metrics(validation, expected_utility)
    safe_rows = _rerank_for_metrics(validation, safe_score)
    ordinal_report = benchmark_metrics(ordinal_rows, k=args.k)
    safe_report = benchmark_metrics(safe_rows, k=args.k)

    report = {
        "status": "validation_only_phase3_baseline",
        "input": str(args.input),
        "train_pair_count": len(train),
        "validation_pair_count": len(validation),
        "train_query_count": len({row["query_id"] for row in train}),
        "validation_query_count": len({row["query_id"] for row in validation}),
        "feature_count": len(vectorizer.feature_names_),
        "severe_penalty": args.severe_penalty,
        "baseline_v25": _metric_summary(baseline),
        "ordinal_expected_utility": _metric_summary(ordinal_report),
        "ordinal_minus_severe": _metric_summary(safe_report),
        "benchmark_split_evaluated": False,
    }

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
            "Phase-3 leakage-safe learned baseline for Thai Words writer relevance. "
            "Fits on train and reports validation only."
        )
    )
    parser.add_argument(
        "--input",
        default="evaluation/writer_relevance_50_annotations.approved.jsonl",
    )
    parser.add_argument(
        "--output",
        default="evaluation/writer_relevance_phase3_baseline_report.json",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--severe-penalty", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.k < 1:
        raise ValueError("--k must be at least 1.")
    if args.severe_penalty < 0:
        raise ValueError("--severe-penalty must be non-negative.")

    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
