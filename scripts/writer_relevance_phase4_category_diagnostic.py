#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer

from scripts.writer_relevance_phase3_baseline import (
    BinaryProbabilityModel,
    _metric_summary,
    _rerank_for_metrics,
    _validate_split_integrity,
)
from thai_substitutability import benchmark_metrics, read_jsonl
from thai_writer_runtime import writer_feature_dict, writer_text_pair


CATEGORY_MODES = ("include", "none", "omit")


def _learned_safe_scores(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    *,
    seed: int,
    severe_penalty: float,
    category_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_features = [
        writer_feature_dict(row, category_mode=category_mode)
        for row in train
    ]
    validation_features = [
        writer_feature_dict(row, category_mode=category_mode)
        for row in validation
    ]

    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(train_features)
    x_validation = vectorizer.transform(validation_features)

    utilities = np.asarray(
        [int(row["annotation"]["utility"]) for row in train],
        dtype=np.int64,
    )
    severe = np.asarray(
        [
            int(
                row["annotation"]["semantic_relation"]
                in {"opposite_misleading", "sense_mismatch", "unrelated"}
            )
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
        cumulative.append(model.predict_positive(x_validation))

    expected_utility = sum(cumulative)
    severe_model = BinaryProbabilityModel(
        seed=seed + 10,
        balanced=True,
    ).fit(x_train, severe)
    severe_probability = severe_model.predict_positive(x_validation)

    return (
        expected_utility - (severe_penalty * severe_probability),
        {
            "feature_count": len(vectorizer.feature_names_),
            "category_mode": category_mode,
        },
    )


def _load_crossencoder(
    model_path: str | Path,
    *,
    max_length: int,
    device: str | None,
) -> Any:
    from sentence_transformers import CrossEncoder

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"CrossEncoder checkpoint not found: {path}")
    return CrossEncoder(
        str(path),
        num_labels=1,
        max_length=max_length,
        device=device,
    )


def _neural_scores(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    category_mode: str,
    batch_size: int,
) -> np.ndarray:
    pairs = [
        writer_text_pair(row, category_mode=category_mode)
        for row in rows
    ]
    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    return np.asarray(scores, dtype=np.float64).reshape(-1)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.input)
    _validate_split_integrity(rows)

    train = [row for row in rows if row.get("split") == "train"]
    validation = [row for row in rows if row.get("split") == "validation"]

    report: dict[str, Any] = {
        "status": "phase4_runtime_category_diagnostic",
        "input": str(args.input),
        "train_pair_count": len(train),
        "validation_pair_count": len(validation),
        "train_query_count": len({row["query_id"] for row in train}),
        "validation_query_count": len({row["query_id"] for row in validation}),
        "benchmark_split_evaluated": False,
        "benchmark_pairs_used_for_selection": 0,
        "diagnostic_only": True,
        "learned": {},
        "neural": {},
    }

    baseline_report = benchmark_metrics(validation, k=args.k)
    report["v25_validation"] = _metric_summary(baseline_report)

    for mode in CATEGORY_MODES:
        learned_scores, meta = _learned_safe_scores(
            train,
            validation,
            seed=args.seed,
            severe_penalty=args.severe_penalty,
            category_mode=mode,
        )
        reranked = _rerank_for_metrics(validation, learned_scores)
        learned_report = benchmark_metrics(reranked, k=args.k)
        report["learned"][mode] = {
            "meta": meta,
            "metrics": _metric_summary(learned_report),
        }

    if args.model_path:
        model = _load_crossencoder(
            args.model_path,
            max_length=args.max_length,
            device=args.device,
        )
        for mode in CATEGORY_MODES:
            scores = _neural_scores(
                model,
                validation,
                category_mode=mode,
                batch_size=args.eval_batch_size,
            )
            reranked = _rerank_for_metrics(validation, scores)
            neural_report = benchmark_metrics(reranked, k=args.k)
            report["neural"][mode] = {
                "metrics": _metric_summary(neural_report),
            }
        report["neural_model_path"] = str(args.model_path)
        report["neural_device"] = str(getattr(model, "device", "unknown"))
    else:
        report["neural_skipped_reason"] = (
            "No --model-path supplied. Learned category ablation was still evaluated."
        )

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
            "Phase-4 development-only category contract diagnostic. "
            "Uses frozen train/validation only and never evaluates benchmark rows."
        )
    )
    parser.add_argument(
        "--input",
        default="evaluation/writer_relevance_50_annotations.approved.jsonl",
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--output",
        default="evaluation/writer_relevance_phase4_category_diagnostic.json",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--severe-penalty", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.k < 1:
        raise ValueError("--k must be >= 1.")
    if args.eval_batch_size < 1:
        raise ValueError("--eval-batch-size must be >= 1.")
    if args.max_length < 32:
        raise ValueError("--max-length must be >= 32.")
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
