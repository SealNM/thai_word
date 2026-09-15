#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.writer_relevance_phase3_baseline import (
    _metric_summary,
    _rerank_for_metrics,
    _validate_split_integrity,
)
from thai_substitutability import benchmark_metrics, read_jsonl


DEFAULT_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def _clip_text(value: Any, *, max_chars: int = 700) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _retrieval_evidence(row: dict[str, Any]) -> str:
    retrieval = row.get("retrieval") or {}
    parts = [
        f"v25_rank={retrieval.get('v25_rank', '')}",
        f"v25_score={retrieval.get('score', '')}",
        f"relation_tier={retrieval.get('relation_tier', '')}",
        f"lexical_score={retrieval.get('lexical_score', '')}",
        f"lexical_rank={retrieval.get('lexical_rank', '')}",
        f"dense_similarity={retrieval.get('dense_similarity', '')}",
        f"dense_rank={retrieval.get('dense_rank', '')}",
        f"relation_hint={retrieval.get('relation_hint') or '<none>'}",
        f"lexical_form={retrieval.get('lexical_form') or '<none>'}",
        f"sense_resolution={retrieval.get('sense_resolution') or '<none>'}",
    ]
    return " | ".join(parts)


def text_pair(row: dict[str, Any]) -> tuple[str, str]:
    query = row.get("query") or {}
    candidate = row.get("candidate") or {}

    target_text = "\n".join(
        [
            f"คำเป้าหมาย: {_clip_text(query.get('word'), max_chars=120)}",
            f"ความหมายเป้าหมาย: {_clip_text(query.get('definition'))}",
            f"หมวด: {_clip_text(query.get('category'), max_chars=120) or '<none>'}",
        ]
    )
    candidate_text = "\n".join(
        [
            f"คำที่พิจารณา: {_clip_text(candidate.get('word'), max_chars=120)}",
            f"ความหมายคำที่พิจารณา: {_clip_text(candidate.get('definition'))}",
            f"หลักฐาน retrieval: {_retrieval_evidence(row)}",
        ]
    )
    return target_text, candidate_text


def utility_target(row: dict[str, Any]) -> float:
    utility = int(row["annotation"]["utility"])
    if utility not in {0, 1, 2, 3}:
        raise ValueError(f"Invalid writer utility: {utility!r}")
    return utility / 3.0


def _fit_model(train: list[dict[str, Any]], args: argparse.Namespace) -> tuple[Any, int]:
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    model = CrossEncoder(
        args.model,
        num_labels=1,
        max_length=args.max_length,
    )
    examples = [
        InputExample(texts=list(text_pair(row)), label=utility_target(row))
        for row in train
    ]
    generator = None
    try:
        import torch

        generator = torch.Generator()
        generator.manual_seed(args.seed)
    except Exception:
        generator = None

    loader = DataLoader(
        examples,
        shuffle=True,
        batch_size=args.batch_size,
        generator=generator,
    )
    total_steps = max(1, len(loader) * args.epochs)
    warmup_steps = int(round(total_steps * args.warmup_ratio))

    model.fit(
        train_dataloader=loader,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        output_path=str(args.model_output) if args.model_output else None,
        save_best_model=False,
        show_progress_bar=not args.no_progress,
    )
    return model, warmup_steps


def _predict(model: Any, rows: list[dict[str, Any]], args: argparse.Namespace) -> np.ndarray:
    pairs = [text_pair(row) for row in rows]
    scores = model.predict(
        pairs,
        batch_size=args.eval_batch_size,
        show_progress_bar=not args.no_progress,
    )
    return np.asarray(scores, dtype=np.float64).reshape(-1)


def _delta_summary(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    keys = (
        "useful_rate_at_k_mean",
        "high_utility_rate_at_k_mean",
        "noise_rate_at_k_mean",
        "severe_error_rate_at_k_mean",
        "relation_diversity_at_k_mean",
        "ndcg_at_k_mean",
        "mrr_high_utility_mean",
    )
    return {
        key: float(candidate[key]) - float(reference[key])
        for key in keys
        if key in candidate and key in reference
    }


def _load_learned_floor(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    floor = payload.get("ordinal_minus_severe")
    if not isinstance(floor, dict):
        raise ValueError(
            "Learned floor report must contain an 'ordinal_minus_severe' metric summary."
        )
    return floor


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.input)
    _validate_split_integrity(rows)

    train = [row for row in rows if row.get("split") == "train"]
    validation = [row for row in rows if row.get("split") == "validation"]

    # Benchmark rows remain present only for split-integrity validation. Their labels/text
    # are never supplied to the model or to validation metrics in this experiment.
    model, warmup_steps = _fit_model(train, args)
    scores = _predict(model, validation, args)

    baseline_report = benchmark_metrics(validation, k=args.k)
    reranked = _rerank_for_metrics(validation, scores)
    candidate_report = benchmark_metrics(reranked, k=args.k)

    baseline_summary = _metric_summary(baseline_report)
    candidate_summary = _metric_summary(candidate_report)
    learned_floor = _load_learned_floor(args.learned_floor_report)

    report: dict[str, Any] = {
        "status": "validation_only_phase3_crossencoder",
        "input": str(args.input),
        "model": args.model,
        "objective": "writer_utility_soft_binary_regression_0_to_1",
        "train_pair_count": len(train),
        "validation_pair_count": len(validation),
        "train_query_count": len({row["query_id"] for row in train}),
        "validation_query_count": len({row["query_id"] for row in validation}),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "max_length": args.max_length,
        "seed": args.seed,
        "baseline_v25": baseline_summary,
        "cross_encoder_utility": candidate_summary,
        "delta_vs_v25": _delta_summary(candidate_summary, baseline_summary),
        "benchmark_split_evaluated": False,
        "benchmark_pairs_used_for_training": 0,
    }

    if learned_floor is not None:
        report["learned_floor"] = learned_floor
        report["delta_vs_learned_floor"] = _delta_summary(
            candidate_summary,
            learned_floor,
        )

    if args.include_per_query:
        report["cross_encoder_validation_queries"] = candidate_report.get("queries", {})

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
            "Phase-3 validation-only cross-encoder experiment for Thai Words writer "
            "utility ranking. Fits on train, evaluates on validation, and never scores "
            "the frozen benchmark split."
        )
    )
    parser.add_argument(
        "--input",
        default="evaluation/writer_relevance_50_annotations.approved.jsonl",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output",
        default="evaluation/writer_relevance_phase3_crossencoder_report.json",
    )
    parser.add_argument("--model-output", default=None)
    parser.add_argument(
        "--learned-floor-report",
        default="evaluation/writer_relevance_phase3_baseline_validation_report.json",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-per-query", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.k < 1:
        raise ValueError("--k must be at least 1.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("Batch sizes must be at least 1.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("--warmup-ratio must be in [0, 1).")
    if args.max_length < 32:
        raise ValueError("--max-length must be at least 32.")

    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
