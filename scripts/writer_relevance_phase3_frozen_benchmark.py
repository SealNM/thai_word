#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from scripts.writer_relevance_phase3_baseline import (
    _metric_summary,
    _rerank_for_metrics,
    _validate_split_integrity,
)
from scripts.writer_relevance_phase3_crossencoder import (
    _delta_summary,
    _predict,
    text_pair,
)
from scripts.writer_relevance_phase3_hybrid import (
    _learned_safe_scores,
    _query_rank_normalize,
)
from thai_substitutability import benchmark_metrics, read_jsonl


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def _validate_locked_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("status") != "locked_before_frozen_benchmark":
        raise ValueError("Candidate manifest is not locked for frozen benchmark evaluation.")

    hybrid = manifest.get("hybrid") or {}
    if hybrid.get("alpha_must_not_be_reselected_on_benchmark") is not True:
        raise ValueError("Locked manifest must forbid alpha reselection on benchmark.")
    alpha = float(hybrid.get("alpha", -1))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Locked alpha must be in [0, 1], got {alpha}.")

    policy = manifest.get("benchmark_policy") or {}
    for key in ("no_hyperparameter_changes_after_opening", "no_alpha_search", "no_model_selection"):
        if policy.get(key) is not True:
            raise ValueError(f"Benchmark policy must set {key}=true.")


def _validate_dataset_against_freeze(
    rows: list[dict[str, Any]],
    *,
    input_path: str | Path,
    manifest: dict[str, Any],
    split_manifest: dict[str, Any],
) -> None:
    _validate_split_integrity(rows)

    expected_hash = str((manifest.get("dataset") or {}).get("sha256") or "")
    observed_hash = _sha256(input_path)
    if not expected_hash or observed_hash != expected_hash:
        raise ValueError(
            "Approved dataset SHA-256 mismatch: "
            f"expected={expected_hash!r}, observed={observed_hash!r}."
        )

    frozen_splits = split_manifest.get("splits") or {}
    for split_name in ("train", "validation", "benchmark"):
        split_cfg = frozen_splits.get(split_name) or {}
        observed_rows = [row for row in rows if row.get("split") == split_name]
        observed_queries = sorted({str(row["query_id"]) for row in observed_rows})
        expected_queries = sorted(str(q) for q in split_cfg.get("queries", []))

        if observed_queries != expected_queries:
            raise ValueError(
                f"Frozen {split_name} query IDs mismatch. "
                f"Expected {expected_queries}, observed {observed_queries}."
            )
        if len(observed_rows) != int(split_cfg.get("pair_count", -1)):
            raise ValueError(
                f"Frozen {split_name} pair count mismatch: "
                f"expected {split_cfg.get('pair_count')}, observed {len(observed_rows)}."
            )


def _load_saved_crossencoder(
    model_path: str | Path,
    *,
    max_length: int,
    device: str | None,
) -> Any:
    from sentence_transformers import CrossEncoder

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Saved fine-tuned model not found: {path}")

    return CrossEncoder(
        str(path),
        num_labels=1,
        max_length=max_length,
        device=device,
    )


def _fixed_hybrid_scores(
    rows: list[dict[str, Any]],
    learned_scores: np.ndarray,
    neural_scores: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    learned_rank = _query_rank_normalize(rows, learned_scores)
    neural_rank = _query_rank_normalize(rows, neural_scores)
    return ((1.0 - alpha) * learned_rank) + (alpha * neural_rank)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirm_frozen_benchmark:
        raise ValueError(
            "Frozen benchmark is locked. Re-run with --confirm-frozen-benchmark only "
            "after the candidate configuration and saved checkpoint have been verified."
        )

    manifest = _load_json(args.candidate_manifest)
    split_manifest = _load_json(args.split_manifest)
    _validate_locked_manifest(manifest)

    rows = read_jsonl(args.input)
    _validate_dataset_against_freeze(
        rows,
        input_path=args.input,
        manifest=manifest,
        split_manifest=split_manifest,
    )

    train = [row for row in rows if row.get("split") == "train"]
    benchmark = [row for row in rows if row.get("split") == "benchmark"]

    dataset_cfg = manifest["dataset"]
    if len(train) != int(dataset_cfg["train_pair_count"]):
        raise ValueError("Train pair count does not match locked candidate manifest.")
    if len(benchmark) != int(dataset_cfg["benchmark_pair_count"]):
        raise ValueError("Benchmark pair count does not match locked candidate manifest.")

    learned_cfg = manifest["learned_baseline"]
    learned_scores, learned_meta = _learned_safe_scores(
        train,
        benchmark,
        seed=int(learned_cfg["seed"]),
        severe_penalty=float(learned_cfg["severe_penalty"]),
    )

    neural_cfg = manifest["neural_model"]
    model = _load_saved_crossencoder(
        args.model_path,
        max_length=int(neural_cfg["max_length"]),
        device=args.device,
    )
    predict_args = SimpleNamespace(
        eval_batch_size=int(neural_cfg["eval_batch_size"]),
        no_progress=bool(args.no_progress),
    )
    neural_scores = _predict(model, benchmark, predict_args)

    alpha = float(manifest["hybrid"]["alpha"])
    hybrid_scores = _fixed_hybrid_scores(
        benchmark,
        learned_scores,
        neural_scores,
        alpha=alpha,
    )

    v25_report = benchmark_metrics(benchmark, k=args.k)
    learned_report = benchmark_metrics(
        _rerank_for_metrics(benchmark, learned_scores),
        k=args.k,
    )
    neural_report = benchmark_metrics(
        _rerank_for_metrics(benchmark, neural_scores),
        k=args.k,
    )
    hybrid_report = benchmark_metrics(
        _rerank_for_metrics(benchmark, hybrid_scores),
        k=args.k,
    )

    v25_summary = _metric_summary(v25_report)
    learned_summary = _metric_summary(learned_report)
    neural_summary = _metric_summary(neural_report)
    hybrid_summary = _metric_summary(hybrid_report)

    report: dict[str, Any] = {
        "status": "final_frozen_benchmark_phase3_locked_candidate",
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "candidate_manifest": str(args.candidate_manifest),
        "split_manifest": str(args.split_manifest),
        "saved_model_path": str(args.model_path),
        "base_model": neural_cfg["base_model"],
        "locked_alpha": alpha,
        "normalization": manifest["hybrid"]["normalization"],
        "train_pair_count": len(train),
        "benchmark_pair_count": len(benchmark),
        "train_query_count": len({row["query_id"] for row in train}),
        "benchmark_query_count": len({row["query_id"] for row in benchmark}),
        "learned_baseline_meta": learned_meta,
        "v25": v25_summary,
        "learned_baseline": learned_summary,
        "neural_only": neural_summary,
        "locked_hybrid": hybrid_summary,
        "delta_hybrid_vs_v25": _delta_summary(hybrid_summary, v25_summary),
        "delta_hybrid_vs_learned": _delta_summary(hybrid_summary, learned_summary),
        "delta_hybrid_vs_neural": _delta_summary(hybrid_summary, neural_summary),
        "selection_performed_on_benchmark": False,
        "alpha_reselected_on_benchmark": False,
        "benchmark_opened": True,
        "benchmark_evaluation_count_this_run": 1,
    }
    if args.include_per_query:
        report["benchmark_queries"] = {
            "v25": v25_report.get("queries", {}),
            "learned_baseline": learned_report.get("queries", {}),
            "neural_only": neural_report.get("queries", {}),
            "locked_hybrid": hybrid_report.get("queries", {}),
        }

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
            "One-shot evaluation of the locked Phase-3 candidate on the frozen "
            "10-query benchmark. No model selection or alpha search is permitted."
        )
    )
    parser.add_argument(
        "--input",
        default="evaluation/writer_relevance_50_annotations.approved.jsonl",
    )
    parser.add_argument(
        "--candidate-manifest",
        default="evaluation/writer_relevance_phase3_locked_candidate.json",
    )
    parser.add_argument(
        "--split-manifest",
        default="evaluation/writer_relevance_50_split_manifest.json",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--output",
        default="evaluation/writer_relevance_phase3_frozen_benchmark_report.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--include-per-query", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--confirm-frozen-benchmark", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.k < 1:
        raise ValueError("--k must be at least 1.")
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
