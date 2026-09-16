#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from scripts.writer_relevance_phase3_baseline import _metric_summary, _rerank_for_metrics
from scripts.writer_relevance_phase3_crossencoder import _delta_summary
from scripts.writer_relevance_phase3_hybrid import _query_rank_normalize
from thai_substitutability import benchmark_metrics, read_jsonl, validate_rows
from thai_writer_learned import LEARNED_RANKER_METADATA_FILENAME, LearnedWriterRanker
from thai_writer_neural import NeuralWriterRanker
from thai_writer_runtime import runtime_row_from_v25_result

EXPECTED_SHA = "7d719f22bf7834ab24bfacd91b3535871f05aa5db1e9273579e0e76a8f7c204c"
PHASE3_BLOB = "7dd0796bd9ef8d719f0febdb62342cb6dbb01d17"
ALPHA = 0.5
POOL = 30


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return value


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_blob_sha1(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _validate_gate(gate: dict[str, Any]) -> None:
    if gate.get("status") != "phase4_post_freeze_evaluation_open":
        raise ValueError("Phase-4 evaluation gate is not open.")
    if gate.get("holdout_opened_for_model_evaluation") is not True:
        raise ValueError("Holdout must be explicitly opened after label freeze.")
    policy = gate.get("policy") or {}
    for key in (
        "no_retraining", "no_threshold_tuning", "no_alpha_search",
        "no_candidate_reselection", "no_architecture_change",
        "single_controlled_evaluation_cycle",
    ):
        if policy.get(key) is not True:
            raise ValueError(f"Evaluation policy must set {key}=true.")
    runtime = gate.get("runtime_contract") or {}
    if float(runtime.get("alpha", -1)) != ALPHA:
        raise ValueError("Locked alpha must remain 0.5.")
    if int(runtime.get("rerank_pool", -1)) != POOL:
        raise ValueError("Locked rerank pool must remain 30.")
    if runtime.get("category_mode") != "omit":
        raise ValueError("Production category mode must remain 'omit'.")


def _validate_phase3_manifest(path: str | Path) -> dict[str, Any]:
    if _git_blob_sha1(path) != PHASE3_BLOB:
        raise ValueError("Phase-3 locked-candidate manifest changed after freeze.")
    manifest = _load_json(path)
    frozen = manifest.get("frozen_benchmark") or {}
    if frozen.get("status") != "completed":
        raise ValueError("Phase-3 frozen benchmark is not complete.")
    if frozen.get("selection_performed_on_benchmark") is not False:
        raise ValueError("Phase-3 manifest is not selection-safe.")
    if frozen.get("alpha_reselected_on_benchmark") is not False:
        raise ValueError("Phase-3 alpha was reselected on benchmark.")
    if float(frozen.get("locked_alpha", -1)) != ALPHA:
        raise ValueError("Phase-3 locked alpha mismatch.")
    checkpoint = manifest.get("reproducibility_checkpoint") or {}
    if checkpoint.get("artifact_verified") is not True:
        raise ValueError("Phase-3 checkpoint is not marked verified.")
    return manifest


def _validate_holdout(rows: list[dict[str, Any]], path: str | Path) -> None:
    if _sha256(path) != EXPECTED_SHA:
        raise ValueError("Approved Phase-4 holdout SHA-256 mismatch.")
    errors = validate_rows(rows, require_labels=True)
    if errors:
        raise ValueError("\n".join(errors[:20]))
    if len(rows) != 600:
        raise ValueError(f"Expected 600 rows, got {len(rows)}.")
    counts = Counter(str(row["query_id"]) for row in rows)
    if len(counts) != 20 or set(counts.values()) != {30}:
        raise ValueError("Holdout must be exactly 20 queries x 30 candidates.")
    if any(row.get("split") != "phase4_holdout" for row in rows):
        raise ValueError("Every row must use split='phase4_holdout'.")


def _validate_learned(path: str | Path) -> dict[str, Any]:
    meta = _load_json(Path(path) / LEARNED_RANKER_METADATA_FILENAME)
    expected = {
        "dataset_sha256": "6e767583302a6df75c9b76d86fc73cbda150c98fbbe7c95b912a650a04a1a515",
        "category_mode": "omit",
        "seed": 42,
        "severe_penalty": 1.0,
        "train_pair_count": 930,
        "train_query_count": 31,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Learned artifact metadata mismatch for {key}.")
    return meta


def _runtime_row(row: dict[str, Any]) -> dict[str, Any]:
    query, candidate, retrieval = row["query"], row["candidate"], row["retrieval"]
    v25 = {
        "word": candidate["word"],
        "definition": candidate.get("definition"),
        "score": retrieval.get("score"),
        "relation_tier": retrieval.get("relation_tier"),
        "relation_hint": retrieval.get("relation_hint"),
        "lexical_form": retrieval.get("lexical_form"),
        "sense_resolution": retrieval.get("sense_resolution"),
        "lexical_score": retrieval.get("lexical_score"),
        "lexical_rank": retrieval.get("lexical_rank"),
        "dense_similarity": retrieval.get("dense_similarity"),
        "dense_rank": retrieval.get("dense_rank"),
        "matched_candidate_sense": {
            "sense": candidate.get("sense"),
            "definition": candidate.get("definition"),
        },
    }
    return runtime_row_from_v25_result(
        query=str(query["word"]),
        query_definition=str(query.get("definition") or ""),
        query_sense=int(query["sense"]),
        candidate=v25,
        v25_rank=int(retrieval["v25_rank"]),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirm_phase4_holdout:
        raise ValueError("Use --confirm-phase4-holdout for the single controlled evaluation.")
    gate = _load_json(args.gate)
    _validate_gate(gate)
    phase3 = _validate_phase3_manifest(args.phase3_manifest)
    rows = read_jsonl(args.input)
    _validate_holdout(rows, args.input)
    learned_meta = _validate_learned(args.learned_ranker)

    runtime_rows = [_runtime_row(row) for row in rows]
    learned = LearnedWriterRanker.load(args.learned_ranker)
    learned_scores = np.asarray(
        [item["safe_score"] for item in learned.score_rows(runtime_rows)], dtype=np.float64
    )

    expected_name = Path(str((phase3.get("reproducibility_checkpoint") or {}).get("saved_model_path"))).name
    if Path(args.neural_model).name != expected_name:
        raise ValueError("Neural checkpoint directory does not match Phase-3 locked artifact.")
    neural = NeuralWriterRanker(
        args.neural_model, device=args.device, batch_size=args.eval_batch_size,
        max_length=384, allow_download=False,
    )
    neural_scores = np.asarray(neural.score_rows(runtime_rows), dtype=np.float64)

    learned_rank = _query_rank_normalize(rows, learned_scores)
    neural_rank = _query_rank_normalize(rows, neural_scores)
    hybrid_scores = (0.5 * learned_rank) + (0.5 * neural_rank)

    raw_reports = {
        "v25": benchmark_metrics(rows, k=args.k),
        "learned_production": benchmark_metrics(_rerank_for_metrics(rows, learned_scores), k=args.k),
        "neural_production": benchmark_metrics(_rerank_for_metrics(rows, neural_scores), k=args.k),
        "locked_hybrid_production": benchmark_metrics(_rerank_for_metrics(rows, hybrid_scores), k=args.k),
    }
    summaries = {name: _metric_summary(report) for name, report in raw_reports.items()}
    report = {
        "status": "phase4_post_freeze_single_controlled_evaluation",
        "input_sha256": _sha256(args.input),
        "label_boundary_commit": gate.get("label_boundary_commit"),
        "runtime_contract": gate.get("runtime_contract"),
        "learned_metadata": learned_meta,
        "neural_runtime": neural.runtime_info(),
        **summaries,
        "delta_hybrid_vs_v25": _delta_summary(summaries["locked_hybrid_production"], summaries["v25"]),
        "delta_hybrid_vs_learned": _delta_summary(summaries["locked_hybrid_production"], summaries["learned_production"]),
        "delta_hybrid_vs_neural": _delta_summary(summaries["locked_hybrid_production"], summaries["neural_production"]),
        "selection_performed_on_holdout": False,
        "retraining_performed": False,
        "threshold_tuning_performed": False,
        "alpha_search_performed": False,
        "candidate_reselection_performed": False,
        "architecture_change_performed": False,
        "evaluation_count_this_run": 1,
    }
    if args.include_per_query:
        report["queries"] = {name: value["queries"] for name, value in raw_reports.items()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="One-shot Phase-4 frozen holdout evaluation; no tuning allowed.")
    p.add_argument("--input", default="evaluation/writer_relevance_phase4_holdout_annotations.approved.jsonl")
    p.add_argument("--gate", default="evaluation/writer_relevance_phase4_evaluation_gate.json")
    p.add_argument("--phase3-manifest", default="evaluation/writer_relevance_phase3_locked_candidate.json")
    p.add_argument("--learned-ranker", default="artifacts/writer-reranker")
    p.add_argument("--neural-model", default="artifacts/phase3/bge-reranker-v2-m3-locked")
    p.add_argument("--output", default="evaluation/writer_relevance_phase4_holdout_evaluation_report.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--include-per-query", action="store_true")
    p.add_argument("--confirm-phase4-holdout", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.k < 1 or args.eval_batch_size < 1:
        raise ValueError("k and eval-batch-size must be >= 1.")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
