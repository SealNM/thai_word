#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import resource
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any

from thai_writer_runtime import runtime_row_from_v25_result
from thai_writer_search import DEFAULT_RERANK_POOL, WriterSearch


def _peak_rss_mb() -> float:
    # Kaggle/Linux reports ru_maxrss in KiB.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(float(value) / 1024.0, 3)


def _cuda_snapshot() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False}

    if not torch.cuda.is_available():
        return {"available": False}

    return {
        "available": True,
        "device_count": int(torch.cuda.device_count()),
        "allocated_mb": round(float(torch.cuda.memory_allocated() / (1024 ** 2)), 3),
        "reserved_mb": round(float(torch.cuda.memory_reserved() / (1024 ** 2)), 3),
        "max_allocated_mb": round(
            float(torch.cuda.max_memory_allocated() / (1024 ** 2)), 3
        ),
        "max_reserved_mb": round(
            float(torch.cuda.max_memory_reserved() / (1024 ** 2)), 3
        ),
    }


def _reset_cuda_peak() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _query_context(results: list[dict[str, Any]]) -> tuple[str, int | None]:
    if not results:
        return "", None
    payload = results[0].get("query_sense")
    if not isinstance(payload, dict):
        return "", None
    sense = payload.get("sense")
    return (
        str(payload.get("definition") or ""),
        int(sense) if sense is not None else None,
    )


def _runtime_rows(
    query: str,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    definition, sense = _query_context(results)
    return [
        runtime_row_from_v25_result(
            query=query,
            query_definition=definition,
            candidate=item,
            v25_rank=index,
            query_sense=sense,
        )
        for index, item in enumerate(results, start=1)
    ]


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": round(float(statistics.mean(values)), 6),
        "median": round(float(statistics.median(values)), 6),
        "min": round(float(min(values)), 6),
        "max": round(float(max(values)), 6),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1.")
    if args.rerank_pool < 1:
        raise ValueError("--rerank-pool must be >= 1.")
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1.")
    if args.rerank_pool < args.top_k:
        raise ValueError("--rerank-pool must be >= --top-k.")

    startup_started = perf_counter()
    searcher = WriterSearch.from_paths(
        lexical_index=args.index,
        dense_index=args.dense_index,
        learned_ranker=args.learned_ranker,
        neural_model=args.neural_model,
        mode="required",
        dense_device=args.dense_device,
        neural_device=args.neural_device,
        neural_batch_size=args.neural_batch_size,
        neural_max_length=args.neural_max_length,
    )
    startup_seconds = perf_counter() - startup_started

    baseline_started = perf_counter()
    v25_results = searcher.v25.search(
        args.query,
        top_k=args.rerank_pool,
        sense=args.sense,
        lexical_pool=args.lexical_pool,
        dense_pool=args.dense_pool,
        lexical_weight=args.lexical_weight,
        dense_weight=args.dense_weight,
        rrf_k=args.rrf_k,
    )
    v25_seconds = perf_counter() - baseline_started
    runtime_rows = _runtime_rows(args.query, v25_results)

    learned_started = perf_counter()
    assert searcher.learned is not None
    searcher.learned.score_rows(runtime_rows)
    learned_seconds = perf_counter() - learned_started

    _reset_cuda_peak()
    cold_started = perf_counter()
    cold_results = searcher.search(
        args.query,
        top_k=args.top_k,
        sense=args.sense,
        rerank_pool=args.rerank_pool,
        lexical_pool=args.lexical_pool,
        dense_pool=args.dense_pool,
        lexical_weight=args.lexical_weight,
        dense_weight=args.dense_weight,
        rrf_k=args.rrf_k,
    )
    cold_writer_seconds = perf_counter() - cold_started
    cold_cuda = _cuda_snapshot()

    assert searcher.neural is not None
    warm_neural_times: list[float] = []
    for _ in range(args.repeats):
        started = perf_counter()
        searcher.neural.score_rows(runtime_rows)
        warm_neural_times.append(perf_counter() - started)

    _reset_cuda_peak()
    warm_writer_times: list[float] = []
    for _ in range(args.repeats):
        started = perf_counter()
        searcher.search(
            args.query,
            top_k=args.top_k,
            sense=args.sense,
            rerank_pool=args.rerank_pool,
            lexical_pool=args.lexical_pool,
            dense_pool=args.dense_pool,
            lexical_weight=args.lexical_weight,
            dense_weight=args.dense_weight,
            rrf_k=args.rrf_k,
        )
        warm_writer_times.append(perf_counter() - started)
    warm_cuda = _cuda_snapshot()

    neural_info = searcher.neural.runtime_info()

    report = {
        "status": "phase4_runtime_profile",
        "query": args.query,
        "sense": args.sense,
        "top_k": args.top_k,
        "rerank_pool": args.rerank_pool,
        "candidate_count": len(v25_results),
        "repeats": args.repeats,
        "startup_seconds_excluding_neural_weights": round(startup_seconds, 6),
        "v25_search_seconds": round(v25_seconds, 6),
        "learned_score_seconds": round(learned_seconds, 6),
        "cold_writer_search_seconds": round(cold_writer_seconds, 6),
        "neural_model_load_seconds": neural_info.get("load_seconds"),
        "warm_neural_score_seconds": _summary(warm_neural_times),
        "warm_writer_search_seconds": _summary(warm_writer_times),
        "peak_rss_mb": _peak_rss_mb(),
        "cold_cuda": cold_cuda,
        "warm_cuda": warm_cuda,
        "runtime": searcher.runtime_info(),
        "cold_result_words": [item["word"] for item in cold_results],
        "quality_selection_performed": False,
        "phase3_benchmark_used": False,
    }

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile Phase-4 writer search runtime without changing ranking."
    )
    parser.add_argument("query")
    parser.add_argument("--index", default="artifacts/v1")
    parser.add_argument("--dense-index", required=True)
    parser.add_argument("--learned-ranker", default="artifacts/writer-reranker")
    parser.add_argument("--neural-model", required=True)
    parser.add_argument("--sense", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rerank-pool", type=int, default=DEFAULT_RERANK_POOL)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--lexical-pool", type=int, default=300)
    parser.add_argument("--dense-pool", type=int, default=300)
    parser.add_argument("--lexical-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-device", default=None)
    parser.add_argument("--neural-device", default=None)
    parser.add_argument("--neural-batch-size", type=int, default=4)
    parser.add_argument("--neural-max-length", type=int, default=384)
    parser.add_argument(
        "--output",
        default="evaluation/writer_relevance_phase4_runtime_profile.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
