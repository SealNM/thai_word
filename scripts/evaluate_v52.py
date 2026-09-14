#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_hybrid_v2 import HybridSearcher
from thai_lexical_v1 import list_senses, load_artifacts
from thai_instruct_v52 import (
    DEFAULT_CTXL_INSTRUCT,
    build_ctxl_scorer,
    rank_ctxl_candidates,
    resolve_ctxl_dtype,
    score_ctxl_candidates,
)


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("queries"), list):
        raise ValueError("Evaluation config must contain a 'queries' array.")
    return data


def _words(results: list[dict[str, Any]], count: int) -> list[str]:
    return [str(item["word"]) for item in results[:count]]


def _fmt_seconds(value: float) -> str:
    value = max(0.0, float(value))
    minutes, seconds = divmod(int(round(value)), 60)
    if minutes:
        return f"{minutes:02d}:{seconds:02d}"
    return f"{seconds}s"


def _status(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ContextualAI's instruction-following multilingual reranker "
            "on Thai Words shared V2.5 top-50 candidate pools."
        )
    )
    parser.add_argument("--config", default="evaluation/v1_queries.json")
    parser.add_argument("--index", default="artifacts/v1")
    parser.add_argument("--dense-index", required=True)
    parser.add_argument("--candidate-pool", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dense-device", default=None)
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--mode",
        action="append",
        choices=["instruct", "fusion"],
        help="Repeat to select variants. Defaults to instruct + light V2.5 fusion.",
    )
    parser.add_argument("--v25-rank-weight", type=float, default=0.1)
    parser.add_argument("--instruct-rank-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    modes = args.mode or ["instruct", "fusion"]
    config = _load_config(args.config)
    specs = config["queries"]
    total_queries = len(specs)

    dense_device = args.dense_device or args.device
    reranker_device = args.reranker_device or args.device

    _status("[setup] Loading lexical artifacts...")
    lexical = load_artifacts(args.index)

    _status("[setup] Loading V2.5 dense retriever...")
    started = perf_counter()
    v25 = HybridSearcher.from_paths(
        lexical,
        args.dense_index,
        device=dense_device,
    )
    _status(f"[setup] V2.5 ready in {_fmt_seconds(perf_counter() - started)}")

    _status(
        f"[setup] Precomputing shared top-{args.candidate_pool} candidate pools..."
    )
    pools: list[dict[str, Any]] = []
    pool_started = perf_counter()

    for index, spec in enumerate(specs, start=1):
        query = str(spec["query"])
        sense = spec.get("sense")
        if sense is not None:
            sense = int(sense)
        category = spec.get("category")
        if category is not None:
            category = str(category)

        candidates = v25.search(
            query,
            top_k=max(args.top_k, args.candidate_pool),
            sense=sense,
            lexical_pool=max(300, args.candidate_pool),
            dense_pool=max(300, args.candidate_pool),
        )
        pools.append(
            {
                "query": query,
                "sense": sense,
                "category": category,
                "candidates": candidates,
            }
        )
        _status(
            f"[setup {index}/{total_queries}] {query}: "
            f"{len(candidates)} candidates cached"
        )

    _status(
        f"[setup] Candidate pools ready in "
        f"{_fmt_seconds(perf_counter() - pool_started)}"
    )

    dtype = resolve_ctxl_dtype(reranker_device)
    _status(
        f"[model] Loading {DEFAULT_CTXL_INSTRUCT} "
        f"(dtype={dtype}, batch={args.batch_size})..."
    )
    model_started = perf_counter()
    scorer = build_ctxl_scorer(
        device=reranker_device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    model_load_seconds = perf_counter() - model_started
    _status(
        f"[model] ContextualAI reranker ready in "
        f"{_fmt_seconds(model_load_seconds)}"
    )

    report: dict[str, Any] = {
        "config": args.config,
        "index": args.index,
        "dense_index": args.dense_index,
        "model": DEFAULT_CTXL_INSTRUCT,
        "dtype": dtype,
        "batch_size": args.batch_size,
        "candidate_pool": args.candidate_pool,
        "top_k": args.top_k,
        "modes": modes,
        "model_load_seconds": round(float(model_load_seconds), 3),
        "queries": [],
    }

    run_started = perf_counter()
    completed_seconds = 0.0

    for query_index, pool in enumerate(pools, start=1):
        query = pool["query"]
        sense = pool["sense"]
        category = pool["category"]
        candidates = pool["candidates"]
        prefix = f"[ctxl {query_index}/{total_queries}]"

        _status(
            f"\n{prefix} {query} [{category}] — scoring "
            f"{len(candidates)} candidates with custom instruction..."
        )

        query_started = perf_counter()
        scored = score_ctxl_candidates(
            query,
            candidates,
            scorer,
            category=category,
        )
        query_seconds = perf_counter() - query_started
        completed_seconds += query_seconds

        baseline = sorted(
            (dict(item) for item in scored),
            key=lambda item: int(item["v25_rank"]),
        )[: args.top_k]

        variants = {
            mode: rank_ctxl_candidates(
                scored,
                mode=mode,
                top_k=args.top_k,
                v25_weight=args.v25_rank_weight,
                instruct_weight=args.instruct_rank_weight,
                rrf_k=args.rrf_k,
            )
            for mode in modes
        }

        report["queries"].append(
            {
                "query": query,
                "category": category,
                "sense": sense,
                "available_senses": list_senses(lexical, query),
                "query_seconds": round(float(query_seconds), 4),
                "v25": baseline,
                "v52": variants,
            }
        )

        remaining = total_queries - query_index
        average = completed_seconds / query_index
        eta = average * remaining

        _status(
            f"{prefix} done in {_fmt_seconds(query_seconds)} · "
            f"ETA {_fmt_seconds(eta)}"
        )
        _status("V2.5          : " + " | ".join(_words(baseline, args.top_k)))
        for mode in modes:
            _status(
                f"V5.2 {mode:<8}: "
                + " | ".join(_words(variants[mode], args.top_k))
            )

    run_seconds = perf_counter() - run_started
    report["query_seconds"] = round(float(run_seconds), 3)
    report["seconds_per_query"] = round(
        float(run_seconds / max(1, total_queries)),
        4,
    )

    _status(
        f"\n[done] {total_queries}/{total_queries} in "
        f"{_fmt_seconds(run_seconds)} "
        f"({report['seconds_per_query']:.2f}s/query)"
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _status(f"[done] Saved report: {out}")


if __name__ == "__main__":
    main()
