#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_hybrid_v2 import HybridSearcher
from thai_lexical_v1 import list_senses, load_artifacts
from thai_listwise_v5 import annotate_listwise_scores, rank_v5_candidates
from thai_generative_v51 import resolve_v51_ranker


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


def _cleanup_gpu() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare V5 Jina listwise against V5.1 Qwen3.5-0.8B generative "
            "listwise ranking on the exact same V2.5 candidate pools."
        )
    )
    parser.add_argument("--config", default="evaluation/v1_queries.json")
    parser.add_argument("--index", default="artifacts/v1")
    parser.add_argument("--dense-index", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-pool", type=int, default=50)
    parser.add_argument(
        "--ranker",
        action="append",
        choices=["jina-v3.5", "qwen3.5-0.8b"],
        help="Repeat to select rankers. Defaults to Jina then Qwen3.5.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=["listwise", "fusion"],
        help="Repeat to select variants. Defaults to listwise + light V2.5 fusion.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dense-device", default=None)
    parser.add_argument("--ranker-device", default=None)
    parser.add_argument("--qwen-dtype", default=None)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=384)
    parser.add_argument("--v25-rank-weight", type=float, default=0.2)
    parser.add_argument("--listwise-rank-weight", type=float, default=1.0)
    parser.add_argument("--v5-rrf-k", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    ranker_names = args.ranker or ["jina-v3.5", "qwen3.5-0.8b"]
    modes = args.mode or ["listwise", "fusion"]

    config = _load_config(args.config)
    specs = config["queries"]
    total_queries = len(specs)

    _status("[setup] Loading lexical artifacts...")
    lexical = load_artifacts(args.index)

    dense_device = args.dense_device or args.device
    ranker_device = args.ranker_device or args.device

    _status("[setup] Loading V2.5 dense retriever...")
    load_started = perf_counter()
    v25 = HybridSearcher.from_paths(
        lexical,
        args.dense_index,
        device=dense_device,
    )
    _status(
        f"[setup] V2.5 ready in {_fmt_seconds(perf_counter() - load_started)}"
    )

    _status(
        f"[setup] Precomputing the same top-{args.candidate_pool} candidate pool "
        "for every ranker..."
    )
    pool_started = perf_counter()
    pools: list[dict[str, Any]] = []

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
            lexical_pool=max(args.candidate_pool, 300),
            dense_pool=max(args.candidate_pool, 300),
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

    report: dict[str, Any] = {
        "config": args.config,
        "index": args.index,
        "dense_index": args.dense_index,
        "candidate_pool": args.candidate_pool,
        "top_k": args.top_k,
        "rankers": ranker_names,
        "modes": modes,
        "models": [],
    }

    for model_index, ranker_name in enumerate(ranker_names, start=1):
        _cleanup_gpu()
        _status(
            f"\n##### [{model_index}/{len(ranker_names)}] Loading {ranker_name} #####"
        )
        model_started = perf_counter()
        resolved_name, ranker = resolve_v51_ranker(
            ranker_name,
            device=ranker_device,
            qwen_dtype=args.qwen_dtype,
            qwen_max_new_tokens=args.qwen_max_new_tokens,
        )
        model_load_seconds = perf_counter() - model_started
        _status(
            f"[model] {resolved_name} ready in "
            f"{_fmt_seconds(model_load_seconds)}"
        )

        model_row: dict[str, Any] = {
            "ranker": resolved_name,
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

            prefix = f"[{resolved_name} {query_index}/{total_queries}]"
            _status(
                f"\n{prefix} {query} [{category}] — ranking "
                f"{len(candidates)} shared candidates..."
            )

            query_started = perf_counter()
            scored = annotate_listwise_scores(
                query,
                candidates,
                ranker,
                category=category,
            )
            query_seconds = perf_counter() - query_started
            completed_seconds += query_seconds

            baseline = sorted(
                (dict(item) for item in scored),
                key=lambda item: int(item["v25_rank"]),
            )[: args.top_k]

            variants = {
                mode: rank_v5_candidates(
                    scored,
                    mode=mode,
                    top_k=args.top_k,
                    v25_weight=args.v25_rank_weight,
                    listwise_weight=args.listwise_rank_weight,
                    rrf_k=args.v5_rrf_k,
                )
                for mode in modes
            }

            generation = None
            if hasattr(ranker, "last_output"):
                generation = {
                    "parse_complete": bool(
                        getattr(ranker, "last_parse_complete", False)
                    ),
                    "parse_strategy": str(
                        getattr(ranker, "last_parse_strategy", "unknown")
                    ),
                    "parsed_count": int(
                        getattr(ranker, "last_parsed_count", 0)
                    ),
                    "raw_output": str(getattr(ranker, "last_output", "")),
                }

            row = {
                "query": query,
                "category": category,
                "sense": sense,
                "available_senses": list_senses(lexical, query),
                "query_seconds": round(float(query_seconds), 4),
                "v25": baseline,
                "v5": variants,
                "generation": generation,
            }
            model_row["queries"].append(row)

            remaining = total_queries - query_index
            average = completed_seconds / query_index
            eta = average * remaining

            _status(
                f"{prefix} done in {_fmt_seconds(query_seconds)} · "
                f"ETA {_fmt_seconds(eta)}"
            )
            if generation is not None:
                parse_label = (
                    "complete" if generation["parse_complete"] else "recovered"
                )
                _status(
                    f"{prefix} generation parse: {parse_label} "
                    f"({generation['parse_strategy']})"
                )

            _status("V2.5       : " + " | ".join(_words(baseline, args.top_k)))
            for mode in modes:
                _status(
                    f"V5.1 {mode:<7}: "
                    + " | ".join(_words(variants[mode], args.top_k))
                )

        run_seconds = perf_counter() - run_started
        model_row["query_seconds"] = round(float(run_seconds), 3)
        model_row["seconds_per_query"] = round(
            float(run_seconds / max(1, total_queries)),
            4,
        )
        report["models"].append(model_row)

        _status(
            f"\n[model done] {resolved_name}: "
            f"{_fmt_seconds(run_seconds)} total · "
            f"{model_row['seconds_per_query']:.2f}s/query"
        )

        del ranker
        _cleanup_gpu()

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _status(f"\n[done] Saved report: {out}")


if __name__ == "__main__":
    main()
