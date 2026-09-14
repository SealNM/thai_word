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
from thai_reranker_v4 import (
    CrossEncoderPairScorer,
    V4Searcher,
    load_tnc_commonness,
    rank_v4_candidates,
    resolve_reranker_profile,
)


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("queries"), list):
        raise ValueError("Evaluation config must contain a 'queries' array.")
    return data


def _words(results: list[dict[str, Any]], count: int) -> list[str]:
    return [str(item["word"]) for item in results[:count]]


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
            "Compare Thai Words V4.3 rerankers with category-aware lexical "
            "substitutability prompts."
        )
    )
    parser.add_argument("--config", default="evaluation/v1_queries.json")
    parser.add_argument("--index", default="artifacts/v1")
    parser.add_argument("--dense-index", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-pool", type=int, default=50)
    parser.add_argument(
        "--mode",
        action="append",
        choices=[
            "rerank",
            "fusion",
            "protected",
            "commonness",
            "gated-commonness",
        ],
        help=(
            "Repeat to select ranking variants. V4.3 defaults to rerank + fusion "
            "so the 0.6B vs 4B model comparison stays clean."
        ),
    )
    parser.add_argument(
        "--reranker",
        action="append",
        help=(
            "Repeat to compare models. Defaults to qwen3-0.6b-v4.3 and "
            "qwen3-4b-v4.3 sequentially."
        ),
    )
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--no-instruction", action="store_true")
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=None,
        help="Override profile default. V4.3 defaults: 0.6B=16, 4B=4.",
    )
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dense-device", default=None)
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument(
        "--ignore-category",
        action="store_true",
        help="Control run: do not add evaluation category / grammatical role to the query.",
    )
    parser.add_argument("--v25-rank-weight", type=float, default=0.35)
    parser.add_argument("--reranker-rank-weight", type=float, default=1.0)
    parser.add_argument(
        "--commonness-source",
        choices=["none", "tnc"],
        default="tnc",
    )
    parser.add_argument(
        "--commonness-weight",
        action="append",
        type=float,
        help="Only used when gated-commonness is explicitly selected.",
    )
    parser.add_argument("--commonness-promotion-cap", type=int, default=4)
    parser.add_argument("--gate-lexical-tier", type=int, default=4)
    parser.add_argument("--gate-reranker-top", type=int, default=12)
    parser.add_argument("--gate-v25-top", type=int, default=20)
    parser.add_argument("--gate-strict-reranker-top", type=int, default=5)
    parser.add_argument("--gate-wide-v25-top", type=int, default=30)
    parser.add_argument("--v4-rrf-k", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    modes = args.mode or ["rerank", "fusion"]
    rerankers = args.reranker or [
        "qwen3-0.6b-v4.3",
        "qwen3-4b-v4.3",
    ]
    commonness_weights = args.commonness_weight or [0.75]
    config = _load_config(args.config)
    lexical = load_artifacts(args.index)

    dense_device = args.dense_device or args.device
    reranker_device = args.reranker_device or args.device

    commonness: dict[str, int] | None = None
    if args.commonness_source == "tnc":
        commonness = load_tnc_commonness()

    v25_load_started = perf_counter()
    v25 = HybridSearcher.from_paths(
        lexical,
        args.dense_index,
        device=dense_device,
    )
    v25_load_seconds = perf_counter() - v25_load_started

    report: dict[str, Any] = {
        "config": args.config,
        "index": args.index,
        "dense_index": args.dense_index,
        "top_k": args.top_k,
        "candidate_pool": args.candidate_pool,
        "category_context_enabled": not args.ignore_category,
        "modes": modes,
        "rerankers": rerankers,
        "v25_model_load_seconds": round(float(v25_load_seconds), 3),
        "commonness": {
            "source": args.commonness_source,
            "weights": commonness_weights,
            "promotion_cap": args.commonness_promotion_cap,
            "entries": len(commonness or {}),
        },
        "models": [],
    }

    all_started = perf_counter()

    for reranker_name in rerankers:
        profile = resolve_reranker_profile(reranker_name)
        instruction = profile.get("instruction")
        if args.instruction is not None:
            instruction = args.instruction
        if args.no_instruction:
            instruction = None

        batch_size = (
            args.reranker_batch_size
            if args.reranker_batch_size is not None
            else int(profile.get("default_batch_size", 16))
        )

        print(
            f"\n##### RERANKER: {profile['key']} "
            f"({profile['model_id']}, batch={batch_size}) #####"
        )

        model_load_started = perf_counter()
        scorer = CrossEncoderPairScorer(
            str(profile["model_id"]),
            instruction=instruction,
            device=reranker_device,
            batch_size=batch_size,
            max_length=args.max_length,
            model_kwargs=profile.get("model_kwargs"),
        )
        searcher = V4Searcher(
            v25=v25,
            scorer=scorer,
            commonness=commonness,
            commonness_source=args.commonness_source,
        )
        model_load_seconds = perf_counter() - model_load_started

        model_row: dict[str, Any] = {
            "profile": profile["key"],
            "model_id": profile["model_id"],
            "instruction_enabled": bool(instruction),
            "batch_size": batch_size,
            "model_kwargs": profile.get("model_kwargs") or {},
            "model_load_seconds": round(float(model_load_seconds), 3),
            "queries": [],
        }

        model_query_started = perf_counter()

        for spec in config["queries"]:
            query = str(spec["query"])
            sense = spec.get("sense")
            if sense is not None:
                sense = int(sense)
            category = None if args.ignore_category else spec.get("category")
            if category is not None:
                category = str(category)

            query_started = perf_counter()
            scored = searcher.retrieve_and_score(
                query,
                sense=sense,
                category=category,
                candidate_pool=max(args.top_k, args.candidate_pool),
            )
            query_seconds = perf_counter() - query_started

            baseline = sorted(
                (dict(item) for item in scored),
                key=lambda item: int(item["v25_rank"]),
            )[: args.top_k]

            variants: dict[str, list[dict[str, Any]]] = {}
            for mode in modes:
                if mode == "gated-commonness":
                    for weight in commonness_weights:
                        label = f"gated-commonness-{weight:g}"
                        variants[label] = rank_v4_candidates(
                            scored,
                            mode="gated-commonness",
                            top_k=args.top_k,
                            v25_weight=args.v25_rank_weight,
                            reranker_weight=args.reranker_rank_weight,
                            commonness_weight=weight,
                            commonness_promotion_cap=args.commonness_promotion_cap,
                            rrf_k=args.v4_rrf_k,
                            gate_lexical_tier=args.gate_lexical_tier,
                            gate_reranker_top=args.gate_reranker_top,
                            gate_v25_top=args.gate_v25_top,
                            gate_strict_reranker_top=args.gate_strict_reranker_top,
                            gate_wide_v25_top=args.gate_wide_v25_top,
                        )
                else:
                    variants[mode] = rank_v4_candidates(
                        scored,
                        mode=mode,
                        top_k=args.top_k,
                        v25_weight=args.v25_rank_weight,
                        reranker_weight=args.reranker_rank_weight,
                        rrf_k=args.v4_rrf_k,
                    )

            row = {
                "query": query,
                "category": spec.get("category"),
                "category_passed_to_reranker": category,
                "sense": sense,
                "available_senses": list_senses(lexical, query),
                "query_seconds": round(float(query_seconds), 4),
                "v25": baseline,
                "v4": variants,
            }
            model_row["queries"].append(row)

            print(f"\n=== {query} [{spec.get('category')}] ===")
            print(
                "V2.5                    :",
                " | ".join(_words(baseline, args.top_k)),
            )
            for label, results in variants.items():
                print(
                    f"V4 {label:<22}: "
                    + " | ".join(_words(results, args.top_k))
                )

        model_query_seconds = perf_counter() - model_query_started
        model_row["query_seconds"] = round(float(model_query_seconds), 3)
        model_row["query_count"] = len(model_row["queries"])
        model_row["seconds_per_query"] = round(
            float(model_query_seconds / max(1, len(model_row["queries"]))),
            4,
        )
        report["models"].append(model_row)

        del searcher
        del scorer
        _cleanup_gpu()

    report["total_seconds"] = round(float(perf_counter() - all_started), 3)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
