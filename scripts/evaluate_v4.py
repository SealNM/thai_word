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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare V2.5 with Thai Words V4/V4.1/V4.2 variants."
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
        help="Repeat to select variants. Defaults to rerank, fusion, gated-commonness.",
    )
    parser.add_argument("--reranker", default="qwen3-0.6b-v4.2")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--no-instruction", action="store_true")
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dense-device", default=None)
    parser.add_argument("--reranker-device", default=None)
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
        help="Repeat to sweep V4.2 familiarity strength. Defaults to 0.5 and 1.0.",
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

    modes = args.mode or ["rerank", "fusion", "gated-commonness"]
    commonness_weights = args.commonness_weight or [0.5, 1.0]
    config = _load_config(args.config)
    lexical = load_artifacts(args.index)

    dense_device = args.dense_device or args.device
    reranker_device = args.reranker_device or args.device

    commonness: dict[str, int] | None = None
    if args.commonness_source == "tnc":
        commonness = load_tnc_commonness()

    load_started = perf_counter()
    v25 = HybridSearcher.from_paths(
        lexical,
        args.dense_index,
        device=dense_device,
    )
    profile = resolve_reranker_profile(args.reranker)
    instruction = profile.get("instruction")
    if args.instruction is not None:
        instruction = args.instruction
    if args.no_instruction:
        instruction = None

    scorer = CrossEncoderPairScorer(
        str(profile["model_id"]),
        instruction=instruction,
        device=reranker_device,
        batch_size=args.reranker_batch_size,
        max_length=args.max_length,
    )
    searcher = V4Searcher(
        v25=v25,
        scorer=scorer,
        commonness=commonness,
        commonness_source=args.commonness_source,
    )
    load_seconds = perf_counter() - load_started

    report: dict[str, Any] = {
        "config": args.config,
        "index": args.index,
        "dense_index": args.dense_index,
        "top_k": args.top_k,
        "candidate_pool": args.candidate_pool,
        "reranker": {
            "profile": profile["key"],
            "model_id": profile["model_id"],
            "instruction_enabled": bool(instruction),
            "batch_size": args.reranker_batch_size,
        },
        "modes": modes,
        "commonness": {
            "source": args.commonness_source,
            "weights": commonness_weights,
            "promotion_cap": args.commonness_promotion_cap,
            "entries": len(commonness or {}),
        },
        "semantic_gate": {
            "lexical_tier": args.gate_lexical_tier,
            "reranker_top": args.gate_reranker_top,
            "v25_top": args.gate_v25_top,
            "strict_reranker_top": args.gate_strict_reranker_top,
            "wide_v25_top": args.gate_wide_v25_top,
        },
        "model_load_seconds": round(float(load_seconds), 3),
        "queries": [],
    }

    total_started = perf_counter()
    for spec in config["queries"]:
        query = str(spec["query"])
        sense = spec.get("sense")
        if sense is not None:
            sense = int(sense)

        query_started = perf_counter()
        scored = searcher.retrieve_and_score(
            query,
            sense=sense,
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
            "sense": sense,
            "available_senses": list_senses(lexical, query),
            "query_seconds": round(float(query_seconds), 4),
            "v25": baseline,
            "v4": variants,
        }
        report["queries"].append(row)

        print(f"\n=== {query} [{spec.get('category')}] ===")
        print("V2.5                    :", " | ".join(_words(baseline, args.top_k)))
        for label, results in variants.items():
            print(
                f"V4 {label:<22}: "
                + " | ".join(_words(results, args.top_k))
            )

    total_seconds = perf_counter() - total_started
    report["query_seconds"] = round(float(total_seconds), 3)
    report["query_count"] = len(report["queries"])
    report["seconds_per_query"] = round(
        float(total_seconds / max(1, len(report["queries"]))),
        4,
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
