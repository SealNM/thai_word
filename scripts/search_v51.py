#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_hybrid_v2 import HybridSearcher
from thai_lexical_v1 import list_senses, load_artifacts
from thai_generative_v51 import resolve_v51_ranker
from thai_listwise_v5 import V5Searcher


def _status(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search Thai Words V5.1 with instruction-following listwise ranking."
    )
    parser.add_argument("query")
    parser.add_argument("--index", default="artifacts/v1")
    parser.add_argument("--dense-index", required=True)
    parser.add_argument("--sense", type=int, default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-pool", type=int, default=50)
    parser.add_argument(
        "--ranker",
        choices=["jina-v3.5", "qwen3.5-0.8b"],
        default="qwen3.5-0.8b",
    )
    parser.add_argument("--mode", choices=["listwise", "fusion"], default="listwise")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dense-device", default=None)
    parser.add_argument("--ranker-device", default=None)
    parser.add_argument("--qwen-dtype", default=None)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=192)
    parser.add_argument("--qwen-output-count", type=int, default=15)
    parser.add_argument("--v25-rank-weight", type=float, default=0.2)
    parser.add_argument("--listwise-rank-weight", type=float, default=1.0)
    parser.add_argument("--v5-rrf-k", type=int, default=20)
    parser.add_argument("--list-senses", action="store_true")
    args = parser.parse_args()

    lexical = load_artifacts(args.index)
    if args.list_senses:
        print(json.dumps(list_senses(lexical, args.query), ensure_ascii=False, indent=2))
        return

    dense_device = args.dense_device or args.device
    ranker_device = args.ranker_device or args.device

    _status("[1/3] Loading V2.5 retriever...")
    v25 = HybridSearcher.from_paths(
        lexical,
        args.dense_index,
        device=dense_device,
    )

    _status(f"[2/3] Loading {args.ranker}...")
    started = perf_counter()
    resolved_name, ranker = resolve_v51_ranker(
        args.ranker,
        device=ranker_device,
        qwen_dtype=args.qwen_dtype,
        qwen_max_new_tokens=args.qwen_max_new_tokens,
        qwen_output_count=args.qwen_output_count,
    )
    _status(
        f"[2/3] {resolved_name} ready in {perf_counter() - started:.1f}s"
    )

    _status(
        f"[3/3] Ranking {args.candidate_pool} candidates jointly for "
        f"{args.query!r}..."
    )
    searcher = V5Searcher(v25=v25, ranker=ranker)
    results = searcher.search(
        args.query,
        sense=args.sense,
        category=args.category,
        top_k=args.top_k,
        candidate_pool=args.candidate_pool,
        mode=args.mode,
        v25_rank_weight=args.v25_rank_weight,
        listwise_rank_weight=args.listwise_rank_weight,
        v5_rrf_k=args.v5_rrf_k,
    )
    _status("[3/3] Done.")
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
