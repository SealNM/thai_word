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
from thai_listwise_v5 import DEFAULT_JINA_LISTWISE, JinaListwiseRanker, V5Searcher


def status(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search Thai Words V5 with listwise lexical reranking."
    )
    parser.add_argument("query")
    parser.add_argument("--index", default="artifacts/v1")
    parser.add_argument("--dense-index", required=True)
    parser.add_argument("--sense", type=int, default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-pool", type=int, default=40)
    parser.add_argument("--model", default=DEFAULT_JINA_LISTWISE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dense-device", default=None)
    parser.add_argument("--listwise-device", default=None)
    parser.add_argument("--mode", choices=["listwise", "fusion"], default="listwise")
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
    listwise_device = args.listwise_device or args.device

    status("[1/3] Loading V2.5 retriever...")
    v25 = HybridSearcher.from_paths(
        lexical,
        args.dense_index,
        device=dense_device,
    )

    status(f"[2/3] Loading {args.model}...")
    model_started = perf_counter()
    ranker = JinaListwiseRanker(
        args.model,
        device=listwise_device,
    )
    status(f"[2/3] Model ready in {perf_counter() - model_started:.1f}s")

    status(
        f"[3/3] Reranking {args.candidate_pool} candidates jointly for "
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
    status("[3/3] Done.")
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
