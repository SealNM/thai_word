#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_lexical_v1 import list_senses, load_artifacts
from thai_writer_search import DEFAULT_RERANK_POOL, RERANKER_MODES, WriterSearch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search Thai Words with the Phase-4 writer reranker."
    )
    parser.add_argument("query", help="Thai headword or phrase.")
    parser.add_argument("--index", default="artifacts/v1")
    parser.add_argument("--dense-index", required=True)
    parser.add_argument("--learned-ranker", default="artifacts/writer-reranker")
    parser.add_argument("--neural-model", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rerank-pool", type=int, default=DEFAULT_RERANK_POOL)
    parser.add_argument("--sense", type=int, default=None)
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
        "--reranker-mode",
        choices=sorted(RERANKER_MODES),
        default="optional",
    )
    parser.add_argument("--list-senses", action="store_true")
    parser.add_argument("--include-runtime", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.list_senses:
        lexical = load_artifacts(args.index)
        print(json.dumps(list_senses(lexical, args.query), ensure_ascii=False, indent=2))
        return

    searcher = WriterSearch.from_paths(
        lexical_index=args.index,
        dense_index=args.dense_index,
        learned_ranker=args.learned_ranker,
        neural_model=args.neural_model,
        dense_device=args.dense_device,
        neural_device=args.neural_device,
        neural_batch_size=args.neural_batch_size,
        neural_max_length=args.neural_max_length,
        mode=args.reranker_mode,
    )
    results = searcher.search(
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

    payload = {
        "query": args.query,
        "sense": args.sense,
        "reranker_mode": args.reranker_mode,
        "reranker_status": searcher.last_reranker_status,
        "reranker_error": searcher.last_reranker_error,
        "results": results,
    }
    if args.include_runtime:
        payload["runtime"] = searcher.runtime_info()

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
