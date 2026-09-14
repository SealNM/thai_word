#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_hybrid_v2 import HybridSearcher
from thai_lexical_v1 import list_senses, load_artifacts
from thai_reranker_v4 import (
    CrossEncoderPairScorer,
    V4Searcher,
    load_tnc_commonness,
    resolve_reranker_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search Thai Words V4.1: V2.5 + strict reranking + commonness."
    )
    parser.add_argument("query", help="Thai headword or phrase.")
    parser.add_argument("--index", default="artifacts/v1")
    parser.add_argument("--dense-index", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--sense", type=int, default=None)
    parser.add_argument("--candidate-pool", type=int, default=50)
    parser.add_argument(
        "--mode",
        choices=["rerank", "fusion", "protected", "commonness"],
        default="commonness",
    )
    parser.add_argument(
        "--reranker",
        default="qwen3-0.6b-v4.1",
        help="Profile name (qwen3-0.6b-v4.1, qwen3-0.6b, bge-v2-m3) or model id.",
    )
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--no-instruction", action="store_true")
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default=None, help="Default device for both models.")
    parser.add_argument("--dense-device", default=None)
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument("--lexical-pool", type=int, default=300)
    parser.add_argument("--dense-pool", type=int, default=300)
    parser.add_argument("--lexical-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--v25-rrf-k", type=int, default=60)
    parser.add_argument("--v25-rank-weight", type=float, default=0.35)
    parser.add_argument("--reranker-rank-weight", type=float, default=1.0)
    parser.add_argument("--commonness-source", choices=["none", "tnc"], default="tnc")
    parser.add_argument("--commonness-rank-weight", type=float, default=0.5)
    parser.add_argument("--v4-rrf-k", type=int, default=20)
    parser.add_argument("--list-senses", action="store_true")
    args = parser.parse_args()

    lexical = load_artifacts(args.index)
    if args.list_senses:
        print(json.dumps(list_senses(lexical, args.query), ensure_ascii=False, indent=2))
        return

    dense_device = args.dense_device or args.device
    reranker_device = args.reranker_device or args.device

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

    commonness = None
    if args.commonness_source == "tnc":
        commonness = load_tnc_commonness()

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

    results = searcher.search(
        args.query,
        top_k=args.top_k,
        sense=args.sense,
        candidate_pool=args.candidate_pool,
        mode=args.mode,
        lexical_pool=args.lexical_pool,
        dense_pool=args.dense_pool,
        lexical_weight=args.lexical_weight,
        dense_weight=args.dense_weight,
        v25_rrf_k=args.v25_rrf_k,
        v25_rank_weight=args.v25_rank_weight,
        reranker_rank_weight=args.reranker_rank_weight,
        commonness_rank_weight=args.commonness_rank_weight,
        v4_rrf_k=args.v4_rrf_k,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
