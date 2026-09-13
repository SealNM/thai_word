#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_hybrid_v2 import HybridSearcher
from thai_lexical_v1 import list_senses, load_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Thai Lexical Semantic V2 hybrid index.")
    parser.add_argument("query", help="Thai headword or phrase.")
    parser.add_argument("--index", default="artifacts/v1", help="V1 lexical artifact directory.")
    parser.add_argument("--dense-index", required=True, help="V2 dense artifact directory.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--sense", type=int, default=None)
    parser.add_argument("--lexical-pool", type=int, default=300)
    parser.add_argument("--dense-pool", type=int, default=300)
    parser.add_argument("--lexical-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--device", default=None)
    parser.add_argument("--list-senses", action="store_true")
    args = parser.parse_args()

    lexical = load_artifacts(args.index)
    if args.list_senses:
        print(json.dumps(list_senses(lexical, args.query), ensure_ascii=False, indent=2))
        return

    searcher = HybridSearcher.from_paths(
        lexical,
        args.dense_index,
        device=args.device,
    )
    results = searcher.search(
        args.query,
        top_k=args.top_k,
        sense=args.sense,
        lexical_pool=args.lexical_pool,
        dense_pool=args.dense_pool,
        lexical_weight=args.lexical_weight,
        dense_weight=args.dense_weight,
        rrf_k=args.rrf_k,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
