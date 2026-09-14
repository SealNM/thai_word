#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_dense_v26 import load_gemini_dense_index
from thai_hybrid_v2 import HybridSearcher
from thai_lexical_v1 import list_senses, load_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search Thai Words V2.6 with Gemini Embedding 2."
    )
    parser.add_argument("query")
    parser.add_argument("--index", default="artifacts/v1")
    parser.add_argument("--dense-index", required=True)
    parser.add_argument("--sense", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--lexical-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--list-senses", action="store_true")
    args = parser.parse_args()

    lexical = load_artifacts(args.index)
    if args.list_senses:
        print(
            json.dumps(
                list_senses(lexical, args.query),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    dense, encoder = load_gemini_dense_index(
        args.dense_index,
        lexical_artifacts=lexical,
    )
    searcher = HybridSearcher(
        lexical=lexical,
        dense=dense,
        encoder=encoder,
    )

    started = perf_counter()
    results = searcher.search(
        args.query,
        top_k=args.top_k,
        sense=args.sense,
        lexical_weight=args.lexical_weight,
        dense_weight=args.dense_weight,
        rrf_k=args.rrf_k,
    )
    print(
        json.dumps(
            {
                "model": dense.metadata.get("model_key"),
                "seconds": round(perf_counter() - started, 4),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
