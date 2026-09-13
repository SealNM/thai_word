#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_lexical_v1 import list_senses, load_artifacts, search


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Thai dictionary semantic V1.")
    parser.add_argument("query", help="Thai headword or phrase to search.")
    parser.add_argument("--index", default="artifacts/v1", help="Artifact directory.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-pool", type=int, default=250)
    parser.add_argument(
        "--sense",
        type=int,
        default=None,
        help="1-based dictionary sense for exact-headword queries. Defaults to sense 1.",
    )
    parser.add_argument(
        "--list-senses",
        action="store_true",
        help="Print available senses for an exact headword and exit.",
    )
    args = parser.parse_args()

    artifacts = load_artifacts(args.index)

    if args.list_senses:
        senses = list_senses(artifacts, args.query)
        print(json.dumps(senses, ensure_ascii=False, indent=2))
        return

    results = search(
        artifacts,
        args.query,
        top_k=args.top_k,
        candidate_pool=args.candidate_pool,
        sense=args.sense,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
