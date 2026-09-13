#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_lexical_v1 import load_artifacts, search


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Thai dictionary semantic V1.")
    parser.add_argument("query", help="Thai headword to search.")
    parser.add_argument("--index", default="artifacts/v1", help="Artifact directory.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-pool", type=int, default=250)
    args = parser.parse_args()

    artifacts = load_artifacts(args.index)
    results = search(
        artifacts,
        args.query,
        top_k=args.top_k,
        candidate_pool=args.candidate_pool,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
