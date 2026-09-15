#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from thai_substitutability import read_jsonl
from thai_writer_learned import (
    LearnedWriterRanker,
    PRODUCTION_CATEGORY_MODE,
    sha256_file,
)


def run(args: argparse.Namespace) -> dict:
    rows = read_jsonl(args.input)
    train = [row for row in rows if row.get("split") == "train"]
    if not train:
        raise ValueError("No train rows found.")

    ranker = LearnedWriterRanker.fit(
        train,
        seed=args.seed,
        severe_penalty=args.severe_penalty,
        category_mode=PRODUCTION_CATEGORY_MODE,
    )

    metadata = ranker.metadata(
        dataset_sha256=sha256_file(args.input),
        train_pair_count=len(train),
        train_query_count=len({str(row["query_id"]) for row in train}),
    )
    ranker.save(args.output, metadata=metadata)

    report = {
        "status": "phase4_learned_ranker_built",
        "input": str(args.input),
        "output": str(args.output),
        "category_mode": PRODUCTION_CATEGORY_MODE,
        "train_pair_count": metadata["train_pair_count"],
        "train_query_count": metadata["train_query_count"],
        "feature_count": metadata["feature_count"],
        "dataset_sha256": metadata["dataset_sha256"],
        "seed": metadata["seed"],
        "severe_penalty": metadata["severe_penalty"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Phase-4 category-free persisted learned writer-ranker artifact "
            "from the train split only."
        )
    )
    parser.add_argument(
        "--input",
        default="evaluation/writer_relevance_50_annotations.approved.jsonl",
    )
    parser.add_argument(
        "--output",
        default="artifacts/writer-reranker",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--severe-penalty", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.severe_penalty < 0:
        raise ValueError("--severe-penalty must be non-negative.")
    run(args)


if __name__ == "__main__":
    main()
