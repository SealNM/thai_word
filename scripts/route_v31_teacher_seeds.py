#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_v3_data import read_jsonl, write_jsonl
from thai_v31_router import route_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Route V3 teacher seeds through high-confidence rules and local-Qwen selection."
    )
    parser.add_argument(
        "--input",
        default="artifacts/v3/teacher_seeds.jsonl",
    )
    parser.add_argument(
        "--output",
        default="artifacts/v3_1/routed_teacher_seeds.jsonl",
    )
    parser.add_argument("--max-local-candidates", type=int, default=8)
    args = parser.parse_args()

    seeds = read_jsonl(args.input)
    if not seeds:
        raise SystemExit(f"No teacher seeds found at {args.input}")

    routed = [
        route_seed(seed, max_local_candidates=args.max_local_candidates)
        for seed in seeds
    ]
    write_jsonl(args.output, routed)

    summary = {
        "version": "3.1",
        "input_tasks": len(seeds),
        "output_tasks": len(routed),
        "source_candidates": sum(
            row.get("routing", {}).get("source_candidate_count", 0)
            for row in routed
        ),
        "auto_labeled": sum(
            row.get("routing", {}).get("auto_labeled", 0)
            for row in routed
        ),
        "local_candidates": sum(
            row.get("routing", {}).get("local_candidates", 0)
            for row in routed
        ),
        "dropped_candidates": sum(
            row.get("routing", {}).get("dropped_candidates", 0)
            for row in routed
        ),
        "max_local_candidates": args.max_local_candidates,
        "output": args.output,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
