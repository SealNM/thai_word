#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_v3_data import read_jsonl, write_jsonl
from thai_v31_router import build_audit_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select low-confidence and sampled V3.1 local labels for Gemini audit."
    )
    parser.add_argument(
        "--input",
        default="artifacts/v3_1/local_teacher_labels.jsonl",
    )
    parser.add_argument(
        "--output",
        default="artifacts/v3_1/gemini_audit_seeds.jsonl",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--random-audit-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if not rows:
        raise SystemExit(f"No local teacher labels found at {args.input}")

    audits = []
    for row in rows:
        audit = build_audit_seed(
            row,
            confidence_threshold=args.confidence_threshold,
            random_audit_fraction=args.random_audit_fraction,
            random_seed=args.seed,
        )
        if audit is not None:
            audits.append(audit)

    write_jsonl(args.output, audits)
    print(
        json.dumps(
            {
                "version": "3.1",
                "rows_input": len(rows),
                "audit_tasks": len(audits),
                "audit_candidates": sum(
                    len(row.get("candidates", [])) for row in audits
                ),
                "confidence_threshold": args.confidence_threshold,
                "random_audit_fraction": args.random_audit_fraction,
                "output": args.output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
