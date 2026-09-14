#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_lexical_v1 import load_artifacts
from thai_v3_data import (
    compile_training_data,
    load_holdout_words,
    read_jsonl,
    validate_teacher_record,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate V3 teacher labels and compile training triplets."
    )
    parser.add_argument(
        "--input",
        default="artifacts/v3/teacher_labels.jsonl",
    )
    parser.add_argument("--index", default="artifacts/v1")
    parser.add_argument(
        "--holdout",
        default="evaluation/v3_holdout_words.json",
    )
    parser.add_argument(
        "--triplets-output",
        default="artifacts/v3/training_triplets.jsonl",
    )
    parser.add_argument(
        "--graded-output",
        default="artifacts/v3/graded_pairs.jsonl",
    )
    parser.add_argument(
        "--report-output",
        default="artifacts/v3/dataset_report.json",
    )
    parser.add_argument("--min-positive-confidence", type=float, default=0.80)
    parser.add_argument("--min-negative-confidence", type=float, default=0.80)
    parser.add_argument("--min-replaceability", type=int, default=2)
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if not rows:
        raise SystemExit(f"No teacher labels found at {args.input}")

    lexical = load_artifacts(args.index)
    dictionary_words = set(lexical.word_to_index)
    holdout = load_holdout_words(args.holdout)

    invalid: list[dict[str, object]] = []
    for row_number, row in enumerate(rows, start=1):
        errors = validate_teacher_record(
            row,
            dictionary_words=dictionary_words,
            holdout_words=holdout,
        )
        if errors:
            invalid.append(
                {
                    "row": row_number,
                    "seed_id": row.get("seed_id"),
                    "errors": errors,
                }
            )

    if invalid:
        print(
            json.dumps(
                {
                    "valid": False,
                    "invalid_records": len(invalid),
                    "examples": invalid[:10],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(
            "Teacher dataset validation failed. Fix invalid rows before training."
        )

    triplets, graded_pairs, report = compile_training_data(
        rows,
        holdout_words=holdout,
        dictionary_words=dictionary_words,
        min_positive_confidence=args.min_positive_confidence,
        min_negative_confidence=args.min_negative_confidence,
        min_replaceability=args.min_replaceability,
        negatives_per_positive=args.negatives_per_positive,
    )

    if not triplets:
        raise SystemExit(
            "Validation passed but no training triplets met the thresholds. "
            "Inspect relation distribution or lower thresholds deliberately."
        )

    write_jsonl(args.triplets_output, triplets)
    write_jsonl(args.graded_output, graded_pairs)

    report.update(
        {
            "valid": True,
            "teacher_records": len(rows),
            "holdout_words": len(holdout),
            "triplets_output": args.triplets_output,
            "graded_output": args.graded_output,
        }
    )

    target = Path(args.report_output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
