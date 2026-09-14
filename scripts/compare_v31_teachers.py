#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_v3_data import read_jsonl
from thai_v31_router import LOCAL_LABEL_SOURCE


POSITIVE = {"synonym", "near_synonym"}
NEGATIVE = {"antonym", "associated", "unrelated"}
GRADED = {"subtype", "supertype", "manner"}


def _coarse(relation: str) -> str:
    if relation in POSITIVE:
        return "positive"
    if relation in NEGATIVE:
        return "negative"
    if relation in GRADED:
        return "graded"
    return relation


def _map(rows: list[dict]) -> dict[tuple[str, str], dict]:
    result: dict[tuple[str, str], dict] = {}
    for row in rows:
        seed_id = str(row.get("seed_id", ""))
        for candidate in row.get("candidates", []):
            judgment = candidate.get("judgment")
            if not isinstance(judgment, dict):
                continue
            result[(seed_id, str(candidate.get("word", "")))] = judgment
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare V3.1 local-Qwen relation labels against Gemini labels."
    )
    parser.add_argument(
        "--local",
        default="artifacts/v3_1/local_teacher_labels.jsonl",
    )
    parser.add_argument(
        "--gemini",
        default="artifacts/v3_1/gemini_benchmark_labels.jsonl",
    )
    parser.add_argument(
        "--all-sources",
        action="store_true",
        help="Also compare rule-auto labels. Default compares Qwen-local labels only.",
    )
    args = parser.parse_args()

    local_rows = read_jsonl(args.local)
    gemini_rows = read_jsonl(args.gemini)
    if not local_rows or not gemini_rows:
        raise SystemExit("Both local and Gemini label files are required.")

    local = _map(local_rows)
    gemini = _map(gemini_rows)
    common = sorted(set(local) & set(gemini))

    relation_matrix: Counter[str] = Counter()
    compared = exact = coarse = 0
    confidence_delta = 0.0

    for key in common:
        left = local[key]
        if (
            not args.all_sources
            and left.get("label_source") != LOCAL_LABEL_SOURCE
        ):
            continue
        right = gemini[key]

        left_relation = str(left.get("relation"))
        right_relation = str(right.get("relation"))
        compared += 1
        exact += int(left_relation == right_relation)
        coarse += int(_coarse(left_relation) == _coarse(right_relation))
        relation_matrix[f"{left_relation} -> {right_relation}"] += 1
        try:
            confidence_delta += abs(
                float(left.get("confidence")) - float(right.get("confidence"))
            )
        except (TypeError, ValueError):
            pass

    if not compared:
        raise SystemExit("No comparable candidate judgments found.")

    report = {
        "version": "3.1",
        "compared_candidates": compared,
        "relation_exact_agreement": round(exact / compared, 4),
        "coarse_agreement": round(coarse / compared, 4),
        "mean_abs_confidence_delta": round(confidence_delta / compared, 4),
        "local_source_filter": None if args.all_sources else LOCAL_LABEL_SOURCE,
        "relation_confusions": dict(relation_matrix.most_common()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
