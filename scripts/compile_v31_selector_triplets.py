#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_lexical_v1 import normalize_text
from thai_v3_data import load_holdout_words, read_jsonl, sense_text, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile direct V3.1 selector output into contrastive triplets."
    )
    parser.add_argument(
        "--input",
        default="artifacts/v3_1/local_triplet_selections.jsonl",
    )
    parser.add_argument(
        "--holdout",
        default="evaluation/v3_holdout_words.json",
    )
    parser.add_argument(
        "--output",
        default="artifacts/v3_1/training_triplets_selector.jsonl",
    )
    parser.add_argument(
        "--report-output",
        default="artifacts/v3_1/selector_report.json",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if not rows:
        raise SystemExit(f"No selector rows found at {args.input}")

    holdout = load_holdout_words(args.holdout)
    triplets: list[dict[str, str]] = []
    stats = Counter()
    seen: set[tuple[str, str, str]] = set()

    for row in rows:
        anchor = row.get("anchor") or {}
        anchor_word = normalize_text(anchor.get("word", ""))
        anchor_definition = normalize_text(anchor.get("definition", ""))
        if not anchor_word or not anchor_definition:
            stats["invalid_anchor"] += 1
            continue
        if anchor_word in holdout:
            raise SystemExit(f"Holdout leakage in anchor: {anchor_word}")

        selection = row.get("selection") or {}
        positives = selection.get("positive") or []
        negatives = selection.get("hard_negative") or []

        stats["rows"] += 1
        stats["positives_selected"] += len(positives)
        stats["hard_negatives_selected"] += len(negatives)

        if not positives:
            stats["rows_without_positive"] += 1
            continue
        if not negatives:
            stats["rows_without_hard_negative"] += 1
            continue

        anchor_text = sense_text(anchor_word, anchor_definition)

        for positive in positives:
            pos_word = normalize_text(positive.get("word", ""))
            pos_def = normalize_text(positive.get("definition", ""))
            if pos_word in holdout:
                raise SystemExit(f"Holdout leakage in positive: {pos_word}")
            if not pos_word or not pos_def or pos_word == anchor_word:
                continue

            positive_text = sense_text(pos_word, pos_def)

            for negative in negatives:
                neg_word = normalize_text(negative.get("word", ""))
                neg_def = normalize_text(negative.get("definition", ""))
                if neg_word in holdout:
                    raise SystemExit(f"Holdout leakage in negative: {neg_word}")
                if not neg_word or not neg_def or neg_word in {anchor_word, pos_word}:
                    continue

                negative_text = sense_text(neg_word, neg_def)
                key = (anchor_text, positive_text, negative_text)
                if key in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(key)
                triplets.append(
                    {
                        "anchor": anchor_text,
                        "positive": positive_text,
                        "negative": negative_text,
                    }
                )

    write_jsonl(args.output, triplets)

    report = {
        "version": "3.1-selector",
        "input_rows": len(rows),
        "triplets": len(triplets),
        "stats": dict(sorted(stats.items())),
        "output": args.output,
    }

    target = Path(args.report_output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
