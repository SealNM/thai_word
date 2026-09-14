#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_v3_data import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize and inspect V3.1 teacher labels."
    )
    parser.add_argument(
        "--input",
        default="artifacts/v3_1/local_teacher_labels.jsonl",
    )
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if not rows:
        raise SystemExit(f"No teacher labels found at {args.input}")

    relations: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    confidences: list[float] = []
    replaceability: Counter[int] = Counter()
    low_confidence: list[dict] = []

    for row in rows:
        anchor = row.get("anchor", {})
        for candidate in row.get("candidates", []):
            judgment = candidate.get("judgment")
            if not isinstance(judgment, dict):
                continue

            relation = str(judgment.get("relation", "unknown"))
            source = str(judgment.get("label_source", "unknown"))
            relations[relation] += 1
            sources[source] += 1

            try:
                confidence = float(judgment.get("confidence"))
                confidences.append(confidence)
            except (TypeError, ValueError):
                confidence = None

            try:
                replaceability[int(judgment.get("replaceability"))] += 1
            except (TypeError, ValueError):
                pass

            if confidence is not None and confidence < 0.75:
                low_confidence.append(
                    {
                        "anchor": anchor.get("word"),
                        "candidate": candidate.get("word"),
                        "relation": relation,
                        "confidence": confidence,
                        "reason": judgment.get("reason"),
                    }
                )

    summary = {
        "rows": len(rows),
        "judgments": sum(relations.values()),
        "relations": dict(relations.most_common()),
        "label_sources": dict(sources.most_common()),
        "replaceability": {
            str(key): value for key, value in sorted(replaceability.items())
        },
        "mean_confidence": (
            round(sum(confidences) / len(confidences), 4)
            if confidences
            else None
        ),
        "low_confidence_under_0_75": len(low_confidence),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n=== EXAMPLES ===")
    shown = 0
    for row in rows:
        anchor = row.get("anchor", {})
        print(
            f"\n[{anchor.get('word')}] sense {anchor.get('sense')}: "
            f"{anchor.get('definition')}"
        )
        for candidate in row.get("candidates", []):
            judgment = candidate.get("judgment")
            if not isinstance(judgment, dict):
                continue
            print(
                "  "
                f"{candidate.get('word')} -> {judgment.get('relation')} "
                f"(replace={judgment.get('replaceability')}, "
                f"conf={judgment.get('confidence')}, "
                f"source={judgment.get('label_source')}) "
                f"| {judgment.get('reason')}"
            )
        shown += 1
        if shown >= args.examples:
            break

    if low_confidence:
        print("\n=== LOW CONFIDENCE ===")
        for item in low_confidence[:20]:
            print(
                f"{item['anchor']} -> {item['candidate']}: "
                f"{item['relation']} conf={item['confidence']} "
                f"| {item['reason']}"
            )


if __name__ == "__main__":
    main()
