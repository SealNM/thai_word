#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_lexical_v1 import list_senses, load_artifacts, search


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("queries"), list):
        raise ValueError("Evaluation config must be an object with a 'queries' array.")
    return data


def _compact_result(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "word": item["word"],
        "score": item["score"],
        "relation_tier": item.get("relation_tier"),
        "relation_hint": item.get("relation_hint"),
        "lexical_form": item.get("lexical_form"),
        "definition": item.get("definition"),
        "matched_candidate_sense": item.get("matched_candidate_sense"),
        "signals": item.get("signals"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small cross-domain manual evaluation suite."
    )
    parser.add_argument(
        "--config",
        default="evaluation/v1_queries.json",
        help="Evaluation query config.",
    )
    parser.add_argument("--index", default="artifacts/v1", help="Artifact directory.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the JSON report.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    artifacts = load_artifacts(args.index)

    report: dict[str, Any] = {
        "config": args.config,
        "index": args.index,
        "top_k": args.top_k,
        "queries": [],
    }

    for spec in config["queries"]:
        query = str(spec["query"])
        requested_sense = spec.get("sense")
        available_senses = list_senses(artifacts, query)

        if requested_sense is not None:
            requested_sense = int(requested_sense)

        try:
            results = search(
                artifacts,
                query,
                top_k=args.top_k,
                sense=requested_sense,
            )
            error = None
        except ValueError as exc:
            results = []
            error = str(exc)

        selected_sense = None
        if results:
            selected_sense = results[0].get("query_sense")
        elif available_senses and requested_sense is not None:
            if 1 <= requested_sense <= len(available_senses):
                selected_sense = available_senses[requested_sense - 1]
        elif available_senses:
            selected_sense = available_senses[0]

        tier_counts: dict[str, int] = {}
        for item in results:
            tier = str(item.get("relation_tier"))
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        report["queries"].append(
            {
                "query": query,
                "category": spec.get("category"),
                "requested_sense": requested_sense,
                "selected_sense": selected_sense,
                "available_senses": available_senses,
                "tier_counts": tier_counts,
                "error": error,
                "results": [_compact_result(item) for item in results],
            }
        )

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
