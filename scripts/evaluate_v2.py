#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from time import perf_counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_hybrid_v2 import HybridSearcher
from thai_lexical_v1 import list_senses, load_artifacts, search as lexical_search


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("queries"), list):
        raise ValueError("Evaluation config must contain a 'queries' array.")
    return data


def _words(results: list[dict[str, Any]], count: int) -> list[str]:
    return [item["word"] for item in results[:count]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare V1 lexical ranking with one or more V2 dense models."
    )
    parser.add_argument("--config", default="evaluation/v1_queries.json")
    parser.add_argument("--index", default="artifacts/v1")
    parser.add_argument(
        "--dense-index",
        action="append",
        required=True,
        help="Dense artifact directory. Repeat to compare multiple models.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--lexical-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    lexical = load_artifacts(args.index)

    report: dict[str, Any] = {
        "config": args.config,
        "index": args.index,
        "top_k": args.top_k,
        "fusion": {
            "method": "weighted_rrf",
            "lexical_weight": args.lexical_weight,
            "dense_weight": args.dense_weight,
            "rrf_k": args.rrf_k,
        },
        "models": [],
        "queries": [],
    }

    for spec in config["queries"]:
        query = str(spec["query"])
        sense = spec.get("sense")
        if sense is not None:
            sense = int(sense)

        available = list_senses(lexical, query)
        baseline = lexical_search(
            lexical,
            query,
            top_k=args.top_k,
            sense=sense,
        )
        report["queries"].append(
            {
                "query": query,
                "category": spec.get("category"),
                "sense": sense,
                "available_senses": available,
                "v1": baseline,
                "v2": [],
            }
        )

    # Load one dense model at a time to keep peak Colab memory low.
    for path in args.dense_index:
        load_started = perf_counter()
        searcher = HybridSearcher.from_paths(lexical, path, device=args.device)
        load_seconds = perf_counter() - load_started
        model_id = searcher.dense.metadata.get("model_id")
        model_report = {
            "dense_index": path,
            "model_id": model_id,
            "dimensions": searcher.dense.metadata.get("dimensions"),
            "model_load_seconds": round(float(load_seconds), 3),
        }
        report["models"].append(model_report)

        query_started = perf_counter()
        for item in report["queries"]:
            results = searcher.search(
                item["query"],
                top_k=args.top_k,
                sense=item["sense"],
                lexical_weight=args.lexical_weight,
                dense_weight=args.dense_weight,
                rrf_k=args.rrf_k,
            )
            item["v2"].append(
                {
                    "dense_index": path,
                    "model_id": model_id,
                    "results": results,
                }
            )

        query_seconds = perf_counter() - query_started
        model_report["query_seconds"] = round(float(query_seconds), 3)
        model_report["query_count"] = len(report["queries"])
        model_report["seconds_per_query"] = round(
            float(query_seconds / max(1, len(report["queries"]))),
            4,
        )

        del searcher
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    for item in report["queries"]:
        print(f"\n=== {item['query']} [{item.get('category')}] ===")
        senses = item.get("available_senses") or []
        if len(senses) > 1 and item.get("sense") is None:
            print(f"WARNING: {len(senses)} raw senses; config has no explicit sense.")
        print("V1      :", " | ".join(_words(item["v1"], args.top_k)))
        for model in item["v2"]:
            label = model["model_id"]
            print(f"V2 {label}: " + " | ".join(_words(model["results"], args.top_k)))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
