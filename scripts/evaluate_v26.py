#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_hybrid_v2 import HybridSearcher
from thai_lexical_v1 import list_senses, load_artifacts
from thai_reranker_v26 import TNCFrequencyModel, V26Config, rerank_v26


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
        description="Compare V2.5 with V2.6 bounded TNC rarity reranking."
    )
    parser.add_argument("--config", default="evaluation/v1_queries.json")
    parser.add_argument("--index", default="artifacts/v1")
    parser.add_argument("--dense-index", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-pool", type=int, default=50)
    parser.add_argument("--rerank-window", type=int, default=20)
    parser.add_argument("--max-promotion", type=int, default=4)
    parser.add_argument("--max-demotion", type=int, default=4)
    parser.add_argument(
        "--dense-similarity-tolerance",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--token-proxy-discount",
        type=float,
        default=0.10,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    lexical = load_artifacts(args.index)
    baseline = HybridSearcher.from_paths(
        lexical,
        args.dense_index,
        device=args.device,
    )
    config = V26Config(
        candidate_pool=args.candidate_pool,
        rerank_window=args.rerank_window,
        max_promotion=args.max_promotion,
        max_demotion=args.max_demotion,
        dense_similarity_tolerance=args.dense_similarity_tolerance,
        token_proxy_discount=args.token_proxy_discount,
    )
    frequency_model = TNCFrequencyModel(
        token_proxy_discount=args.token_proxy_discount
    )

    eval_config = _load_config(args.config)
    report: dict[str, Any] = {
        "config": args.config,
        "index": args.index,
        "dense_index": args.dense_index,
        "top_k": args.top_k,
        "v26": {
            "policy": "v2.5 + bounded negative rarity prior",
            "candidate_pool": config.candidate_pool,
            "rerank_window": config.rerank_window,
            "max_promotion": config.max_promotion,
            "max_demotion": config.max_demotion,
            "dense_similarity_tolerance": config.dense_similarity_tolerance,
            "token_proxy_discount": config.token_proxy_discount,
            "frequency_source": "PyThaiNLP Thai National Corpus (TNC)",
        },
        "queries": [],
    }

    started = perf_counter()
    for spec in eval_config["queries"]:
        query = str(spec["query"])
        sense = spec.get("sense")
        if sense is not None:
            sense = int(sense)

        pool = max(args.candidate_pool, args.top_k)
        v25_pool = baseline.search(
            query,
            top_k=pool,
            sense=sense,
        )
        v26_pool = rerank_v26(
            v25_pool,
            frequency_model=frequency_model,
            config=config,
        )
        row = {
            "query": query,
            "category": spec.get("category"),
            "sense": sense,
            "available_senses": list_senses(lexical, query),
            "v25": v25_pool[: args.top_k],
            "v26": v26_pool[: args.top_k],
            "v26_diagnostics": [
                {
                    "word": item["word"],
                    "relation_tier": item.get("relation_tier"),
                    "dense_similarity": item.get("dense_similarity"),
                    "v26": item.get("v26"),
                }
                for item in v26_pool[: args.rerank_window]
            ],
        }
        report["queries"].append(row)

        print(f"\n=== {query} [{spec.get('category')}] ===")
        print(
            "V2.5:",
            " | ".join(_words(row["v25"], args.top_k)),
        )
        print(
            "V2.6:",
            " | ".join(_words(row["v26"], args.top_k)),
        )
        moved = [
            f"{item['word']}({item['v26']['movement']:+d})"
            for item in v26_pool[: args.rerank_window]
            if item.get("v26", {}).get("movement")
        ]
        if moved:
            print("moves:", " | ".join(moved))

    report["seconds"] = round(perf_counter() - started, 3)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
