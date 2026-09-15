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
from thai_pairwise_v27 import (
    PairwiseRanker,
    TNCFamiliarity,
    rerank_v27,
)


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("queries"),
        list,
    ):
        raise ValueError(
            "Evaluation config must contain a 'queries' array."
        )
    return payload


def _words(
    items: list[dict[str, Any]],
    count: int,
) -> str:
    return " | ".join(
        str(item["word"])
        for item in items[:count]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen V2.5 baseline with the learned "
            "V2.7 pairwise ranker."
        )
    )
    parser.add_argument(
        "--config",
        default="evaluation/v1_queries.json",
    )
    parser.add_argument(
        "--index",
        default="artifacts/v1",
    )
    parser.add_argument(
        "--dense-index",
        default="artifacts/v2/embeddinggemma-300m-256",
    )
    parser.add_argument(
        "--model",
        default="artifacts/v27/pairwise_ranker.joblib",
    )
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output",
        default="artifacts/v27/evaluation.json",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    lexical = load_artifacts(args.index)
    baseline = HybridSearcher.from_paths(
        lexical,
        args.dense_index,
        device=args.device,
    )
    ranker = PairwiseRanker.load(args.model)
    familiarity = TNCFamiliarity()

    report: dict[str, Any] = {
        "config": args.config,
        "index": args.index,
        "dense_index": args.dense_index,
        "model": args.model,
        "candidate_pool": args.candidate_pool,
        "top_k": args.top_k,
        "model_metadata": ranker.metadata,
        "queries": [],
    }

    started = perf_counter()
    for spec in config["queries"]:
        query = str(spec["query"])
        sense = spec.get("sense")
        if sense is not None:
            sense = int(sense)

        pool_size = max(
            args.top_k,
            args.candidate_pool,
        )
        v25_pool = baseline.search(
            query,
            top_k=pool_size,
            sense=sense,
            lexical_pool=max(300, pool_size),
            dense_pool=max(300, pool_size),
        )
        v27_pool = rerank_v27(
            v25_pool,
            ranker=ranker,
            familiarity=familiarity,
        )

        row = {
            "query": query,
            "category": spec.get("category"),
            "sense": sense,
            "available_senses": list_senses(
                lexical,
                query,
            ),
            "v25": v25_pool[: args.top_k],
            "v27": v27_pool[: args.top_k],
            "v27_diagnostics": [
                {
                    "word": item["word"],
                    "relation_tier": item.get(
                        "relation_tier"
                    ),
                    "protected_relation_tier": (
                        item.get(
                            "protected_relation_tier"
                        )
                    ),
                    "dense_similarity": item.get(
                        "dense_similarity"
                    ),
                    "v27": item.get("v27"),
                }
                for item in v27_pool[:pool_size]
            ],
        }
        report["queries"].append(row)

        print(
            f"\n=== {query} "
            f"[{spec.get('category')}] ==="
        )
        print(
            "V2.5:",
            _words(
                v25_pool,
                args.top_k,
            ),
        )
        print(
            "V2.7:",
            _words(
                v27_pool,
                args.top_k,
            ),
        )
        moved = [
            f"{item['word']}"
            f"({item['v27']['movement']:+d})"
            for item in v27_pool[:pool_size]
            if int(
                item["v27"]["movement"]
            )
            != 0
        ]
        if moved:
            print(
                "moves:",
                " | ".join(moved),
            )

    report["seconds"] = round(
        float(
            perf_counter() - started
        ),
        3,
    )
    target = Path(args.output)
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    target.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
