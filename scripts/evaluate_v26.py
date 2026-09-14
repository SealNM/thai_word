#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_dense_v26 import load_gemini_dense_index
from thai_hybrid_v2 import HybridSearcher
from thai_lexical_v1 import list_senses, load_artifacts


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("queries"), list):
        raise ValueError("Evaluation config must contain a 'queries' array.")
    return data


def _words(results: list[dict[str, Any]], top_k: int) -> str:
    return " | ".join(str(item["word"]) for item in results[:top_k])


def _fmt(value: float) -> str:
    minutes, seconds = divmod(int(round(max(0.0, value))), 60)
    return f"{minutes:02d}:{seconds:02d}" if minutes else f"{seconds}s"


def _cleanup() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare V2.5 EmbeddingGemma with V2.6 Gemini Embedding 2 "
            "using the same lexical fusion."
        )
    )
    parser.add_argument("--config", default="evaluation/v1_queries.json")
    parser.add_argument("--index", default="artifacts/v1")
    parser.add_argument(
        "--baseline-index",
        default="artifacts/v2/embeddinggemma-300m-256",
    )
    parser.add_argument(
        "--gemini-index",
        action="append",
        required=True,
        help="Repeat for 256d / 768d V2.6 indexes.",
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

    queries: list[dict[str, Any]] = []
    for spec in config["queries"]:
        sense = spec.get("sense")
        if sense is not None:
            sense = int(sense)
        queries.append(
            {
                "query": str(spec["query"]),
                "sense": sense,
                "category": spec.get("category"),
                "available_senses": list_senses(lexical, str(spec["query"])),
                "results": {},
                "timings": {},
            }
        )

    print("[baseline] Loading V2.5 EmbeddingGemma...", flush=True)
    baseline_started = perf_counter()
    baseline = HybridSearcher.from_paths(
        lexical,
        args.baseline_index,
        device=args.device,
    )
    print(
        f"[baseline] Ready in {_fmt(perf_counter() - baseline_started)}",
        flush=True,
    )

    for index, row in enumerate(queries, start=1):
        started = perf_counter()
        result = baseline.search(
            row["query"],
            top_k=args.top_k,
            sense=row["sense"],
            lexical_weight=args.lexical_weight,
            dense_weight=args.dense_weight,
            rrf_k=args.rrf_k,
        )
        row["results"]["v2.5"] = result
        row["timings"]["v2.5"] = round(perf_counter() - started, 4)
        print(
            f"[baseline {index}/{len(queries)}] {row['query']}: "
            + _words(result, args.top_k),
            flush=True,
        )

    del baseline
    _cleanup()

    model_rows: list[dict[str, Any]] = []
    for path in args.gemini_index:
        dense, encoder = load_gemini_dense_index(
            path,
            lexical_artifacts=lexical,
        )
        label = str(dense.metadata.get("model_key", Path(path).name))
        searcher = HybridSearcher(
            lexical=lexical,
            dense=dense,
            encoder=encoder,
        )

        print(
            f"\n[v2.6] Evaluating {label} "
            f"({dense.metadata.get('dimensions')}d)...",
            flush=True,
        )
        run_started = perf_counter()
        query_seconds = 0.0

        for index, row in enumerate(queries, start=1):
            started = perf_counter()
            result = searcher.search(
                row["query"],
                top_k=args.top_k,
                sense=row["sense"],
                lexical_weight=args.lexical_weight,
                dense_weight=args.dense_weight,
                rrf_k=args.rrf_k,
            )
            elapsed = perf_counter() - started
            query_seconds += elapsed
            row["results"][label] = result
            row["timings"][label] = round(elapsed, 4)
            print(
                f"[{label} {index}/{len(queries)}] {row['query']} "
                f"({_fmt(elapsed)}): "
                + _words(result, args.top_k),
                flush=True,
            )

        model_rows.append(
            {
                "dense_index": path,
                "model_key": label,
                "model_id": dense.metadata.get("model_id"),
                "dimensions": dense.metadata.get("dimensions"),
                "query_seconds": round(query_seconds, 4),
                "seconds_per_query": round(
                    query_seconds / max(1, len(queries)),
                    4,
                ),
                "wall_seconds": round(perf_counter() - run_started, 4),
            }
        )
        del searcher
        del encoder
        _cleanup()

    print("\n===== V2.5 vs V2.6 =====", flush=True)
    for row in queries:
        print(
            f"\n=== {row['query']} [{row.get('category')}] ===",
            flush=True,
        )
        print(
            "V2.5 EmbeddingGemma: "
            + _words(row["results"]["v2.5"], args.top_k),
            flush=True,
        )
        for model in model_rows:
            label = model["model_key"]
            print(
                f"V2.6 {label}: "
                + _words(row["results"][label], args.top_k),
                flush=True,
            )

    if args.output:
        report = {
            "config": args.config,
            "index": args.index,
            "baseline_index": args.baseline_index,
            "gemini_indexes": args.gemini_index,
            "fusion": {
                "lexical_weight": args.lexical_weight,
                "dense_weight": args.dense_weight,
                "rrf_k": args.rrf_k,
            },
            "models": model_rows,
            "queries": queries,
        }
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\n[done] Saved report: {out}", flush=True)


if __name__ == "__main__":
    main()
