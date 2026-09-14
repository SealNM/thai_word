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

from thai_gemma_v53 import DEFAULT_GEMMA4_E2B_QAT, Gemma4ListwiseJudge
from thai_hybrid_v2 import HybridSearcher
from thai_lexical_v1 import load_artifacts
from thai_listwise_v5 import JinaListwiseRanker, annotate_listwise_scores


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("queries"), list):
        raise ValueError("Evaluation config must contain a 'queries' array.")
    return data


def _words(items: list[dict[str, Any]]) -> str:
    return " | ".join(str(item["word"]) for item in items)


def _fmt_seconds(value: float) -> str:
    value = max(0.0, float(value))
    minutes, seconds = divmod(int(round(value)), 60)
    if minutes:
        return f"{minutes:02d}:{seconds:02d}"
    return f"{seconds}s"


def _status(message: str) -> None:
    print(message, flush=True)


def _cleanup_gpu() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _candidate_payload(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in ranked:
        candidate = dict(item["candidate"])
        candidate["gemma_rank"] = int(item["rank"])
        result.append(candidate)
    return result


def _generation_meta(judge: Gemma4ListwiseJudge) -> dict[str, Any]:
    return {
        "parse_complete": bool(judge.last_parse_complete),
        "parse_strategy": judge.last_parse_strategy,
        "explicit_count": int(judge.last_explicit_count),
        "requested_count": int(judge.last_requested_count),
        "raw_output": judge.last_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Gemma 4 E2B as a Thai lexical listwise judge, both directly "
            "on V2.5 top-50 and after Jina semantic narrowing."
        )
    )
    parser.add_argument("--config", default="evaluation/v1_queries.json")
    parser.add_argument("--index", default="artifacts/v1")
    parser.add_argument("--dense-index", required=True)
    parser.add_argument("--candidate-pool", type=int, default=50)
    parser.add_argument("--jina-pool", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dense-device", default=None)
    parser.add_argument("--gemma-device", default=None)
    parser.add_argument("--jina-device", default=None)
    parser.add_argument("--gemma-model", default=DEFAULT_GEMMA4_E2B_QAT)
    parser.add_argument("--gemma-max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        action="append",
        choices=["direct", "jina-gemma"],
        help="Repeat to select modes. Defaults to direct + jina-gemma.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    modes = args.mode or ["direct", "jina-gemma"]
    config = _load_config(args.config)
    specs = config["queries"]
    total_queries = len(specs)

    dense_device = args.dense_device or args.device
    jina_device = args.jina_device or args.device
    gemma_device = args.gemma_device or args.device

    _status("[setup] Loading lexical artifacts...")
    lexical = load_artifacts(args.index)

    _status("[setup] Loading V2.5 dense retriever...")
    started = perf_counter()
    v25 = HybridSearcher.from_paths(
        lexical,
        args.dense_index,
        device=dense_device,
    )
    _status(f"[setup] V2.5 ready in {_fmt_seconds(perf_counter() - started)}")

    pools: list[dict[str, Any]] = []
    _status(
        f"[setup] Precomputing shared top-{args.candidate_pool} V2.5 pools..."
    )
    pool_started = perf_counter()
    for index, spec in enumerate(specs, start=1):
        query = str(spec["query"])
        sense = spec.get("sense")
        if sense is not None:
            sense = int(sense)
        category = spec.get("category")
        if category is not None:
            category = str(category)

        candidates = v25.search(
            query,
            top_k=max(args.top_k, args.candidate_pool),
            sense=sense,
            lexical_pool=max(300, args.candidate_pool),
            dense_pool=max(300, args.candidate_pool),
        )
        pools.append(
            {
                "query": query,
                "sense": sense,
                "category": category,
                "candidates": candidates,
            }
        )
        _status(
            f"[setup {index}/{total_queries}] {query}: "
            f"{len(candidates)} candidates cached"
        )
    _status(
        f"[setup] V2.5 pools ready in "
        f"{_fmt_seconds(perf_counter() - pool_started)}"
    )

    if "jina-gemma" in modes:
        _status("[jina] Loading semantic narrowing model...")
        jina_started = perf_counter()
        jina = JinaListwiseRanker(device=jina_device)
        _status(
            f"[jina] Ready in {_fmt_seconds(perf_counter() - jina_started)}"
        )

        jina_completed = 0.0
        for index, pool in enumerate(pools, start=1):
            query_started = perf_counter()
            scored = annotate_listwise_scores(
                pool["query"],
                pool["candidates"],
                jina,
                category=pool["category"],
            )
            ranked = sorted(
                scored,
                key=lambda item: (
                    int(item["listwise_rank"]),
                    int(item["v25_rank"]),
                ),
            )
            pool["jina_candidates"] = ranked[: max(args.top_k, args.jina_pool)]
            elapsed = perf_counter() - query_started
            jina_completed += elapsed
            eta = (jina_completed / index) * (total_queries - index)
            _status(
                f"[jina {index}/{total_queries}] {pool['query']}: "
                f"top-{len(pool['jina_candidates'])} ready in "
                f"{_fmt_seconds(elapsed)} · ETA {_fmt_seconds(eta)}"
            )

        del jina
        _cleanup_gpu()
        _status("[jina] Semantic narrowing cached; model released from GPU.")

    _status(
        f"[gemma] Loading {args.gemma_model} "
        "(first run may download ~2.5 GB)..."
    )
    gemma_started = perf_counter()
    judge = Gemma4ListwiseJudge(
        args.gemma_model,
        device=gemma_device,
        max_new_tokens=args.gemma_max_new_tokens,
        seed=args.seed,
    )
    gemma_load_seconds = perf_counter() - gemma_started
    _status(
        f"[gemma] Ready in {_fmt_seconds(gemma_load_seconds)}"
    )

    report: dict[str, Any] = {
        "config": args.config,
        "index": args.index,
        "dense_index": args.dense_index,
        "gemma_model": args.gemma_model,
        "candidate_pool": args.candidate_pool,
        "jina_pool": args.jina_pool,
        "top_k": args.top_k,
        "modes": modes,
        "seed": args.seed,
        "gemma_model_load_seconds": round(float(gemma_load_seconds), 3),
        "queries": [],
    }

    run_started = perf_counter()
    completed = 0.0

    for query_index, pool in enumerate(pools, start=1):
        prefix = f"[gemma {query_index}/{total_queries}]"
        _status(
            f"\n{prefix} {pool['query']} [{pool['category']}]"
        )
        query_started = perf_counter()

        row: dict[str, Any] = {
            "query": pool["query"],
            "category": pool["category"],
            "sense": pool["sense"],
            "v25": [dict(item) for item in pool["candidates"][: args.top_k]],
            "modes": {},
        }

        if "direct" in modes:
            _status(
                f"{prefix} direct — Gemma judging "
                f"{len(pool['candidates'])} candidates..."
            )
            direct_started = perf_counter()
            direct_raw = judge.rank(
                pool["query"],
                pool["candidates"],
                category=pool["category"],
                top_k=args.top_k,
            )
            direct_seconds = perf_counter() - direct_started
            direct = _candidate_payload(direct_raw)
            direct_meta = _generation_meta(judge)
            row["modes"]["direct"] = {
                "seconds": round(float(direct_seconds), 4),
                "results": direct,
                "generation": direct_meta,
            }
            status = "complete" if direct_meta["parse_complete"] else "INCOMPLETE"
            _status(
                f"{prefix} direct {status}: "
                f"{direct_meta['explicit_count']}/{direct_meta['requested_count']} IDs "
                f"in {_fmt_seconds(direct_seconds)}"
            )
            if not direct_meta["parse_complete"]:
                preview = direct_meta["raw_output"].replace("\n", " ")[:300]
                _status(f"{prefix} direct raw: {preview!r}")
            _status("Gemma direct     : " + _words(direct))

        if "jina-gemma" in modes:
            narrowed = pool.get("jina_candidates", [])
            _status(
                f"{prefix} jina→gemma — Gemma judging "
                f"{len(narrowed)} Jina-semantic candidates..."
            )
            cascade_started = perf_counter()
            cascade_raw = judge.rank(
                pool["query"],
                narrowed,
                category=pool["category"],
                top_k=args.top_k,
            )
            cascade_seconds = perf_counter() - cascade_started
            cascade = _candidate_payload(cascade_raw)
            cascade_meta = _generation_meta(judge)
            row["modes"]["jina-gemma"] = {
                "seconds": round(float(cascade_seconds), 4),
                "results": cascade,
                "generation": cascade_meta,
            }
            status = (
                "complete" if cascade_meta["parse_complete"] else "INCOMPLETE"
            )
            _status(
                f"{prefix} jina→gemma {status}: "
                f"{cascade_meta['explicit_count']}/{cascade_meta['requested_count']} IDs "
                f"in {_fmt_seconds(cascade_seconds)}"
            )
            if not cascade_meta["parse_complete"]:
                preview = cascade_meta["raw_output"].replace("\n", " ")[:300]
                _status(f"{prefix} jina→gemma raw: {preview!r}")
            _status("Jina → Gemma    : " + _words(cascade))

        query_seconds = perf_counter() - query_started
        completed += query_seconds
        remaining = total_queries - query_index
        eta = (completed / query_index) * remaining
        row["query_seconds"] = round(float(query_seconds), 4)
        report["queries"].append(row)

        _status(
            f"{prefix} done in {_fmt_seconds(query_seconds)} · "
            f"ETA {_fmt_seconds(eta)}"
        )

    run_seconds = perf_counter() - run_started
    report["query_seconds"] = round(float(run_seconds), 3)
    report["seconds_per_query"] = round(
        float(run_seconds / max(1, total_queries)),
        4,
    )

    _status(
        f"\n[done] {total_queries}/{total_queries} in "
        f"{_fmt_seconds(run_seconds)} "
        f"({report['seconds_per_query']:.2f}s/query including selected modes)"
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _status(f"[done] Saved report: {out}")


if __name__ == "__main__":
    main()
