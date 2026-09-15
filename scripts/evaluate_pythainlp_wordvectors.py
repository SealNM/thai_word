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

from thai_hybrid_v2 import HybridSearcher
from thai_lexical_v1 import load_artifacts
from thai_pythainlp_wv import (
    build_index,
    load_pythainlp_model,
    search_headwords,
    search_senses,
)


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
        raise ValueError("Evaluation config must contain a 'queries' array.")
    return payload


def _words(rows: list[dict[str, Any]]) -> str:
    return " | ".join(str(row["word"]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare V2.5 EmbeddingGemma against PyThaiNLP static word-vector "
            "retrieval using both headword cosine and mean sense-definition vectors."
        )
    )
    parser.add_argument("--config", default="evaluation/v1_queries.json")
    parser.add_argument("--index", default="artifacts/v1")
    parser.add_argument(
        "--dense-index",
        default="artifacts/v2/embeddinggemma-300m-256",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        choices=[
            "thai2fit_wv",
            "ltw2v",
            "ltw2v_v1.0_15_window",
            "ltw2v_v1.0_5_window",
        ],
        help="Repeat for multiple models. Default: thai2fit_wv only.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output",
        default="artifacts/pythainlp-wordvectors/evaluation.json",
    )
    args = parser.parse_args()

    models = args.model or ["thai2fit_wv"]
    lexical = load_artifacts(args.index)
    baseline = HybridSearcher.from_paths(
        lexical,
        args.dense_index,
        device=args.device,
    )
    config = _load_config(args.config)

    try:
        from pythainlp.tokenize import word_tokenize
    except ImportError as exc:
        raise SystemExit(
            "PyThaiNLP is required. Install requirements-wordvectors.txt"
        ) from exc

    def tokenizer(text: str):
        return word_tokenize(
            text,
            engine="newmm",
            keep_whitespace=False,
        )

    report: dict[str, Any] = {
        "config": args.config,
        "index": args.index,
        "dense_index": args.dense_index,
        "top_k": args.top_k,
        "models": {},
    }

    baseline_rows: dict[str, list[dict[str, Any]]] = {}
    for spec in config["queries"]:
        query = str(spec["query"])
        sense = spec.get("sense")
        baseline_rows[query] = baseline.search(
            query,
            sense=int(sense) if sense is not None else None,
            top_k=args.top_k,
        )

    for model_name in models:
        print(f"\n[setup] loading PyThaiNLP word vector: {model_name}", flush=True)
        started = perf_counter()
        model = load_pythainlp_model(model_name)
        index = build_index(
            lexical,
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
        )
        build_seconds = perf_counter() - started

        print(
            f"[setup] {model_name}: {index.vector_size}d · "
            f"headword coverage={index.headword_coverage:.1%} · "
            f"sense coverage={index.sense_coverage:.1%} · "
            f"built in {build_seconds:.1f}s",
            flush=True,
        )

        model_report: dict[str, Any] = {
            "vector_size": index.vector_size,
            "headword_coverage": round(index.headword_coverage, 6),
            "sense_coverage": round(index.sense_coverage, 6),
            "build_seconds": round(build_seconds, 3),
            "queries": [],
        }

        for spec in config["queries"]:
            query = str(spec["query"])
            sense = spec.get("sense")
            sense_value = int(sense) if sense is not None else None

            headword_rows, headword_meta = search_headwords(
                lexical,
                index,
                query,
                model=model,
                tokenizer=tokenizer,
                top_k=args.top_k,
            )
            sense_rows, sense_meta = search_senses(
                lexical,
                index,
                query,
                sense=sense_value,
                model=model,
                tokenizer=tokenizer,
                top_k=args.top_k,
            )

            print(f"\n=== {query} [{spec.get('category')}] ===")
            print("V2.5                 :", _words(baseline_rows[query]))
            print(f"{model_name} headword:", _words(headword_rows) or "<OOV>")
            print(f"{model_name} sense   :", _words(sense_rows) or "<OOV>")

            model_report["queries"].append(
                {
                    "query": query,
                    "category": spec.get("category"),
                    "sense": sense_value,
                    "v25": baseline_rows[query],
                    "headword": headword_rows,
                    "headword_meta": headword_meta,
                    "sense_mean": sense_rows,
                    "sense_meta": sense_meta,
                }
            )

        report["models"][model_name] = model_report

        del index
        del model
        gc.collect()

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
