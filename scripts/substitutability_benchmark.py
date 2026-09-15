#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_hybrid_v2 import HybridSearcher
from thai_lexical_v1 import load_artifacts
from thai_substitutability import (
    SCHEMA_VERSION,
    benchmark_metrics,
    read_jsonl,
    stable_pair_id,
    validate_rows,
    write_jsonl,
)


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("queries"), list):
        raise ValueError("Config must contain a 'queries' array.")
    return data


def export_candidates(args: argparse.Namespace) -> None:
    if args.candidates < 1:
        raise ValueError("--candidates must be at least 1.")

    config = _load_config(args.config)
    lexical = load_artifacts(args.index)
    searcher = HybridSearcher.from_paths(
        lexical,
        args.dense_index,
        device=args.device,
    )

    rows: list[dict[str, Any]] = []
    for spec in config["queries"]:
        query = str(spec["query"]).strip()
        sense = spec.get("sense")
        if sense is not None:
            sense = int(sense)

        entry_index = lexical.word_to_index.get(query)
        if entry_index is None:
            raise ValueError(
                f"Query {query!r} is not an exact dictionary headword. "
                "Benchmark targets must resolve to stored senses."
            )
        available_senses = lexical.entry_to_senses[entry_index]
        if sense is None and len(available_senses) != 1:
            raise ValueError(
                f"Query {query!r} has {len(available_senses)} senses; "
                "set an explicit sense in the benchmark config."
            )

        results = searcher.search(
            query,
            sense=sense,
            top_k=args.candidates,
            lexical_pool=max(300, args.candidates),
            dense_pool=max(300, args.candidates),
        )
        if not results:
            raise ValueError(f"No V2.5 candidates returned for query {query!r}.")

        resolved_query_sense = results[0].get("query_sense")
        if not isinstance(resolved_query_sense, dict):
            raise ValueError(
                f"Query {query!r} did not resolve to a dictionary sense. "
                "Benchmark targets must be exact dictionary headwords."
            )

        resolved_sense = int(resolved_query_sense["sense"])
        query_id = f"{query}#{resolved_sense}"

        for rank, item in enumerate(results, start=1):
            matched = item.get("matched_candidate_sense") or {}
            candidate_sense = matched.get("sense")
            if candidate_sense is not None:
                candidate_sense = int(candidate_sense)

            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "pair_id": stable_pair_id(
                        query,
                        resolved_sense,
                        str(item["word"]),
                        candidate_sense,
                    ),
                    "query_id": query_id,
                    "query": {
                        "word": query,
                        "sense": resolved_sense,
                        "definition": resolved_query_sense["definition"],
                        "category": spec.get("category"),
                    },
                    "candidate": {
                        "word": item["word"],
                        "sense": candidate_sense,
                        "definition": matched.get("definition") or item.get("definition"),
                    },
                    "retrieval": {
                        "system": "v2.5",
                        "v25_rank": rank,
                        "score": item.get("score"),
                        "relation_tier": item.get("relation_tier"),
                        "relation_hint": item.get("relation_hint"),
                        "lexical_form": item.get("lexical_form"),
                        "sense_resolution": item.get("sense_resolution"),
                        "lexical_score": item.get("lexical_score"),
                        "lexical_rank": item.get("lexical_rank"),
                        "dense_similarity": item.get("dense_similarity"),
                        "dense_rank": item.get("dense_rank"),
                    },
                    "annotation": {
                        "utility": None,
                        "relation": None,
                        "notes": "",
                    },
                    "split": None,
                }
            )

    write_jsonl(args.output, rows)
    print(
        f"Wrote {len(rows)} annotation pairs from {len(config['queries'])} "
        f"queries to {args.output}"
    )


def validate_annotations(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.path)
    errors = validate_rows(rows, require_labels=not args.allow_unlabeled)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

    mode = "template" if args.allow_unlabeled else "labeled benchmark"
    print(f"OK: {len(rows)} rows validated as {mode}.")


def evaluate_baseline(args: argparse.Namespace) -> None:
    if args.k < 1:
        raise ValueError("--k must be at least 1.")

    rows = read_jsonl(args.path)
    errors = validate_rows(rows, require_labels=True)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

    report = benchmark_metrics(rows, k=args.k)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Thai Words writer lexical-relevance benchmark utilities."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser(
        "export",
        help="Export deterministic V2.5 candidates for human annotation.",
    )
    export.add_argument(
        "--config",
        default="evaluation/substitutability_pilot_queries.json",
    )
    export.add_argument("--index", default="artifacts/v1")
    export.add_argument(
        "--dense-index",
        default="artifacts/v2/embeddinggemma-300m-256",
    )
    export.add_argument("--candidates", type=int, default=30)
    export.add_argument("--device", default=None)
    export.add_argument(
        "--output",
        default="evaluation/substitutability_annotations.jsonl",
    )
    export.set_defaults(func=export_candidates)

    validate = subparsers.add_parser(
        "validate",
        help="Validate an annotation template or labeled JSONL file.",
    )
    validate.add_argument("path")
    validate.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help="Allow null utility/relation labels in a fresh export.",
    )
    validate.set_defaults(func=validate_annotations)

    metrics = subparsers.add_parser(
        "metrics",
        help="Measure frozen V2.5 ranking against human writer-utility labels.",
    )
    metrics.add_argument("path")
    metrics.add_argument("--k", type=int, default=10)
    metrics.add_argument("--output", default=None)
    metrics.set_defaults(func=evaluate_baseline)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
