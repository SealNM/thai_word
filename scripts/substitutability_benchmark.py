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
    RELATIONS,
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



ANNOTATION_RELATIONS = (
    "direct",
    "near_register",
    "subtype",
    "manner_action",
    "scene_context",
    "effect_state",
    "literary_imagery",
    "weak_related",
    "opposite_misleading",
    "sense_mismatch",
    "unrelated",
    "unclear",
)

if set(ANNOTATION_RELATIONS) != set(RELATIONS):
    raise RuntimeError("Annotation relation menu is out of sync with schema relations.")


def _is_labeled(row: dict[str, Any]) -> bool:
    annotation = row.get("annotation")
    return (
        isinstance(annotation, dict)
        and annotation.get("utility") in {0, 1, 2, 3}
        and annotation.get("relation") in RELATIONS
    )


def _matches_query(row: dict[str, Any], query_filter: str | None) -> bool:
    if not query_filter:
        return True
    query_filter = query_filter.strip()
    query = row.get("query") or {}
    return query_filter in {
        str(row.get("query_id", "")),
        str(query.get("word", "")),
    }


def _show_annotation_row(
    row: dict[str, Any],
    *,
    position: int,
    total: int,
    labeled_count: int,
) -> None:
    query = row["query"]
    candidate = row["candidate"]
    retrieval = row["retrieval"]
    print()
    print("=" * 72)
    print(
        f"[{position}/{total}] labeled={labeled_count} | "
        f"{row['query_id']} | V2.5 #{retrieval['v25_rank']}"
    )
    print(f"QUERY     : {query['word']} — {query.get('definition') or '-'}")
    print(
        f"CANDIDATE : {candidate['word']} — "
        f"{candidate.get('definition') or '-'}"
    )
    hint = retrieval.get("relation_hint")
    if hint:
        print(f"V2.5 hint : {hint}")
    print()


def _prompt_utility() -> int | str:
    while True:
        raw = input(
            "Writer utility [3=สูงมาก, 2=ชัดเจน, 1=พอมีประโยชน์, "
            "0=ไม่ช่วย] (s=ข้าม, q=ออก): "
        ).strip().lower()
        if raw in {"s", "q"}:
            return raw
        if raw in {"0", "1", "2", "3"}:
            return int(raw)
        print("กรุณาเลือก 0, 1, 2, 3, s หรือ q")


def _prompt_relation(utility: int) -> str | None:
    severe = {"opposite_misleading", "sense_mismatch", "unrelated"}
    print("Relation:")
    for index, relation in enumerate(ANNOTATION_RELATIONS, start=1):
        print(f"  {index:>2}. {relation}")

    while True:
        raw = input("เลือก relation (เลข, b=ย้อนกลับ utility): ").strip().lower()
        if raw == "b":
            return None
        if not raw.isdigit():
            print("กรุณาเลือกหมายเลข relation หรือ b")
            continue
        index = int(raw)
        if not 1 <= index <= len(ANNOTATION_RELATIONS):
            print("หมายเลข relation อยู่นอกช่วง")
            continue
        relation = ANNOTATION_RELATIONS[index - 1]
        if utility > 0 and relation in severe:
            print(
                f"{relation} เป็น severe error และ schema กำหนดให้ utility ต้องเป็น 0"
            )
            continue
        return relation


def annotate_interactively(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.path)
    errors = validate_rows(rows, require_labels=False)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

    eligible = [
        index
        for index, row in enumerate(rows)
        if _matches_query(row, args.query)
        and (args.review or not _is_labeled(row))
    ]

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1.")
        eligible = eligible[: args.limit]

    if not eligible:
        print("No matching unlabeled rows. Nothing to annotate.")
        return

    total_matching = sum(
        1 for row in rows if _matches_query(row, args.query)
    )
    labeled_matching = sum(
        1
        for row in rows
        if _matches_query(row, args.query) and _is_labeled(row)
    )

    print(
        f"Writer Relevance annotation: {len(eligible)} row(s) queued; "
        f"{labeled_matching}/{total_matching} matching rows already labeled."
    )
    print("Autosave: every completed label is written immediately.")

    completed_this_run = 0
    cursor = 0
    while cursor < len(eligible):
        row_index = eligible[cursor]
        row = rows[row_index]
        _show_annotation_row(
            row,
            position=cursor + 1,
            total=len(eligible),
            labeled_count=labeled_matching + completed_this_run,
        )

        utility_or_command = _prompt_utility()
        if utility_or_command == "q":
            break
        if utility_or_command == "s":
            cursor += 1
            continue

        utility = int(utility_or_command)
        relation = _prompt_relation(utility)
        if relation is None:
            continue

        previous_labeled = _is_labeled(row)
        row["annotation"]["utility"] = utility
        row["annotation"]["relation"] = relation

        row_errors = validate_rows([row], require_labels=True)
        if row_errors:
            for error in row_errors:
                print(f"ERROR: {error}", file=sys.stderr)
            row["annotation"]["utility"] = None
            row["annotation"]["relation"] = None
            continue

        write_jsonl(args.path, rows)
        if not previous_labeled:
            completed_this_run += 1
        print(
            f"Saved: {row['query']['word']} -> {row['candidate']['word']} "
            f"| utility={utility} | relation={relation}"
        )
        cursor += 1

    remaining = sum(
        1
        for row in rows
        if _matches_query(row, args.query) and not _is_labeled(row)
    )
    print(
        f"Session complete: labeled {completed_this_run} new row(s); "
        f"{remaining} matching row(s) remain unlabeled."
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


    annotate = subparsers.add_parser(
        "annotate",
        help="Interactively label writer utility and relation with autosave/resume.",
    )
    annotate.add_argument(
        "path",
        nargs="?",
        default="evaluation/substitutability_annotations.jsonl",
    )
    annotate.add_argument(
        "--query",
        default=None,
        help="Only annotate one query headword or exact query_id, e.g. ฝน or ฝน#1.",
    )
    annotate.add_argument(
        "--review",
        action="store_true",
        help="Include already-labeled rows so their labels can be reviewed/replaced.",
    )
    annotate.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Annotate at most this many queued rows in the session.",
    )
    annotate.set_defaults(func=annotate_interactively)

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
