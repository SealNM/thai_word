#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_hybrid_v2 import HybridSearcher
from thai_lexical_v1 import load_artifacts, normalize_text
from thai_v3_data import load_holdout_words, write_jsonl


def _bound_form(word: str) -> bool:
    word = normalize_text(word)
    return word.startswith("-") or word.endswith("-")


def _seed_id(word: str, sense: int, definition: str) -> str:
    payload = f"{word}\0{sense}\0{definition}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def _candidate_from_result(item: dict[str, Any]) -> dict[str, Any]:
    matched = item.get("matched_candidate_sense") or {}
    definition = normalize_text(
        matched.get("definition") or item.get("definition") or ""
    )
    return {
        "word": normalize_text(item["word"]),
        "sense": matched.get("sense"),
        "definition": definition,
        "evidence": {
            "relation_tier": item.get("relation_tier"),
            "relation_hint": item.get("relation_hint"),
            "lexical_form": item.get("lexical_form"),
            "sense_resolution": item.get("sense_resolution"),
            "lexical_score": item.get("lexical_score"),
            "lexical_rank": item.get("lexical_rank"),
            "dense_similarity": item.get("dense_similarity"),
            "dense_rank": item.get("dense_rank"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build grounded V3 teacher tasks from dictionary senses and the V2.5 "
            "hybrid retriever. This script does not call an LLM."
        )
    )
    parser.add_argument("--index", default="artifacts/v1")
    parser.add_argument(
        "--dense-index",
        default="artifacts/v2/embeddinggemma-300m-256",
        help="V2.5 dense artifact directory used to mine candidates.",
    )
    parser.add_argument(
        "--holdout",
        default="evaluation/v3_holdout_words.json",
        help="Frozen headwords excluded from both anchors and candidates.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/v3/teacher_seeds.jsonl",
    )
    parser.add_argument("--candidate-count", type=int, default=24)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of senses to sample. Use 0 to process all eligible senses.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--include-bound-forms",
        action="store_true",
        help="Include dictionary forms beginning/ending in '-' as anchors/candidates.",
    )
    args = parser.parse_args()

    lexical = load_artifacts(args.index)
    holdout = load_holdout_words(args.holdout)
    searcher = HybridSearcher.from_paths(
        lexical,
        args.dense_index,
        device=args.device,
    )

    eligible: list[int] = []
    for sense_id, sense in enumerate(lexical.senses):
        entry = lexical.entries[int(sense["entry_index"])]
        word = normalize_text(entry["word"])
        if word in holdout:
            continue
        if not args.include_bound_forms and _bound_form(word):
            continue
        eligible.append(sense_id)

    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    if args.sample_size > 0:
        eligible = eligible[: args.sample_size]

    rows: list[dict[str, Any]] = []
    skipped_no_candidates = 0

    for position, sense_id in enumerate(eligible, start=1):
        sense = lexical.senses[sense_id]
        entry_index = int(sense["entry_index"])
        entry = lexical.entries[entry_index]
        word = normalize_text(entry["word"])
        definition = normalize_text(sense["definition"])
        sense_index = int(sense["sense_index"])

        results = searcher.search(
            word,
            sense=sense_index,
            top_k=max(args.candidate_count * 2, args.candidate_count + 10),
            lexical_pool=300,
            dense_pool=300,
        )

        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in results:
            candidate = _candidate_from_result(item)
            candidate_word = candidate["word"]
            if not candidate_word or candidate_word == word:
                continue
            if candidate_word in holdout:
                continue
            if candidate_word in seen:
                continue
            if not args.include_bound_forms and _bound_form(candidate_word):
                continue
            if not candidate["definition"]:
                continue
            seen.add(candidate_word)
            candidates.append(candidate)
            if len(candidates) >= args.candidate_count:
                break

        if not candidates:
            skipped_no_candidates += 1
            continue

        rows.append(
            {
                "schema_version": 1,
                "seed_id": _seed_id(word, sense_index, definition),
                "anchor": {
                    "word": word,
                    "sense": sense_index,
                    "definition": definition,
                },
                "candidates": candidates,
                "mining": {
                    "dense_index": args.dense_index,
                    "candidate_count": len(candidates),
                    "sample_seed": args.seed,
                },
            }
        )

        if position % 100 == 0:
            print(
                f"prepared {position}/{len(eligible)} senses "
                f"({len(rows)} tasks, {skipped_no_candidates} without candidates)",
                file=sys.stderr,
            )

    count = write_jsonl(args.output, rows)
    report = {
        "output": args.output,
        "tasks": count,
        "eligible_senses_sampled": len(eligible),
        "skipped_without_candidates": skipped_no_candidates,
        "holdout_words": len(holdout),
        "candidate_count": args.candidate_count,
        "sample_seed": args.seed,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
