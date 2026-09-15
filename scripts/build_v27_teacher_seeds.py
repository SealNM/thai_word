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
from thai_pairwise_v27 import (
    FEATURE_NAMES,
    TNCFamiliarity,
    extract_features,
)
from thai_v27_data import load_holdout_words, write_jsonl


def _seed_id(word: str, sense: int, definition: str) -> str:
    payload = f"{word}\0{sense}\0{definition}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def _is_bound_form(word: str) -> bool:
    word = normalize_text(word)
    return word.startswith("-") or word.endswith("-")


def _candidate_payload(
    item: dict[str, Any],
    *,
    v25_rank: int,
    familiarity: TNCFamiliarity,
) -> dict[str, Any]:
    matched = item.get("matched_candidate_sense")
    definition = ""
    if isinstance(matched, dict):
        definition = normalize_text(matched.get("definition", ""))
    if not definition:
        definition = normalize_text(item.get("definition", ""))

    features = extract_features(
        item,
        v25_rank=v25_rank,
        familiarity=familiarity,
    )
    return {
        "word": normalize_text(item.get("word", "")),
        "definition": definition,
        "v25_rank": v25_rank,
        "result": {
            "score": item.get("score"),
            "protected_relation_tier": item.get(
                "protected_relation_tier"
            ),
            "relation_tier": item.get("relation_tier"),
            "relation_hint": item.get("relation_hint"),
            "lexical_form": item.get("lexical_form"),
            "sense_resolution": item.get("sense_resolution"),
            "lexical_score": item.get("lexical_score"),
            "lexical_rank": item.get("lexical_rank"),
            "dense_similarity": item.get("dense_similarity"),
            "dense_rank": item.get("dense_rank"),
            "sense_count": item.get("sense_count"),
        },
        "features": {
            name: float(features[name])
            for name in FEATURE_NAMES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mine V2.7 offline-teacher tasks from V2.5. Frozen holdout "
            "words are excluded as both anchors and candidates."
        )
    )
    parser.add_argument("--index", default="artifacts/v1")
    parser.add_argument(
        "--dense-index",
        default="artifacts/v2/embeddinggemma-300m-256",
    )
    parser.add_argument(
        "--holdout",
        default="evaluation/v27_holdout_words.json",
    )
    parser.add_argument(
        "--output",
        default="artifacts/v27/teacher_seeds.jsonl",
    )
    parser.add_argument("--sample-size", type=int, default=96)
    parser.add_argument("--candidate-count", type=int, default=18)
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    lexical = load_artifacts(args.index)
    holdout = load_holdout_words(args.holdout)
    baseline = HybridSearcher.from_paths(
        lexical,
        args.dense_index,
        device=args.device,
    )
    familiarity = TNCFamiliarity()

    eligible: list[int] = []
    for sense_id, sense in enumerate(lexical.senses):
        entry = lexical.entries[int(sense["entry_index"])]
        word = normalize_text(entry["word"])
        if not word or word in holdout or _is_bound_form(word):
            continue
        eligible.append(sense_id)

    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    if args.sample_size > 0:
        eligible = eligible[: args.sample_size]

    rows: list[dict[str, Any]] = []
    skipped = 0
    requested_pool = max(
        args.candidate_count * 2,
        args.candidate_count + 12,
    )

    for position, sense_id in enumerate(eligible, start=1):
        sense_record = lexical.senses[sense_id]
        entry = lexical.entries[int(sense_record["entry_index"])]
        word = normalize_text(entry["word"])
        sense_index = int(sense_record["sense_index"])
        definition = normalize_text(sense_record["definition"])

        raw = baseline.search(
            word,
            sense=sense_index,
            top_k=requested_pool,
            lexical_pool=300,
            dense_pool=300,
        )

        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for v25_rank, item in enumerate(raw, start=1):
            candidate_word = normalize_text(item.get("word", ""))
            if (
                not candidate_word
                or candidate_word == word
                or candidate_word in holdout
                or candidate_word in seen
            ):
                continue
            payload = _candidate_payload(
                item,
                v25_rank=v25_rank,
                familiarity=familiarity,
            )
            if not payload["definition"]:
                continue
            seen.add(candidate_word)
            candidates.append(payload)
            if len(candidates) >= args.candidate_count:
                break

        if len(candidates) < max(
            8,
            min(args.candidate_count, 12),
        ):
            skipped += 1
            continue

        rows.append(
            {
                "schema_version": 1,
                "seed_id": _seed_id(
                    word,
                    sense_index,
                    definition,
                ),
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
                    "holdout_file": args.holdout,
                },
            }
        )

        if position % 20 == 0 or position == len(eligible):
            print(
                f"[{position}/{len(eligible)}] "
                f"tasks={len(rows)} skipped={skipped}",
                file=sys.stderr,
                flush=True,
            )

    written = write_jsonl(args.output, rows)
    print(
        json.dumps(
            {
                "output": args.output,
                "tasks": written,
                "eligible_sampled": len(eligible),
                "skipped": skipped,
                "candidate_count": args.candidate_count,
                "holdout_words": len(holdout),
                "feature_count": len(FEATURE_NAMES),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
