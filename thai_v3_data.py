from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from thai_lexical_v1 import normalize_text


RELATIONS = {
    "synonym",
    "near_synonym",
    "antonym",
    "subtype",
    "supertype",
    "manner",
    "associated",
    "unrelated",
    "uncertain",
}

REGISTERS = {
    "neutral",
    "literary",
    "formal",
    "colloquial",
    "archaic",
    "technical",
    "slang",
    "unknown",
}

POSITIVE_RELATIONS = {"synonym", "near_synonym"}
HARD_NEGATIVE_RELATIONS = {"antonym", "associated", "unrelated"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = Path(path)
    if not source.exists():
        return rows
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {source}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"JSONL row at {source}:{line_number} must be an object."
                )
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_holdout_words(path: str | Path) -> set[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        words = data.get("words")
    else:
        words = data

    if not isinstance(words, list):
        raise ValueError("Holdout file must be an array or contain a 'words' array.")

    normalized = {normalize_text(word) for word in words if normalize_text(word)}
    if not normalized:
        raise ValueError("Holdout word set is empty.")
    return normalized


def sense_text(word: str, definition: str) -> str:
    return normalize_text(f"{word}: {definition}")


def _number(value: Any, *, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric.") from exc


def validate_teacher_record(
    row: dict[str, Any],
    *,
    dictionary_words: set[str] | None = None,
    holdout_words: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    anchor = row.get("anchor")
    if not isinstance(anchor, dict):
        return ["anchor must be an object"]

    anchor_word = normalize_text(anchor.get("word", ""))
    if not anchor_word:
        errors.append("anchor.word is required")

    if holdout_words and anchor_word in holdout_words:
        errors.append(f"holdout leakage in anchor: {anchor_word}")

    candidates = row.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        errors.append("candidates must be a non-empty array")
        return errors

    seen_words: set[str] = set()
    for index, candidate in enumerate(candidates):
        prefix = f"candidates[{index}]"
        if not isinstance(candidate, dict):
            errors.append(f"{prefix} must be an object")
            continue

        word = normalize_text(candidate.get("word", ""))
        if not word:
            errors.append(f"{prefix}.word is required")
            continue
        if word == anchor_word:
            errors.append(f"{prefix} duplicates anchor word")
        if word in seen_words:
            errors.append(f"{prefix} duplicates candidate {word}")
        seen_words.add(word)

        if dictionary_words is not None and word not in dictionary_words:
            errors.append(f"{prefix}.word not found in dictionary: {word}")
        if holdout_words and word in holdout_words:
            errors.append(f"holdout leakage in candidate: {word}")

        judgment = candidate.get("judgment")
        if not isinstance(judgment, dict):
            errors.append(f"{prefix}.judgment is required")
            continue

        relation = judgment.get("relation")
        if relation not in RELATIONS:
            errors.append(f"{prefix}.judgment.relation invalid: {relation!r}")

        try:
            relatedness = int(judgment.get("semantic_relatedness"))
        except (TypeError, ValueError):
            errors.append(f"{prefix}.judgment.semantic_relatedness must be 0-4")
        else:
            if relatedness < 0 or relatedness > 4:
                errors.append(f"{prefix}.judgment.semantic_relatedness must be 0-4")

        try:
            replaceability = int(judgment.get("replaceability"))
        except (TypeError, ValueError):
            errors.append(f"{prefix}.judgment.replaceability must be 0-3")
        else:
            if replaceability < 0 or replaceability > 3:
                errors.append(f"{prefix}.judgment.replaceability must be 0-3")

        try:
            confidence = _number(
                judgment.get("confidence"),
                field=f"{prefix}.judgment.confidence",
            )
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if confidence < 0.0 or confidence > 1.0:
                errors.append(f"{prefix}.judgment.confidence must be 0-1")

        register = judgment.get("register", "unknown")
        if register not in REGISTERS:
            errors.append(f"{prefix}.judgment.register invalid: {register!r}")

    return errors


def _negative_hardness(candidate: dict[str, Any]) -> tuple[int, int, float]:
    evidence = candidate.get("evidence") or {}
    dense_rank = evidence.get("dense_rank")
    lexical_rank = evidence.get("lexical_rank")
    dense_similarity = evidence.get("dense_similarity")

    dense_rank_sort = int(dense_rank) if dense_rank is not None else 10**9
    lexical_rank_sort = int(lexical_rank) if lexical_rank is not None else 10**9
    dense_similarity_sort = (
        -float(dense_similarity) if dense_similarity is not None else 1.0
    )
    return dense_rank_sort, lexical_rank_sort, dense_similarity_sort


def compile_training_data(
    rows: Iterable[dict[str, Any]],
    *,
    holdout_words: set[str],
    dictionary_words: set[str] | None = None,
    min_positive_confidence: float = 0.80,
    min_negative_confidence: float = 0.80,
    min_replaceability: int = 2,
    negatives_per_positive: int = 2,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    triplets: list[dict[str, str]] = []
    graded_pairs: list[dict[str, Any]] = []
    relation_counts: Counter[str] = Counter()
    skipped = Counter()
    pair_seen: set[tuple[str, str, str]] = set()

    for row_index, row in enumerate(rows):
        errors = validate_teacher_record(
            row,
            dictionary_words=dictionary_words,
            holdout_words=holdout_words,
        )
        if errors:
            skipped["invalid_records"] += 1
            continue

        anchor_info = row["anchor"]
        anchor_word = normalize_text(anchor_info["word"])
        anchor_definition = normalize_text(anchor_info["definition"])
        anchor = sense_text(anchor_word, anchor_definition)

        positives: list[dict[str, Any]] = []
        negatives: list[dict[str, Any]] = []

        for candidate in row["candidates"]:
            judgment = candidate["judgment"]
            relation = str(judgment["relation"])
            relation_counts[relation] += 1

            confidence = float(judgment["confidence"])
            replaceability = int(judgment["replaceability"])
            relatedness = int(judgment["semantic_relatedness"])
            candidate_word = normalize_text(candidate["word"])
            candidate_definition = normalize_text(candidate["definition"])
            candidate_text = sense_text(candidate_word, candidate_definition)

            graded_pairs.append(
                {
                    "anchor": anchor,
                    "candidate": candidate_text,
                    "anchor_word": anchor_word,
                    "candidate_word": candidate_word,
                    "relation": relation,
                    "semantic_relatedness": relatedness,
                    "replaceability": replaceability,
                    "confidence": confidence,
                    "register": judgment.get("register", "unknown"),
                    "reason": judgment.get("reason", ""),
                }
            )

            if (
                relation in POSITIVE_RELATIONS
                and confidence >= min_positive_confidence
                and replaceability >= min_replaceability
            ):
                positives.append(candidate)
                continue

            is_hard_negative = False
            if relation == "antonym" and confidence >= min_negative_confidence:
                is_hard_negative = True
            elif (
                relation in {"associated", "unrelated"}
                and confidence >= min_negative_confidence
                and replaceability == 0
            ):
                is_hard_negative = True

            if is_hard_negative:
                negatives.append(candidate)

        if not positives:
            skipped["anchors_without_positive"] += 1
            continue
        if not negatives:
            skipped["anchors_without_hard_negative"] += 1
            continue

        negatives.sort(key=_negative_hardness)

        for positive in positives:
            positive_text = sense_text(
                normalize_text(positive["word"]),
                normalize_text(positive["definition"]),
            )
            for negative in negatives[: max(1, negatives_per_positive)]:
                negative_text = sense_text(
                    normalize_text(negative["word"]),
                    normalize_text(negative["definition"]),
                )
                key = (anchor, positive_text, negative_text)
                if key in pair_seen:
                    skipped["duplicate_triplets"] += 1
                    continue
                pair_seen.add(key)
                triplets.append(
                    {
                        "anchor": anchor,
                        "positive": positive_text,
                        "negative": negative_text,
                    }
                )

    report = {
        "records_input": row_index + 1 if "row_index" in locals() else 0,
        "triplets": len(triplets),
        "graded_pairs": len(graded_pairs),
        "relation_counts": dict(sorted(relation_counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "thresholds": {
            "min_positive_confidence": min_positive_confidence,
            "min_negative_confidence": min_negative_confidence,
            "min_replaceability": min_replaceability,
            "negatives_per_positive": negatives_per_positive,
        },
    }
    return triplets, graded_pairs, report
