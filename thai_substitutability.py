from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 3

SEMANTIC_RELATIONS = frozenset(
    {
        "direct",
        "subtype",
        "broader_concept",
        "manner_action",
        "scene_context",
        "effect_state",
        "weak_related",
        "opposite_misleading",
        "sense_mismatch",
        "unrelated",
        "unclear",
    }
)

STYLE_TAGS = frozenset(
    {
        "literary",
        "archaic",
        "formal",
        "colloquial",
        "technical",
        "dialect",
        "figurative",
        "other",
        "unknown",
    }
)

SEVERE_ERROR_RELATIONS = frozenset(
    {"opposite_misleading", "sense_mismatch", "unrelated"}
)


def stable_pair_id(
    query_word: str,
    query_sense: int | None,
    candidate_word: str,
    candidate_sense: int | None,
) -> str:
    raw = json.dumps(
        [query_word, query_sense, candidate_word, candidate_sense],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"pair_{digest}"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object.")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def validate_annotation_row(
    row: dict[str, Any],
    *,
    require_labels: bool = True,
) -> list[str]:
    errors: list[str] = []
    pair_id = row.get("pair_id", "<missing pair_id>")

    if row.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"{pair_id}: schema_version must be {SCHEMA_VERSION}, "
            f"got {row.get('schema_version')!r}"
        )

    query = row.get("query")
    candidate = row.get("candidate")
    retrieval = row.get("retrieval")
    annotation = row.get("annotation")

    if not isinstance(query, dict) or not query.get("word"):
        errors.append(f"{pair_id}: query.word is required.")
    if not isinstance(candidate, dict) or not candidate.get("word"):
        errors.append(f"{pair_id}: candidate.word is required.")
    if not isinstance(retrieval, dict):
        errors.append(f"{pair_id}: retrieval object is required.")
    elif not isinstance(retrieval.get("v25_rank"), int) or retrieval["v25_rank"] < 1:
        errors.append(f"{pair_id}: retrieval.v25_rank must be a positive integer.")

    if not isinstance(annotation, dict):
        errors.append(f"{pair_id}: annotation object is required.")
        return errors

    utility = annotation.get("utility")
    semantic_relation = annotation.get("semantic_relation")
    style_tags = annotation.get("style_tags")

    if require_labels:
        if utility not in {0, 1, 2, 3}:
            errors.append(f"{pair_id}: annotation.utility must be one of 0,1,2,3.")
        if semantic_relation not in SEMANTIC_RELATIONS:
            errors.append(
                f"{pair_id}: annotation.semantic_relation must be one of "
                f"{sorted(SEMANTIC_RELATIONS)}."
            )
        if not isinstance(style_tags, list):
            errors.append(f"{pair_id}: annotation.style_tags must be a list.")
    else:
        if utility is not None and utility not in {0, 1, 2, 3}:
            errors.append(f"{pair_id}: annotation.utility must be one of 0,1,2,3 or null.")
        if (
            semantic_relation is not None
            and semantic_relation not in SEMANTIC_RELATIONS
        ):
            errors.append(
                f"{pair_id}: annotation.semantic_relation must be one of "
                f"{sorted(SEMANTIC_RELATIONS)} or null."
            )
        if style_tags is not None and not isinstance(style_tags, list):
            errors.append(
                f"{pair_id}: annotation.style_tags must be a list or null."
            )

    if isinstance(style_tags, list):
        invalid_tags = [tag for tag in style_tags if tag not in STYLE_TAGS]
        if invalid_tags:
            errors.append(
                f"{pair_id}: unsupported style tag(s): {sorted(set(invalid_tags))}."
            )
        if len(style_tags) != len(set(style_tags)):
            errors.append(f"{pair_id}: annotation.style_tags must not contain duplicates.")
        if "unknown" in style_tags and len(style_tags) > 1:
            errors.append(
                f"{pair_id}: style tag 'unknown' cannot be combined with other tags."
            )

    if (
        semantic_relation in SEVERE_ERROR_RELATIONS
        and utility in {1, 2, 3}
    ):
        errors.append(
            f"{pair_id}: semantic relation {semantic_relation!r} is a severe error "
            "and requires utility 0."
        )

    return errors


def validate_rows(
    rows: list[dict[str, Any]],
    *,
    require_labels: bool = True,
) -> list[str]:
    errors: list[str] = []
    seen_pair_ids: set[str] = set()
    ranks_by_query: dict[str, set[int]] = defaultdict(set)

    for row in rows:
        pair_id = str(row.get("pair_id", ""))
        if pair_id and pair_id in seen_pair_ids:
            errors.append(f"{pair_id}: duplicate pair_id.")
        if pair_id:
            seen_pair_ids.add(pair_id)

        query_id = str(row.get("query_id", ""))
        retrieval = row.get("retrieval")
        if query_id and isinstance(retrieval, dict):
            rank = retrieval.get("v25_rank")
            if isinstance(rank, int):
                if rank in ranks_by_query[query_id]:
                    errors.append(
                        f"{pair_id or query_id}: duplicate V2.5 rank {rank} "
                        f"inside query {query_id!r}."
                    )
                ranks_by_query[query_id].add(rank)

        errors.extend(validate_annotation_row(row, require_labels=require_labels))

    return errors


def _dcg(utilities: list[int], k: int) -> float:
    total = 0.0
    for index, utility in enumerate(utilities[:k], start=1):
        gain = (2**utility) - 1
        total += gain / math.log2(index + 1)
    return total


def query_metrics(
    rows: list[dict[str, Any]],
    *,
    k: int = 10,
) -> dict[str, Any]:
    ranked = sorted(rows, key=lambda row: int(row["retrieval"]["v25_rank"]))
    utilities = [int(row["annotation"]["utility"]) for row in ranked]
    relations = [str(row["annotation"]["semantic_relation"]) for row in ranked]
    top_utilities = utilities[:k]
    top_relations = relations[:k]

    visible_count = len(top_utilities)
    noise_count = sum(value == 0 for value in top_utilities)
    useful_count = sum(value >= 1 for value in top_utilities)
    high_utility_count = sum(value >= 2 for value in top_utilities)
    severe_error_count = sum(
        relation in SEVERE_ERROR_RELATIONS for relation in top_relations
    )

    useful_relations = {
        relation
        for utility, relation in zip(top_utilities, top_relations)
        if utility >= 1
    }

    ideal = sorted(utilities, reverse=True)
    ideal_dcg = _dcg(ideal, k)
    ndcg = (_dcg(utilities, k) / ideal_dcg) if ideal_dcg > 0 else 0.0

    reciprocal_rank = 0.0
    for index, value in enumerate(utilities, start=1):
        if value >= 2:
            reciprocal_rank = 1.0 / index
            break

    denominator = max(1, visible_count)
    return {
        "candidate_count": len(ranked),
        "visible_count": visible_count,
        "noise_at_k": noise_count,
        "noise_rate_at_k": noise_count / denominator,
        "useful_at_k": useful_count,
        "useful_rate_at_k": useful_count / denominator,
        "high_utility_at_k": high_utility_count,
        "high_utility_rate_at_k": high_utility_count / denominator,
        "severe_error_at_k": severe_error_count,
        "severe_error_rate_at_k": severe_error_count / denominator,
        "relation_diversity_at_k": len(useful_relations),
        "ndcg_at_k": ndcg,
        "mrr_high_utility": reciprocal_rank,
    }


def benchmark_metrics(
    rows: list[dict[str, Any]],
    *,
    k: int = 10,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_id"])].append(row)

    per_query = {
        query_id: query_metrics(group_rows, k=k)
        for query_id, group_rows in sorted(grouped.items())
    }

    mean_keys = [
        "noise_at_k",
        "noise_rate_at_k",
        "useful_at_k",
        "useful_rate_at_k",
        "high_utility_at_k",
        "high_utility_rate_at_k",
        "severe_error_at_k",
        "severe_error_rate_at_k",
        "relation_diversity_at_k",
        "ndcg_at_k",
        "mrr_high_utility",
    ]

    if not per_query:
        return {
            "query_count": 0,
            "k": k,
            **{f"{key}_mean": 0.0 for key in mean_keys},
            "queries": {},
        }

    summary = {
        f"{key}_mean": sum(float(item[key]) for item in per_query.values())
        / len(per_query)
        for key in mean_keys
    }
    return {
        "query_count": len(per_query),
        "k": k,
        **summary,
        "queries": per_query,
    }
