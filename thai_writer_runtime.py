from __future__ import annotations

from typing import Any

import numpy as np


WRITER_RUNTIME_SCHEMA_VERSION = 1
CATEGORY_MODES = {"include", "none", "omit"}


def _num(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return float(value)
    return default


def _reciprocal(value: Any) -> float:
    number = _num(value, 0.0)
    return 1.0 / number if number > 0 else 0.0


def _clip_text(value: Any, *, max_chars: int = 700) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _candidate_definition(item: dict[str, Any]) -> str:
    matched = item.get("matched_candidate_sense")
    if isinstance(matched, dict) and matched.get("definition"):
        return str(matched["definition"])
    return str(item.get("definition") or "")


def runtime_row_from_v25_result(
    *,
    query: str,
    query_definition: str,
    candidate: dict[str, Any],
    v25_rank: int,
    query_sense: int | None = None,
    query_category: str | None = None,
) -> dict[str, Any]:
    """Convert one V2.5 result into the annotation-free Phase-4 inference contract."""
    if v25_rank < 1:
        raise ValueError("v25_rank must be >= 1.")
    if not candidate.get("word"):
        raise ValueError("candidate must contain a word.")

    query_payload: dict[str, Any] = {
        "word": str(query),
        "definition": str(query_definition or ""),
    }
    if query_sense is not None:
        query_payload["sense"] = int(query_sense)
    if query_category is not None:
        query_payload["category"] = str(query_category)

    retrieval = {
        "system": "v2.5",
        "v25_rank": int(v25_rank),
        "score": _num(candidate.get("score")),
        "relation_tier": int(_num(candidate.get("relation_tier"))),
        "relation_hint": str(candidate.get("relation_hint") or "<none>"),
        "lexical_form": str(candidate.get("lexical_form") or "<none>"),
        "sense_resolution": str(candidate.get("sense_resolution") or "<none>"),
        "lexical_score": _num(candidate.get("lexical_score")),
        "lexical_rank": candidate.get("lexical_rank"),
        "dense_similarity": candidate.get("dense_similarity"),
        "dense_rank": candidate.get("dense_rank"),
    }

    return {
        "runtime_schema_version": WRITER_RUNTIME_SCHEMA_VERSION,
        "query": query_payload,
        "candidate": {
            "word": str(candidate["word"]),
            "definition": _candidate_definition(candidate),
        },
        "retrieval": retrieval,
    }


def writer_feature_dict(
    row: dict[str, Any],
    *,
    category_mode: str = "include",
) -> dict[str, Any]:
    if category_mode not in CATEGORY_MODES:
        raise ValueError(f"Unknown category_mode: {category_mode!r}")

    retrieval = row.get("retrieval") or {}
    query = row.get("query") or {}

    features: dict[str, Any] = {
        "v25_score": _num(retrieval.get("score")),
        "v25_rr": _reciprocal(retrieval.get("v25_rank")),
        "relation_tier": _num(retrieval.get("relation_tier")),
        "lexical_score": _num(retrieval.get("lexical_score")),
        "lexical_rr": _reciprocal(retrieval.get("lexical_rank")),
        "dense_similarity": _num(retrieval.get("dense_similarity")),
        "dense_rr": _reciprocal(retrieval.get("dense_rank")),
        "has_lexical_rank": float(retrieval.get("lexical_rank") is not None),
        "has_dense_rank": float(retrieval.get("dense_rank") is not None),
        "relation_hint": str(retrieval.get("relation_hint") or "<none>"),
        "lexical_form": str(retrieval.get("lexical_form") or "<none>"),
        "sense_resolution": str(retrieval.get("sense_resolution") or "<none>"),
    }

    if category_mode == "include":
        features["query_category"] = str(query.get("category") or "<none>")
    elif category_mode == "none":
        features["query_category"] = "<none>"

    return features


def writer_text_pair(
    row: dict[str, Any],
    *,
    category_mode: str = "include",
) -> tuple[str, str]:
    if category_mode not in CATEGORY_MODES:
        raise ValueError(f"Unknown category_mode: {category_mode!r}")

    query = row.get("query") or {}
    candidate = row.get("candidate") or {}
    retrieval = row.get("retrieval") or {}

    target_lines = [
        f"คำเป้าหมาย: {_clip_text(query.get('word'), max_chars=120)}",
        f"ความหมายเป้าหมาย: {_clip_text(query.get('definition'))}",
    ]
    if category_mode == "include":
        target_lines.append(
            f"หมวด: {_clip_text(query.get('category'), max_chars=120) or '<none>'}"
        )
    elif category_mode == "none":
        target_lines.append("หมวด: <none>")

    evidence = " | ".join(
        [
            f"v25_rank={retrieval.get('v25_rank', '')}",
            f"v25_score={retrieval.get('score', '')}",
            f"relation_tier={retrieval.get('relation_tier', '')}",
            f"lexical_score={retrieval.get('lexical_score', '')}",
            f"lexical_rank={retrieval.get('lexical_rank', '')}",
            f"dense_similarity={retrieval.get('dense_similarity', '')}",
            f"dense_rank={retrieval.get('dense_rank', '')}",
            f"relation_hint={retrieval.get('relation_hint') or '<none>'}",
            f"lexical_form={retrieval.get('lexical_form') or '<none>'}",
            f"sense_resolution={retrieval.get('sense_resolution') or '<none>'}",
        ]
    )
    candidate_text = "\n".join(
        [
            f"คำที่พิจารณา: {_clip_text(candidate.get('word'), max_chars=120)}",
            f"ความหมายคำที่พิจารณา: {_clip_text(candidate.get('definition'))}",
            f"หลักฐาน retrieval: {evidence}",
        ]
    )
    return "\n".join(target_lines), candidate_text
