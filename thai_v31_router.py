from __future__ import annotations

import copy
import random
from typing import Any

from thai_lexical_v1 import normalize_text


AUTO_LABEL_SOURCE = "dictionary_rule"
LOCAL_LABEL_SOURCE = "qwen_local"
GEMINI_LABEL_SOURCE = "gemini_audit"


def _evidence(candidate: dict[str, Any]) -> dict[str, Any]:
    evidence = candidate.get("evidence")
    return evidence if isinstance(evidence, dict) else {}


def auto_judgment(candidate: dict[str, Any]) -> dict[str, Any] | None:
    """Return only high-precision rule labels.

    V3.1 intentionally auto-labels a narrow subset. Tier 5 standalone candidates
    are exact/direct dictionary glosses in the V1 lexical architecture. Lower
    tiers remain for a teacher because subtype, context, and ambiguity can look
    deceptively similar to synonyms.
    """
    evidence = _evidence(candidate)
    relation_tier = int(evidence.get("relation_tier") or 0)
    lexical_form = str(evidence.get("lexical_form") or "standalone")
    relation_hint = str(evidence.get("relation_hint") or "")
    safe_hints = {"direct_gloss_or_synonym", "mutual_definition_reference"}

    if (
        relation_tier >= 5
        and lexical_form == "standalone"
        and relation_hint in safe_hints
    ):
        return {
            "relation": "synonym",
            "semantic_relatedness": 4,
            "replaceability": 3,
            "confidence": 0.99,
            "register": "unknown",
            "reason": "นิยามพจนานุกรมชี้เป็นคำเทียบตรงระดับสูงสุด",
            "label_source": AUTO_LABEL_SOURCE,
        }
    return None


def _rank_value(value: Any) -> int:
    if value is None:
        return 10**9
    try:
        return int(value)
    except (TypeError, ValueError):
        return 10**9


def _candidate_priority(candidate: dict[str, Any]) -> tuple[int, int, float, str]:
    evidence = _evidence(candidate)
    dense_rank = _rank_value(evidence.get("dense_rank"))
    lexical_rank = _rank_value(evidence.get("lexical_rank"))
    dense_similarity = evidence.get("dense_similarity")
    try:
        dense_similarity_sort = -float(dense_similarity)
    except (TypeError, ValueError):
        dense_similarity_sort = 1.0
    return dense_rank, lexical_rank, dense_similarity_sort, normalize_text(candidate.get("word", ""))


def select_local_candidates(
    candidates: list[dict[str, Any]],
    *,
    max_candidates: int = 8,
) -> list[int]:
    """Pick a compact but diverse set of uncertain candidates for local judging."""
    if max_candidates <= 0:
        return []

    pending = [
        (index, candidate)
        for index, candidate in enumerate(candidates)
        if not isinstance(candidate.get("judgment"), dict)
    ]
    if len(pending) <= max_candidates:
        return [index for index, _ in pending]

    selected: list[int] = []
    selected_set: set[int] = set()

    def add(index: int) -> None:
        if index not in selected_set and len(selected) < max_candidates:
            selected.append(index)
            selected_set.add(index)

    # Dense-nearest items are the most useful hard-negative candidates.
    dense_sorted = sorted(
        pending,
        key=lambda item: (
            _rank_value(_evidence(item[1]).get("dense_rank")),
            _candidate_priority(item[1]),
        ),
    )
    for index, _ in dense_sorted[: max(1, max_candidates // 2)]:
        add(index)

    # Preserve strong lexical-but-not-safe-enough-for-rules evidence.
    lexical_sorted = sorted(
        pending,
        key=lambda item: (
            -int(_evidence(item[1]).get("relation_tier") or 0),
            _rank_value(_evidence(item[1]).get("lexical_rank")),
            _candidate_priority(item[1]),
        ),
    )
    for index, _ in lexical_sorted:
        add(index)
        if len(selected) >= max_candidates - 1:
            break

    # Keep one lower-ranked retrieved item when possible. It often supplies an
    # associated/unrelated hard negative instead of eight near-duplicates.
    tail = sorted(pending, key=lambda item: _candidate_priority(item[1]), reverse=True)
    for index, _ in tail:
        if index not in selected_set:
            if len(selected) >= max_candidates:
                selected[-1] = index
            else:
                add(index)
            break

    return selected[:max_candidates]


def route_seed(
    seed: dict[str, Any],
    *,
    max_local_candidates: int = 8,
) -> dict[str, Any]:
    routed = copy.deepcopy(seed)
    candidates = routed.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("seed.candidates must be an array")

    auto_count = 0
    for candidate in candidates:
        if isinstance(candidate.get("judgment"), dict):
            continue
        judgment = auto_judgment(candidate)
        if judgment is not None:
            candidate["judgment"] = judgment
            auto_count += 1

    selected_indices = set(
        select_local_candidates(
            candidates,
            max_candidates=max_local_candidates,
        )
    )

    routed_candidates: list[dict[str, Any]] = []
    local_count = 0
    dropped_count = 0
    for index, candidate in enumerate(candidates):
        if isinstance(candidate.get("judgment"), dict):
            routed_candidates.append(candidate)
        elif index in selected_indices:
            item = copy.deepcopy(candidate)
            item["route"] = LOCAL_LABEL_SOURCE
            routed_candidates.append(item)
            local_count += 1
        else:
            dropped_count += 1

    routed["candidates"] = routed_candidates
    routed["routing"] = {
        "version": "3.1",
        "auto_labeled": auto_count,
        "local_candidates": local_count,
        "dropped_candidates": dropped_count,
        "source_candidate_count": len(candidates),
    }
    return routed


def local_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = row.get("candidates")
    if not isinstance(candidates, list):
        return []
    return [
        candidate
        for candidate in candidates
        if not isinstance(candidate.get("judgment"), dict)
    ]


def candidate_needs_audit(
    candidate: dict[str, Any],
    *,
    confidence_threshold: float = 0.75,
) -> bool:
    judgment = candidate.get("judgment")
    if not isinstance(judgment, dict):
        return True

    if judgment.get("label_source") == AUTO_LABEL_SOURCE:
        return False

    if judgment.get("relation") == "uncertain":
        return True

    try:
        confidence = float(judgment.get("confidence"))
    except (TypeError, ValueError):
        return True
    return confidence < confidence_threshold


def build_audit_seed(
    row: dict[str, Any],
    *,
    confidence_threshold: float = 0.75,
    random_audit_fraction: float = 0.02,
    random_seed: int = 42,
) -> dict[str, Any] | None:
    candidates = row.get("candidates")
    if not isinstance(candidates, list):
        return None

    rng = random.Random(f"{random_seed}:{row.get('seed_id', '')}")
    selected: list[dict[str, Any]] = []

    for candidate in candidates:
        judgment = candidate.get("judgment")
        source = judgment.get("label_source") if isinstance(judgment, dict) else None
        needs_review = candidate_needs_audit(
            candidate,
            confidence_threshold=confidence_threshold,
        )

        random_review = (
            source == LOCAL_LABEL_SOURCE
            and random_audit_fraction > 0
            and rng.random() < random_audit_fraction
        )

        if not (needs_review or random_review):
            continue

        item = {
            key: copy.deepcopy(value)
            for key, value in candidate.items()
            if key not in {"judgment", "route"}
        }
        selected.append(item)

    if not selected:
        return None

    return {
        "schema_version": row.get("schema_version", 1),
        "seed_id": row.get("seed_id"),
        "anchor": copy.deepcopy(row["anchor"]),
        "candidates": selected,
        "audit": {
            "version": "3.1",
            "source": "local_teacher_labels",
            "confidence_threshold": confidence_threshold,
            "random_audit_fraction": random_audit_fraction,
        },
    }


def merge_audit_labels(
    local_row: dict[str, Any],
    audit_row: dict[str, Any],
) -> dict[str, Any]:
    if local_row.get("seed_id") != audit_row.get("seed_id"):
        raise ValueError("Cannot merge teacher rows with different seed_id values.")

    merged = copy.deepcopy(local_row)
    audit_by_word: dict[str, dict[str, Any]] = {}
    for candidate in audit_row.get("candidates", []):
        judgment = candidate.get("judgment")
        if not isinstance(judgment, dict):
            continue
        updated = copy.deepcopy(judgment)
        updated["label_source"] = GEMINI_LABEL_SOURCE
        audit_by_word[normalize_text(candidate.get("word", ""))] = updated

    for candidate in merged.get("candidates", []):
        word = normalize_text(candidate.get("word", ""))
        if word in audit_by_word:
            candidate["judgment"] = audit_by_word[word]

    merged["audit_merge"] = {
        "version": "3.1",
        "replaced_candidates": len(audit_by_word),
    }
    return merged
