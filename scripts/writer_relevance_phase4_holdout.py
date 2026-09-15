#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from thai_hybrid_v2 import HybridSearcher
from thai_lexical_v1 import list_senses, load_artifacts
from thai_substitutability import SCHEMA_VERSION, stable_pair_id, validate_rows, write_jsonl


TARGET_COUNT = 20
CANDIDATES_PER_QUERY = 30
EXPECTED_PAIR_COUNT = TARGET_COUNT * CANDIDATES_PER_QUERY
EXPECTED_GROUPS = {
    "noun_scene": 5,
    "verb_action": 5,
    "adjective_state": 5,
    "emotion_abstract": 5,
}


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object.")
    return payload


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _old_headwords(old_targets: dict[str, Any]) -> set[str]:
    queries = old_targets.get("queries")
    if not isinstance(queries, list):
        raise ValueError("Old target file must contain a queries array.")
    return {
        str(item.get("query", "")).strip()
        for item in queries
        if isinstance(item, dict) and str(item.get("query", "")).strip()
    }


def validate_target_config(
    config: dict[str, Any],
    *,
    old_targets: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    queries = config.get("queries")
    if not isinstance(queries, list):
        return ["config.queries must be an array."]

    if len(queries) != TARGET_COUNT:
        errors.append(
            f"Phase-4 holdout must contain exactly {TARGET_COUNT} targets; "
            f"found {len(queries)}."
        )

    words: list[str] = []
    groups: Counter[str] = Counter()
    for index, item in enumerate(queries, start=1):
        if not isinstance(item, dict):
            errors.append(f"queries[{index}] must be an object.")
            continue
        word = str(item.get("query", "")).strip()
        group = str(item.get("group", "")).strip()
        intended = str(item.get("intended", "")).strip()
        if not word:
            errors.append(f"queries[{index}].query is required.")
        else:
            words.append(word)
        if group:
            groups[group] += 1
        else:
            errors.append(f"{word or f'queries[{index}]'}: group is required.")
        if not intended:
            errors.append(f"{word or f'queries[{index}]'}: intended meaning is required.")

    duplicates = sorted({word for word in words if words.count(word) > 1})
    if duplicates:
        errors.append(f"Duplicate Phase-4 target headword(s): {duplicates}.")

    old_words = _old_headwords(old_targets)
    overlap = sorted(set(words) & old_words)
    if overlap:
        errors.append(
            "Phase-4 holdout headwords overlap the original 50-target set: "
            + ", ".join(overlap)
        )

    if set(groups) != set(EXPECTED_GROUPS):
        errors.append(
            f"Phase-4 groups must be exactly {sorted(EXPECTED_GROUPS)}; "
            f"found {sorted(groups)}."
        )
    for group, expected in EXPECTED_GROUPS.items():
        if groups.get(group, 0) != expected:
            errors.append(
                f"Group {group!r} must contain {expected} targets; "
                f"found {groups.get(group, 0)}."
            )

    return errors


def apply_sense_decisions(
    report: dict[str, Any],
    decisions: dict[str, Any],
    *,
    expected_config_sha256: str,
) -> dict[str, Any]:
    if decisions.get("config_sha256") != expected_config_sha256:
        raise ValueError(
            "Sense decisions were created for a different target config hash."
        )

    items = decisions.get("decisions")
    if not isinstance(items, list):
        raise ValueError("Sense decisions file must contain a decisions array.")
    if len(items) != TARGET_COUNT:
        raise ValueError(
            f"Sense decisions must contain exactly {TARGET_COUNT} targets; "
            f"found {len(items)}."
        )

    decision_map: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each sense decision must be an object.")
        query = str(item.get("query", "")).strip()
        sense = item.get("sense")
        if not query or not isinstance(sense, int):
            raise ValueError("Each sense decision requires query and integer sense.")
        if query in decision_map:
            raise ValueError(f"Duplicate sense decision for {query!r}.")
        decision_map[query] = sense

    targets = report.get("targets")
    if not isinstance(targets, list):
        raise ValueError("Sense report must contain targets array.")

    report_words = {
        str(item.get("query", "")).strip()
        for item in targets
        if isinstance(item, dict)
    }
    if set(decision_map) != report_words:
        missing = sorted(report_words - set(decision_map))
        extra = sorted(set(decision_map) - report_words)
        raise ValueError(
            "Sense decisions must cover exactly the inspected targets. "
            f"missing={missing}, extra={extra}"
        )

    for target in targets:
        query = str(target.get("query", "")).strip()
        selected = decision_map[query]
        senses = target.get("senses")
        if not isinstance(senses, list) or not any(
            isinstance(item, dict) and int(item.get("sense", -1)) == selected
            for item in senses
        ):
            raise ValueError(
                f"{query}: decided sense {selected} is not present in inspected senses."
            )
        target["recommended_sense"] = selected
        if len(senses) > 1:
            target["status"] = "reviewed"
        else:
            target["status"] = "unique"

    report["status"] = "phase4_holdout_sense_review_complete"
    report["resolved_count"] = TARGET_COUNT
    report["needs_review_count"] = 0
    report["missing_headword_count"] = 0
    return report


def inspect_targets(args: argparse.Namespace) -> dict[str, Any]:
    config = _read_json(args.config)
    old_targets = _read_json(args.old_targets)
    errors = validate_target_config(config, old_targets=old_targets)
    if errors:
        raise ValueError("\n".join(errors))

    lexical = load_artifacts(args.index)
    targets: list[dict[str, Any]] = []
    counts = Counter()

    for spec in config["queries"]:
        query = str(spec["query"]).strip()
        senses = list_senses(lexical, query)
        recommended: int | None = None
        if not senses:
            status = "missing_headword"
        elif len(senses) == 1:
            status = "unique"
            recommended = int(senses[0]["sense"])
        else:
            status = "needs_review"

        counts[status] += 1
        targets.append(
            {
                "query": query,
                "group": spec.get("group"),
                "category": spec.get("category"),
                "intended": spec.get("intended"),
                "status": status,
                "recommended_sense": recommended,
                "senses": senses,
            }
        )

    report = {
        "status": "phase4_holdout_sense_inspection",
        "selection_policy": "headwords frozen before candidate/model output",
        "config": str(args.config),
        "config_sha256": _sha256_file(args.config),
        "old_targets": str(args.old_targets),
        "old_targets_sha256": _sha256_file(args.old_targets),
        "target_count": len(targets),
        "unique_sense_count": counts["unique"],
        "needs_review_count": counts["needs_review"],
        "missing_headword_count": counts["missing_headword"],
        "targets": targets,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    return report


def freeze_targets(args: argparse.Namespace) -> dict[str, Any]:
    config = _read_json(args.config)
    report = _read_json(args.report)
    old_targets = _read_json(args.old_targets)

    errors = validate_target_config(config, old_targets=old_targets)
    if errors:
        raise ValueError("\n".join(errors))

    expected_config_hash = report.get("config_sha256")
    actual_config_hash = _sha256_file(args.config)
    if expected_config_hash != actual_config_hash:
        raise ValueError(
            "Target config changed after sense inspection. "
            "Re-run inspection; do not silently replace holdout headwords."
        )

    decisions_path = getattr(args, "decisions", None)
    decisions = None
    if decisions_path:
        decisions = _read_json(decisions_path)
        report = apply_sense_decisions(
            report,
            decisions,
            expected_config_sha256=actual_config_hash,
        )

    report_targets = report.get("targets")
    if not isinstance(report_targets, list):
        raise ValueError("Sense report must contain targets array.")
    by_word = {
        str(item.get("query", "")).strip(): item
        for item in report_targets
        if isinstance(item, dict)
    }

    frozen_queries: list[dict[str, Any]] = []
    freeze_errors: list[str] = []
    for spec in config["queries"]:
        query = str(spec["query"]).strip()
        target = by_word.get(query)
        if target is None:
            freeze_errors.append(f"{query}: missing from sense inspection report.")
            continue

        selected = target.get("recommended_sense")
        senses = target.get("senses")
        if not isinstance(selected, int):
            freeze_errors.append(
                f"{query}: set recommended_sense to an integer after reviewing senses."
            )
            continue
        if not isinstance(senses, list) or not any(
            int(item.get("sense", -1)) == selected
            for item in senses
            if isinstance(item, dict)
        ):
            freeze_errors.append(
                f"{query}: recommended_sense={selected} is not present in inspected senses."
            )
            continue

        selected_definition = next(
            str(item.get("definition", ""))
            for item in senses
            if isinstance(item, dict) and int(item.get("sense", -1)) == selected
        )
        frozen_queries.append(
            {
                "query": query,
                "sense": selected,
                "definition": selected_definition,
                "group": spec.get("group"),
                "category": spec.get("category"),
                "intended": spec.get("intended"),
            }
        )

    if freeze_errors:
        raise ValueError("\n".join(freeze_errors))

    frozen = {
        "description": (
            "Phase-4 fresh writer-relevance holdout. Headwords were frozen before "
            "candidate/model output; senses were resolved only from the V1 lexical artifact."
        ),
        "status": "frozen_before_candidate_export",
        "schema_version": SCHEMA_VERSION,
        "target_count": TARGET_COUNT,
        "candidates_per_query": CANDIDATES_PER_QUERY,
        "expected_pair_count": EXPECTED_PAIR_COUNT,
        "old_target_headword_overlap": 0,
        "queries": frozen_queries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "status": "phase4_holdout_frozen_before_candidate_export",
        "schema_version": SCHEMA_VERSION,
        "target_file": str(output),
        "target_file_sha256": _sha256_file(output),
        "headword_source_file": str(args.config),
        "headword_source_sha256": actual_config_hash,
        "sense_report_file": str(args.report),
        "sense_report_sha256": _sha256_file(args.report),
        "sense_decisions_file": str(decisions_path) if decisions_path else None,
        "sense_decisions_sha256": (
            _sha256_file(decisions_path) if decisions_path else None
        ),
        "old_targets_file": str(args.old_targets),
        "old_targets_sha256": _sha256_file(args.old_targets),
        "target_count": TARGET_COUNT,
        "candidates_per_query": CANDIDATES_PER_QUERY,
        "expected_pair_count": EXPECTED_PAIR_COUNT,
        "headword_overlap_with_original_50": 0,
        "candidate_exported": False,
        "writer_reranker_used_for_target_selection": False,
        "writer_reranker_used_for_candidate_export": False,
        "quality_selection_performed": False,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def export_candidates(args: argparse.Namespace) -> dict[str, Any]:
    frozen = _read_json(args.config)
    manifest = _read_json(args.manifest)

    if frozen.get("status") != "frozen_before_candidate_export":
        raise ValueError("Holdout target file is not frozen for candidate export.")
    if int(frozen.get("target_count", -1)) != TARGET_COUNT:
        raise ValueError("Frozen holdout target count mismatch.")
    if int(frozen.get("candidates_per_query", -1)) != CANDIDATES_PER_QUERY:
        raise ValueError("Frozen holdout candidate count mismatch.")

    expected_hash = manifest.get("target_file_sha256")
    actual_hash = _sha256_file(args.config)
    if expected_hash != actual_hash:
        raise ValueError(
            "Frozen holdout target file changed after manifest creation. "
            "Refusing candidate export."
        )
    if manifest.get("candidate_exported") is True:
        raise ValueError(
            "Manifest already records a candidate export. "
            "Create a new explicitly versioned artifact instead of silently replacing it."
        )

    lexical = load_artifacts(args.index)
    searcher = HybridSearcher.from_paths(
        lexical,
        args.dense_index,
        device=args.device,
    )

    rows: list[dict[str, Any]] = []
    for spec in frozen["queries"]:
        query = str(spec["query"]).strip()
        sense = int(spec["sense"])
        results = searcher.search(
            query,
            sense=sense,
            top_k=CANDIDATES_PER_QUERY,
            lexical_pool=300,
            dense_pool=300,
        )
        if len(results) != CANDIDATES_PER_QUERY:
            raise ValueError(
                f"{query}#{sense}: expected {CANDIDATES_PER_QUERY} V2.5 candidates, "
                f"got {len(results)}."
            )

        resolved_query_sense = results[0].get("query_sense")
        if not isinstance(resolved_query_sense, dict):
            raise ValueError(f"{query}#{sense}: query sense did not resolve.")
        resolved_sense = int(resolved_query_sense["sense"])
        if resolved_sense != sense:
            raise ValueError(
                f"{query}: frozen sense {sense} resolved as {resolved_sense}."
            )

        query_id = f"{query}#{sense}"
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
                        sense,
                        str(item["word"]),
                        candidate_sense,
                    ),
                    "query_id": query_id,
                    "query": {
                        "word": query,
                        "sense": sense,
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
                        "semantic_relation": None,
                        "style_tags": None,
                        "legacy_relation": None,
                        "notes": "",
                    },
                    "split": "phase4_holdout",
                }
            )

    if len(rows) != EXPECTED_PAIR_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PAIR_COUNT} exported pairs, got {len(rows)}."
        )
    errors = validate_rows(rows, require_labels=False)
    if errors:
        raise ValueError("\n".join(errors[:20]))

    output = Path(args.output)
    write_jsonl(output, rows)

    export_manifest = {
        "status": "phase4_holdout_candidates_exported_unlabeled",
        "schema_version": SCHEMA_VERSION,
        "target_file": str(args.config),
        "target_file_sha256": actual_hash,
        "source_manifest": str(args.manifest),
        "candidate_file": str(output),
        "candidate_file_sha256": _sha256_file(output),
        "target_count": TARGET_COUNT,
        "candidate_count_per_query": CANDIDATES_PER_QUERY,
        "pair_count": len(rows),
        "candidate_system": "v2.5",
        "writer_reranker_used": False,
        "quality_selection_performed": False,
        "labels_present": False,
        "holdout_opened_for_model_evaluation": False,
    }
    export_manifest_path = Path(args.export_manifest)
    export_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    export_manifest_path.write_text(
        json.dumps(export_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(export_manifest, ensure_ascii=False, indent=2))
    return export_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the fresh Phase-4 writer-relevance holdout without model-selection leakage."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="Inspect dictionary senses for the frozen headword list.")
    inspect.add_argument("--config", default="evaluation/writer_relevance_phase4_holdout_targets.json")
    inspect.add_argument("--old-targets", default="evaluation/writer_relevance_50_targets.json")
    inspect.add_argument("--index", default="artifacts/v1")
    inspect.add_argument("--output", default="evaluation/writer_relevance_phase4_holdout_sense_report.json")
    inspect.set_defaults(func=inspect_targets)

    freeze = subparsers.add_parser("freeze", help="Freeze reviewed sense IDs before candidate export.")
    freeze.add_argument("--config", default="evaluation/writer_relevance_phase4_holdout_targets.json")
    freeze.add_argument("--report", default="evaluation/writer_relevance_phase4_holdout_sense_report.json")
    freeze.add_argument(
        "--decisions",
        default="evaluation/writer_relevance_phase4_holdout_sense_decisions.json",
        help="Reviewed query->sense decisions locked from dictionary definitions only.",
    )
    freeze.add_argument("--old-targets", default="evaluation/writer_relevance_50_targets.json")
    freeze.add_argument("--output", default="evaluation/writer_relevance_phase4_holdout_frozen_queries.json")
    freeze.add_argument("--manifest", default="evaluation/writer_relevance_phase4_holdout_manifest.json")
    freeze.set_defaults(func=freeze_targets)

    export = subparsers.add_parser("export", help="Export exactly 30 V2.5 candidates per frozen target.")
    export.add_argument("--config", default="evaluation/writer_relevance_phase4_holdout_frozen_queries.json")
    export.add_argument("--manifest", default="evaluation/writer_relevance_phase4_holdout_manifest.json")
    export.add_argument("--index", default="artifacts/v1")
    export.add_argument("--dense-index", default="artifacts/v2/embeddinggemma-300m-256")
    export.add_argument("--device", default=None)
    export.add_argument("--output", default="evaluation/writer_relevance_phase4_holdout_annotations.jsonl")
    export.add_argument("--export-manifest", default="evaluation/writer_relevance_phase4_holdout_export_manifest.json")
    export.set_defaults(func=export_candidates)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
