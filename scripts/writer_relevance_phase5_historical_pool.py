#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from scripts.writer_relevance_phase4_labels_materialize import materialize
from thai_substitutability import read_jsonl, validate_rows, write_jsonl

SOURCE_50_FILE = "evaluation/writer_relevance_50_annotations.approved.jsonl"
SOURCE_PHASE4_FILE = "evaluation/writer_relevance_phase4_holdout_annotations.approved.jsonl"
SOURCE_50_SHA256 = "6e767583302a6df75c9b76d86fc73cbda150c98fbbe7c95b912a650a04a1a515"
SOURCE_PHASE4_SHA256 = "7d719f22bf7834ab24bfacd91b3535871f05aa5db1e9273579e0e76a8f7c204c"
SOURCE_50_QUERY_COUNT = 50
SOURCE_PHASE4_QUERY_COUNT = 20
SOURCE_50_PAIR_COUNT = 1500
SOURCE_PHASE4_PAIR_COUNT = 600
TOTAL_QUERY_COUNT = 70
TOTAL_PAIR_COUNT = 2100
SCHEMA_VERSION = 3
PHASE5_DATE = "2026-09-16"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _query_ids(rows: list[dict[str, Any]], *, source_name: str) -> set[str]:
    query_ids: set[str] = set()
    missing: list[int] = []
    for index, row in enumerate(rows, start=1):
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            missing.append(index)
            continue
        query_ids.add(query_id)
    if missing:
        preview = ", ".join(str(value) for value in missing[:10])
        raise ValueError(f"{source_name}: missing/invalid query_id at row(s): {preview}")
    return query_ids


def _load_verified_source(
    path: Path,
    *,
    source_name: str,
    expected_sha256: str,
    expected_query_count: int,
    expected_pair_count: int,
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{source_name}: required source file not found: {path}")

    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{source_name}: SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )

    rows = read_jsonl(path)
    errors = validate_rows(rows, require_labels=True)
    if errors:
        raise ValueError(
            f"{source_name}: Writer Relevance schema v3 validation failed:\n"
            + "\n".join(errors[:20])
        )

    query_ids = _query_ids(rows, source_name=source_name)
    if len(query_ids) != expected_query_count:
        raise ValueError(
            f"{source_name}: expected {expected_query_count} unique query_id values, "
            f"got {len(query_ids)}"
        )
    if len(rows) != expected_pair_count:
        raise ValueError(
            f"{source_name}: expected {expected_pair_count} pairs, got {len(rows)}"
        )

    summary = {
        "source_name": source_name,
        "file": str(path),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "query_count": len(query_ids),
        "pair_count": len(rows),
        "schema_version": SCHEMA_VERSION,
        "validation_passed": True,
    }
    return rows, query_ids, summary


def _with_provenance(
    rows: list[dict[str, Any]],
    *,
    source_name: str,
    source_file: str,
    source_sha256: str,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        copied = copy.deepcopy(row)
        if "phase5_provenance" in copied:
            raise ValueError(
                f"{source_name}: row {copied.get('pair_id', '<unknown>')} already "
                "contains phase5_provenance"
            )
        copied["phase5_provenance"] = {
            "source": source_name,
            "source_file": source_file,
            "source_sha256": source_sha256,
        }
        enriched.append(copied)
    return enriched


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        write_jsonl(temporary, rows)
        digest = sha256_file(temporary)
        os.replace(temporary, path)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(rendered, encoding="utf-8")
        digest = sha256_file(temporary)
        os.replace(temporary, path)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def build_historical_pool(
    source_50: Path,
    source_phase4: Path,
    output: Path,
    manifest_path: Path,
    *,
    source_50_sha256: str = SOURCE_50_SHA256,
    source_phase4_sha256: str = SOURCE_PHASE4_SHA256,
    source_50_query_count: int = SOURCE_50_QUERY_COUNT,
    source_phase4_query_count: int = SOURCE_PHASE4_QUERY_COUNT,
    source_50_pair_count: int = SOURCE_50_PAIR_COUNT,
    source_phase4_pair_count: int = SOURCE_PHASE4_PAIR_COUNT,
) -> dict[str, Any]:
    rows_50, ids_50, summary_50 = _load_verified_source(
        source_50,
        source_name="writer_relevance_50",
        expected_sha256=source_50_sha256,
        expected_query_count=source_50_query_count,
        expected_pair_count=source_50_pair_count,
    )
    rows_phase4, ids_phase4, summary_phase4 = _load_verified_source(
        source_phase4,
        source_name="phase4_consumed_holdout",
        expected_sha256=source_phase4_sha256,
        expected_query_count=source_phase4_query_count,
        expected_pair_count=source_phase4_pair_count,
    )

    overlap = sorted(ids_50 & ids_phase4)
    if overlap:
        raise ValueError(
            "Duplicate query_id values across historical sources: "
            + ", ".join(overlap)
        )

    combined = _with_provenance(
        rows_50,
        source_name="writer_relevance_50",
        source_file=SOURCE_50_FILE,
        source_sha256=summary_50["actual_sha256"],
    )
    combined.extend(
        _with_provenance(
            rows_phase4,
            source_name="phase4_consumed_holdout",
            source_file=SOURCE_PHASE4_FILE,
            source_sha256=summary_phase4["actual_sha256"],
        )
    )

    combined_errors = validate_rows(combined, require_labels=True)
    if combined_errors:
        raise ValueError(
            "Combined historical pool validation failed:\n"
            + "\n".join(combined_errors[:20])
        )

    combined_ids = _query_ids(combined, source_name="phase5_historical_70")
    expected_total_queries = source_50_query_count + source_phase4_query_count
    expected_total_pairs = source_50_pair_count + source_phase4_pair_count
    if len(combined_ids) != expected_total_queries:
        raise ValueError(
            f"Combined pool expected {expected_total_queries} unique query_id values, "
            f"got {len(combined_ids)}"
        )
    if len(combined) != expected_total_pairs:
        raise ValueError(
            f"Combined pool expected {expected_total_pairs} pairs, got {len(combined)}"
        )

    output_sha256 = _atomic_write_jsonl(output, combined)
    manifest = {
        "status": "phase5_historical_pool_frozen",
        "schema_version": SCHEMA_VERSION,
        "date": PHASE5_DATE,
        "output_file": str(output),
        "output_sha256": output_sha256,
        "query_count": len(combined_ids),
        "pair_count": len(combined),
        "duplicate_query_ids_across_sources": overlap,
        "sources": [summary_50, summary_phase4],
        "validation": {
            "passed": True,
            "error_count": 0,
            "writer_relevance_schema_v3_passed": True,
            "severe_error_utility_constraint_passed": True,
            "query_ids_unique_across_sources": True,
        },
        "policy": {
            "historical_development_pool": True,
            "phase4_holdout_is_consumed": True,
            "phase4_holdout_is_unbiased_acceptance_set": False,
            "fresh_phase5_acceptance_holdout_opened": False,
            "model_selection_allowed_on_this_pool": True,
        },
    }
    manifest_sha256 = _atomic_write_json(manifest_path, manifest)
    return {
        **manifest,
        "manifest_file": str(manifest_path),
        "manifest_sha256": manifest_sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the frozen Phase-5 70-query historical development pool."
    )
    parser.add_argument(
        "--source-50",
        default=SOURCE_50_FILE,
    )
    parser.add_argument(
        "--source-phase4",
        default=SOURCE_PHASE4_FILE,
    )
    parser.add_argument(
        "--phase4-candidate",
        default="evaluation/writer_relevance_phase4_holdout_annotations.jsonl",
    )
    parser.add_argument(
        "--phase4-labels",
        default="evaluation/writer_relevance_phase4_holdout_labels.approved.json",
    )
    parser.add_argument(
        "--output",
        default="evaluation/writer_relevance_phase5_historical_70.approved.jsonl",
    )
    parser.add_argument(
        "--manifest",
        default="evaluation/writer_relevance_phase5_historical_70_manifest.json",
    )
    parser.add_argument(
        "--no-materialize-phase4",
        action="store_true",
        help="Require the approved Phase-4 JSONL to exist instead of materializing it.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_phase4 = Path(args.source_phase4)
    if not source_phase4.is_file() and not args.no_materialize_phase4:
        materialize(
            Path(args.phase4_candidate),
            Path(args.phase4_labels),
            source_phase4,
        )

    result = build_historical_pool(
        Path(args.source_50),
        source_phase4,
        Path(args.output),
        Path(args.manifest),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
