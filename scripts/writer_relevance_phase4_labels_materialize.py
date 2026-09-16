#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from thai_substitutability import validate_rows, write_jsonl


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def materialize(source: Path, labels_path: Path, output: Path) -> None:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    expected_source_hash = str(labels["source_sha256"])
    actual_source_hash = sha256_file(source)
    if actual_source_hash != expected_source_hash:
        raise ValueError(
            "Source candidate export hash mismatch: "
            f"expected {expected_source_hash}, got {actual_source_hash}"
        )

    rows = read_jsonl(source)
    tokens = labels.get("labels")
    relation_codes = labels.get("relation_codes")
    style_bit_order = labels.get("style_bit_order")
    if not isinstance(tokens, list) or len(tokens) != len(rows):
        raise ValueError("Frozen label count does not match source row count.")
    if len(rows) != int(labels.get("row_count", -1)):
        raise ValueError("Source row count does not match frozen overlay.")
    if not isinstance(relation_codes, list) or not isinstance(style_bit_order, list):
        raise ValueError("Frozen label codebooks are missing.")

    for row, token in zip(rows, tokens):
        utility_raw, relation_raw, mask_raw = str(token).split(":")
        utility = int(utility_raw)
        relation_index = int(relation_raw)
        mask = int(mask_raw, 16)
        annotation = dict(row.get("annotation") or {})
        annotation["utility"] = utility
        annotation["semantic_relation"] = str(relation_codes[relation_index])
        annotation["style_tags"] = [
            str(tag)
            for bit, tag in enumerate(style_bit_order)
            if mask & (1 << bit)
        ]
        annotation["legacy_relation"] = annotation.get("legacy_relation")
        annotation["notes"] = annotation.get("notes", "")
        row["annotation"] = annotation

    errors = validate_rows(rows, require_labels=True)
    if errors:
        raise ValueError("\n".join(errors[:20]))

    write_jsonl(output, rows)
    expected_output_hash = str(labels["materialized_sha256"])
    actual_output_hash = sha256_file(output)
    if actual_output_hash != expected_output_hash:
        raise ValueError(
            "Materialized approved JSONL hash mismatch: "
            f"expected {expected_output_hash}, got {actual_output_hash}"
        )

    print(
        json.dumps(
            {
                "status": "phase4_holdout_labels_materialized",
                "rows": len(rows),
                "source_sha256": actual_source_hash,
                "output": str(output),
                "output_sha256": actual_output_hash,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize the frozen Phase-4 writer-relevance labels."
    )
    parser.add_argument(
        "--source",
        default="evaluation/writer_relevance_phase4_holdout_annotations.jsonl",
    )
    parser.add_argument(
        "--labels",
        default="evaluation/writer_relevance_phase4_holdout_labels.approved.json",
    )
    parser.add_argument(
        "--output",
        default="evaluation/writer_relevance_phase4_holdout_annotations.approved.jsonl",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    materialize(Path(args.source), Path(args.labels), Path(args.output))


if __name__ == "__main__":
    main()
