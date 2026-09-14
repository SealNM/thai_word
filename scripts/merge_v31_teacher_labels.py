#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_v3_data import read_jsonl, write_jsonl
from thai_v31_router import merge_audit_labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Gemini audit judgments back into V3.1 local teacher labels."
    )
    parser.add_argument(
        "--local",
        default="artifacts/v3_1/local_teacher_labels.jsonl",
    )
    parser.add_argument(
        "--audit",
        default="artifacts/v3_1/gemini_audit_labels.jsonl",
    )
    parser.add_argument(
        "--output",
        default="artifacts/v3_1/final_teacher_labels.jsonl",
    )
    args = parser.parse_args()

    local_rows = read_jsonl(args.local)
    if not local_rows:
        raise SystemExit(f"No local teacher labels found at {args.local}")

    audit_rows = read_jsonl(args.audit)
    audit_by_seed = {
        row.get("seed_id"): row
        for row in audit_rows
        if row.get("seed_id")
    }

    merged = []
    replaced = 0
    for row in local_rows:
        seed_id = row.get("seed_id")
        audit = audit_by_seed.get(seed_id)
        if audit is None:
            merged.append(row)
            continue
        item = merge_audit_labels(row, audit)
        replaced += int(item.get("audit_merge", {}).get("replaced_candidates", 0))
        merged.append(item)

    write_jsonl(args.output, merged)
    print(
        json.dumps(
            {
                "version": "3.1",
                "local_rows": len(local_rows),
                "audit_rows": len(audit_rows),
                "final_rows": len(merged),
                "replaced_candidates": replaced,
                "output": args.output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
