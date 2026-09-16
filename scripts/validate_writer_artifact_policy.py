#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_writer_artifact_policy import load_writer_artifact_registry


DEFAULT_REGISTRY = "evaluation/writer_relevance_artifact_registry.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the writer-reranker benchmark/reproduction/production artifact policy."
    )
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    args = parser.parse_args()

    payload = load_writer_artifact_registry(args.registry)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "phase3_locked_artifact": payload["phase3_locked_benchmark"]["artifact_id"],
                "phase4_reproduction_artifact": payload["phase4_reproduction"]["artifact_id"],
                "production_status": payload["production_policy"]["status"],
                "production_name_pattern": payload["production_policy"]["name_pattern"],
                "fresh_holdout_required_for_new_quality_claim": payload["production_policy"][
                    "requires_fresh_holdout_for_quality_claim"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
