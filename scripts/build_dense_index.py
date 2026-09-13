#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_dense_v2 import MODEL_PROFILES, build_dense_index
from thai_lexical_v1 import load_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a sense-level dense embedding index for Thai Lexical Semantic V2."
    )
    parser.add_argument("--index", default="artifacts/v1", help="V1 lexical artifact directory.")
    parser.add_argument(
        "--model",
        default="e5-small",
        help=(
            "Model key or Hugging Face model id. Built-ins: "
            + ", ".join(sorted(MODEL_PROFILES))
        ),
    )
    parser.add_argument("--output", required=True, help="Dense artifact output directory.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        default=None,
        help="Optional sentence-transformers device, e.g. cuda or cpu.",
    )
    args = parser.parse_args()

    lexical = load_artifacts(args.index)
    metadata = build_dense_index(
        lexical,
        args.output,
        model=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
