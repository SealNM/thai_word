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
        help="Optional device, e.g. cuda or cpu. CUDA requests safely fall back to CPU when unavailable.",
    )
    parser.add_argument(
        "--native-retrieval",
        action="store_true",
        help=(
            "Use SentenceTransformer.encode_query/encode_document for a custom "
            "retrieval model path/id. Built-in EmbeddingGemma profiles already "
            "enable this automatically."
        ),
    )
    parser.add_argument(
        "--truncate-dim",
        type=int,
        default=None,
        help="Optional Matryoshka output dimension, e.g. 256.",
    )
    parser.add_argument(
        "--model-key",
        default=None,
        help="Optional display key stored in dense metadata.",
    )
    args = parser.parse_args()

    profile_overrides = {}
    if args.native_retrieval:
        profile_overrides.update(
            {
                "query_method": "encode_query",
                "document_method": "encode_document",
                "query_prefix": "",
                "document_prefix": "",
            }
        )
    if args.truncate_dim is not None:
        profile_overrides["truncate_dim"] = args.truncate_dim
    if args.model_key:
        profile_overrides["key"] = args.model_key

    lexical = load_artifacts(args.index)
    metadata = build_dense_index(
        lexical,
        args.output,
        model=args.model,
        batch_size=args.batch_size,
        device=args.device,
        profile_overrides=profile_overrides or None,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
