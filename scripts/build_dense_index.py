#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_dense_v2 import MODEL_PROFILES, build_dense_index, resolve_model_profile
from thai_lexical_v1 import load_artifacts


EMBEDDINGGEMMA_MODEL_ID = "google/embeddinggemma-300m"


def _looks_like_gated_hf_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        name = type(current).__name__
        text = str(current)
        if name == "GatedRepoError":
            return True
        if (
            "401" in text
            and (
                "gated repo" in text.lower()
                or "restricted" in text.lower()
                or "embeddinggemma" in text.lower()
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _gated_model_help(model: str) -> str:
    profile = resolve_model_profile(model)
    model_id = str(profile["model_id"])
    token_state = "set" if os.environ.get("HF_TOKEN") else "not set"

    return (
        f"Cannot access gated Hugging Face model: {model_id}\n\n"
        "EmbeddingGemma requires accepting Google's usage license with the same "
        "Hugging Face account used by this runtime.\n"
        "Model page: https://huggingface.co/google/embeddinggemma-300m\n\n"
        "After accepting the license, authenticate before running this command.\n"
        "Recommended for Kaggle: store a Hugging Face read token as a private "
        "Kaggle secret named HF_TOKEN, then expose it as the HF_TOKEN environment "
        "variable before importing/running Hugging Face libraries.\n\n"
        "You can also authenticate interactively with:\n"
        "  hf auth login\n"
        "and verify with:\n"
        "  hf auth whoami\n\n"
        f"Current process HF_TOKEN: {token_state}\n\n"
        "Do not switch embedding models for this rebuild: the V2.5 baseline uses "
        "embeddinggemma-300m-256."
    )


def build_parser() -> argparse.ArgumentParser:
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
    return parser


def run(args: argparse.Namespace) -> dict:
    lexical = load_artifacts(args.index)
    try:
        metadata = build_dense_index(
            lexical,
            args.output,
            model=args.model,
            batch_size=args.batch_size,
            device=args.device,
        )
    except Exception as exc:
        if _looks_like_gated_hf_error(exc):
            raise SystemExit(_gated_model_help(args.model)) from None
        raise

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
