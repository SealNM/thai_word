#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_lexical_v1 import (
    DEFAULT_DEFINITION_FIELD,
    DEFAULT_ID_FIELD,
    DEFAULT_INPUT_PATH,
    DEFAULT_WORD_FIELD,
    build_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Thai dictionary semantic V1 artifacts.")
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT_PATH, help=f"Path to dictionary JSON. Default: {DEFAULT_INPUT_PATH}")
    parser.add_argument("--output", default="artifacts/v1", help="Artifact output directory.")
    parser.add_argument("--id-field", default=DEFAULT_ID_FIELD)
    parser.add_argument("--word-field", default=DEFAULT_WORD_FIELD)
    parser.add_argument("--definition-field", default=DEFAULT_DEFINITION_FIELD)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-features", type=int, default=None)
    args = parser.parse_args()

    metadata = build_index(
        args.input,
        args.output,
        id_field=args.id_field,
        word_field=args.word_field,
        definition_field=args.definition_field,
        min_df=args.min_df,
        max_features=args.max_features,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
