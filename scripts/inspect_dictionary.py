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
    DEFAULT_WORD_FIELD,
    inspect_records,
    load_json_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Thai dictionary JSON schema and coverage.")
    parser.add_argument("input", help="Path to dictionary JSON.")
    parser.add_argument("--id-field", default=DEFAULT_ID_FIELD)
    parser.add_argument("--word-field", default=DEFAULT_WORD_FIELD)
    parser.add_argument("--definition-field", default=DEFAULT_DEFINITION_FIELD)
    args = parser.parse_args()

    records = load_json_records(args.input)
    report = inspect_records(
        records,
        id_field=args.id_field,
        word_field=args.word_field,
        definition_field=args.definition_field,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if report["definition_coverage"] == 0:
        raise SystemExit(
            "\nCannot build semantic V1 yet: no definitions were found in the selected definition field."
        )


if __name__ == "__main__":
    main()
