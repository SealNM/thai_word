#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_dense_v26 import (
    DEFAULT_GEMINI_EMBEDDING_MODEL,
    RECOMMENDED_DIMENSIONS,
    build_gemini_dense_index,
)
from thai_lexical_v1 import load_artifacts


def _fmt(value: float) -> str:
    minutes, seconds = divmod(int(round(max(0.0, value))), 60)
    return f"{minutes:02d}:{seconds:02d}" if minutes else f"{seconds}s"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Thai Words V2.6 dense index with Gemini Embedding 2."
    )
    parser.add_argument("--index", default="artifacts/v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_GEMINI_EMBEDDING_MODEL)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from row 0 instead of resuming a compatible partial build.",
    )
    args = parser.parse_args()

    if args.dimension not in RECOMMENDED_DIMENSIONS and args.dimension != 256:
        print(
            f"[note] {args.dimension}d is supported, but the V2.6 experiment "
            "focuses on 256d parity and Google's recommended 768d control.",
            flush=True,
        )

    print("[setup] Loading V1 lexical artifacts...", flush=True)
    lexical = load_artifacts(args.index)
    total = len(lexical.senses)
    print(
        f"[setup] {total} dictionary senses ready; "
        f"Gemini {args.dimension}d build starting...",
        flush=True,
    )

    started = perf_counter()
    last_completed = 0

    def progress(completed: int, rows: int, batch_seconds: float) -> None:
        nonlocal last_completed
        processed = completed - last_completed
        last_completed = completed
        overall = perf_counter() - started
        rate = completed / max(overall, 1e-9)
        remaining = (rows - completed) / max(rate, 1e-9)
        print(
            f"[build] {completed}/{rows} senses · "
            f"batch {processed} in {_fmt(batch_seconds)} · "
            f"ETA {_fmt(remaining)}",
            flush=True,
        )

    metadata = build_gemini_dense_index(
        lexical,
        args.output,
        dimension=args.dimension,
        model_id=args.model,
        batch_size=args.batch_size,
        resume=not args.no_resume,
        retries=args.retries,
        progress=progress,
    )

    print("[done] Dense index built.", flush=True)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
