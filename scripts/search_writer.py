#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_lexical_v1 import list_senses, load_artifacts
from thai_writer_search import DEFAULT_RERANK_POOL, RERANKER_MODES, WriterSearch


def _missing_base_artifacts(index: str | Path, dense_index: str | Path) -> list[str]:
    index_path = Path(index)
    dense_path = Path(dense_index)
    required = [
        index_path / "entries.json",
        index_path / "senses.json",
        index_path / "metadata.json",
        dense_path / "dense_metadata.json",
        dense_path / "dense_embeddings.npy",
    ]
    return [str(path) for path in required if not path.exists()]


def _base_artifact_error(
    *,
    index: str | Path,
    dense_index: str | Path,
    dense_device: str | None,
) -> str:
    missing = _missing_base_artifacts(index, dense_index)
    if not missing:
        return ""

    device = dense_device or "cuda"
    missing_lines = "\n".join(f"  - {path}" for path in missing)
    return (
        "Thai Words base retrieval artifacts are missing.\n"
        "Writer reranker fallback still requires the V2.5 base index.\n\n"
        "Missing files:\n"
        f"{missing_lines}\n\n"
        "Build them with:\n"
        f"  python scripts/build_index.py --output {index}\n"
        "  python scripts/build_dense_index.py "
        f"--index {index} --model embeddinggemma-300m-256 "
        f"--output {dense_index} --device {device}\n\n"
        "These are V1/V2.5 retrieval artifacts, not the writer-reranker checkpoint."
    )


def _validate_base_artifacts(
    *,
    index: str | Path,
    dense_index: str | Path,
    dense_device: str | None,
) -> None:
    message = _base_artifact_error(
        index=index,
        dense_index=dense_index,
        dense_device=dense_device,
    )
    if message:
        raise SystemExit(message)


def _validate_lexical_artifact(index: str | Path) -> None:
    entries = Path(index) / "entries.json"
    if entries.exists():
        return
    raise SystemExit(
        "V1 lexical artifact is missing. Build it first with:\n"
        f"  python scripts/build_index.py --output {index}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search Thai Words with the Phase-4 writer reranker."
    )
    parser.add_argument("query", help="Thai headword or phrase.")
    parser.add_argument("--index", default="artifacts/v1")
    parser.add_argument("--dense-index", required=True)
    parser.add_argument("--learned-ranker", default="artifacts/writer-reranker")
    parser.add_argument("--neural-model", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rerank-pool", type=int, default=DEFAULT_RERANK_POOL)
    parser.add_argument("--sense", type=int, default=None)
    parser.add_argument("--lexical-pool", type=int, default=300)
    parser.add_argument("--dense-pool", type=int, default=300)
    parser.add_argument("--lexical-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-device", default=None)
    parser.add_argument("--neural-device", default=None)
    parser.add_argument("--neural-batch-size", type=int, default=4)
    parser.add_argument("--neural-max-length", type=int, default=384)
    parser.add_argument(
        "--reranker-mode",
        choices=sorted(RERANKER_MODES),
        default="optional",
    )
    parser.add_argument("--list-senses", action="store_true")
    parser.add_argument("--include-runtime", action="store_true")
    return parser


def run(args: argparse.Namespace):
    if args.list_senses:
        _validate_lexical_artifact(args.index)
        lexical = load_artifacts(args.index)
        payload = list_senses(lexical, args.query)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    _validate_base_artifacts(
        index=args.index,
        dense_index=args.dense_index,
        dense_device=args.dense_device,
    )

    searcher = WriterSearch.from_paths(
        lexical_index=args.index,
        dense_index=args.dense_index,
        learned_ranker=args.learned_ranker,
        neural_model=args.neural_model,
        dense_device=args.dense_device,
        neural_device=args.neural_device,
        neural_batch_size=args.neural_batch_size,
        neural_max_length=args.neural_max_length,
        mode=args.reranker_mode,
    )
    results = searcher.search(
        args.query,
        top_k=args.top_k,
        sense=args.sense,
        rerank_pool=args.rerank_pool,
        lexical_pool=args.lexical_pool,
        dense_pool=args.dense_pool,
        lexical_weight=args.lexical_weight,
        dense_weight=args.dense_weight,
        rrf_k=args.rrf_k,
    )

    payload = {
        "query": args.query,
        "sense": args.sense,
        "reranker_mode": args.reranker_mode,
        "reranker_status": searcher.last_reranker_status,
        "reranker_error": searcher.last_reranker_error,
        "results": results,
    }
    if args.include_runtime:
        payload["runtime"] = searcher.runtime_info()

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
