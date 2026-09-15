#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_v27_data import append_jsonl, read_jsonl


DEFAULT_MODEL = "google/gemma-4-E2B-it-qat-mobile-transformers"

SYSTEM_PROMPT = """You are a ranking engine for a Thai dictionary/search tool used by fiction writers.

Rank only by lexical usefulness for replacing the target Thai word.

Apply these rules in order:
1. Preserve the intended dictionary sense.
2. Preserve grammatical role.
3. Prefer candidates that can replace the target naturally in a Thai sentence.
4. True synonyms and strong near-synonyms must beat associated words, subtypes,
   manner words, objects, causes/effects, and compounds with a different lexical role.
5. Among equally valid substitutes, prefer common contemporary Thai before formal,
   literary, archaic, technical, or rare dictionary vocabulary.
6. Literary or archaic alternatives are still useful and should remain below equally
   accurate common alternatives rather than being removed.
7. Do not reward a candidate merely because it is frequent.

Return only the requested JSON array of candidate IDs, ordered best to worst.
"""


def _prompt(seed: dict[str, Any], top_k: int) -> str:
    anchor = seed["anchor"]
    candidates = seed["candidates"]
    requested = min(max(1, int(top_k)), len(candidates))
    lines = [
        "TARGET:",
        f"word: {anchor['word']}",
        f"dictionary sense: {anchor['definition']}",
        "",
        "CANDIDATES:",
    ]
    for index, candidate in enumerate(candidates):
        lines.append(
            f"[{index}] {candidate['word']} — "
            f"{candidate['definition']}"
        )
    lines.extend(
        [
            "",
            f"Return exactly the best {requested} candidate IDs.",
            "Output one JSON array only, for example: [3, 7, 1, 12]",
            "Every returned ID must be unique. Never repeat an ID.",
            "Do not add explanations, markdown, labels, or scores.",
        ]
    )
    return "\n".join(lines)


def parse_ranking(
    output_text: str,
    *,
    candidate_count: int,
    expected_count: int,
) -> list[int]:
    target = min(
        max(1, int(expected_count)),
        candidate_count,
    )
    best: list[int] = []
    text = str(output_text or "").strip()

    for match in re.finditer(r"\[[^\[\]]*\]", text):
        try:
            raw = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, list):
            continue
        parsed: list[int] = []
        for value in raw:
            if isinstance(value, bool):
                continue
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if (
                0 <= index < candidate_count
                and index not in parsed
            ):
                parsed.append(index)
        if len(parsed) > len(best):
            best = parsed

    return best[:target]


class GemmaTeacher:
    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        max_new_tokens: int = 96,
        seed: int = 42,
    ) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForMultimodalLM,
                AutoProcessor,
            )
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Gemma teacher requires a recent Transformers build "
                "with AutoModelForMultimodalLM plus accelerate."
            ) from exc

        if (
            device
            and str(device).lower().startswith("cuda")
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA was requested for the V2.7 Gemma teacher, but "
                "torch.cuda.is_available() is False. Enable a GPU runtime "
                "and install a CUDA-enabled PyTorch build before labeling."
            )

        self.torch = torch
        self.model_id = model_id
        self.max_new_tokens = max(
            32,
            int(max_new_tokens),
        )
        self.seed = int(seed)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            dtype="auto",
            device_map={"": device} if device else "auto",
        )
        self.model.eval()

    def rank(
        self,
        seed: dict[str, Any],
        *,
        top_k: int,
        attempt: int = 0,
    ) -> tuple[list[int], str]:
        prompt = _prompt(seed, top_k)
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(self.model.device)
        input_len = int(inputs["input_ids"].shape[-1])

        generation_seed = self.seed + max(0, int(attempt))
        self.torch.manual_seed(generation_seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(generation_seed)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            use_cache=True,
        )
        generated = outputs[0][input_len:]
        raw = self.processor.decode(
            generated,
            skip_special_tokens=True,
        ).strip()
        ranking = parse_ranking(
            raw,
            candidate_count=len(seed["candidates"]),
            expected_count=top_k,
        )
        return ranking, raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Label V2.7 pairwise training seeds with Gemma 4 offline."
        )
    )
    parser.add_argument(
        "--input",
        default="artifacts/v27/teacher_seeds.jsonl",
    )
    parser.add_argument(
        "--output",
        default="artifacts/v27/teacher_labels.jsonl",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Retry incomplete/duplicate rankings up to this many generations.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum new labels this run; 0 means all pending seeds.",
    )
    parser.add_argument(
        "--resume-from",
        action="append",
        default=[],
        help=(
            "Additional JSONL label file whose seed_ids count as completed. "
            "Repeat this option to union multiple prior outputs."
        ),
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Split pending seeds into this many deterministic worker shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard to process after completed seed_ids are removed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first prompt without loading Gemma.",
    )
    args = parser.parse_args()

    seeds = read_jsonl(args.input)
    if not seeds:
        raise SystemExit(
            f"No seeds found at {args.input}"
        )

    if args.dry_run:
        print(SYSTEM_PROMPT)
        print("\n--- USER PROMPT ---\n")
        print(_prompt(seeds[0], args.top_k))
        return

    completed_sources = [args.output, *args.resume_from]
    completed: set[str] = set()
    for completed_path in completed_sources:
        completed.update(
            str(row.get("seed_id"))
            for row in read_jsonl(completed_path)
            if row.get("seed_id")
        )

    pending = [
        row
        for row in seeds
        if str(row.get("seed_id")) not in completed
    ]

    shard_count = max(1, int(args.shard_count))
    shard_index = int(args.shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        raise SystemExit(
            f"--shard-index must be in [0, {shard_count - 1}], "
            f"got {shard_index}"
        )
    pending = pending[shard_index::shard_count]

    if args.limit > 0:
        pending = pending[: args.limit]

    if not pending:
        print(
            json.dumps(
                {
                    "pending": 0,
                    "existing": len(completed),
                    "output": args.output,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(
        f"[setup] loading {args.model} for "
        f"{len(pending)} pending tasks "
        f"(shard {shard_index + 1}/{shard_count})",
        flush=True,
    )
    teacher = GemmaTeacher(
        args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    successes = 0
    failures = 0
    total_started = perf_counter()

    for position, seed in enumerate(pending, start=1):
        started = perf_counter()
        requested = min(
            args.top_k,
            len(seed["candidates"]),
        )
        try:
            max_attempts = max(1, int(args.max_attempts))
            ranking: list[int] = []
            raw = ""
            used_attempts = 0

            for attempt in range(max_attempts):
                used_attempts = attempt + 1
                ranking, raw = teacher.rank(
                    seed,
                    top_k=requested,
                    attempt=attempt,
                )
                if len(ranking) == requested:
                    break

                print(
                    f"[{position}/{len(pending)}] "
                    f"{seed['anchor']['word']}: retry "
                    f"{used_attempts}/{max_attempts} — "
                    f"expected {requested} unique IDs, "
                    f"got {len(ranking)}; raw={raw[:180]!r}",
                    file=sys.stderr,
                    flush=True,
                )

            if len(ranking) != requested:
                raise ValueError(
                    f"incomplete ranking after {used_attempts} attempts: "
                    f"expected {requested} unique IDs, got {len(ranking)}; "
                    f"raw={raw[:300]!r}"
                )

            labeled = dict(seed)
            labeled["teacher"] = {
                "provider": "gemma-local",
                "model": args.model,
                "generated_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "requested_count": requested,
                "generation_attempts": used_attempts,
                "ranking_indices": ranking,
                "ranking_words": [
                    seed["candidates"][index]["word"]
                    for index in ranking
                ],
                "raw_output": raw,
            }
            append_jsonl(args.output, labeled)
            successes += 1
            elapsed = perf_counter() - started
            avg = (
                perf_counter() - total_started
            ) / position
            eta = avg * (len(pending) - position)
            print(
                f"[{position}/{len(pending)}] "
                f"{seed['anchor']['word']}: "
                f"{requested}/{requested} in "
                f"{elapsed:.1f}s · ETA {eta/60:.1f}m",
                flush=True,
            )
        except Exception as exc:
            failures += 1
            print(
                f"[{position}/{len(pending)}] ERROR "
                f"{seed.get('seed_id')}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    print(
        json.dumps(
            {
                "model": args.model,
                "selected": len(pending),
                "successes": successes,
                "failures": failures,
                "existing_before_run": len(completed),
                "shard_index": shard_index,
                "shard_count": shard_count,
                "output": args.output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
