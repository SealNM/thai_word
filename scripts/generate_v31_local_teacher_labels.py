#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_v3_data import RELATIONS, REGISTERS, append_jsonl, read_jsonl
from thai_v31_router import LOCAL_LABEL_SOURCE, local_candidates


JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

RELATION_CODES = {
    "s": "synonym",
    "n": "near_synonym",
    "a": "antonym",
    "t": "subtype",
    "g": "supertype",
    "m": "manner",
    "x": "associated",
    "u": "unrelated",
    "?": "uncertain",
}

REGISTER_CODES = {
    "n": "neutral",
    "l": "literary",
    "f": "formal",
    "c": "colloquial",
    "a": "archaic",
    "t": "technical",
    "s": "slang",
    "u": "unknown",
}


def _prompt(row: dict[str, Any]) -> str:
    anchor = row["anchor"]
    candidates = [
        [index, candidate["word"], candidate["definition"]]
        for index, candidate in enumerate(local_candidates(row))
    ]

    return f"""
จัดประเภทความสัมพันธ์คำไทยสำหรับระบบค้นคำของนักเขียน

คำหลัก: {anchor["word"]}
ความหมาย: {anchor["definition"]}

candidate แต่ละรายการคือ [id,คำ,นิยาม]:
{json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))}

relation code:
s=synonym ใช้แทนตรงมาก
n=near_synonym ใกล้และใช้แทนได้หลายบริบทแต่มี nuance
a=antonym ตรงข้าม
t=subtype ชนิดย่อย/รูปเฉพาะ
g=supertype ความหมายกว้างกว่า
m=manner วิธี/ลักษณะเฉพาะของการกระทำ
x=associated เกี่ยวข้องแต่ใช้แทนไม่ได้
u=unrelated ไม่เกี่ยวข้อง
?=uncertain หลักฐานไม่พอ

register code:
n=neutral,l=literary,f=formal,c=colloquial,a=archaic,t=technical,s=slang,u=unknown

ตอบ JSON อย่างเดียว:
{{"j":[[id,relation,relatedness,replaceability,confidence,register],...]}}

relatedness=0..4
replaceability=0..3
confidence=0..100

กฎ:
- ต้องตอบครบทุก id และเรียงตาม id
- antonym ต้อง replaceability=0
- คำที่แค่เกี่ยวข้องกันไม่ใช่ synonym
- ห้ามเพิ่ม candidate
""".strip()


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = JSON_FENCE_RE.sub("", text.strip()).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Local teacher did not return a JSON object.")
        parsed = json.loads(cleaned[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Local teacher JSON must be an object.")
    return parsed


def _validate_compact_judgments(
    result: dict[str, Any],
    expected_words: list[str],
) -> dict[str, dict[str, Any]]:
    judgments = result.get("j")
    if not isinstance(judgments, list):
        raise ValueError("Local teacher response is missing compact 'j' array.")

    by_word: dict[str, dict[str, Any]] = {}
    seen_ids: set[int] = set()

    for item in judgments:
        if not isinstance(item, list) or len(item) != 6:
            raise ValueError(
                "Every compact judgment must be "
                "[id,relation,relatedness,replaceability,confidence,register]."
            )

        candidate_id = int(item[0])
        if candidate_id < 0 or candidate_id >= len(expected_words):
            raise ValueError(f"Candidate id out of range: {candidate_id}")
        if candidate_id in seen_ids:
            raise ValueError(f"Duplicate candidate id: {candidate_id}")
        seen_ids.add(candidate_id)

        relation_code = str(item[1])
        register_code = str(item[5])
        if relation_code not in RELATION_CODES:
            raise ValueError(f"Invalid relation code: {relation_code!r}")
        if register_code not in REGISTER_CODES:
            raise ValueError(f"Invalid register code: {register_code!r}")

        relation = RELATION_CODES[relation_code]
        register = REGISTER_CODES[register_code]
        if relation not in RELATIONS or register not in REGISTERS:
            raise ValueError("Compact code resolved outside V3 schema.")

        relatedness = int(item[2])
        replaceability = int(item[3])
        confidence_raw = int(item[4])

        if not 0 <= relatedness <= 4:
            raise ValueError(f"semantic_relatedness out of range for id {candidate_id}")
        if not 0 <= replaceability <= 3:
            raise ValueError(f"replaceability out of range for id {candidate_id}")
        if not 0 <= confidence_raw <= 100:
            raise ValueError(f"confidence out of range for id {candidate_id}")

        if relation == "antonym":
            replaceability = 0

        word = expected_words[candidate_id]
        by_word[word] = {
            "relation": relation,
            "semantic_relatedness": relatedness,
            "replaceability": replaceability,
            "confidence": round(confidence_raw / 100.0, 4),
            "register": register,
            "reason": "",
            "label_source": LOCAL_LABEL_SOURCE,
        }

    missing_ids = [
        index for index in range(len(expected_words)) if index not in seen_ids
    ]
    if missing_ids:
        raise ValueError(f"Missing compact candidate ids: {missing_ids}")
    return by_word


def _load_model(model_id: str, *, four_bit: bool):
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Local teacher dependencies are missing. Install requirements.txt first."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "V3.1 local Qwen teacher requires a CUDA GPU for the recommended path. "
            "Enable a Colab GPU runtime."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs["dtype"] = torch.float16
    else:
        kwargs["dtype"] = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return tokenizer, model


def _chat_text(tokenizer, prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "ตอบตาม schema เท่านั้น ห้ามอธิบายนอก JSON "
                "ห้ามเพิ่มหรือลด candidate"
            ),
        },
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def _generate_batch(
    tokenizer,
    model,
    prompts: list[str],
    *,
    max_new_tokens: int,
) -> list[str]:
    import torch

    texts = [_chat_text(tokenizer, prompt) for prompt in prompts]
    inputs = tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    input_width = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return [
        tokenizer.decode(
            output[input_width:],
            skip_special_tokens=True,
        ).strip()
        for output in outputs
    ]


def _apply_judgments(
    row: dict[str, Any],
    judgments: dict[str, dict[str, Any]],
    *,
    model_id: str,
) -> dict[str, Any]:
    output = json.loads(json.dumps(row, ensure_ascii=False))
    for candidate in output["candidates"]:
        if isinstance(candidate.get("judgment"), dict):
            continue
        word = candidate["word"]
        candidate["judgment"] = judgments[word]
        candidate.pop("route", None)

    output["teacher"] = {
        "provider": "local",
        "model": model_id,
        "version": "3.1-compact-batch",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return output


def _rules_only_row(row: dict[str, Any]) -> dict[str, Any]:
    output = json.loads(json.dumps(row, ensure_ascii=False))
    output["teacher"] = {
        "provider": "rules_only",
        "version": "3.1-compact-batch",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return output


def _repair_one(
    row: dict[str, Any],
    raw: str,
    tokenizer,
    model,
    *,
    max_new_tokens: int,
) -> dict[str, dict[str, Any]]:
    expected_words = [candidate["word"] for candidate in local_candidates(row)]
    prompt = (
        _prompt(row)
        + "\n\nคำตอบก่อนหน้าผิด schema ให้ตอบใหม่เป็น JSON compact ที่ถูกต้องเท่านั้น:"
        + "\n"
        + raw
    )
    repaired = _generate_batch(
        tokenizer,
        model,
        [prompt],
        max_new_tokens=max_new_tokens,
    )[0]
    return _validate_compact_judgments(
        _extract_json(repaired),
        expected_words,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label routed V3.1 candidates locally with batched compact Qwen output."
    )
    parser.add_argument(
        "--input",
        default="artifacts/v3_1/routed_teacher_seeds.jsonl",
    )
    parser.add_argument(
        "--output",
        default="artifacts/v3_1/local_teacher_labels.jsonl",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of dictionary senses generated together. Use 2 if GPU memory is tight.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=320,
        help="Compact output budget for <=8 candidates.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Load the local teacher without bitsandbytes 4-bit quantization.",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    rows = read_jsonl(args.input)
    if not rows:
        raise SystemExit(f"No routed seeds found at {args.input}")

    completed = {
        row.get("seed_id")
        for row in read_jsonl(args.output)
        if row.get("seed_id")
    }
    pending = [row for row in rows if row.get("seed_id") not in completed]
    if args.limit > 0:
        pending = pending[: args.limit]

    if not pending:
        print(
            json.dumps(
                {
                    "pending": 0,
                    "resume_existing": len(completed),
                    "output": args.output,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    rules_only = [row for row in pending if not local_candidates(row)]
    llm_rows = [row for row in pending if local_candidates(row)]

    for row in rules_only:
        append_jsonl(args.output, _rules_only_row(row))

    tokenizer = model = None
    load_seconds = 0.0
    if llm_rows:
        load_start = time.perf_counter()
        tokenizer, model = _load_model(
            args.model,
            four_bit=not args.no_4bit,
        )
        load_seconds = time.perf_counter() - load_start

    success = len(rules_only)
    failures = 0
    repairs = 0
    generation_start = time.perf_counter()

    for start in range(0, len(llm_rows), args.batch_size):
        batch = llm_rows[start : start + args.batch_size]
        prompts = [_prompt(row) for row in batch]

        try:
            raws = _generate_batch(
                tokenizer,
                model,
                prompts,
                max_new_tokens=args.max_new_tokens,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "CUDA out of memory during batched generation. "
                    "Retry with --batch-size 2 or --batch-size 1."
                ) from exc
            raise

        for offset, (row, raw) in enumerate(zip(batch, raws), start=1):
            ordinal = start + offset
            expected_words = [
                candidate["word"] for candidate in local_candidates(row)
            ]
            try:
                judgments = _validate_compact_judgments(
                    _extract_json(raw),
                    expected_words,
                )
            except Exception:
                repairs += 1
                try:
                    judgments = _repair_one(
                        row,
                        raw,
                        tokenizer,
                        model,
                        max_new_tokens=args.max_new_tokens,
                    )
                except Exception as exc:
                    failures += 1
                    print(
                        f"[{ordinal}/{len(llm_rows)}] ERROR "
                        f"{row.get('seed_id')}: {exc}",
                        file=sys.stderr,
                    )
                    continue

            labeled = _apply_judgments(
                row,
                judgments,
                model_id=args.model,
            )
            append_jsonl(args.output, labeled)
            success += 1
            print(
                f"[{ordinal}/{len(llm_rows)}] labeled {row.get('seed_id')} "
                f"({row['anchor']['word']})",
                file=sys.stderr,
            )

    generation_seconds = time.perf_counter() - generation_start
    llm_completed = max(0, success - len(rules_only))
    senses_per_minute = (
        (llm_completed / generation_seconds) * 60.0
        if generation_seconds > 0
        else None
    )

    print(
        json.dumps(
            {
                "version": "3.1-compact-batch",
                "model": args.model,
                "four_bit": not args.no_4bit,
                "batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
                "selected": len(pending),
                "rules_only": len(rules_only),
                "successes": success,
                "failures": failures,
                "repair_attempts": repairs,
                "resume_existing": len(completed),
                "model_load_seconds": round(load_seconds, 2),
                "generation_seconds": round(generation_seconds, 2),
                "llm_senses_per_minute": (
                    round(senses_per_minute, 3)
                    if senses_per_minute is not None
                    else None
                ),
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
