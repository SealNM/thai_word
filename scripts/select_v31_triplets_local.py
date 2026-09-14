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

from thai_v3_data import append_jsonl, read_jsonl
from thai_v31_router import AUTO_LABEL_SOURCE, local_candidates


JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _prompt(row: dict[str, Any]) -> str:
    anchor = row["anchor"]
    candidates = [
        [index, candidate["word"], candidate["definition"]]
        for index, candidate in enumerate(local_candidates(row))
    ]
    return f"""
เลือกข้อมูล contrastive training สำหรับระบบค้นคำภาษาไทย

คำหลัก: {anchor["word"]}
ความหมายที่ต้องยึด: {anchor["definition"]}

candidate [id,คำ,นิยาม]:
{json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))}

ให้เลือก:
- p = positive: คำที่ใช้แทนคำหลักได้จริงในความหมายนี้อย่างปลอดภัย
- n = hard negative: คำที่เกี่ยวข้อง/ใกล้จนระบบค้นอาจสับสน แต่ใช้แทนคำหลักไม่ได้
- ถ้าไม่มั่นใจ ไม่ต้องเลือก

กฎ:
1. positive ต้องเข้มงวดมาก อย่าเลือกเพียงเพราะอยู่หัวข้อเดียวกัน
2. ชนิดย่อย วัตถุที่เกี่ยวข้อง ผู้กระทำ สถานที่ หรือคำที่เป็นเพียงส่วนประกอบ ไม่ใช่ positive เว้นแต่ใช้แทนได้จริง
3. hard negative ควรใกล้ความหมายแต่ไม่ interchangeable; อย่าเลือกคำที่ไม่เกี่ยวข้องเลยถ้ามีตัวเลือกที่ดีกว่า
4. เลือก positive ได้ 0-2 ตัว และ hard negative ได้ 0-2 ตัว
5. id ห้ามซ้ำระหว่าง p และ n
6. confidence 0-100; เลือกเฉพาะที่มั่นใจ >=80

ตอบ JSON เท่านั้น:
{{"p":[[id,confidence],...],"n":[[id,confidence],...]}}
""".strip()


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = JSON_FENCE_RE.sub("", text.strip()).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Teacher did not return a JSON object.")
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Teacher JSON must be an object.")
    return parsed


def _validate_selection(
    result: dict[str, Any],
    candidate_count: int,
) -> dict[str, list[list[int]]]:
    output: dict[str, list[list[int]]] = {}
    used: set[int] = set()

    for key in ("p", "n"):
        values = result.get(key, [])
        if not isinstance(values, list) or len(values) > 2:
            raise ValueError(f"{key} must be an array with at most 2 choices.")

        parsed: list[list[int]] = []
        for item in values:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(f"{key} choice must be [id,confidence].")
            candidate_id = int(item[0])
            confidence = int(item[1])
            if not 0 <= candidate_id < candidate_count:
                raise ValueError(f"candidate id out of range: {candidate_id}")
            if not 80 <= confidence <= 100:
                raise ValueError(
                    f"selected candidate confidence must be 80-100: {confidence}"
                )
            if candidate_id in used:
                raise ValueError(f"candidate id selected more than once: {candidate_id}")
            used.add(candidate_id)
            parsed.append([candidate_id, confidence])
        output[key] = parsed

    return output


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
        raise RuntimeError("A CUDA GPU is required for the recommended local teacher path.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    kwargs: dict[str, Any] = {"device_map": "auto", "low_cpu_mem_usage": True}
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
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "ตอบ JSON ตาม schema เท่านั้น และเลือกแบบ conservative",
            },
            {"role": "user", "content": prompt},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )


def _generate_batch(tokenizer, model, prompts: list[str], *, max_new_tokens: int) -> list[str]:
    import torch

    texts = [_chat_text(tokenizer, prompt) for prompt in prompts]
    inputs = tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    width = inputs["input_ids"].shape[1]

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
        tokenizer.decode(output[width:], skip_special_tokens=True).strip()
        for output in outputs
    ]


def _apply_selection(row: dict[str, Any], selection: dict[str, list[list[int]]], model_id: str) -> dict[str, Any]:
    output = json.loads(json.dumps(row, ensure_ascii=False))
    pending = local_candidates(output)

    selected = {"positive": [], "hard_negative": []}
    for key, target in (("p", "positive"), ("n", "hard_negative")):
        for candidate_id, confidence in selection[key]:
            candidate = pending[candidate_id]
            selected[target].append(
                {
                    "word": candidate["word"],
                    "definition": candidate["definition"],
                    "confidence": confidence / 100.0,
                    "source": "qwen_local_selector",
                    "evidence": candidate.get("evidence", {}),
                }
            )

    # Preserve only safe rule-auto positives as additional positives.
    for candidate in output.get("candidates", []):
        judgment = candidate.get("judgment")
        if (
            isinstance(judgment, dict)
            and judgment.get("label_source") == AUTO_LABEL_SOURCE
            and judgment.get("relation") == "synonym"
        ):
            selected["positive"].append(
                {
                    "word": candidate["word"],
                    "definition": candidate["definition"],
                    "confidence": float(judgment.get("confidence", 0.99)),
                    "source": AUTO_LABEL_SOURCE,
                    "evidence": candidate.get("evidence", {}),
                }
            )

    output["selection"] = selected
    output["selection_teacher"] = {
        "provider": "local",
        "model": model_id,
        "version": "3.1-selector",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select V3.1 positives and hard negatives directly with local Qwen."
    )
    parser.add_argument("--input", default="artifacts/v3_1/routed_teacher_seeds.jsonl")
    parser.add_argument("--output", default="artifacts/v3_1/local_triplet_selections.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

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
        print(json.dumps({"pending": 0, "resume_existing": len(completed)}, indent=2))
        return

    load_start = time.perf_counter()
    tokenizer, model = _load_model(args.model, four_bit=not args.no_4bit)
    load_seconds = time.perf_counter() - load_start

    success = failures = repairs = 0
    generation_start = time.perf_counter()

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        raws = _generate_batch(
            tokenizer,
            model,
            [_prompt(row) for row in batch],
            max_new_tokens=args.max_new_tokens,
        )

        for offset, (row, raw) in enumerate(zip(batch, raws), start=1):
            ordinal = start + offset
            count = len(local_candidates(row))
            try:
                selection = _validate_selection(_extract_json(raw), count)
            except Exception:
                repairs += 1
                try:
                    repair_prompt = (
                        _prompt(row)
                        + "\nคำตอบก่อนหน้าผิด schema ตอบ JSON ใหม่เท่านั้น:\n"
                        + raw
                    )
                    repaired = _generate_batch(
                        tokenizer,
                        model,
                        [repair_prompt],
                        max_new_tokens=args.max_new_tokens,
                    )[0]
                    selection = _validate_selection(_extract_json(repaired), count)
                except Exception as exc:
                    failures += 1
                    print(
                        f"[{ordinal}/{len(pending)}] ERROR {row.get('seed_id')}: {exc}",
                        file=sys.stderr,
                    )
                    continue

            append_jsonl(
                args.output,
                _apply_selection(row, selection, args.model),
            )
            success += 1
            print(
                f"[{ordinal}/{len(pending)}] selected {row.get('seed_id')} "
                f"({row['anchor']['word']})",
                file=sys.stderr,
            )

    generation_seconds = time.perf_counter() - generation_start
    rate = success / generation_seconds * 60 if generation_seconds else None
    print(
        json.dumps(
            {
                "version": "3.1-selector",
                "selected_tasks": len(pending),
                "successes": success,
                "failures": failures,
                "repair_attempts": repairs,
                "batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
                "model_load_seconds": round(load_seconds, 2),
                "generation_seconds": round(generation_seconds, 2),
                "senses_per_minute": round(rate, 3) if rate is not None else None,
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
