#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_v3_data import RELATIONS, REGISTERS, append_jsonl, read_jsonl
from thai_v31_router import LOCAL_LABEL_SOURCE, local_candidates


JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _prompt(row: dict[str, Any]) -> str:
    anchor = row["anchor"]
    candidates = [
        {
            "word": candidate["word"],
            "definition": candidate["definition"],
        }
        for candidate in local_candidates(row)
    ]

    return f"""
คุณเป็นผู้เชี่ยวชาญภาษาไทยและกำลังช่วยสร้างข้อมูลฝึกสำหรับระบบค้นคำของนักเขียน

คำหลัก:
- คำ: {anchor["word"]}
- sense {anchor["sense"]}: {anchor["definition"]}

ตัดสิน candidate ทุกคำต่อไปนี้ตาม sense นี้เท่านั้น:
{json.dumps(candidates, ensure_ascii=False)}

relation เลือกหนึ่งค่าเท่านั้น:
synonym, near_synonym, antonym, subtype, supertype, manner, associated, unrelated, uncertain

เกณฑ์:
- synonym = ใช้แทนกันได้ตรงมาก
- near_synonym = ใกล้เคียงและใช้แทนกันได้หลายบริบท แต่มี nuance/register ต่าง
- antonym = ตรงข้าม/contrast ชัด
- subtype = candidate เป็นชนิดย่อยหรือรูปเฉพาะของคำหลัก
- supertype = candidate กว้างกว่าคำหลัก
- manner = วิธี/ลักษณะเฉพาะของการกระทำ เช่น กระซิบ ต่อ พูด
- associated = เกี่ยวข้องกันแต่ใช้แทนกันไม่ได้
- unrelated = ไม่เกี่ยวข้องอย่างมีนัยสำคัญ
- uncertain = หลักฐานไม่พอ

semantic_relatedness: 0-4
replaceability: 0-3
confidence: 0.0-1.0
register: neutral, literary, formal, colloquial, archaic, technical, slang, unknown

คำเตือน:
- antonym อาจ related สูง แต่ replaceability ต้อง 0
- คำที่อยู่ในสถานการณ์เดียวกันไม่ได้แปลว่าเป็น synonym
- อย่าสร้าง candidate ใหม่
- ต้องตอบครบทุก candidate
- reason ใช้ภาษาไทยสั้น ๆ

ตอบเป็น JSON เท่านั้น รูปแบบ:
{{
  "judgments": [
    {{
      "candidate_word": "คำเดิมตามรายการ",
      "relation": "synonym",
      "semantic_relatedness": 4,
      "replaceability": 3,
      "confidence": 0.95,
      "register": "neutral",
      "reason": "เหตุผลสั้น ๆ"
    }}
  ]
}}
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


def _validate_judgments(
    result: dict[str, Any],
    expected_words: list[str],
) -> dict[str, dict[str, Any]]:
    judgments = result.get("judgments")
    if not isinstance(judgments, list):
        raise ValueError("Local teacher response is missing judgments array.")

    expected = set(expected_words)
    by_word: dict[str, dict[str, Any]] = {}

    for judgment in judgments:
        if not isinstance(judgment, dict):
            raise ValueError("Each judgment must be an object.")
        word = str(judgment.get("candidate_word", "")).strip()
        if word not in expected:
            raise ValueError(f"Unexpected candidate_word: {word!r}")
        if word in by_word:
            raise ValueError(f"Duplicate judgment for {word!r}")

        relation = judgment.get("relation")
        if relation not in RELATIONS:
            raise ValueError(f"Invalid relation for {word!r}: {relation!r}")

        relatedness = int(judgment.get("semantic_relatedness"))
        replaceability = int(judgment.get("replaceability"))
        confidence = float(judgment.get("confidence"))
        register = judgment.get("register", "unknown")

        if not 0 <= relatedness <= 4:
            raise ValueError(f"semantic_relatedness out of range for {word!r}")
        if not 0 <= replaceability <= 3:
            raise ValueError(f"replaceability out of range for {word!r}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence out of range for {word!r}")
        if register not in REGISTERS:
            raise ValueError(f"Invalid register for {word!r}: {register!r}")

        if relation == "antonym":
            replaceability = 0

        by_word[word] = {
            "relation": relation,
            "semantic_relatedness": relatedness,
            "replaceability": replaceability,
            "confidence": confidence,
            "register": register,
            "reason": str(judgment.get("reason", "")).strip(),
            "label_source": LOCAL_LABEL_SOURCE,
        }

    missing = [word for word in expected_words if word not in by_word]
    if missing:
        raise ValueError(f"Missing judgments: {missing}")
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
        if torch.cuda.is_bf16_supported():
            kwargs["dtype"] = torch.bfloat16
        else:
            kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return tokenizer, model


def _generate(
    tokenizer,
    model,
    prompt: str,
    *,
    max_new_tokens: int,
) -> str:
    import torch

    messages = [
        {
            "role": "system",
            "content": (
                "ตอบตามคำสั่งอย่างแม่นยำ ห้ามอธิบายนอก JSON "
                "และห้ามเพิ่ม candidate ที่ผู้ใช้ไม่ได้ให้"
            ),
        },
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _label_row(
    row: dict[str, Any],
    tokenizer,
    model,
    *,
    model_id: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    pending = local_candidates(row)
    if not pending:
        output = json.loads(json.dumps(row, ensure_ascii=False))
        output["teacher"] = {
            "provider": "rules_only",
            "version": "3.1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        return output

    expected_words = [candidate["word"] for candidate in pending]
    prompt = _prompt(row)
    raw = _generate(
        tokenizer,
        model,
        prompt,
        max_new_tokens=max_new_tokens,
    )

    try:
        judgments = _validate_judgments(_extract_json(raw), expected_words)
    except Exception:
        repair_prompt = (
            prompt
            + "\n\nคำตอบก่อนหน้ามีรูปแบบไม่ถูกต้อง แก้ให้เป็น JSON ที่ valid "
            "และต้องมี candidate ครบตามรายการเท่านั้น\nคำตอบก่อนหน้า:\n"
            + raw
        )
        repaired = _generate(
            tokenizer,
            model,
            repair_prompt,
            max_new_tokens=max_new_tokens,
        )
        judgments = _validate_judgments(
            _extract_json(repaired),
            expected_words,
        )

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
        "version": "3.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label routed V3.1 candidates locally with Qwen."
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
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Load the local teacher without bitsandbytes 4-bit quantization.",
    )
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

    needs_model = any(local_candidates(row) for row in pending)
    tokenizer = model = None
    if needs_model:
        tokenizer, model = _load_model(
            args.model,
            four_bit=not args.no_4bit,
        )

    success = 0
    failures = 0
    for index, row in enumerate(pending, start=1):
        try:
            labeled = _label_row(
                row,
                tokenizer,
                model,
                model_id=args.model,
                max_new_tokens=args.max_new_tokens,
            )
            append_jsonl(args.output, labeled)
            success += 1
            print(
                f"[{index}/{len(pending)}] labeled {row.get('seed_id')} "
                f"({row['anchor']['word']})",
                file=sys.stderr,
            )
        except Exception as exc:
            failures += 1
            print(
                f"[{index}/{len(pending)}] ERROR {row.get('seed_id')}: {exc}",
                file=sys.stderr,
            )

    print(
        json.dumps(
            {
                "version": "3.1",
                "model": args.model,
                "four_bit": not args.no_4bit,
                "selected": len(pending),
                "successes": success,
                "failures": failures,
                "resume_existing": len(completed),
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
