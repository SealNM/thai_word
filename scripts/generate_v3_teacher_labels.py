#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_v3_data import RELATIONS, REGISTERS, append_jsonl, read_jsonl


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "judgments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_word": {"type": "string"},
                        "relation": {
                            "type": "string",
                            "enum": sorted(RELATIONS),
                        },
                        "semantic_relatedness": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 4,
                        },
                        "replaceability": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 3,
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "register": {
                            "type": "string",
                            "enum": sorted(REGISTERS),
                        },
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "candidate_word",
                        "relation",
                        "semantic_relatedness",
                        "replaceability",
                        "confidence",
                        "register",
                        "reason",
                    ],
                },
            }
        },
        "required": ["judgments"],
    }


def _prompt(seed: dict[str, Any]) -> str:
    anchor = seed["anchor"]
    candidates = [
        {
            "word": candidate["word"],
            "definition": candidate["definition"],
        }
        for candidate in seed["candidates"]
    ]

    return f"""
คุณเป็นผู้เชี่ยวชาญภาษาไทย ทำหน้าที่สร้างข้อมูลฝึกสำหรับระบบค้นคำของนักเขียนนิยาย

เป้าหมายไม่ใช่แค่วัดว่า 'เกี่ยวข้องกันไหม' แต่ต้องแยกให้ได้ว่าคำใดใช้แทนคำหลักได้จริง คำใดเป็นเพียงชนิดย่อย/ลักษณะวิธี/คำที่เกี่ยวข้อง และคำใดเป็นคำตรงข้าม

คำหลัก:
- คำ: {anchor["word"]}
- ความหมายที่กำลังพิจารณา (sense {anchor["sense"]}): {anchor["definition"]}

ให้ตัดสิน candidate ทุกคำด้านล่างตามความหมายที่ระบุเท่านั้น:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

relation ใช้ได้เฉพาะ:
- synonym: ความหมายเดียวกันหรือใช้แทนกันได้ตรงมาก
- near_synonym: ใกล้เคียงและมักใช้แทนกันได้ แต่มี nuance/register ต่าง
- antonym: ความหมายตรงข้ามหรือ contrast ชัด
- subtype: candidate เป็นชนิดย่อย/รูปแบบเฉพาะของคำหลัก
- supertype: candidate กว้างกว่าคำหลัก
- manner: เป็นวิธี/ลักษณะการกระทำ เช่น กระซิบ เทียบกับ พูด
- associated: เกี่ยวข้องกัน แต่ไม่ควรใช้แทนกัน
- unrelated: ไม่เกี่ยวข้องอย่างมีนัยสำคัญกับ sense นี้
- uncertain: หลักฐานไม่พอหรือกำกวมเกินไป

semantic_relatedness:
0 = ไม่เกี่ยวข้อง
1 = เกี่ยวข้องห่าง ๆ
2 = เกี่ยวข้องชัด แต่คนละแนวคิด
3 = ใกล้ความหมายมาก
4 = แทบเป็นความหมายเดียวกัน

replaceability:
0 = ใช้แทนกันไม่ได้
1 = ใช้แทนได้เฉพาะบริบทแคบมาก
2 = ใช้แทนกันได้หลายบริบท แต่มี nuance
3 = ใช้แทนกันได้โดยตรงเกือบทั้งหมดใน sense นี้

register เลือกหนึ่งค่า: {", ".join(sorted(REGISTERS))}

ข้อกำหนดสำคัญ:
1. ต้องให้ผลครบทุก candidate และ candidate_word ต้องสะกดตรงกับรายการที่ให้มา
2. อย่าสร้าง candidate ใหม่
3. อย่าตัดสินจากรูปคำเพียงอย่างเดียว ให้อิงนิยาม
4. คำที่อยู่ในประโยค/สถานการณ์เดียวกันแต่แทนกันไม่ได้ ให้เป็น associated ไม่ใช่ synonym
5. antonym อาจมี semantic_relatedness สูงได้เพราะอยู่แกนความหมายเดียวกัน แต่ replaceability ต้องเป็น 0
6. ถ้าไม่แน่ใจให้ใช้ uncertain และลด confidence
7. reason สั้น กระชับ ไม่เกินหนึ่งประโยค
""".strip()


def _attach_judgments(
    seed: dict[str, Any],
    result: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    judgments = result.get("judgments")
    if not isinstance(judgments, list):
        raise ValueError("Teacher response is missing judgments array.")

    expected = [candidate["word"] for candidate in seed["candidates"]]
    by_word: dict[str, dict[str, Any]] = {}
    for judgment in judgments:
        if not isinstance(judgment, dict):
            raise ValueError("Every teacher judgment must be an object.")
        word = str(judgment.get("candidate_word", ""))
        if word in by_word:
            raise ValueError(f"Duplicate teacher judgment for {word!r}.")
        by_word[word] = judgment

    missing = [word for word in expected if word not in by_word]
    extra = [word for word in by_word if word not in set(expected)]
    if missing or extra:
        raise ValueError(
            f"Teacher candidate mismatch. missing={missing[:5]} extra={extra[:5]}"
        )

    output = dict(seed)
    output_candidates = []
    for candidate in seed["candidates"]:
        item = dict(candidate)
        judgment = dict(by_word[candidate["word"]])
        judgment.pop("candidate_word", None)
        item["judgment"] = judgment
        output_candidates.append(item)

    output["candidates"] = output_candidates
    output["teacher"] = {
        "provider": "gemini",
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label grounded V3 teacher candidates with Gemini structured output."
    )
    parser.add_argument(
        "--input",
        default="artifacts/v3/teacher_seeds.jsonl",
    )
    parser.add_argument(
        "--output",
        default="artifacts/v3/teacher_labels.jsonl",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("THAI_WORD_TEACHER_MODEL", "gemini-3.8-flash"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum new teacher calls this run. Use 0 for all remaining seeds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first prompt/schema without calling Gemini.",
    )
    args = parser.parse_args()

    seeds = read_jsonl(args.input)
    if not seeds:
        raise SystemExit(f"No teacher seeds found at {args.input}")

    if args.dry_run:
        print(_prompt(seeds[0]))
        print("\n--- RESPONSE SCHEMA ---")
        print(json.dumps(_response_schema(), ensure_ascii=False, indent=2))
        return

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY is not set. Put the key in a Colab secret/environment "
            "variable; never commit it to the repository."
        )

    try:
        from google import genai
    except ImportError as exc:
        raise SystemExit(
            "google-genai is required. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    completed = {
        row.get("seed_id")
        for row in read_jsonl(args.output)
        if row.get("seed_id")
    }

    pending = [seed for seed in seeds if seed.get("seed_id") not in completed]
    if args.limit > 0:
        pending = pending[: args.limit]

    client = genai.Client(api_key=api_key)
    successes = 0
    failures = 0

    for index, seed in enumerate(pending, start=1):
        seed_id = seed.get("seed_id", "<unknown>")
        try:
            interaction = client.interactions.create(
                model=args.model,
                input=_prompt(seed),
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": _response_schema(),
                },
            )
            parsed = json.loads(interaction.output_text)
            labeled = _attach_judgments(seed, parsed, model=args.model)
            append_jsonl(args.output, labeled)
            successes += 1
            print(
                f"[{index}/{len(pending)}] labeled {seed_id} "
                f"({seed['anchor']['word']})",
                file=sys.stderr,
            )
        except Exception as exc:
            failures += 1
            print(
                f"[{index}/{len(pending)}] ERROR {seed_id}: {exc}",
                file=sys.stderr,
            )

    summary = {
        "model": args.model,
        "pending_selected": len(pending),
        "successes": successes,
        "failures": failures,
        "output": args.output,
        "resume_existing": len(completed),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
