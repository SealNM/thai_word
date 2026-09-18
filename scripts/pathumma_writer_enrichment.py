#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thai_substitutability import (
    SEMANTIC_RELATIONS,
    SEVERE_ERROR_RELATIONS,
    STYLE_TAGS,
    read_jsonl,
    validate_rows,
)

DEFAULT_MODEL_ID = "nectec/pathumma-llm-4b-think-4.0.0"
EXPERIMENT_SCHEMA_VERSION = 3
PROMPT_VERSION = 3
EVIDENCE_STATUSES = {
    "sufficient",
    "ambiguous",
    "cross_reference_only",
    "missing",
}
FORBIDDEN_INPUT_MARKERS = (
    "writer_relevance_phase5_holdout",
    "phase5_acceptance",
)

SYSTEM_PROMPT = """คุณเป็นผู้ช่วยตรวจความสัมพันธ์เชิงความหมายของคำภาษาไทยสำหรับนักเขียน
หน้าที่นี้เป็น semantic judge เท่านั้น ไม่ต้องจำแนก style/register/mood/usage ในรอบนี้
ให้ตัดสินจากข้อมูลที่ให้มาเท่านั้น ห้ามเติมความรู้หรือเดาความหมายที่ไม่ได้อยู่ใน definition

เกณฑ์ utility:
3 = ใช้แทนหรือช่วยนักเขียนได้ตรงมากในความหมายนี้
2 = เกี่ยวข้องและมีประโยชน์ชัดเจน แต่แทนตรง ๆ ไม่ได้เสมอ
1 = เกี่ยวข้องอ่อน ๆ หรือใช้ได้เฉพาะบริบทจำกัด
0 = ไม่ช่วยในความหมายเป้าหมายนี้ หรือข้อมูลไม่พอให้ยืนยันว่าช่วยได้

semantic_relation:
- direct = ความหมายตรงกันหรือใกล้เคียงมากใน sense เป้าหมาย
- subtype = candidate เป็นชนิดย่อย/กรณีเฉพาะที่แคบกว่าคำค้น
- broader_concept = candidate เป็นแนวคิดหรือหมวดที่กว้างกว่าคำค้น
- manner_action = candidate เป็นอาการ วิธีการ หรือการกระทำที่เกี่ยวข้องกับคำค้น
- scene_context = candidate เป็นสิ่ง/เหตุการณ์ที่มักอยู่ในฉากหรือบริบทเดียวกัน
- effect_state = candidate เป็นผล สภาพ หรือภาวะที่เกิดตามมาจากคำค้น
- weak_related = เกี่ยวข้องทางความหมายแบบอ้อมหรืออ่อน แต่ยังมีประโยชน์บางบริบท
- opposite_misleading = ความหมายตรงข้ามหรือทำให้ตีความกลับด้าน
- sense_mismatch = ตัวคำดูเกี่ยวข้อง แต่ sense ที่ definition ระบุไม่ตรงกับ sense เป้าหมาย
- unrelated = definition มีข้อมูลเพียงพอและแสดงว่าไม่มีความสัมพันธ์เชิงความหมายที่เป็นประโยชน์
- unclear = definition ไม่มีข้อมูลพอหรือกำกวมจนจำแนกความสัมพันธ์ไม่ได้

กฎ evidence สำคัญ:
- candidate มี definition_raw, definition_head และ definition_status
- definition_head คือส่วนหัวของคำนิยามที่ตัดตัวอย่างหลังคำว่า "เช่น" ออก เพื่อช่วยไม่ให้ตัวอย่างเก่ากลบ gloss หลัก
- ถ้า definition_status = cross_reference_only หรือ missing: ต้องตอบ utility = 0 และ semantic_relation = unclear ห้ามเดาว่าคำอ้างอิงนั้นหมายถึงอะไร
- ถ้า definition_head ระบุคำค้นหรือคำพ้องตรง ๆ เช่น "แห้ง, พร่อง, ลดลง" ให้ถือว่าเป็นหลักฐานสำคัญของความหมาย อย่าปัดทิ้งเพราะตัวอย่างช่วงท้ายกล่าวถึงอีกบริบท
- utility = 0 ไม่ได้แปลว่าต้องเป็น unrelated/sense_mismatch; หากข้อมูลไม่พอให้ใช้ unclear
- ใช้ unrelated เฉพาะเมื่อ definition มีข้อมูลพอจริง ๆ
- ใช้ sense_mismatch เมื่อ definition ระบุ sense อื่นชัดเจน
- evidence_quote ต้องคัดข้อความสั้น ๆ จาก candidate definition จริง ห้ามแต่งข้อความใหม่
- reason ต้องอธิบายจาก evidence_quote เท่านั้น
- confidence อยู่ระหว่าง 0 ถึง 1 และหมายถึงความมั่นใจในการตัดสินจากหลักฐานที่มี
- opposite_misleading, sense_mismatch, unrelated ต้องมี utility = 0
- final answer ต้องเป็น JSON object เท่านั้น ห้ามมี markdown
"""


def normalize_definition_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.strip()


def definition_head(value: Any) -> str:
    text = normalize_definition_text(value)
    if not text:
        return ""
    head = text.split("เช่น", 1)[0].strip()
    return head.rstrip(" ,;:")


def definition_status(value: Any) -> str:
    text = normalize_definition_text(value)
    if not text:
        return "missing"
    if re.fullmatch(r"ดู(?:ใน)?\s+.+?[.]?", text):
        return "cross_reference_only"
    return "substantive"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_historical_input(path: str | Path) -> None:
    lowered = str(path).lower()
    if any(marker in lowered for marker in FORBIDDEN_INPUT_MARKERS):
        raise ValueError(
            "Refusing Phase-5 fresh/acceptance holdout. "
            "Use historical/consumed labeled data only."
        )


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_id"])].append(row)
    return dict(grouped)


def choose_query_ids(
    rows: list[dict[str, Any]],
    *,
    query_limit: int | None,
    seed: int,
) -> list[str]:
    query_ids = sorted(group_rows(rows))
    random.Random(seed).shuffle(query_ids)
    return query_ids if query_limit is None else query_ids[:query_limit]


def shuffled_query_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    query_id = str(rows[0]["query_id"])
    digest = hashlib.sha256(f"{seed}:{query_id}".encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], "big")
    copied = list(rows)
    random.Random(local_seed).shuffle(copied)
    return copied


def iter_batches(rows: list[dict[str, Any]], size: int):
    if size < 1:
        raise ValueError("pairs-per-prompt must be >= 1")
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def build_prompt(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("Cannot build prompt from empty rows.")

    query = rows[0]["query"]
    payload = {
        "query": {
            "word": query.get("word"),
            "sense": query.get("sense"),
            "definition": query.get("definition"),
            "intended": query.get("intended"),
            "category": query.get("category"),
        },
        "candidates": [
            {
                "pair_id": row["pair_id"],
                "word": row["candidate"].get("word"),
                "sense": row["candidate"].get("sense"),
                "definition_raw": row["candidate"].get("definition"),
                "definition_head": definition_head(row["candidate"].get("definition")),
                "definition_status": definition_status(row["candidate"].get("definition")),
            }
            for row in rows
        ],
    }

    output_schema = (
        "FINAL_JSON_SCHEMA:\n"
        "{\n"
        '  "predictions": [\n'
        "    {\n"
        '      "pair_id": string จาก INPUT_JSON เท่านั้น,\n'
        '      "utility": integer 0..3,\n'
        '      "semantic_relation": หนึ่งค่าใน relation ที่อนุญาต,\n'
        '      "evidence_status": sufficient | ambiguous | cross_reference_only | missing,\n'
        '      "evidence_quote": string ที่คัดจาก candidate definition จริง,\n'
        '      "confidence": number 0..1,\n'
        '      "reason": string เหตุผลสั้น ๆ ที่อิง evidence_quote\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "schema นี้ระบุชนิดข้อมูล ไม่ใช่ตัวอย่างคำตอบ"
    )

    return (
        "ประเมิน candidate ทุกตัว ต้องคืน pair_id ให้ครบและห้ามเพิ่ม candidate ใหม่\n\n"
        + "INPUT_JSON:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\n"
        + output_schema
    )


def split_reasoning_and_final(text: str) -> tuple[str, str]:
    if "</think>" not in text:
        return "", text.strip()
    reasoning, final = text.rsplit("</think>", 1)
    return reasoning.replace("<think>", "", 1).strip(), final.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)

    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("No JSON object found in final answer.")
        value = json.loads(candidate[start : end + 1])

    if not isinstance(value, dict):
        raise ValueError("Final answer JSON must be an object.")
    return value


def validate_prediction(
    item: Any,
    expected_pair_ids: set[str],
    source_by_id: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(item, dict):
        return None, ["prediction must be an object"]

    errors: list[str] = []
    pair_id = str(item.get("pair_id") or "")
    utility = item.get("utility")
    relation = str(item.get("semantic_relation") or "")
    confidence = item.get("confidence")
    evidence_status = str(item.get("evidence_status") or "")
    evidence_quote = normalize_definition_text(item.get("evidence_quote"))

    if pair_id not in expected_pair_ids:
        errors.append(f"unknown pair_id {pair_id!r}")
    if isinstance(utility, bool) or utility not in {0, 1, 2, 3}:
        errors.append("utility must be integer 0..3")
    if relation not in SEMANTIC_RELATIONS:
        errors.append(f"unsupported semantic_relation {relation!r}")
    if relation in SEVERE_ERROR_RELATIONS and utility in {1, 2, 3}:
        errors.append(f"{relation} requires utility 0")
    if evidence_status not in EVIDENCE_STATUSES:
        errors.append(f"unsupported evidence_status {evidence_status!r}")

    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        errors.append("confidence must be numeric")
        confidence_value = None
    else:
        confidence_value = float(confidence)
        if not 0 <= confidence_value <= 1:
            errors.append("confidence must be within [0,1]")

    reason = str(item.get("reason") or "").strip()
    if not reason:
        errors.append("reason must be a non-empty string")

    if source_by_id is not None and pair_id in source_by_id:
        source = source_by_id[pair_id]
        definition = source["candidate"].get("definition")
        source_status = definition_status(definition)
        normalized_definition = normalize_definition_text(definition)

        if source_status == "cross_reference_only":
            if evidence_status != "cross_reference_only":
                errors.append("cross-reference-only definition requires matching evidence_status")
            if relation != "unclear":
                errors.append("cross-reference-only definition requires semantic_relation unclear")
            if utility != 0:
                errors.append("cross-reference-only definition requires utility 0")
        elif source_status == "missing":
            if evidence_status != "missing":
                errors.append("missing definition requires evidence_status missing")
            if relation != "unclear":
                errors.append("missing definition requires semantic_relation unclear")
            if utility != 0:
                errors.append("missing definition requires utility 0")

        if source_status != "missing":
            if not evidence_quote:
                errors.append("evidence_quote must be non-empty when definition exists")
            elif evidence_quote not in normalized_definition:
                errors.append("evidence_quote must be copied from candidate definition")

    normalized = {
        "pair_id": pair_id,
        "utility": utility,
        "semantic_relation": relation,
        "evidence_status": evidence_status,
        "evidence_quote": evidence_quote,
        "confidence": confidence_value,
        "reason": reason,
        "style_tags": [],
        "mood_tags": [],
        "usage_tags": [],
    }
    return normalized, errors

def parse_predictions(
    final_text: str,
    expected_pair_ids: set[str],
    source_by_id: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    payload = extract_json_object(final_text)
    items = payload.get("predictions")
    if not isinstance(items, list):
        raise ValueError("Final JSON must contain predictions list.")

    parsed: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()

    for index, item in enumerate(items):
        normalized, item_errors = validate_prediction(
            item,
            expected_pair_ids,
            source_by_id,
        )
        if item_errors:
            errors.extend(f"prediction[{index}]: {msg}" for msg in item_errors)
            continue
        assert normalized is not None
        pair_id = normalized["pair_id"]
        if pair_id in seen:
            errors.append(f"prediction[{index}]: duplicate pair_id {pair_id}")
            continue
        seen.add(pair_id)
        parsed.append(normalized)

    missing = sorted(expected_pair_ids - seen)
    if missing:
        errors.append("missing pair_id(s): " + ", ".join(missing))

    return parsed, errors


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_output_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{target}:{line_no}: row must be object")
            rows.append(value)
    return rows


def binary_stats(gold: list[bool], pred: list[bool]) -> dict[str, Any]:
    tp = sum(g and p for g, p in zip(gold, pred))
    tn = sum((not g) and (not p) for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    fn = sum(g and (not p) for g, p in zip(gold, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(gold) if gold else 0.0,
    }


def confusion_matrix(gold: list[Any], pred: list[Any]) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for gold_value, pred_value in zip(gold, pred):
        matrix[str(gold_value)][str(pred_value)] += 1
    return {
        gold_value: dict(sorted(counts.items()))
        for gold_value, counts in sorted(matrix.items())
    }


def style_jaccard(gold_tags: list[str], pred_tags: list[str]) -> float:
    gold_set = set(gold_tags)
    pred_set = set(pred_tags)
    union = gold_set | pred_set
    if not union:
        return 1.0
    return len(gold_set & pred_set) / len(union)


def evaluate(
    gold_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    gold = {str(row["pair_id"]): row for row in gold_rows}
    pred = {
        str(row["pair_id"]): row
        for row in predictions
        if str(row.get("pair_id", "")) in gold
    }
    ids = [pair_id for pair_id in gold if pair_id in pred]
    attempted = len(gold)
    if not ids:
        return {
            "attempted_pair_count": attempted,
            "valid_prediction_count": 0,
            "valid_prediction_coverage": 0.0,
        }

    gu = [int(gold[i]["annotation"]["utility"]) for i in ids]
    pu = [int(pred[i]["utility"]) for i in ids]
    gr = [str(gold[i]["annotation"]["semantic_relation"]) for i in ids]
    pr = [str(pred[i]["semantic_relation"]) for i in ids]
    relation_totals = Counter(gr)
    relation_correct = Counter(g for g, p in zip(gr, pr) if g == p)

    return {
        "attempted_pair_count": attempted,
        "valid_prediction_count": len(ids),
        "valid_prediction_coverage": len(ids) / attempted if attempted else 0.0,
        "utility_accuracy": sum(g == p for g, p in zip(gu, pu)) / len(ids),
        "utility_mae": sum(abs(g - p) for g, p in zip(gu, pu)) / len(ids),
        "utility_within_1_accuracy": (
            sum(abs(g - p) <= 1 for g, p in zip(gu, pu)) / len(ids)
        ),
        "relation_accuracy": sum(g == p for g, p in zip(gr, pr)) / len(ids),
        "relation_confusion_matrix": confusion_matrix(gr, pr),
        "utility_confusion_matrix": confusion_matrix(gu, pu),
        "relation_accuracy_by_gold": {
            relation: relation_correct[relation] / count
            for relation, count in sorted(relation_totals.items())
        },
        "useful_binary": binary_stats(
            [value >= 1 for value in gu],
            [value >= 1 for value in pu],
        ),
        "high_utility_binary": binary_stats(
            [value >= 2 for value in gu],
            [value >= 2 for value in pu],
        ),
        "severe_error_binary": binary_stats(
            [value in SEVERE_ERROR_RELATIONS for value in gr],
            [value in SEVERE_ERROR_RELATIONS for value in pr],
        ),
    }


def load_model(model_id: str, *, load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": "auto",
    }

    if load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("--load-in-4bit requires CUDA.")
        major, _minor = torch.cuda.get_device_capability()
        compute_dtype = torch.bfloat16 if major >= 8 else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return tokenizer, model


def generate(
    tokenizer,
    model,
    prompt: str,
    *,
    max_new_tokens: int,
    sample: bool,
    temperature: float,
    top_p: float,
) -> tuple[str, str, str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": sample,
    }
    if sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    outputs = model.generate(**inputs, **generation_kwargs)
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    reasoning, final = split_reasoning_and_final(raw)
    return raw, reasoning, final


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_historical_input(args.input)
    rows = read_jsonl(args.input)
    errors = validate_rows(rows, require_labels=True)
    if errors:
        raise ValueError(
            "Input must be valid labeled Writer Relevance v3:\n"
            + "\n".join(errors[:30])
        )

    grouped = group_rows(rows)
    query_ids = choose_query_ids(
        rows,
        query_limit=args.query_limit,
        seed=args.seed,
    )

    selected: list[dict[str, Any]] = []
    batches: list[list[dict[str, Any]]] = []
    for query_id in query_ids:
        query_rows = shuffled_query_rows(grouped[query_id], seed=args.seed)
        if args.candidates_per_query is not None:
            query_rows = query_rows[: args.candidates_per_query]
        selected.extend(query_rows)
        batches.extend(iter_batches(query_rows, args.pairs_per_prompt))

    existing = read_output_jsonl(args.predictions_output)
    completed_ids = {str(row.get("pair_id", "")) for row in existing}

    tokenizer = None
    model = None
    failed_batches = 0
    invalid_batches = 0
    fatal_error: str | None = None

    for batch_index, batch in enumerate(batches, start=1):
        pending = [
            row for row in batch
            if str(row["pair_id"]) not in completed_ids
        ]
        if not pending:
            continue

        expected_ids = {str(row["pair_id"]) for row in pending}
        source_by_id = {str(row["pair_id"]): row for row in pending}
        prompt = build_prompt(pending)
        raw = reasoning = final = ""
        parse_errors: list[str] = []
        parsed: list[dict[str, Any]] = []

        try:
            if tokenizer is None or model is None:
                tokenizer, model = load_model(
                    args.model_id,
                    load_in_4bit=args.load_in_4bit,
                )

            raw, reasoning, final = generate(
                tokenizer,
                model,
                prompt,
                max_new_tokens=args.max_new_tokens,
                sample=args.sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            parsed, parse_errors = parse_predictions(
                final,
                expected_ids,
                source_by_id,
            )
        except Exception as exc:
            failed_batches += 1
            parse_errors = [f"{type(exc).__name__}: {exc}"]
            if tokenizer is None or model is None:
                fatal_error = parse_errors[0]

        if parse_errors:
            invalid_batches += 1

        append_jsonl(
            args.raw_output,
            {
                "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
                "prompt_version": PROMPT_VERSION,
                "batch_index": batch_index,
                "model_id": args.model_id,
                "pair_ids": sorted(expected_ids),
                "prompt": prompt if args.store_prompts else None,
                "raw_output": raw if args.store_reasoning else None,
                "reasoning_trace": reasoning if args.store_reasoning else None,
                "final_text": final,
                "parse_errors": parse_errors,
            },
        )

        gold_by_id = {str(row["pair_id"]): row for row in pending}
        for prediction in parsed:
            pair_id = prediction["pair_id"]
            source = gold_by_id[pair_id]
            append_jsonl(
                args.predictions_output,
                {
                    "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
                    "model_id": args.model_id,
                    "source_sha256": sha256_file(args.input),
                    "prompt_version": PROMPT_VERSION,
                    "query_id": source["query_id"],
                    "query_word": source["query"].get("word"),
                    "candidate_word": source["candidate"].get("word"),
                    "candidate_definition": source["candidate"].get("definition"),
                    "candidate_definition_head": definition_head(source["candidate"].get("definition")),
                    "candidate_definition_status": definition_status(source["candidate"].get("definition")),
                    **prediction,
                },
            )
            completed_ids.add(pair_id)

        print(
            f"[{batch_index}/{len(batches)}] "
            f"valid={len(parsed)}/{len(expected_ids)} "
            f"errors={len(parse_errors)}",
            flush=True,
        )

        if fatal_error is not None:
            break

    all_predictions = read_output_jsonl(args.predictions_output)
    selected_ids = {str(row["pair_id"]) for row in selected}
    selected_predictions = [
        row for row in all_predictions
        if str(row.get("pair_id", "")) in selected_ids
    ]

    gold_by_id = {str(row["pair_id"]): row for row in selected}
    prediction_by_id = {
        str(row["pair_id"]): row
        for row in selected_predictions
        if str(row.get("pair_id", "")) in gold_by_id
    }
    severe_false_positives: list[dict[str, Any]] = []
    for pair_id, prediction in prediction_by_id.items():
        source = gold_by_id[pair_id]
        gold_relation = str(source["annotation"]["semantic_relation"])
        pred_relation = str(prediction["semantic_relation"])
        if (
            gold_relation not in SEVERE_ERROR_RELATIONS
            and pred_relation in SEVERE_ERROR_RELATIONS
        ):
            severe_false_positives.append({
                "pair_id": pair_id,
                "query_word": source["query"].get("word"),
                "candidate_word": source["candidate"].get("word"),
                "gold_utility": source["annotation"].get("utility"),
                "pred_utility": prediction.get("utility"),
                "gold_relation": gold_relation,
                "pred_relation": pred_relation,
                "confidence": prediction.get("confidence"),
                "reason": prediction.get("reason"),
            })

    report = {
        "status": (
            "pathumma_historical_blind_judge_failed"
            if fatal_error
            else "pathumma_historical_blind_judge_experiment"
        ),
        "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model_id": args.model_id,
        "input_file": str(args.input),
        "input_sha256": sha256_file(args.input),
        "query_count": len(query_ids),
        "selected_pair_count": len(selected),
        "seed": args.seed,
        "load_in_4bit": args.load_in_4bit,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature if args.sample else None,
            "top_p": args.top_p if args.sample else None,
            "do_sample": args.sample,
            "mode": "sampled" if args.sample else "deterministic_greedy",
        },
        "semantic_judge_only": True,
        "style_register_enrichment_enabled": False,
        "blind_judge_policy": {
            "human_labels_shown_to_model": False,
            "v25_score_shown_to_model": False,
            "v25_rank_shown_to_model": False,
            "retrieval_relation_hint_shown_to_model": False,
            "candidate_order": "deterministic_shuffle_per_query",
        },
        "failed_batch_count": failed_batches,
        "invalid_batch_count": invalid_batches,
        "fatal_error": fatal_error,
        "metrics": evaluate(selected, selected_predictions),
        "severe_error_false_positives": severe_false_positives,
        "development_only": True,
        "may_auto_promote_to_gold": False,
    }

    output = Path(args.report_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pathumma blind judge over historical Writer Relevance v3 data."
    )
    parser.add_argument(
        "--input",
        default="evaluation/writer_relevance_phase5_historical_70.approved.jsonl",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--query-limit", type=int, default=2)
    parser.add_argument("--candidates-per-query", type=int, default=5)
    parser.add_argument("--pairs-per-prompt", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sampled decoding. Default is deterministic greedy decoding.",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--predictions-output",
        default="experiments/pathumma/pathumma_predictions.jsonl",
    )
    parser.add_argument(
        "--raw-output",
        default="experiments/pathumma/pathumma_raw_generations.jsonl",
    )
    parser.add_argument(
        "--report-output",
        default="experiments/pathumma/pathumma_report.json",
    )
    parser.add_argument("--store-prompts", action="store_true")
    parser.add_argument("--store-reasoning", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get("fatal_error"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
