#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_hybrid_v2 import HybridSearcher
from thai_lexical_v1 import list_senses, load_artifacts
from thai_substitutability import (
    SCHEMA_VERSION,
    SEMANTIC_RELATIONS,
    STYLE_TAGS,
    benchmark_metrics,
    read_jsonl,
    stable_pair_id,
    validate_rows,
    write_jsonl,
)


SEMANTIC_RELATION_MENU = (
    ("direct", "คำแทนโดยตรง / ความหมายตรงกัน"),
    ("subtype", "ชนิดย่อย / รูปแบบเฉพาะของคำค้น"),
    ("broader_concept", "คำหรือแนวคิดที่กว้างกว่าคำค้น"),
    ("manner_action", "อาการ / การกระทำที่ใช้บรรยาย"),
    ("scene_context", "คำประกอบฉาก / บริบทที่เกี่ยวข้อง"),
    ("effect_state", "ผลที่เกิดขึ้น / สภาพที่เกี่ยวข้อง"),
    ("weak_related", "เกี่ยวข้องอยู่บ้าง แต่ค่อนข้างห่าง"),
    ("opposite_misleading", "ความหมายตรงข้าม / ชวนให้เข้าใจผิด"),
    ("sense_mismatch", "จับผิดความหมาย / คนละ sense"),
    ("unrelated", "ไม่เกี่ยวข้อง"),
    ("unclear", "ไม่แน่ใจ / ตัดสินไม่ได้"),
)

STYLE_TAG_MENU = (
    ("literary", "วรรณศิลป์ / ภาษาสละสลวย"),
    ("archaic", "โบราณ / เก่า"),
    ("formal", "ทางการ"),
    ("colloquial", "ภาษาพูด / กันเอง"),
    ("technical", "ศัพท์เฉพาะ / วิชาการ"),
    ("dialect", "ภาษาถิ่น"),
    ("figurative", "เชิงเปรียบเทียบ / ภาพพจน์"),
    ("other", "ลักษณะภาษาอื่น"),
    ("unknown", "ไม่แน่ใจ"),
)

if {item[0] for item in SEMANTIC_RELATION_MENU} != set(SEMANTIC_RELATIONS):
    raise RuntimeError("Semantic relation menu is out of sync with schema.")
if {item[0] for item in STYLE_TAG_MENU} != set(STYLE_TAGS):
    raise RuntimeError("Style-tag menu is out of sync with schema.")


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("queries"), list):
        raise ValueError("Config must contain a 'queries' array.")
    return data


def inspect_targets(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    lexical = load_artifacts(args.index)

    report_rows: list[dict[str, Any]] = []
    missing = 0
    unresolved = 0
    resolved = 0

    for spec in config["queries"]:
        query = str(spec["query"]).strip()
        requested_sense = spec.get("sense")
        senses = list_senses(lexical, query)

        row = {
            "query": query,
            "category": spec.get("category"),
            "intended": spec.get("intended"),
            "requested_sense": requested_sense,
            "status": None,
            "recommended_sense": None,
            "senses": senses,
        }

        if not senses:
            row["status"] = "missing_headword"
            missing += 1
        elif requested_sense is not None:
            requested_sense = int(requested_sense)
            if any(int(item["sense"]) == requested_sense for item in senses):
                row["status"] = "explicit"
                row["recommended_sense"] = requested_sense
                resolved += 1
            else:
                row["status"] = "invalid_explicit_sense"
                unresolved += 1
        elif len(senses) == 1:
            row["status"] = "unique"
            row["recommended_sense"] = int(senses[0]["sense"])
            resolved += 1
        else:
            row["status"] = "needs_review"
            unresolved += 1

        report_rows.append(row)

    report = {
        "config": str(args.config),
        "target_count": len(report_rows),
        "resolved_count": resolved,
        "unresolved_count": unresolved,
        "missing_count": missing,
        "targets": report_rows,
    }

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")

    print(
        f"Target inspection: {resolved} resolved, {unresolved} need review, "
        f"{missing} missing headword(s).",
        file=sys.stderr,
    )


def freeze_targets(args: argparse.Namespace) -> None:
    with Path(args.report).open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    targets = report.get("targets")
    if not isinstance(targets, list):
        raise ValueError("Sense report must contain a 'targets' array.")

    frozen_queries: list[dict[str, Any]] = []
    errors: list[str] = []

    for target in targets:
        query = str(target.get("query", "")).strip()
        selected = target.get("recommended_sense")
        senses = target.get("senses")

        if not query:
            errors.append("<missing query>: query is required.")
            continue
        if not isinstance(senses, list) or not senses:
            errors.append(f"{query}: no dictionary senses are available.")
            continue
        if not isinstance(selected, int):
            errors.append(
                f"{query}: recommended_sense must be set to an integer before freezing."
            )
            continue
        if not any(int(item.get("sense", -1)) == selected for item in senses):
            errors.append(
                f"{query}: recommended_sense={selected} does not exist in the sense report."
            )
            continue

        frozen_queries.append(
            {
                "query": query,
                "sense": selected,
                "category": target.get("category"),
                "intended": target.get("intended"),
            }
        )

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

    frozen = {
        "description": (
            "Frozen Phase-2 writer relevance targets. Sense IDs were verified "
            "against the V2.5 lexical artifact before candidate export."
        ),
        "queries": frozen_queries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Frozen {len(frozen_queries)} target senses to {output}")


def export_candidates(args: argparse.Namespace) -> None:
    if args.candidates < 1:
        raise ValueError("--candidates must be at least 1.")

    config = _load_config(args.config)
    lexical = load_artifacts(args.index)
    searcher = HybridSearcher.from_paths(
        lexical,
        args.dense_index,
        device=args.device,
    )

    rows: list[dict[str, Any]] = []
    for spec in config["queries"]:
        query = str(spec["query"]).strip()
        sense = spec.get("sense")
        if sense is not None:
            sense = int(sense)

        entry_index = lexical.word_to_index.get(query)
        if entry_index is None:
            raise ValueError(
                f"Query {query!r} is not an exact dictionary headword. "
                "Benchmark targets must resolve to stored senses."
            )
        available_senses = lexical.entry_to_senses[entry_index]
        if sense is None and len(available_senses) != 1:
            raise ValueError(
                f"Query {query!r} has {len(available_senses)} senses; "
                "set an explicit sense in the benchmark config."
            )

        results = searcher.search(
            query,
            sense=sense,
            top_k=args.candidates,
            lexical_pool=max(300, args.candidates),
            dense_pool=max(300, args.candidates),
        )
        if not results:
            raise ValueError(f"No V2.5 candidates returned for query {query!r}.")

        resolved_query_sense = results[0].get("query_sense")
        if not isinstance(resolved_query_sense, dict):
            raise ValueError(
                f"Query {query!r} did not resolve to a dictionary sense. "
                "Benchmark targets must be exact dictionary headwords."
            )

        resolved_sense = int(resolved_query_sense["sense"])
        query_id = f"{query}#{resolved_sense}"

        for rank, item in enumerate(results, start=1):
            matched = item.get("matched_candidate_sense") or {}
            candidate_sense = matched.get("sense")
            if candidate_sense is not None:
                candidate_sense = int(candidate_sense)

            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "pair_id": stable_pair_id(
                        query,
                        resolved_sense,
                        str(item["word"]),
                        candidate_sense,
                    ),
                    "query_id": query_id,
                    "query": {
                        "word": query,
                        "sense": resolved_sense,
                        "definition": resolved_query_sense["definition"],
                        "category": spec.get("category"),
                    },
                    "candidate": {
                        "word": item["word"],
                        "sense": candidate_sense,
                        "definition": matched.get("definition") or item.get("definition"),
                    },
                    "retrieval": {
                        "system": "v2.5",
                        "v25_rank": rank,
                        "score": item.get("score"),
                        "relation_tier": item.get("relation_tier"),
                        "relation_hint": item.get("relation_hint"),
                        "lexical_form": item.get("lexical_form"),
                        "sense_resolution": item.get("sense_resolution"),
                        "lexical_score": item.get("lexical_score"),
                        "lexical_rank": item.get("lexical_rank"),
                        "dense_similarity": item.get("dense_similarity"),
                        "dense_rank": item.get("dense_rank"),
                    },
                    "annotation": {
                        "utility": None,
                        "semantic_relation": None,
                        "style_tags": None,
                        "legacy_relation": None,
                        "notes": "",
                    },
                    "split": None,
                }
            )

    write_jsonl(args.output, rows)
    print(
        f"Wrote {len(rows)} annotation pairs from {len(config['queries'])} "
        f"queries to {args.output} (schema v{SCHEMA_VERSION})"
    )


def _default_v3_path(path: str | Path) -> Path:
    source = Path(path)
    if source.suffix == ".jsonl":
        return source.with_name(source.stem + ".v3.jsonl")
    return Path(str(source) + ".v3.jsonl")


def migrate_v2_to_v3(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.path)
    migrated: list[dict[str, Any]] = []
    preserved_utility = 0

    for row in rows:
        version = row.get("schema_version")
        if version == SCHEMA_VERSION:
            migrated.append(row)
            continue
        if version != 2:
            raise ValueError(
                f"{row.get('pair_id', '<missing pair_id>')}: "
                f"cannot migrate schema_version={version!r}; expected 2 or {SCHEMA_VERSION}."
            )

        annotation = row.get("annotation") or {}
        utility = annotation.get("utility")
        if utility in {0, 1, 2, 3}:
            preserved_utility += 1

        migrated_row = dict(row)
        migrated_row["schema_version"] = SCHEMA_VERSION
        migrated_row["annotation"] = {
            "utility": utility,
            "semantic_relation": None,
            "style_tags": None,
            "legacy_relation": annotation.get("relation"),
            "notes": annotation.get("notes", ""),
        }
        migrated.append(migrated_row)

    output = Path(args.output) if args.output else _default_v3_path(args.path)
    errors = validate_rows(migrated, require_labels=False)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

    write_jsonl(output, migrated)
    print(
        f"Migrated {len(migrated)} rows to schema v{SCHEMA_VERSION}: {output}"
    )
    print(
        f"Preserved writer-utility labels for {preserved_utility} row(s). "
        "Old v2 relation labels were kept only as annotation.legacy_relation."
    )


def _is_fully_labeled(row: dict[str, Any]) -> bool:
    annotation = row.get("annotation")
    if not isinstance(annotation, dict):
        return False
    style_tags = annotation.get("style_tags")
    return (
        annotation.get("utility") in {0, 1, 2, 3}
        and annotation.get("semantic_relation") in SEMANTIC_RELATIONS
        and isinstance(style_tags, list)
        and all(tag in STYLE_TAGS for tag in style_tags)
    )


def _matches_query(row: dict[str, Any], query_filter: str | None) -> bool:
    if not query_filter:
        return True
    query_filter = query_filter.strip()
    query = row.get("query") or {}
    return query_filter in {
        str(row.get("query_id", "")),
        str(query.get("word", "")),
    }


def _show_annotation_row(
    row: dict[str, Any],
    *,
    position: int,
    total: int,
    labeled_count: int,
) -> None:
    query = row["query"]
    candidate = row["candidate"]
    retrieval = row["retrieval"]
    annotation = row.get("annotation") or {}

    print()
    print("=" * 78)
    print(
        f"[{position}/{total}] complete={labeled_count} | "
        f"{row['query_id']} | V2.5 #{retrieval['v25_rank']}"
    )
    print(f"คำค้น       : {query['word']} — {query.get('definition') or '-'}")
    print(
        f"คำที่พบ      : {candidate['word']} — "
        f"{candidate.get('definition') or '-'}"
    )

    hint = retrieval.get("relation_hint")
    if hint:
        print(f"V2.5 hint   : {hint}")

    current_utility = annotation.get("utility")
    current_relation = annotation.get("semantic_relation")
    current_style = annotation.get("style_tags")
    legacy_relation = annotation.get("legacy_relation")

    if current_utility in {0, 1, 2, 3}:
        print(f"Utility เดิม : {current_utility}")
    if current_relation in SEMANTIC_RELATIONS:
        print(f"Relation เดิม: {current_relation}")
    if isinstance(current_style, list):
        print(
            "Style เดิม   : "
            + (", ".join(current_style) if current_style else "ทั่วไป/ไม่ทำเครื่องหมาย")
        )
    if legacy_relation:
        print(f"v2 relation : {legacy_relation} (ใช้อ้างอิงเท่านั้น)")
    print()


def _prompt_utility(existing: int | None = None) -> int | str:
    suffix = f", Enter=คง {existing}" if existing in {0, 1, 2, 3} else ""
    while True:
        raw = input(
            "Writer utility [3=สูงมาก, 2=ชัดเจน, 1=พอมีประโยชน์, "
            f"0=ไม่ช่วย] (s=ข้าม, q=ออก{suffix}): "
        ).strip().lower()
        if raw == "" and existing in {0, 1, 2, 3}:
            return int(existing)
        if raw in {"s", "q"}:
            return raw
        if raw in {"0", "1", "2", "3"}:
            return int(raw)
        print("กรุณาเลือก 0, 1, 2, 3, s หรือ q")


def _prompt_semantic_relation(
    utility: int,
    existing: str | None = None,
) -> str:
    print("ความสัมพันธ์ทางความหมาย:")
    for index, (relation, thai) in enumerate(SEMANTIC_RELATION_MENU, start=1):
        print(f"  {index:>2}. {thai} ({relation})")

    suffix = ""
    if existing in SEMANTIC_RELATIONS:
        suffix = f", Enter=คง {existing}"

    while True:
        raw = input(f"เลือก relation (เลข, b=ย้อนกลับ, s=ข้าม, q=ออก{suffix}): ").strip().lower()
        if raw == "" and existing in SEMANTIC_RELATIONS:
            return str(existing)
        if raw in {"b", "s", "q"}:
            return raw
        if not raw.isdigit():
            print("กรุณาเลือกหมายเลข relation, b, s หรือ q")
            continue

        index = int(raw)
        if not 1 <= index <= len(SEMANTIC_RELATION_MENU):
            print("หมายเลข relation อยู่นอกช่วง")
            continue

        relation = SEMANTIC_RELATION_MENU[index - 1][0]
        if utility > 0 and relation in {
            "opposite_misleading",
            "sense_mismatch",
            "unrelated",
        }:
            print(
                f"{relation} เป็น severe error และกำหนดให้ utility ต้องเป็น 0"
            )
            continue
        return relation


def _prompt_style_tags(existing: list[str] | None = None) -> list[str] | str:
    print("สไตล์/ระดับภาษา (เลือกได้หลายข้อ):")
    print("   0. ทั่วไป / ไม่มีลักษณะพิเศษ")
    for index, (tag, thai) in enumerate(STYLE_TAG_MENU, start=1):
        print(f"  {index:>2}. {thai} ({tag})")

    suffix = ""
    if isinstance(existing, list):
        shown = ",".join(existing) if existing else "ทั่วไป"
        suffix = f", Enter=คง {shown}"

    while True:
        raw = input(
            "เลือก style เช่น 0 หรือ 1,2 (b=ย้อนกลับ, s=ข้าม, q=ออก"
            f"{suffix}): "
        ).strip().lower()

        if raw == "" and isinstance(existing, list):
            return list(existing)
        if raw in {"b", "s", "q"}:
            return raw
        if raw == "0":
            return []

        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if not parts or any(not part.isdigit() for part in parts):
            print("กรุณาเลือก 0 หรือหมายเลขคั่นด้วย comma เช่น 1,2")
            continue

        indexes = [int(part) for part in parts]
        if any(index < 1 or index > len(STYLE_TAG_MENU) for index in indexes):
            print("หมายเลข style อยู่นอกช่วง")
            continue

        tags = [STYLE_TAG_MENU[index - 1][0] for index in indexes]
        tags = list(dict.fromkeys(tags))
        if "unknown" in tags and len(tags) > 1:
            print("unknown ใช้ร่วมกับ style อื่นไม่ได้")
            continue
        return tags


def annotate_interactively(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.path)
    errors = validate_rows(rows, require_labels=False)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

    eligible = [
        index
        for index, row in enumerate(rows)
        if _matches_query(row, args.query)
        and (args.review or not _is_fully_labeled(row))
    ]

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1.")
        eligible = eligible[: args.limit]

    if not eligible:
        print("No matching incomplete rows. Nothing to annotate.")
        return

    total_matching = sum(1 for row in rows if _matches_query(row, args.query))
    labeled_matching = sum(
        1
        for row in rows
        if _matches_query(row, args.query) and _is_fully_labeled(row)
    )

    print(
        f"Writer Relevance schema v{SCHEMA_VERSION}: {len(eligible)} row(s) queued; "
        f"{labeled_matching}/{total_matching} matching rows complete."
    )
    print("Autosave: บันทึกทันทีหลังกรอกครบหนึ่งคู่")

    completed_this_run = 0
    cursor = 0
    quit_requested = False

    while cursor < len(eligible) and not quit_requested:
        row_index = eligible[cursor]
        row = rows[row_index]
        annotation = row["annotation"]

        _show_annotation_row(
            row,
            position=cursor + 1,
            total=len(eligible),
            labeled_count=labeled_matching + completed_this_run,
        )

        while True:
            existing_utility = annotation.get("utility")
            if existing_utility in {0, 1, 2, 3} and not args.review:
                utility: int | str = int(existing_utility)
                print(f"คง Writer utility เดิม = {utility}")
            else:
                utility = _prompt_utility(
                    int(existing_utility)
                    if existing_utility in {0, 1, 2, 3}
                    else None
                )

            if utility == "q":
                quit_requested = True
                break
            if utility == "s":
                cursor += 1
                break

            existing_relation = annotation.get("semantic_relation")
            if (
                existing_relation in SEMANTIC_RELATIONS
                and not args.review
            ):
                semantic_relation = str(existing_relation)
                print(f"คง semantic relation เดิม = {semantic_relation}")
            else:
                semantic_relation = _prompt_semantic_relation(
                    int(utility),
                    str(existing_relation)
                    if existing_relation in SEMANTIC_RELATIONS
                    else None,
                )

            if semantic_relation == "q":
                quit_requested = True
                break
            if semantic_relation == "s":
                cursor += 1
                break
            if semantic_relation == "b":
                continue

            existing_style = annotation.get("style_tags")
            if isinstance(existing_style, list) and not args.review:
                style_tags: list[str] | str = list(existing_style)
                shown = ", ".join(style_tags) if style_tags else "ทั่วไป"
                print(f"คง style เดิม = {shown}")
            else:
                style_tags = _prompt_style_tags(
                    list(existing_style)
                    if isinstance(existing_style, list)
                    else None
                )

            if style_tags == "q":
                quit_requested = True
                break
            if style_tags == "s":
                cursor += 1
                break
            if style_tags == "b":
                continue

            new_annotation = dict(annotation)
            new_annotation["utility"] = int(utility)
            new_annotation["semantic_relation"] = str(semantic_relation)
            new_annotation["style_tags"] = list(style_tags)

            previous_annotation = row["annotation"]
            row["annotation"] = new_annotation
            row_errors = validate_rows([row], require_labels=True)
            if row_errors:
                for error in row_errors:
                    print(f"ERROR: {error}", file=sys.stderr)
                row["annotation"] = previous_annotation
                continue

            was_complete = _is_fully_labeled(
                {**row, "annotation": previous_annotation}
            )
            write_jsonl(args.path, rows)
            if not was_complete:
                completed_this_run += 1

            shown_style = ", ".join(style_tags) if style_tags else "unmarked"
            print(
                f"Saved: {row['query']['word']} -> {row['candidate']['word']} "
                f"| utility={utility} | semantic={semantic_relation} "
                f"| style={shown_style}"
            )
            cursor += 1
            break

    remaining = sum(
        1
        for row in rows
        if _matches_query(row, args.query) and not _is_fully_labeled(row)
    )
    print(
        f"Session complete: completed {completed_this_run} new row(s); "
        f"{remaining} matching row(s) remain incomplete."
    )


def validate_annotations(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.path)
    errors = validate_rows(rows, require_labels=not args.allow_unlabeled)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

    mode = "template/partial benchmark" if args.allow_unlabeled else "labeled benchmark"
    print(f"OK: {len(rows)} rows validated as {mode} (schema v{SCHEMA_VERSION}).")


def evaluate_baseline(args: argparse.Namespace) -> None:
    if args.k < 1:
        raise ValueError("--k must be at least 1.")

    rows = read_jsonl(args.path)
    if args.query:
        rows = [row for row in rows if _matches_query(row, args.query)]
        if not rows:
            raise ValueError(f"No rows match --query {args.query!r}.")

    errors = validate_rows(rows, require_labels=True)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

    report = benchmark_metrics(rows, k=args.k)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Thai Words writer lexical-relevance benchmark utilities."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser(
        "inspect-targets",
        help="Inspect dictionary senses before freezing benchmark target senses.",
    )
    inspect.add_argument(
        "--config",
        default="evaluation/writer_relevance_phase2_targets.json",
    )
    inspect.add_argument("--index", default="artifacts/v1")
    inspect.add_argument(
        "--output",
        default="evaluation/writer_relevance_phase2_sense_report.json",
    )
    inspect.set_defaults(func=inspect_targets)

    freeze = subparsers.add_parser(
        "freeze-targets",
        help="Freeze a reviewed sense report into an explicit benchmark config.",
    )
    freeze.add_argument(
        "--report",
        default="evaluation/writer_relevance_phase2_sense_report.json",
    )
    freeze.add_argument(
        "--output",
        default="evaluation/writer_relevance_phase2_frozen_queries.json",
    )
    freeze.set_defaults(func=freeze_targets)

    export = subparsers.add_parser(
        "export",
        help="Export deterministic V2.5 candidates for human annotation.",
    )
    export.add_argument(
        "--config",
        default="evaluation/substitutability_pilot_queries.json",
    )
    export.add_argument("--index", default="artifacts/v1")
    export.add_argument(
        "--dense-index",
        default="artifacts/v2/embeddinggemma-300m-256",
    )
    export.add_argument("--candidates", type=int, default=30)
    export.add_argument("--device", default=None)
    export.add_argument(
        "--output",
        default="evaluation/substitutability_annotations.jsonl",
    )
    export.set_defaults(func=export_candidates)

    migrate = subparsers.add_parser(
        "migrate-v3",
        help="Migrate schema-v2 annotations to v3 while preserving writer utility.",
    )
    migrate.add_argument("path")
    migrate.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Default: <input>.v3.jsonl",
    )
    migrate.set_defaults(func=migrate_v2_to_v3)

    annotate = subparsers.add_parser(
        "annotate",
        help="Interactively label utility, semantics, and style with autosave/resume.",
    )
    annotate.add_argument(
        "path",
        nargs="?",
        default="evaluation/substitutability_annotations.jsonl",
    )
    annotate.add_argument(
        "--query",
        default=None,
        help="Only annotate one query headword or exact query_id, e.g. ฝน or ฝน#1.",
    )
    annotate.add_argument(
        "--review",
        action="store_true",
        help="Review all three axes even when a row is already complete.",
    )
    annotate.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Annotate at most this many queued rows in the session.",
    )
    annotate.set_defaults(func=annotate_interactively)

    validate = subparsers.add_parser(
        "validate",
        help="Validate an annotation template or labeled JSONL file.",
    )
    validate.add_argument("path")
    validate.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help="Allow null/partial annotation axes.",
    )
    validate.set_defaults(func=validate_annotations)

    metrics = subparsers.add_parser(
        "metrics",
        help="Measure frozen V2.5 ranking against human writer-utility labels.",
    )
    metrics.add_argument("path")
    metrics.add_argument("--k", type=int, default=10)
    metrics.add_argument(
        "--query",
        default=None,
        help="Evaluate only one query headword/query_id, useful during pilot labeling.",
    )
    metrics.add_argument("--output", default=None)
    metrics.set_defaults(func=evaluate_baseline)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
