#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thai_pairwise_v27 import FEATURE_NAMES, feature_vector
from thai_v27_data import load_holdout_words, read_jsonl


def _validate_row(
    row: dict[str, Any],
    *,
    holdout: set[str],
) -> list[str]:
    errors: list[str] = []
    anchor = row.get("anchor") or {}
    anchor_word = str(anchor.get("word", "")).strip()
    if not anchor_word:
        errors.append("missing anchor.word")
    if anchor_word in holdout:
        errors.append(
            f"holdout anchor leaked into training: "
            f"{anchor_word}"
        )

    candidates = row.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        errors.append("need at least two candidates")
        return errors

    for index, candidate in enumerate(candidates):
        word = str(candidate.get("word", "")).strip()
        if word in holdout:
            errors.append(
                f"holdout candidate leaked at index "
                f"{index}: {word}"
            )
        features = candidate.get("features")
        if not isinstance(features, dict):
            errors.append(
                f"candidate {index} missing features"
            )
            continue
        missing = [
            name
            for name in FEATURE_NAMES
            if name not in features
        ]
        if missing:
            errors.append(
                f"candidate {index} missing feature(s): "
                f"{missing[:4]}"
            )

    teacher = row.get("teacher")
    if not isinstance(teacher, dict):
        errors.append("missing teacher metadata")
        return errors
    ranking = teacher.get("ranking_indices")
    if not isinstance(ranking, list) or not ranking:
        errors.append("missing teacher.ranking_indices")
        return errors

    seen: set[int] = set()
    for value in ranking:
        try:
            index = int(value)
        except (TypeError, ValueError):
            errors.append(
                f"invalid ranking index {value!r}"
            )
            continue
        if index < 0 or index >= len(candidates):
            errors.append(
                f"ranking index out of range: {index}"
            )
        if index in seen:
            errors.append(
                f"duplicate ranking index: {index}"
            )
        seen.add(index)
    return errors


def _candidate_matrix(
    row: dict[str, Any],
) -> np.ndarray:
    return np.vstack(
        [
            feature_vector(candidate["features"])
            for candidate in row["candidates"]
        ]
    )


def _pair_specs(
    row: dict[str, Any],
    *,
    unselected_per_selected: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    ranking = [
        int(value)
        for value in row["teacher"]["ranking_indices"]
    ]
    selected = set(ranking)
    unselected = [
        index
        for index in range(len(row["candidates"]))
        if index not in selected
    ]

    pairs: list[tuple[int, int]] = []

    for left_pos, better in enumerate(ranking):
        for worse in ranking[left_pos + 1 :]:
            pairs.append((better, worse))

    for better in ranking:
        pool = list(unselected)
        rng.shuffle(pool)
        for worse in pool[
            : max(0, unselected_per_selected)
        ]:
            pairs.append((better, worse))

    return pairs


def build_pairwise_dataset(
    rows: list[dict[str, Any]],
    *,
    unselected_per_selected: int,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    dict[str, int],
]:
    rng = random.Random(seed)
    X: list[np.ndarray] = []
    y: list[int] = []
    groups: list[str] = []
    source_counts: dict[str, int] = defaultdict(int)

    for row in rows:
        matrix = _candidate_matrix(row)
        ranking_set = {
            int(x)
            for x in row["teacher"]["ranking_indices"]
        }
        pairs = _pair_specs(
            row,
            unselected_per_selected=unselected_per_selected,
            rng=rng,
        )
        seed_id = str(row.get("seed_id", ""))
        for better, worse in pairs:
            diff = matrix[better] - matrix[worse]
            X.append(diff)
            y.append(1)
            groups.append(seed_id)
            X.append(-diff)
            y.append(0)
            groups.append(seed_id)

            if (
                better in ranking_set
                and worse in ranking_set
            ):
                source_counts[
                    "selected_vs_selected"
                ] += 2
            else:
                source_counts[
                    "selected_vs_unselected"
                ] += 2

    if not X:
        return (
            np.empty(
                (0, len(FEATURE_NAMES)),
                dtype=np.float64,
            ),
            np.empty((0,), dtype=np.int64),
            groups,
            dict(source_counts),
        )
    return (
        np.vstack(X).astype(np.float64),
        np.asarray(y, dtype=np.int64),
        groups,
        dict(source_counts),
    )


def _split_rows(
    rows: list[dict[str, Any]],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 5 or validation_fraction <= 0:
        return shuffled, []

    val_count = max(
        1,
        int(round(
            len(shuffled) * validation_fraction
        )),
    )
    val_count = min(
        val_count,
        len(shuffled) - 1,
    )
    return (
        shuffled[val_count:],
        shuffled[:val_count],
    )


def _pipeline(c: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", MaxAbsScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=float(c),
                    fit_intercept=False,
                    solver="lbfgs",
                    max_iter=3000,
                    random_state=27,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train V2.7 linear pairwise ranker from "
            "Gemma teacher orderings."
        )
    )
    parser.add_argument(
        "--input",
        default="artifacts/v27/teacher_labels.jsonl",
    )
    parser.add_argument(
        "--holdout",
        default="evaluation/v27_holdout_words.json",
    )
    parser.add_argument(
        "--output",
        default="artifacts/v27/pairwise_ranker.joblib",
    )
    parser.add_argument(
        "--report",
        default="artifacts/v27/training_report.json",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--unselected-per-selected",
        type=int,
        default=3,
    )
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2701)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if not rows:
        raise SystemExit(
            f"No labeled teacher data found at {args.input}"
        )

    holdout = load_holdout_words(args.holdout)
    invalid: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=1):
        errors = _validate_row(
            row,
            holdout=holdout,
        )
        if errors:
            invalid.append(
                {
                    "row": row_number,
                    "seed_id": row.get("seed_id"),
                    "errors": errors,
                }
            )

    if invalid:
        print(
            json.dumps(
                {
                    "valid": False,
                    "invalid_records": len(invalid),
                    "examples": invalid[:10],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(
            "V2.7 training data validation failed."
        )

    train_rows, val_rows = _split_rows(
        rows,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    (
        X_train,
        y_train,
        _,
        train_sources,
    ) = build_pairwise_dataset(
        train_rows,
        unselected_per_selected=(
            args.unselected_per_selected
        ),
        seed=args.seed,
    )
    if len(X_train) == 0:
        raise SystemExit(
            "No pairwise examples were generated."
        )

    validation: dict[str, Any] = {
        "anchor_count": len(val_rows),
        "pair_examples": 0,
        "accuracy": None,
    }
    if val_rows:
        (
            X_val,
            y_val,
            _,
            val_sources,
        ) = build_pairwise_dataset(
            val_rows,
            unselected_per_selected=(
                args.unselected_per_selected
            ),
            seed=args.seed + 1,
        )
        model = _pipeline(args.c)
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        validation = {
            "anchor_count": len(val_rows),
            "pair_examples": int(len(y_val)),
            "accuracy": round(
                float(
                    accuracy_score(
                        y_val,
                        predictions,
                    )
                ),
                6,
            ),
            "sources": val_sources,
        }

    (
        X_all,
        y_all,
        _,
        all_sources,
    ) = build_pairwise_dataset(
        rows,
        unselected_per_selected=(
            args.unselected_per_selected
        ),
        seed=args.seed,
    )
    final_model = _pipeline(args.c)
    final_model.fit(X_all, y_all)

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    bundle = {
        "schema_version": 1,
        "feature_names": FEATURE_NAMES,
        "pipeline": final_model,
        "metadata": {
            "training_rows": len(rows),
            "pair_examples": int(len(y_all)),
            "teacher_models": sorted(
                {
                    str(
                        (
                            row.get("teacher")
                            or {}
                        ).get(
                            "model",
                            "unknown",
                        )
                    )
                    for row in rows
                }
            ),
            "holdout_file": args.holdout,
            "unselected_per_selected": (
                args.unselected_per_selected
            ),
            "c": args.c,
            "seed": args.seed,
        },
    }
    joblib.dump(
        bundle,
        output,
        compress=3,
    )

    scaler = final_model.named_steps["scale"]
    classifier = final_model.named_steps["logreg"]
    effective_weights = (
        classifier.coef_[0]
        / np.where(
            scaler.scale_ == 0,
            1.0,
            scaler.scale_,
        )
    )
    feature_weights = sorted(
        [
            {
                "feature": name,
                "weight": round(
                    float(weight),
                    8,
                ),
            }
            for name, weight in zip(
                FEATURE_NAMES,
                effective_weights,
            )
        ],
        key=lambda item: -abs(item["weight"]),
    )

    report = {
        "valid": True,
        "input": args.input,
        "output": args.output,
        "anchors_total": len(rows),
        "anchors_train_for_validation": len(
            train_rows
        ),
        "training_pair_examples": int(
            len(y_train)
        ),
        "all_pair_examples": int(len(y_all)),
        "training_sources": train_sources,
        "all_sources": all_sources,
        "validation": validation,
        "feature_count": len(FEATURE_NAMES),
        "feature_weights": feature_weights,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
