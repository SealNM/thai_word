from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from thai_substitutability import SEVERE_ERROR_RELATIONS
from thai_writer_runtime import WRITER_RUNTIME_SCHEMA_VERSION, writer_feature_dict


LEARNED_RANKER_ARTIFACT_VERSION = 1
LEARNED_RANKER_FILENAME = "learned_ranker.joblib"
LEARNED_RANKER_METADATA_FILENAME = "learned_ranker_metadata.json"
PRODUCTION_CATEGORY_MODE = "omit"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _BinaryProbabilityModel:
    def __init__(self, *, seed: int, balanced: bool) -> None:
        self.seed = int(seed)
        self.balanced = bool(balanced)
        self.constant: float | None = None
        self.model: LogisticRegression | None = None

    def fit(self, x: Any, y: np.ndarray) -> "_BinaryProbabilityModel":
        classes = np.unique(y)
        if len(classes) == 1:
            self.constant = float(classes[0])
            return self

        self.model = LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced" if self.balanced else None,
            random_state=self.seed,
        )
        self.model.fit(x, y)
        return self

    def predict_positive(self, x: Any) -> np.ndarray:
        if self.constant is not None:
            return np.full(x.shape[0], self.constant, dtype=np.float64)
        if self.model is None:
            raise RuntimeError("Binary probability model has not been fitted.")
        return np.asarray(self.model.predict_proba(x)[:, 1], dtype=np.float64)


@dataclass
class LearnedWriterRanker:
    vectorizer: DictVectorizer
    utility_models: list[_BinaryProbabilityModel]
    severe_model: _BinaryProbabilityModel
    severe_penalty: float
    seed: int
    category_mode: str = PRODUCTION_CATEGORY_MODE
    artifact_version: int = LEARNED_RANKER_ARTIFACT_VERSION
    runtime_schema_version: int = WRITER_RUNTIME_SCHEMA_VERSION

    @classmethod
    def fit(
        cls,
        rows: list[dict[str, Any]],
        *,
        seed: int = 42,
        severe_penalty: float = 1.0,
        category_mode: str = PRODUCTION_CATEGORY_MODE,
    ) -> "LearnedWriterRanker":
        if not rows:
            raise ValueError("At least one training row is required.")
        if severe_penalty < 0:
            raise ValueError("severe_penalty must be non-negative.")
        if category_mode != PRODUCTION_CATEGORY_MODE:
            raise ValueError(
                "Phase-4 production learned ranker is category-free; "
                f"expected category_mode={PRODUCTION_CATEGORY_MODE!r}."
            )

        features = [
            writer_feature_dict(row, category_mode=category_mode)
            for row in rows
        ]
        vectorizer = DictVectorizer(sparse=True)
        x = vectorizer.fit_transform(features)

        utilities = np.asarray(
            [int(row["annotation"]["utility"]) for row in rows],
            dtype=np.int64,
        )
        severe = np.asarray(
            [
                int(
                    row["annotation"]["semantic_relation"]
                    in SEVERE_ERROR_RELATIONS
                )
                for row in rows
            ],
            dtype=np.int64,
        )

        utility_models: list[_BinaryProbabilityModel] = []
        for threshold in (1, 2, 3):
            target = (utilities >= threshold).astype(np.int64)
            utility_models.append(
                _BinaryProbabilityModel(
                    seed=seed + threshold,
                    balanced=True,
                ).fit(x, target)
            )

        severe_model = _BinaryProbabilityModel(
            seed=seed + 10,
            balanced=True,
        ).fit(x, severe)

        return cls(
            vectorizer=vectorizer,
            utility_models=utility_models,
            severe_model=severe_model,
            severe_penalty=float(severe_penalty),
            seed=int(seed),
            category_mode=category_mode,
        )

    def _validate_runtime_contract(self) -> None:
        if self.artifact_version != LEARNED_RANKER_ARTIFACT_VERSION:
            raise ValueError(
                "Learned ranker artifact version mismatch: "
                f"{self.artifact_version} != {LEARNED_RANKER_ARTIFACT_VERSION}."
            )
        if self.runtime_schema_version != WRITER_RUNTIME_SCHEMA_VERSION:
            raise ValueError(
                "Writer runtime schema mismatch: "
                f"{self.runtime_schema_version} != {WRITER_RUNTIME_SCHEMA_VERSION}."
            )
        if self.category_mode != PRODUCTION_CATEGORY_MODE:
            raise ValueError(
                "Learned ranker category contract mismatch: "
                f"{self.category_mode!r} != {PRODUCTION_CATEGORY_MODE!r}."
            )
        if len(self.utility_models) != 3:
            raise ValueError("Learned ranker must contain exactly three utility models.")

    def score_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, float]]:
        self._validate_runtime_contract()
        if not rows:
            return []

        features = [
            writer_feature_dict(row, category_mode=self.category_mode)
            for row in rows
        ]
        x = self.vectorizer.transform(features)

        cumulative = [
            model.predict_positive(x)
            for model in self.utility_models
        ]
        expected_utility = np.sum(np.vstack(cumulative), axis=0)
        severe_probability = self.severe_model.predict_positive(x)
        safe_score = expected_utility - (
            self.severe_penalty * severe_probability
        )

        return [
            {
                "expected_utility": float(expected_utility[index]),
                "severe_probability": float(severe_probability[index]),
                "safe_score": float(safe_score[index]),
            }
            for index in range(len(rows))
        ]

    def metadata(
        self,
        *,
        dataset_sha256: str,
        train_pair_count: int,
        train_query_count: int,
    ) -> dict[str, Any]:
        self._validate_runtime_contract()
        return {
            "artifact_version": self.artifact_version,
            "runtime_schema_version": self.runtime_schema_version,
            "category_mode": self.category_mode,
            "feature_count": len(self.vectorizer.feature_names_),
            "feature_names": list(self.vectorizer.feature_names_),
            "seed": self.seed,
            "severe_penalty": self.severe_penalty,
            "dataset_sha256": dataset_sha256,
            "train_pair_count": int(train_pair_count),
            "train_query_count": int(train_query_count),
            "training_split": "train",
            "utility_thresholds": [1, 2, 3],
            "severe_relations": sorted(SEVERE_ERROR_RELATIONS),
        }

    def save(
        self,
        output_dir: str | Path,
        *,
        metadata: dict[str, Any],
    ) -> None:
        self._validate_runtime_contract()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, output / LEARNED_RANKER_FILENAME)
        (output / LEARNED_RANKER_METADATA_FILENAME).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, output_dir: str | Path) -> "LearnedWriterRanker":
        output = Path(output_dir)
        model_path = output / LEARNED_RANKER_FILENAME
        metadata_path = output / LEARNED_RANKER_METADATA_FILENAME

        if not model_path.exists():
            raise FileNotFoundError(f"Learned ranker artifact not found: {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Learned ranker metadata not found: {metadata_path}"
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata.get("artifact_version", -1)) != LEARNED_RANKER_ARTIFACT_VERSION:
            raise ValueError("Learned ranker metadata artifact version mismatch.")
        if int(metadata.get("runtime_schema_version", -1)) != WRITER_RUNTIME_SCHEMA_VERSION:
            raise ValueError("Learned ranker metadata runtime schema mismatch.")
        if metadata.get("category_mode") != PRODUCTION_CATEGORY_MODE:
            raise ValueError("Learned ranker metadata category contract mismatch.")

        model = joblib.load(model_path)
        if not isinstance(model, cls):
            raise TypeError("learned_ranker.joblib does not contain LearnedWriterRanker.")
        model._validate_runtime_contract()

        if len(model.vectorizer.feature_names_) != int(metadata.get("feature_count", -1)):
            raise ValueError("Learned ranker feature count does not match metadata.")
        if list(model.vectorizer.feature_names_) != list(metadata.get("feature_names", [])):
            raise ValueError("Learned ranker feature names do not match metadata.")

        return model
