from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from thai_writer_runtime import writer_text_pair


WRITER_RERANKER_MODEL_ENV = "THAI_WORD_WRITER_RERANKER_MODEL_PATH"
PRODUCTION_CATEGORY_MODE = "omit"


def _create_crossencoder(
    source: str,
    *,
    max_length: int,
    device: str | None,
) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        source,
        num_labels=1,
        max_length=max_length,
        device=device,
    )


class NeuralWriterRanker:
    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        device: str | None = None,
        max_length: int = 384,
        batch_size: int = 4,
        allow_download: bool = False,
    ) -> None:
        if max_length < 32:
            raise ValueError("max_length must be >= 32.")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        self.model_path = str(model_path) if model_path is not None else None
        self.device = device
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.allow_download = bool(allow_download)

        self._model: Any | None = None
        self._resolved_source: str | None = None
        self._load_seconds: float | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _resolve_source(self) -> str:
        source = self.model_path or os.environ.get(WRITER_RERANKER_MODEL_ENV)
        if not source:
            raise ValueError(
                "No neural writer-reranker model path configured. "
                f"Pass model_path or set {WRITER_RERANKER_MODEL_ENV}."
            )

        path = Path(source)
        if path.exists():
            return str(path)

        if not self.allow_download:
            raise FileNotFoundError(
                "Neural writer-reranker checkpoint not found locally: "
                f"{source}. Runtime model download is disabled by default."
            )

        return str(source)

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        source = self._resolve_source()
        started = perf_counter()
        model = _create_crossencoder(
            source,
            max_length=self.max_length,
            device=self.device,
        )
        self._load_seconds = perf_counter() - started
        self._resolved_source = source
        self._model = model
        return model

    def score_rows(self, rows: list[dict[str, Any]]) -> list[float]:
        if not rows:
            return []

        model = self._load_model()
        pairs = [
            writer_text_pair(row, category_mode=PRODUCTION_CATEGORY_MODE)
            for row in rows
        ]
        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        array = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(array) != len(rows):
            raise RuntimeError(
                "CrossEncoder returned a score count that does not match input rows."
            )
        if not np.all(np.isfinite(array)):
            raise RuntimeError("CrossEncoder returned non-finite neural scores.")
        return [float(value) for value in array]

    def warmup(self) -> dict[str, Any]:
        """Load the model and run one category-free synthetic inference."""
        row = {
            "query": {
                "word": "ทดสอบ",
                "definition": "ข้อมูลสำหรับอุ่นโมเดลก่อนรับคำค้นจริง",
            },
            "candidate": {
                "word": "ทดลอง",
                "definition": "ทำเพื่อพิสูจน์หรือทดสอบ",
            },
            "retrieval": {
                "v25_rank": 1,
                "score": 0.0,
                "relation_tier": 0,
                "lexical_score": 0.0,
                "lexical_rank": None,
                "dense_similarity": 0.0,
                "dense_rank": 1,
                "relation_hint": "definition_similar",
                "lexical_form": "standalone",
                "sense_resolution": "dense_only",
            },
        }
        self.score_rows([row])
        return self.runtime_info()

    def runtime_info(self) -> dict[str, Any]:
        model = self._model
        return {
            "loaded": self.loaded,
            "configured_model_path": self.model_path,
            "environment_variable": WRITER_RERANKER_MODEL_ENV,
            "resolved_source": self._resolved_source,
            "requested_device": self.device,
            "device": (
                str(getattr(model, "device", "unknown"))
                if model is not None
                else None
            ),
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "allow_download": self.allow_download,
            "load_seconds": (
                round(float(self._load_seconds), 6)
                if self._load_seconds is not None
                else None
            ),
            "category_mode": PRODUCTION_CATEGORY_MODE,
        }
