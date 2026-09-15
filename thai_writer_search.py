from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from thai_hybrid_v2 import HybridSearcher
from thai_lexical_v1 import load_artifacts
from thai_writer_learned import LearnedWriterRanker
from thai_writer_neural import NeuralWriterRanker
from thai_writer_runtime import runtime_row_from_v25_result


WRITER_HYBRID_ALPHA = 0.5
DEFAULT_RERANK_POOL = 30


def _rank_normalize(scores: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("Writer reranker scores must be finite.")
    if len(values) == 0:
        return np.asarray([], dtype=np.float64)
    if len(values) == 1:
        return np.asarray([1.0], dtype=np.float64)

    ranks = rankdata(values, method="average")
    return np.asarray((ranks - 1.0) / (len(values) - 1.0), dtype=np.float64)


def _query_context(v25_results: list[dict[str, Any]]) -> tuple[str, int | None]:
    if not v25_results:
        return "", None

    payload = v25_results[0].get("query_sense")
    if not isinstance(payload, dict):
        return "", None

    definition = str(payload.get("definition") or "")
    sense_value = payload.get("sense")
    sense = int(sense_value) if sense_value is not None else None
    return definition, sense


@dataclass
class WriterSearch:
    v25: HybridSearcher
    learned: LearnedWriterRanker
    neural: NeuralWriterRanker
    alpha: float = WRITER_HYBRID_ALPHA

    def __post_init__(self) -> None:
        if float(self.alpha) != WRITER_HYBRID_ALPHA:
            raise ValueError(
                "Phase-4 WriterSearch uses the locked alpha=0.5. "
                "Changing alpha requires a new model-selection cycle and holdout."
            )

    @classmethod
    def from_paths(
        cls,
        *,
        lexical_index: str | Path,
        dense_index: str | Path,
        learned_ranker: str | Path,
        neural_model: str | Path | None = None,
        dense_device: str | None = None,
        neural_device: str | None = None,
        neural_batch_size: int = 4,
        neural_max_length: int = 384,
    ) -> "WriterSearch":
        lexical = load_artifacts(lexical_index)
        v25 = HybridSearcher.from_paths(
            lexical,
            str(dense_index),
            device=dense_device,
        )
        learned = LearnedWriterRanker.load(learned_ranker)
        neural = NeuralWriterRanker(
            neural_model,
            device=neural_device,
            batch_size=neural_batch_size,
            max_length=neural_max_length,
        )
        return cls(v25=v25, learned=learned, neural=neural)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        sense: int | None = None,
        rerank_pool: int = DEFAULT_RERANK_POOL,
        lexical_pool: int = 300,
        dense_pool: int = 300,
        lexical_weight: float = 1.0,
        dense_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")
        if rerank_pool < top_k:
            raise ValueError("rerank_pool must be >= top_k.")
        if rerank_pool < 1:
            raise ValueError("rerank_pool must be >= 1.")

        v25_results = self.v25.search(
            query,
            top_k=rerank_pool,
            sense=sense,
            lexical_pool=lexical_pool,
            dense_pool=dense_pool,
            lexical_weight=lexical_weight,
            dense_weight=dense_weight,
            rrf_k=rrf_k,
        )
        if not v25_results:
            return []

        query_definition, selected_sense = _query_context(v25_results)
        runtime_rows = [
            runtime_row_from_v25_result(
                query=query,
                query_definition=query_definition,
                candidate=item,
                v25_rank=index,
                query_sense=selected_sense,
            )
            for index, item in enumerate(v25_results, start=1)
        ]

        learned_scores = self.learned.score_rows(runtime_rows)
        neural_scores = self.neural.score_rows(runtime_rows)

        if len(learned_scores) != len(v25_results):
            raise RuntimeError("Learned scorer returned the wrong number of rows.")
        if len(neural_scores) != len(v25_results):
            raise RuntimeError("Neural scorer returned the wrong number of rows.")

        learned_safe = np.asarray(
            [float(item["safe_score"]) for item in learned_scores],
            dtype=np.float64,
        )
        neural_raw = np.asarray(neural_scores, dtype=np.float64)

        learned_rank = _rank_normalize(learned_safe)
        neural_rank = _rank_normalize(neural_raw)
        hybrid = ((1.0 - self.alpha) * learned_rank) + (self.alpha * neural_rank)

        enriched: list[dict[str, Any]] = []
        for index, item in enumerate(v25_results):
            learned_item = learned_scores[index]
            clone = dict(item)
            clone.update(
                {
                    "original_v25_rank": index + 1,
                    "writer_hybrid_score": float(hybrid[index]),
                    "writer_learned_rank_score": float(learned_rank[index]),
                    "writer_neural_rank_score": float(neural_rank[index]),
                    "writer_learned_safe_score": float(learned_item["safe_score"]),
                    "writer_expected_utility": float(
                        learned_item["expected_utility"]
                    ),
                    "writer_severe_probability": float(
                        learned_item["severe_probability"]
                    ),
                    "writer_neural_score": float(neural_raw[index]),
                    "writer_alpha": float(self.alpha),
                    "reranker_status": "writer_reranked",
                }
            )
            enriched.append(clone)

        enriched.sort(
            key=lambda item: (
                -float(item["writer_hybrid_score"]),
                int(item["original_v25_rank"]),
            )
        )

        for final_rank, item in enumerate(enriched, start=1):
            item["final_rank"] = final_rank

        return enriched[:top_k]

    def runtime_info(self) -> dict[str, Any]:
        return {
            "alpha": float(self.alpha),
            "default_rerank_pool": DEFAULT_RERANK_POOL,
            "learned_category_mode": self.learned.category_mode,
            "neural": self.neural.runtime_info(),
        }
