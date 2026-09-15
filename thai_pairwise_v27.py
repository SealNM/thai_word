from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np

from thai_hybrid_v2 import HybridSearcher
from thai_lexical_v1 import SearchArtifacts, normalize_text


FEATURE_NAMES: tuple[str, ...] = (
    "v25_reciprocal_rank",
    "v25_fusion_score",
    "protected_relation_tier",
    "relation_tier",
    "direct_relation",
    "weak_relation",
    "lexical_score",
    "lexical_reciprocal_rank",
    "has_lexical_rank",
    "dense_similarity",
    "dense_reciprocal_rank",
    "has_dense_rank",
    "bound_form",
    "dense_only_resolution",
    "tnc_log1p_count",
    "tnc_missing",
    "sense_count_log1p",
    "word_length_log1p",
    "definition_length_log1p",
)


@dataclass(frozen=True)
class FamiliarityEvidence:
    exact_count: int
    proxy_count: float
    effective_count: float
    missing: bool


class TNCFamiliarity:
    """Runtime-safe Thai National Corpus familiarity signal.

    Frequency is only one model feature. It does not directly reorder candidates.
    Multi-token dictionary forms missing as exact TNC entries may use a discounted
    token proxy so familiar phrases are distinguishable from genuinely unseen forms.
    """

    def __init__(
        self,
        *,
        frequencies: Mapping[str, int] | None = None,
        tokenizer: Callable[[str], list[str]] | None = None,
        token_proxy_discount: float = 0.10,
    ) -> None:
        if frequencies is None:
            try:
                from pythainlp.corpus.tnc import unigram_word_freqs
            except ImportError as exc:
                raise RuntimeError(
                    "PyThaiNLP is required for V2.7 TNC familiarity features."
                ) from exc
            frequencies = unigram_word_freqs()

        self.frequencies = {
            normalize_text(word): int(count)
            for word, count in frequencies.items()
            if normalize_text(word) and int(count) > 0
        }
        self.token_proxy_discount = max(0.0, float(token_proxy_discount))

        if tokenizer is None:
            try:
                from pythainlp.tokenize import word_tokenize
            except ImportError as exc:
                raise RuntimeError(
                    "PyThaiNLP is required for V2.7 Thai tokenization."
                ) from exc

            def tokenizer(text: str) -> list[str]:
                return word_tokenize(text, engine="newmm", keep_whitespace=False)

        self.tokenizer = tokenizer

    def measure(self, word: str) -> FamiliarityEvidence:
        word = normalize_text(word)
        exact = int(self.frequencies.get(word, 0))
        proxy = 0.0
        if exact <= 0:
            tokens = [
                normalize_text(token)
                for token in self.tokenizer(word)
                if normalize_text(token)
            ]
            if len(tokens) > 1:
                counts = [int(self.frequencies.get(token, 0)) for token in tokens]
                if counts and all(count > 0 for count in counts):
                    proxy = min(counts) * self.token_proxy_discount

        effective = max(float(exact), float(proxy))
        return FamiliarityEvidence(
            exact_count=exact,
            proxy_count=round(proxy, 3),
            effective_count=round(effective, 3),
            missing=effective <= 0,
        )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _safe_rank_reciprocal(value: Any) -> float:
    try:
        rank = int(value)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 / rank if rank > 0 else 0.0


def _candidate_definition(item: Mapping[str, Any]) -> str:
    matched = item.get("matched_candidate_sense")
    if isinstance(matched, Mapping):
        definition = normalize_text(matched.get("definition", ""))
        if definition:
            return definition
    return normalize_text(item.get("definition", ""))


def extract_features(
    item: Mapping[str, Any],
    *,
    v25_rank: int,
    familiarity: TNCFamiliarity,
) -> dict[str, float]:
    word = normalize_text(item.get("word", ""))
    definition = _candidate_definition(item)
    lexical_rank = item.get("lexical_rank")
    dense_rank = item.get("dense_rank")
    relation_tier = int(item.get("relation_tier", 0) or 0)
    protected_tier = int(item.get("protected_relation_tier", 0) or 0)
    lexical_form = str(item.get("lexical_form", ""))
    sense_resolution = str(item.get("sense_resolution", ""))
    frequency = familiarity.measure(word)

    return {
        "v25_reciprocal_rank": _safe_rank_reciprocal(v25_rank),
        "v25_fusion_score": _safe_float(item.get("score")),
        "protected_relation_tier": float(protected_tier),
        "relation_tier": float(relation_tier),
        "direct_relation": 1.0 if relation_tier >= 4 else 0.0,
        "weak_relation": 1.0 if relation_tier <= 1 else 0.0,
        "lexical_score": _safe_float(item.get("lexical_score")),
        "lexical_reciprocal_rank": _safe_rank_reciprocal(lexical_rank),
        "has_lexical_rank": 1.0 if lexical_rank is not None else 0.0,
        "dense_similarity": _safe_float(item.get("dense_similarity")),
        "dense_reciprocal_rank": _safe_rank_reciprocal(dense_rank),
        "has_dense_rank": 1.0 if dense_rank is not None else 0.0,
        "bound_form": 1.0
        if lexical_form == "bound_form" or word.startswith("-") or word.endswith("-")
        else 0.0,
        "dense_only_resolution": 1.0 if sense_resolution == "dense_only" else 0.0,
        "tnc_log1p_count": math.log1p(max(0.0, frequency.effective_count)),
        "tnc_missing": 1.0 if frequency.missing else 0.0,
        "sense_count_log1p": math.log1p(max(0, int(item.get("sense_count", 1) or 1))),
        "word_length_log1p": math.log1p(len(word)),
        "definition_length_log1p": math.log1p(len(definition)),
    }


def feature_vector(features: Mapping[str, float]) -> np.ndarray:
    return np.asarray(
        [_safe_float(features.get(name)) for name in FEATURE_NAMES],
        dtype=np.float64,
    )


def featurize_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    familiarity: TNCFamiliarity,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    vectors: list[np.ndarray] = []
    for rank, item in enumerate(candidates, start=1):
        features = extract_features(item, v25_rank=rank, familiarity=familiarity)
        rows.append(features)
        vectors.append(feature_vector(features))
    if not vectors:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64), rows
    return np.vstack(vectors), rows


@dataclass
class PairwiseRanker:
    pipeline: Any
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "PairwiseRanker":
        bundle = joblib.load(path)
        if not isinstance(bundle, dict):
            raise ValueError("V2.7 model bundle must be a dictionary.")
        stored_names = tuple(bundle.get("feature_names", ()))
        if stored_names != FEATURE_NAMES:
            raise ValueError(
                "V2.7 feature schema mismatch. "
                f"model={stored_names!r} runtime={FEATURE_NAMES!r}"
            )
        pipeline = bundle.get("pipeline")
        if pipeline is None or not hasattr(pipeline, "decision_function"):
            raise ValueError(
                "V2.7 model bundle is missing a decision_function pipeline."
            )
        return cls(
            pipeline=pipeline,
            feature_names=stored_names,
            metadata=dict(bundle.get("metadata") or {}),
        )

    def score_matrix(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected feature matrix (*, {len(FEATURE_NAMES)}), got {matrix.shape}."
            )
        scores = np.asarray(
            self.pipeline.decision_function(matrix),
            dtype=np.float64,
        )
        return scores.reshape(-1)


def rerank_v27(
    candidates: list[dict[str, Any]],
    *,
    ranker: PairwiseRanker,
    familiarity: TNCFamiliarity,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    matrix, feature_rows = featurize_candidates(
        candidates,
        familiarity=familiarity,
    )
    scores = ranker.score_matrix(matrix)

    annotated: list[dict[str, Any]] = []
    for original_rank, (raw, score, features) in enumerate(
        zip(candidates, scores, feature_rows),
        start=1,
    ):
        item = dict(raw)
        item["v27"] = {
            "original_rank": original_rank,
            "final_rank": original_rank,
            "movement": 0,
            "model_score": round(float(score), 8),
            "features": {
                name: round(float(features[name]), 8)
                for name in FEATURE_NAMES
            },
        }
        annotated.append(item)

    annotated.sort(
        key=lambda item: (
            -int(item.get("protected_relation_tier", 0) or 0),
            -float(item["v27"]["model_score"]),
            int(item["v27"]["original_rank"]),
        )
    )

    for final_rank, item in enumerate(annotated, start=1):
        original_rank = int(item["v27"]["original_rank"])
        item["v27"]["final_rank"] = final_rank
        item["v27"]["movement"] = original_rank - final_rank
    return annotated


@dataclass
class PairwiseSearcher:
    baseline: HybridSearcher
    ranker: PairwiseRanker
    familiarity: TNCFamiliarity
    candidate_pool: int = 30

    @classmethod
    def from_paths(
        cls,
        lexical: SearchArtifacts,
        dense_index: str,
        model_path: str | Path,
        *,
        device: str | None = None,
        candidate_pool: int = 30,
    ) -> "PairwiseSearcher":
        return cls(
            baseline=HybridSearcher.from_paths(
                lexical,
                dense_index,
                device=device,
            ),
            ranker=PairwiseRanker.load(model_path),
            familiarity=TNCFamiliarity(),
            candidate_pool=max(10, int(candidate_pool)),
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        sense: int | None = None,
        lexical_pool: int = 300,
        dense_pool: int = 300,
        lexical_weight: float = 1.0,
        dense_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        pool = max(top_k, self.candidate_pool)
        candidates = self.baseline.search(
            query,
            top_k=pool,
            sense=sense,
            lexical_pool=max(pool, lexical_pool),
            dense_pool=max(pool, dense_pool),
            lexical_weight=lexical_weight,
            dense_weight=dense_weight,
            rrf_k=rrf_k,
        )
        return rerank_v27(
            candidates,
            ranker=self.ranker,
            familiarity=self.familiarity,
        )[:top_k]
