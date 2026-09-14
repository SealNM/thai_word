from __future__ import annotations

from typing import Any

from thai_hybrid_v2 import weighted_rrf
from thai_reranker_v4 import (
    CrossEncoderPairScorer,
    annotate_reranker_scores,
)


DEFAULT_CTXL_INSTRUCT = "ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b"

THAI_WORDS_INSTRUCT_PROMPT = (
    "Rank Thai dictionary candidates for lexical substitution in fiction writing. "
    "Preserve the intended dictionary sense and grammatical role first. Prefer words "
    "that can naturally replace the target in a Thai sentence over words that are only "
    "topically or semantically associated. Strongly penalize cause/effect relations, "
    "objects or agents, different parts of speech, and manner/subtype changes that alter "
    "the meaning. Among candidates that are equally valid substitutes, prefer common "
    "contemporary Thai before formal, literary, archaic, technical, or rare dictionary "
    "vocabulary. Keep useful literary alternatives lower rather than removing them."
)


def resolve_ctxl_dtype(device: str | None = None) -> str:
    try:
        import torch
    except ImportError:
        return "float32"

    if not torch.cuda.is_available():
        return "float32"

    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float16"


def build_ctxl_scorer(
    *,
    device: str | None = None,
    batch_size: int = 16,
    max_length: int | None = None,
) -> CrossEncoderPairScorer:
    dtype = resolve_ctxl_dtype(device)
    return CrossEncoderPairScorer(
        DEFAULT_CTXL_INSTRUCT,
        instruction=THAI_WORDS_INSTRUCT_PROMPT,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        model_kwargs={"torch_dtype": dtype},
    )


def score_ctxl_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    scorer: CrossEncoderPairScorer,
    *,
    category: str | None = None,
) -> list[dict[str, Any]]:
    return annotate_reranker_scores(
        query,
        candidates,
        scorer,
        category=category,
    )


def rank_ctxl_candidates(
    candidates: list[dict[str, Any]],
    *,
    mode: str = "instruct",
    top_k: int = 20,
    v25_weight: float = 0.1,
    instruct_weight: float = 1.0,
    rrf_k: int = 20,
) -> list[dict[str, Any]]:
    if mode not in {"instruct", "fusion"}:
        raise ValueError("mode must be 'instruct' or 'fusion'")
    if top_k <= 0:
        return []

    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)

        if mode == "fusion":
            score = weighted_rrf(
                lexical_rank=int(item["v25_rank"]),
                dense_rank=int(item["reranker_rank"]),
                lexical_weight=v25_weight,
                dense_weight=instruct_weight,
                k=rrf_k,
            )
        else:
            score = float(item["reranker_score"])

        item["v52_mode"] = mode
        item["v52_score"] = float(score)
        item["score"] = float(score)
        ranked.append(item)

    if mode == "instruct":
        ranked.sort(
            key=lambda item: (
                int(item["reranker_rank"]),
                int(item["v25_rank"]),
                item["word"],
            )
        )
    else:
        ranked.sort(
            key=lambda item: (
                -float(item["v52_score"]),
                int(item["reranker_rank"]),
                int(item["v25_rank"]),
                item["word"],
            )
        )

    for rank, item in enumerate(ranked, start=1):
        item["v52_rank"] = rank

    return ranked[:top_k]
