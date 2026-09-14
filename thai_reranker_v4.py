from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from thai_hybrid_v2 import HybridSearcher, weighted_rrf
from thai_lexical_v1 import normalize_text


DEFAULT_QWEN_RERANKER = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_QWEN_RERANKER_4B = "Qwen/Qwen3-Reranker-4B"
DEFAULT_BGE_RERANKER = "BAAI/bge-reranker-v2-m3"

WRITER_RERANK_INSTRUCTION = (
    "Given a Thai dictionary query and a candidate dictionary entry, score how useful "
    "the candidate is as a substitute or closely usable alternative for a writer. "
    "Prefer exact synonyms and near-synonyms, including literary, archaic, poetic, "
    "formal, colloquial, or register-shifted alternatives when they preserve the core "
    "meaning. Penalize antonyms, words that are only topically associated, accidental "
    "definition overlap, and candidates that substantially change the core meaning.\n"
)

WRITER_RERANK_INSTRUCTION_V41 = (
    "Rank Thai dictionary candidates by lexical substitutability for a writer, not by "
    "general semantic relatedness. A strong candidate should be able to replace the "
    "query in a natural sentence while preserving the intended dictionary sense and "
    "roughly the same grammatical role. Prefer contemporary, commonly used Thai words "
    "before rare, literary, archaic, poetic, or highly formal alternatives when both "
    "are equally accurate. Rare or literary alternatives are still useful, but should "
    "usually rank below equally accurate common words. Strongly penalize antonyms, "
    "topically associated words, cause/effect relations, objects or agents, compounds "
    "and collocations that merely contain the query concept, subtype or manner changes "
    "that alter the action, and accidental definition overlap. Do not reward a candidate "
    "only because its dictionary definition mentions the query word.\n"
)

RERANKER_PROFILES: dict[str, dict[str, Any]] = {
    "qwen3-0.6b": {
        "model_id": DEFAULT_QWEN_RERANKER,
        "instruction": WRITER_RERANK_INSTRUCTION,
    },
    "qwen3-0.6b-v4.1": {
        "model_id": DEFAULT_QWEN_RERANKER,
        "instruction": WRITER_RERANK_INSTRUCTION_V41,
    },
    "qwen3-0.6b-v4.2": {
        "model_id": DEFAULT_QWEN_RERANKER,
        "instruction": WRITER_RERANK_INSTRUCTION_V41,
    },
    "qwen3-0.6b-v4.3": {
        "model_id": DEFAULT_QWEN_RERANKER,
        "instruction": WRITER_RERANK_INSTRUCTION_V41,
        "default_batch_size": 16,
    },
    "qwen3-4b-v4.3": {
        "model_id": DEFAULT_QWEN_RERANKER_4B,
        "instruction": WRITER_RERANK_INSTRUCTION_V41,
        "default_batch_size": 4,
        "model_kwargs": {"torch_dtype": "float16"},
    },
    "bge-v2-m3": {
        "model_id": DEFAULT_BGE_RERANKER,
        "instruction": None,
    },
}

SAFE_RELATION_HINTS = {
    "direct_gloss_or_synonym",
    "mutual_definition_reference",
}

VALID_MODES = {
    "rerank",
    "fusion",
    "protected",
    "commonness",
    "gated-commonness",
}


class PairScorer(Protocol):
    def score(self, query_text: str, documents: list[str]) -> list[float]:
        ...


def resolve_reranker_profile(name_or_model: str) -> dict[str, Any]:
    if name_or_model in RERANKER_PROFILES:
        profile = dict(RERANKER_PROFILES[name_or_model])
        profile["key"] = name_or_model
        return profile
    return {
        "key": name_or_model,
        "model_id": name_or_model,
        "instruction": None,
    }


def build_writer_query(
    query: str,
    query_sense: dict[str, Any] | None,
    *,
    category: str | None = None,
) -> str:
    query = normalize_text(query)
    lines = [f"Thai query word: {query}"]
    category = str(category or "").strip()
    if category:
        lines.append(f"Query category / grammatical role: {category}")
    if isinstance(query_sense, dict):
        definition = normalize_text(query_sense.get("definition", ""))
        if definition:
            lines.append(f"Intended dictionary sense: {definition}")
    return "\n".join(lines)


def build_candidate_document(candidate: dict[str, Any]) -> str:
    word = normalize_text(candidate.get("word", ""))
    definition = ""

    matched = candidate.get("matched_candidate_sense")
    if isinstance(matched, dict):
        definition = normalize_text(matched.get("definition", ""))

    if not definition:
        definition = normalize_text(candidate.get("definition", ""))

    lines = [f"Thai candidate word: {word}"]
    if definition:
        lines.append(f"Dictionary meaning: {definition}")
    return "\n".join(lines)


def is_high_precision_lexical(candidate: dict[str, Any]) -> bool:
    try:
        relation_tier = int(candidate.get("relation_tier") or 0)
    except (TypeError, ValueError):
        relation_tier = 0

    return (
        relation_tier >= 5
        and str(candidate.get("lexical_form") or "standalone") == "standalone"
        and str(candidate.get("relation_hint") or "") in SAFE_RELATION_HINTS
    )


def load_tnc_commonness() -> dict[str, int]:
    try:
        from pythainlp.corpus.tnc import unigram_word_freqs
    except ImportError as exc:
        raise RuntimeError(
            "PyThaiNLP TNC frequency data is required for V4 commonness ranking."
        ) from exc

    values = unigram_word_freqs()
    result: dict[str, int] = {}
    for word, raw_count in values.items():
        word = normalize_text(word)
        if not word:
            continue
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if count > 0:
            result[word] = count
    return result


def _default_commonness_tokenizer(text: str) -> list[str]:
    try:
        from pythainlp.tokenize import word_tokenize
    except ImportError:
        return [text] if text else []

    tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
    return [normalize_text(token) for token in tokens if normalize_text(token)]


def _commonness_features(
    word: str,
    frequency: dict[str, int],
    tokenizer: Callable[[str], list[str]],
) -> tuple[int, list[str], list[int], float]:
    word = normalize_text(word)
    try:
        exact_count = max(0, int(frequency.get(word, 0)))
    except (TypeError, ValueError):
        exact_count = 0

    tokens = tokenizer(word)
    if not tokens:
        tokens = [word] if word else []

    token_counts: list[int] = []
    for token in tokens:
        try:
            token_counts.append(max(0, int(frequency.get(token, 0))))
        except (TypeError, ValueError):
            token_counts.append(0)

    exact_log = math.log1p(exact_count)
    token_logs = [math.log1p(count) for count in token_counts if count > 0]
    token_average = sum(token_logs) / len(token_logs) if token_logs else 0.0

    if len(tokens) <= 1:
        familiarity = exact_log
    else:
        familiarity = max(
            exact_log,
            (0.55 * exact_log) + (0.45 * token_average),
        )

    return exact_count, tokens, token_counts, float(familiarity)


def annotate_commonness(
    candidates: list[dict[str, Any]],
    frequency: dict[str, int] | None,
    *,
    source: str = "tnc",
    tokenizer: Callable[[str], list[str]] | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    frequency = frequency or {}
    tokenizer = tokenizer or _default_commonness_tokenizer

    features = [
        _commonness_features(
            normalize_text(item.get("word", "")),
            frequency,
            tokenizer,
        )
        for item in candidates
    ]

    positive_scores = sorted(
        {round(feature[3], 12) for feature in features if feature[3] > 0.0},
        reverse=True,
    )
    rank_by_score = {
        score: rank
        for rank, score in enumerate(positive_scores, start=1)
    }

    annotated: list[dict[str, Any]] = []
    for candidate, feature in zip(candidates, features, strict=True):
        exact_count, tokens, token_counts, familiarity = feature
        item = dict(candidate)
        item["commonness_source"] = source
        item["commonness_count"] = exact_count
        item["commonness_tokens"] = tokens
        item["commonness_token_counts"] = token_counts
        item["commonness_score"] = round(familiarity, 8)
        item["commonness_rank"] = (
            rank_by_score.get(round(familiarity, 12))
            if familiarity > 0.0
            else None
        )
        annotated.append(item)
    return annotated


def semantic_commonness_eligible(
    candidate: dict[str, Any],
    *,
    lexical_tier: int = 4,
    reranker_top: int = 12,
    v25_top: int = 20,
    strict_reranker_top: int = 5,
    wide_v25_top: int = 30,
) -> bool:
    try:
        relation_tier = int(candidate.get("relation_tier") or 0)
    except (TypeError, ValueError):
        relation_tier = 0
    lexical_form = str(candidate.get("lexical_form") or "standalone")
    if relation_tier >= lexical_tier and lexical_form == "standalone":
        return True

    try:
        reranker_rank = int(candidate["reranker_rank"])
        v25_rank = int(candidate["v25_rank"])
    except (KeyError, TypeError, ValueError):
        return False

    if reranker_rank <= reranker_top and v25_rank <= v25_top:
        return True
    if reranker_rank <= strict_reranker_top and v25_rank <= wide_v25_top:
        return True
    return False


class CrossEncoderPairScorer:
    def __init__(
        self,
        model_id: str,
        *,
        instruction: str | None = None,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Sentence Transformers is required for V4 reranking. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        kwargs: dict[str, Any] = {}
        if device:
            kwargs["device"] = device
        if max_length is not None:
            kwargs["max_length"] = int(max_length)
        if model_kwargs:
            resolved_model_kwargs = dict(model_kwargs)
            dtype_name = resolved_model_kwargs.get("torch_dtype")
            if dtype_name in {"float16", "bfloat16", "float32"}:
                import torch

                resolved_model_kwargs["torch_dtype"] = getattr(torch, str(dtype_name))
            kwargs["model_kwargs"] = resolved_model_kwargs

        self.model_id = model_id
        self.instruction = instruction
        self.batch_size = max(1, int(batch_size))
        self.model = CrossEncoder(model_id, **kwargs)

    def score(self, query_text: str, documents: list[str]) -> list[float]:
        if not documents:
            return []

        pairs = [(query_text, document) for document in documents]
        predict_kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "show_progress_bar": False,
        }
        if self.instruction:
            predict_kwargs["prompt"] = self.instruction

        raw_scores = self.model.predict(pairs, **predict_kwargs)
        scores = np.asarray(raw_scores)

        if scores.ndim == 0:
            scores = scores.reshape(1)
        elif scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]

        if scores.ndim != 1 or scores.shape[0] != len(documents):
            raise ValueError(
                "Reranker must return exactly one scalar score per candidate; "
                f"got shape {scores.shape} for {len(documents)} candidates."
            )

        return [float(value) for value in scores]


def annotate_reranker_scores(
    query: str,
    candidates: list[dict[str, Any]],
    scorer: PairScorer,
    *,
    category: str | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    query_sense = candidates[0].get("query_sense")
    if not isinstance(query_sense, dict):
        query_sense = None

    query_text = build_writer_query(query, query_sense, category=category)
    documents = [build_candidate_document(item) for item in candidates]
    scores = scorer.score(query_text, documents)

    if len(scores) != len(candidates):
        raise ValueError(
            "Reranker score count does not match candidate count: "
            f"{len(scores)} != {len(candidates)}"
        )

    reranker_order = sorted(
        range(len(candidates)),
        key=lambda index: (-float(scores[index]), index),
    )
    reranker_rank = {
        candidate_index: rank
        for rank, candidate_index in enumerate(reranker_order, start=1)
    }

    annotated: list[dict[str, Any]] = []
    for index, (candidate, reranker_score) in enumerate(
        zip(candidates, scores, strict=True),
        start=1,
    ):
        item = dict(candidate)
        item["v25_rank"] = index
        item["v25_score"] = candidate.get("score")
        item["reranker_score"] = float(reranker_score)
        item["reranker_rank"] = reranker_rank[index - 1]
        item["protected_lexical"] = is_high_precision_lexical(candidate)
        annotated.append(item)

    return annotated


def _fusion_score(
    item: dict[str, Any],
    *,
    v25_weight: float,
    reranker_weight: float,
    rrf_k: int,
) -> float:
    return weighted_rrf(
        lexical_rank=int(item["v25_rank"]),
        dense_rank=int(item["reranker_rank"]),
        lexical_weight=v25_weight,
        dense_weight=reranker_weight,
        k=rrf_k,
    )


def _fusion_sorted(
    candidates: list[dict[str, Any]],
    *,
    v25_weight: float,
    reranker_weight: float,
    rrf_k: int,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        item["fusion_score"] = float(
            _fusion_score(
                item,
                v25_weight=v25_weight,
                reranker_weight=reranker_weight,
                rrf_k=rrf_k,
            )
        )
        ranked.append(item)
    ranked.sort(
        key=lambda item: (
            -float(item["fusion_score"]),
            int(item["reranker_rank"]),
            int(item["v25_rank"]),
            item["word"],
        )
    )
    for rank, item in enumerate(ranked, start=1):
        item["fusion_rank"] = rank
    return ranked


def _rank_gated_commonness(
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
    v25_weight: float,
    reranker_weight: float,
    commonness_weight: float,
    commonness_promotion_cap: int,
    rrf_k: int,
    gate_lexical_tier: int,
    gate_reranker_top: int,
    gate_v25_top: int,
    gate_strict_reranker_top: int,
    gate_wide_v25_top: int,
) -> list[dict[str, Any]]:
    baseline = _fusion_sorted(
        candidates,
        v25_weight=v25_weight,
        reranker_weight=reranker_weight,
        rrf_k=rrf_k,
    )

    eligible_scores = sorted(
        {
            round(float(item.get("commonness_score") or 0.0), 12)
            for item in baseline
            if float(item.get("commonness_score") or 0.0) > 0.0
            and semantic_commonness_eligible(
                item,
                lexical_tier=gate_lexical_tier,
                reranker_top=gate_reranker_top,
                v25_top=gate_v25_top,
                strict_reranker_top=gate_strict_reranker_top,
                wide_v25_top=gate_wide_v25_top,
            )
        },
        reverse=True,
    )
    eligible_rank = {
        score: rank
        for rank, score in enumerate(eligible_scores, start=1)
    }
    eligible_count = len(eligible_scores)

    ranked: list[dict[str, Any]] = []
    cap = max(0, int(commonness_promotion_cap))
    for item in baseline:
        row = dict(item)
        eligible = semantic_commonness_eligible(
            row,
            lexical_tier=gate_lexical_tier,
            reranker_top=gate_reranker_top,
            v25_top=gate_v25_top,
            strict_reranker_top=gate_strict_reranker_top,
            wide_v25_top=gate_wide_v25_top,
        )
        familiarity = float(row.get("commonness_score") or 0.0)
        common_rank = (
            eligible_rank.get(round(familiarity, 12))
            if eligible and familiarity > 0.0
            else None
        )

        promotion = 0
        if common_rank is not None and eligible_count > 0 and cap > 0:
            percentile = (eligible_count - common_rank + 1) / eligible_count
            promotion = min(
                cap,
                max(0, int(round(commonness_weight * cap * percentile))),
            )

        row["semantic_gate_eligible"] = eligible
        row["gated_commonness_rank"] = common_rank
        row["commonness_promotion"] = promotion
        row["adjusted_fusion_rank"] = max(1, int(row["fusion_rank"]) - promotion)
        row["v4_mode"] = "gated-commonness"
        row["v4_score"] = 1.0 / float(row["adjusted_fusion_rank"])
        row["score"] = row["v4_score"]
        ranked.append(row)

    ranked.sort(
        key=lambda item: (
            int(item["adjusted_fusion_rank"]),
            int(item["fusion_rank"]),
            int(item["reranker_rank"]),
            int(item["v25_rank"]),
            item["word"],
        )
    )

    for rank, item in enumerate(ranked, start=1):
        item["v4_rank"] = rank
    return ranked[:top_k]


def rank_v4_candidates(
    candidates: list[dict[str, Any]],
    *,
    mode: str = "gated-commonness",
    top_k: int = 20,
    v25_weight: float = 0.35,
    reranker_weight: float = 1.0,
    commonness_weight: float = 0.5,
    commonness_promotion_cap: int = 4,
    rrf_k: int = 20,
    gate_lexical_tier: int = 4,
    gate_reranker_top: int = 12,
    gate_v25_top: int = 20,
    gate_strict_reranker_top: int = 5,
    gate_wide_v25_top: int = 30,
) -> list[dict[str, Any]]:
    if mode not in VALID_MODES:
        raise ValueError(
            f"Unknown V4 mode {mode!r}; expected one of {sorted(VALID_MODES)}."
        )
    if top_k <= 0:
        return []

    if mode == "gated-commonness":
        return _rank_gated_commonness(
            candidates,
            top_k=top_k,
            v25_weight=v25_weight,
            reranker_weight=reranker_weight,
            commonness_weight=commonness_weight,
            commonness_promotion_cap=commonness_promotion_cap,
            rrf_k=rrf_k,
            gate_lexical_tier=gate_lexical_tier,
            gate_reranker_top=gate_reranker_top,
            gate_v25_top=gate_v25_top,
            gate_strict_reranker_top=gate_strict_reranker_top,
            gate_wide_v25_top=gate_wide_v25_top,
        )

    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        reranker_score = float(item["reranker_score"])
        v25_rank = int(item["v25_rank"])
        reranker_rank = int(item["reranker_rank"])

        if mode in {"fusion", "commonness"}:
            v4_score = weighted_rrf(
                lexical_rank=v25_rank,
                dense_rank=reranker_rank,
                lexical_weight=v25_weight,
                dense_weight=reranker_weight,
                k=rrf_k,
            )
            if mode == "commonness":
                commonness_rank = item.get("commonness_rank")
                if commonness_rank is not None:
                    v4_score += commonness_weight / (rrf_k + int(commonness_rank))
        else:
            v4_score = reranker_score

        item["v4_mode"] = mode
        item["v4_score"] = float(v4_score)
        item["score"] = float(v4_score)
        ranked.append(item)

    if mode == "rerank":
        ranked.sort(
            key=lambda item: (
                -float(item["reranker_score"]),
                int(item["v25_rank"]),
                item["word"],
            )
        )
    elif mode in {"fusion", "commonness"}:
        ranked.sort(
            key=lambda item: (
                -float(item["v4_score"]),
                int(item["reranker_rank"]),
                int(item["v25_rank"]),
                item["word"],
            )
        )
    else:
        ranked.sort(
            key=lambda item: (
                -int(bool(item.get("protected_lexical"))),
                -float(item["reranker_score"]),
                int(item["v25_rank"]),
                item["word"],
            )
        )

    for rank, item in enumerate(ranked, start=1):
        item["v4_rank"] = rank

    return ranked[:top_k]


@dataclass
class V4Searcher:
    v25: HybridSearcher
    scorer: PairScorer
    commonness: dict[str, int] | None = None
    commonness_source: str = "none"

    def retrieve_and_score(
        self,
        query: str,
        *,
        sense: int | None = None,
        category: str | None = None,
        candidate_pool: int = 50,
        lexical_pool: int = 300,
        dense_pool: int = 300,
        lexical_weight: float = 1.0,
        dense_weight: float = 1.0,
        v25_rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        candidate_pool = max(1, int(candidate_pool))
        candidates = self.v25.search(
            query,
            top_k=candidate_pool,
            sense=sense,
            lexical_pool=max(candidate_pool, lexical_pool),
            dense_pool=max(candidate_pool, dense_pool),
            lexical_weight=lexical_weight,
            dense_weight=dense_weight,
            rrf_k=v25_rrf_k,
        )
        scored = annotate_reranker_scores(
            query,
            candidates,
            self.scorer,
            category=category,
        )
        return annotate_commonness(
            scored,
            self.commonness,
            source=self.commonness_source,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        sense: int | None = None,
        category: str | None = None,
        candidate_pool: int = 50,
        mode: str = "gated-commonness",
        lexical_pool: int = 300,
        dense_pool: int = 300,
        lexical_weight: float = 1.0,
        dense_weight: float = 1.0,
        v25_rrf_k: int = 60,
        v25_rank_weight: float = 0.35,
        reranker_rank_weight: float = 1.0,
        commonness_rank_weight: float = 0.75,
        commonness_promotion_cap: int = 4,
        v4_rrf_k: int = 20,
    ) -> list[dict[str, Any]]:
        scored = self.retrieve_and_score(
            query,
            sense=sense,
            category=category,
            candidate_pool=max(top_k, candidate_pool),
            lexical_pool=lexical_pool,
            dense_pool=dense_pool,
            lexical_weight=lexical_weight,
            dense_weight=dense_weight,
            v25_rrf_k=v25_rrf_k,
        )
        return rank_v4_candidates(
            scored,
            mode=mode,
            top_k=top_k,
            v25_weight=v25_rank_weight,
            reranker_weight=reranker_rank_weight,
            commonness_weight=commonness_rank_weight,
            commonness_promotion_cap=commonness_promotion_cap,
            rrf_k=v4_rrf_k,
        )
