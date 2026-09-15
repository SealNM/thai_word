from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from thai_hybrid_v2 import HybridSearcher
from thai_lexical_v1 import SearchArtifacts, normalize_text


HTML_TAG_RE = re.compile(r"<[^>]+>")
LEADING_REFERENCE_RE = re.compile(r"^(?:ดู|ดูที่|ดูคำ|เทียบ)\s+")


@dataclass(frozen=True)
class V26Config:
    candidate_pool: int = 50
    rerank_window: int = 20
    max_promotion: int = 4
    max_demotion: int = 4
    dense_similarity_tolerance: float = 0.03
    token_proxy_discount: float = 0.10


@dataclass(frozen=True)
class FrequencyEvidence:
    exact_count: int
    token_proxy_count: float
    familiarity_count: float
    percentile: float | None
    tokens: tuple[str, ...]
    source: str


class FrequencyModel(Protocol):
    def measure(self, word: str) -> FrequencyEvidence:
        ...


class TNCFrequencyModel:
    """Thai National Corpus familiarity signal with token-aware fallback.

    TNC is used only as a negative rarity prior. A high count never creates a
    positive ranking bonus. Multiword/compound forms may use a heavily
    discounted token proxy when the exact phrase is absent from TNC.
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
                    "PyThaiNLP is required for the V2.6 TNC rarity signal. "
                    "Install dependencies with: pip install -r requirements.txt"
                ) from exc
            frequencies = unigram_word_freqs()

        self.frequencies = {
            normalize_text(word): int(count)
            for word, count in frequencies.items()
            if normalize_text(word) and int(count) > 0
        }
        self.sorted_counts = sorted(self.frequencies.values())
        self.token_proxy_discount = max(0.0, float(token_proxy_discount))

        if tokenizer is None:
            try:
                from pythainlp.tokenize import word_tokenize
            except ImportError as exc:
                raise RuntimeError(
                    "PyThaiNLP is required for Thai tokenization in V2.6."
                ) from exc

            def tokenizer(text: str) -> list[str]:
                return word_tokenize(text, engine="newmm", keep_whitespace=False)

        self.tokenizer = tokenizer

    def _percentile(self, count: float) -> float | None:
        if not self.sorted_counts:
            return None
        if count <= 0:
            return 0.0
        # lower-bound percentile intentionally treats the low-frequency tie
        # block as rare instead of letting a large count=1 block look common.
        return bisect_left(self.sorted_counts, count) / len(self.sorted_counts)

    def measure(self, word: str) -> FrequencyEvidence:
        word = normalize_text(word)
        exact_count = int(self.frequencies.get(word, 0))
        raw_tokens = [normalize_text(token) for token in self.tokenizer(word)]
        tokens = tuple(token for token in raw_tokens if token)

        token_proxy = 0.0
        if exact_count <= 0 and len(tokens) > 1:
            token_counts = [self.frequencies.get(token, 0) for token in tokens]
            if token_counts and all(count > 0 for count in token_counts):
                token_proxy = min(token_counts) * self.token_proxy_discount

        familiarity = max(float(exact_count), float(token_proxy))
        source = "tnc_exact" if exact_count > 0 else (
            "tnc_token_proxy" if token_proxy > 0 else "tnc_missing"
        )
        return FrequencyEvidence(
            exact_count=exact_count,
            token_proxy_count=round(float(token_proxy), 3),
            familiarity_count=round(familiarity, 3),
            percentile=self._percentile(familiarity),
            tokens=tokens,
            source=source,
        )


def rarity_penalty(evidence: FrequencyEvidence, *, relation_tier: int) -> int:
    percentile = evidence.percentile
    if percentile is None:
        return 0
    if percentile < 0.10:
        penalty = 4
    elif percentile < 0.25:
        penalty = 3
    elif percentile < 0.45:
        penalty = 2
    elif percentile < 0.65:
        penalty = 1
    else:
        penalty = 0

    # Strong dictionary relations are allowed to remain visible even when the
    # form is rare. Rarity can reorder direct substitutes, not erase them.
    if relation_tier >= 4:
        return min(penalty, 2)
    if relation_tier == 3:
        return min(penalty, 3)
    return penalty


def structural_penalties(item: Mapping[str, Any]) -> dict[str, int]:
    word = normalize_text(str(item.get("word", "")))
    definition = normalize_text(str(item.get("definition", "")))
    plain_definition = normalize_text(HTML_TAG_RE.sub("", definition))

    penalties = {
        "bound_form": 0,
        "cross_reference_only": 0,
        "very_short_form": 0,
    }
    if item.get("lexical_form") == "bound_form" or word.startswith("-") or word.endswith("-"):
        penalties["bound_form"] = 3
    if LEADING_REFERENCE_RE.match(plain_definition):
        penalties["cross_reference_only"] = 1
    if len(word) == 1:
        penalties["very_short_form"] = 2
    return penalties


def _dense_similarity(item: Mapping[str, Any]) -> float | None:
    value = item.get("dense_similarity")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _can_pass(
    candidate: Mapping[str, Any],
    previous: Mapping[str, Any],
    *,
    config: V26Config,
) -> bool:
    cand_diag = candidate["v26"]
    prev_diag = previous["v26"]

    # Never cross the protected V2.5 lexical bucket boundary.
    if int(candidate.get("protected_relation_tier", 0) or 0) != int(
        previous.get("protected_relation_tier", 0) or 0
    ):
        return False

    # A weaker lexical relation cannot use frequency to jump over a stronger
    # relation. This is the key "commonness never creates relevance" guard.
    if int(candidate.get("relation_tier", 0) or 0) < int(previous.get("relation_tier", 0) or 0):
        return False

    cand_dense = _dense_similarity(candidate)
    prev_dense = _dense_similarity(previous)
    if cand_dense is not None and prev_dense is not None:
        if cand_dense + config.dense_similarity_tolerance < prev_dense:
            return False

    # Promotion only happens when the candidate is strictly less penalized.
    return int(cand_diag["total_penalty"]) < int(prev_diag["total_penalty"])


def rerank_v26(
    results: list[dict[str, Any]],
    *,
    frequency_model: FrequencyModel,
    config: V26Config | None = None,
) -> list[dict[str, Any]]:
    config = config or V26Config()
    annotated: list[dict[str, Any]] = []

    for original_rank, raw_item in enumerate(results, start=1):
        item = dict(raw_item)
        relation_tier = int(item.get("relation_tier", 0) or 0)
        frequency = frequency_model.measure(str(item.get("word", "")))
        rarity = rarity_penalty(frequency, relation_tier=relation_tier)
        structural = structural_penalties(item)
        structural_total = sum(structural.values())
        item["v26"] = {
            "original_rank": original_rank,
            "final_rank": original_rank,
            "movement": 0,
            "rarity_penalty": rarity,
            "structural_penalty": structural_total,
            "structural_penalties": structural,
            "total_penalty": rarity + structural_total,
            "frequency": {
                "source": frequency.source,
                "exact_count": frequency.exact_count,
                "token_proxy_count": frequency.token_proxy_count,
                "familiarity_count": frequency.familiarity_count,
                "percentile": (
                    round(float(frequency.percentile), 6)
                    if frequency.percentile is not None
                    else None
                ),
                "tokens": list(frequency.tokens),
            },
        }
        annotated.append(item)

    window = min(max(0, config.rerank_window), len(annotated))
    working = annotated[:window]

    # Stable bounded insertion/bubble pass. Every candidate may improve by at
    # most max_promotion positions, and no displaced candidate may fall by more
    # than max_demotion positions from its original V2.5 rank.
    for index in range(1, len(working)):
        cursor = index
        while cursor > 0:
            candidate = working[cursor]
            previous = working[cursor - 1]
            original_rank = int(candidate["v26"]["original_rank"])
            current_rank = cursor + 1
            if original_rank - current_rank >= config.max_promotion:
                break

            previous_original_rank = int(previous["v26"]["original_rank"])
            previous_new_rank = cursor + 1
            if previous_new_rank - previous_original_rank > config.max_demotion:
                break

            if not _can_pass(candidate, previous, config=config):
                break
            working[cursor - 1], working[cursor] = candidate, previous
            cursor -= 1

    reranked = working + annotated[window:]
    for final_rank, item in enumerate(reranked, start=1):
        original_rank = int(item["v26"]["original_rank"])
        item["v26"]["final_rank"] = final_rank
        item["v26"]["movement"] = original_rank - final_rank
    return reranked


@dataclass
class LightweightRerankSearcher:
    baseline: HybridSearcher
    frequency_model: FrequencyModel
    config: V26Config

    @classmethod
    def from_paths(
        cls,
        lexical: SearchArtifacts,
        dense_index: str,
        *,
        device: str | None = None,
        config: V26Config | None = None,
    ) -> "LightweightRerankSearcher":
        resolved = config or V26Config()
        baseline = HybridSearcher.from_paths(lexical, dense_index, device=device)
        frequency_model = TNCFrequencyModel(
            token_proxy_discount=resolved.token_proxy_discount
        )
        return cls(
            baseline=baseline,
            frequency_model=frequency_model,
            config=resolved,
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
        pool = max(top_k, self.config.candidate_pool)
        baseline_results = self.baseline.search(
            query,
            top_k=pool,
            sense=sense,
            lexical_pool=max(lexical_pool, pool),
            dense_pool=max(dense_pool, pool),
            lexical_weight=lexical_weight,
            dense_weight=dense_weight,
            rrf_k=rrf_k,
        )
        return rerank_v26(
            baseline_results,
            frequency_model=self.frequency_model,
            config=self.config,
        )[:top_k]
