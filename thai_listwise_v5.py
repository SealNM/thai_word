from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from thai_hybrid_v2 import HybridSearcher, weighted_rrf
from thai_lexical_v1 import normalize_text


DEFAULT_JINA_LISTWISE = "jinaai/jina-reranker-v3.5"
VALID_V5_MODES = {"listwise", "fusion"}

LISTWISE_TASK = (
    "Thai lexical substitution task for fiction writing. Rank candidate dictionary "
    "entries by how naturally they can replace the target word in a Thai sentence while "
    "preserving the intended dictionary sense and grammatical role. Exact synonyms and "
    "strong near-synonyms should come first. Among candidates that preserve meaning "
    "equally well, prefer common contemporary Thai before formal, literary, archaic, or "
    "rare vocabulary. Literary and archaic alternatives are still useful and should "
    "remain in the ranking, but below equally accurate common alternatives. Penalize "
    "words that are only associated with the target, different grammatical roles, "
    "objects/agents/causes/effects, compounds that merely contain the idea, or manner/"
    "subtype changes that materially change the meaning."
)


class ListwiseRanker(Protocol):
    def rank(self, query_text: str, documents: list[str]) -> list[dict[str, Any]]:
        ...


def build_listwise_query(
    query: str,
    query_sense: dict[str, Any] | None,
    *,
    category: str | None = None,
) -> str:
    query = normalize_text(query)
    lines = [
        LISTWISE_TASK,
        f"Target Thai word: {query}",
    ]

    category_text = str(category or "").strip()
    if category_text:
        lines.append(f"Target category / grammatical role: {category_text}")

    if isinstance(query_sense, dict):
        definition = normalize_text(query_sense.get("definition", ""))
        if definition:
            lines.append(f"Intended dictionary sense: {definition}")

    lines.append(
        "Rank the candidate entries against one another, not independently. "
        "The first result should be the most useful natural substitute."
    )
    return "\n".join(lines)


def build_listwise_document(candidate: dict[str, Any]) -> str:
    word = normalize_text(candidate.get("word", ""))
    definition = ""

    matched = candidate.get("matched_candidate_sense")
    if isinstance(matched, dict):
        definition = normalize_text(matched.get("definition", ""))

    if not definition:
        definition = normalize_text(candidate.get("definition", ""))

    if definition:
        return f"Candidate: {word}\nDictionary meaning: {definition}"
    return f"Candidate: {word}"


class JinaListwiseRanker:
    def __init__(
        self,
        model_id: str = DEFAULT_JINA_LISTWISE,
        *,
        device: str | None = None,
        dtype: str = "auto",
    ) -> None:
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "Transformers is required for V5 listwise reranking. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        device_map = device or "auto"
        self.model_id = model_id
        self.model = AutoModel.from_pretrained(
            model_id,
            dtype=dtype,
            trust_remote_code=True,
            device_map=device_map,
        )
        self.model.eval()

    def rank(self, query_text: str, documents: list[str]) -> list[dict[str, Any]]:
        if not documents:
            return []

        raw = self.model.rerank(
            query_text,
            documents,
            top_n=len(documents),
        )

        if not isinstance(raw, list) or len(raw) != len(documents):
            raise ValueError(
                "Listwise reranker must return one ranked result per candidate; "
                f"got {type(raw).__name__} with length "
                f"{len(raw) if isinstance(raw, list) else 'unknown'} for "
                f"{len(documents)} candidates."
            )

        normalized: list[dict[str, Any]] = []
        seen: set[int] = set()
        for rank, result in enumerate(raw, start=1):
            if not isinstance(result, dict):
                raise ValueError("Listwise reranker returned a non-dict result.")

            try:
                index = int(result["index"])
                score = float(result["relevance_score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Each listwise result must contain numeric index and relevance_score."
                ) from exc

            if index < 0 or index >= len(documents):
                raise ValueError(f"Listwise reranker returned invalid index {index}.")
            if index in seen:
                raise ValueError(f"Listwise reranker returned duplicate index {index}.")
            seen.add(index)

            normalized.append(
                {
                    "index": index,
                    "rank": rank,
                    "score": score,
                }
            )

        if len(seen) != len(documents):
            raise ValueError("Listwise reranker did not cover every candidate exactly once.")

        return normalized


def annotate_listwise_scores(
    query: str,
    candidates: list[dict[str, Any]],
    ranker: ListwiseRanker,
    *,
    category: str | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    query_sense = candidates[0].get("query_sense")
    if not isinstance(query_sense, dict):
        query_sense = None

    query_text = build_listwise_query(
        query,
        query_sense,
        category=category,
    )
    documents = [build_listwise_document(item) for item in candidates]
    ranking = ranker.rank(query_text, documents)

    rank_by_index = {
        int(item["index"]): (int(item["rank"]), float(item["score"]))
        for item in ranking
    }

    annotated: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if index not in rank_by_index:
            raise ValueError(f"Missing listwise result for candidate index {index}.")

        listwise_rank, listwise_score = rank_by_index[index]
        item = dict(candidate)
        item["v25_rank"] = index + 1
        item["v25_score"] = candidate.get("score")
        item["listwise_rank"] = listwise_rank
        item["listwise_score"] = listwise_score
        annotated.append(item)

    return annotated


def rank_v5_candidates(
    candidates: list[dict[str, Any]],
    *,
    mode: str = "listwise",
    top_k: int = 20,
    v25_weight: float = 0.2,
    listwise_weight: float = 1.0,
    rrf_k: int = 20,
) -> list[dict[str, Any]]:
    if mode not in VALID_V5_MODES:
        raise ValueError(
            f"Unknown V5 mode {mode!r}; expected one of {sorted(VALID_V5_MODES)}."
        )
    if top_k <= 0:
        return []

    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)

        if mode == "fusion":
            score = weighted_rrf(
                lexical_rank=int(item["v25_rank"]),
                dense_rank=int(item["listwise_rank"]),
                lexical_weight=v25_weight,
                dense_weight=listwise_weight,
                k=rrf_k,
            )
        else:
            score = float(item["listwise_score"])

        item["v5_mode"] = mode
        item["v5_score"] = float(score)
        item["score"] = float(score)
        ranked.append(item)

    if mode == "listwise":
        ranked.sort(
            key=lambda item: (
                int(item["listwise_rank"]),
                int(item["v25_rank"]),
                item["word"],
            )
        )
    else:
        ranked.sort(
            key=lambda item: (
                -float(item["v5_score"]),
                int(item["listwise_rank"]),
                int(item["v25_rank"]),
                item["word"],
            )
        )

    for rank, item in enumerate(ranked, start=1):
        item["v5_rank"] = rank

    return ranked[:top_k]


@dataclass
class V5Searcher:
    v25: HybridSearcher
    ranker: ListwiseRanker

    def retrieve_and_rank(
        self,
        query: str,
        *,
        sense: int | None = None,
        category: str | None = None,
        candidate_pool: int = 40,
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
        return annotate_listwise_scores(
            query,
            candidates,
            self.ranker,
            category=category,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        sense: int | None = None,
        category: str | None = None,
        candidate_pool: int = 40,
        mode: str = "listwise",
        lexical_pool: int = 300,
        dense_pool: int = 300,
        lexical_weight: float = 1.0,
        dense_weight: float = 1.0,
        v25_rrf_k: int = 60,
        v25_rank_weight: float = 0.2,
        listwise_rank_weight: float = 1.0,
        v5_rrf_k: int = 20,
    ) -> list[dict[str, Any]]:
        scored = self.retrieve_and_rank(
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
        return rank_v5_candidates(
            scored,
            mode=mode,
            top_k=top_k,
            v25_weight=v25_rank_weight,
            listwise_weight=listwise_rank_weight,
            rrf_k=v5_rrf_k,
        )
