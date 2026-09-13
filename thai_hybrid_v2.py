from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from thai_dense_v2 import DenseArtifacts, DenseEncoder, load_dense_artifacts, sense_text
from thai_lexical_v1 import SearchArtifacts, normalize_text, search as lexical_search


def weighted_rrf(
    *,
    lexical_rank: int | None,
    dense_rank: int | None,
    lexical_weight: float = 1.0,
    dense_weight: float = 1.0,
    k: int = 60,
) -> float:
    score = 0.0
    if lexical_rank is not None:
        score += lexical_weight / (k + lexical_rank)
    if dense_rank is not None:
        score += dense_weight / (k + dense_rank)
    return score


def lexical_relation_weight(relation_tier: int) -> float:
    """Scale lexical rank evidence by relation quality before RRF fusion."""
    if relation_tier >= 4:
        return 1.0
    if relation_tier == 3:
        return 0.5
    if relation_tier == 2:
        return 0.25
    # Tier 0-1 is weak/component evidence. Let dense semantics decide ordering
    # instead of rewarding a candidate merely for appearing in the lexical pool.
    return 0.0


def _selected_query(
    artifacts: SearchArtifacts,
    query: str,
    sense: int | None,
) -> tuple[str, dict[str, Any] | None, int | None]:
    query = normalize_text(query)
    entry_index = artifacts.word_to_index.get(query)
    if entry_index is None:
        if sense is not None:
            raise ValueError("--sense can only be used for an exact dictionary headword.")
        return query, None, None

    sense_ids = artifacts.entry_to_senses[entry_index]
    if not sense_ids:
        return query, None, entry_index

    if sense is None:
        sense_id = sense_ids[0]
    else:
        if sense < 1 or sense > len(sense_ids):
            raise ValueError(
                f"Sense {sense} is out of range for {query!r}; "
                f"available senses: 1-{len(sense_ids)}"
            )
        sense_id = sense_ids[sense - 1]

    record = artifacts.senses[sense_id]
    return (
        sense_text(artifacts, sense_id),
        {
            "sense": int(record["sense_index"]),
            "definition": record["definition"],
        },
        entry_index,
    )


def _top_dense_entries(
    artifacts: SearchArtifacts,
    dense_scores: np.ndarray,
    *,
    count: int,
    exclude_entry: int | None,
) -> list[tuple[int, float, int]]:
    order = np.argsort(-dense_scores)
    result: list[tuple[int, float, int]] = []
    seen: set[int] = set()

    for sense_id in order:
        entry_index = int(artifacts.senses[int(sense_id)]["entry_index"])
        if exclude_entry is not None and entry_index == exclude_entry:
            continue
        if entry_index in seen:
            continue
        seen.add(entry_index)
        result.append((entry_index, float(dense_scores[int(sense_id)]), int(sense_id)))
        if len(result) >= count:
            break
    return result


@dataclass
class HybridSearcher:
    lexical: SearchArtifacts
    dense: DenseArtifacts
    encoder: DenseEncoder

    @classmethod
    def from_paths(
        cls,
        lexical: SearchArtifacts,
        dense_index: str,
        *,
        device: str | None = None,
    ) -> "HybridSearcher":
        dense = load_dense_artifacts(dense_index, lexical_artifacts=lexical)
        encoder = DenseEncoder(dense.metadata, device=device)
        return cls(lexical=lexical, dense=dense, encoder=encoder)

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
        query_text, query_sense, query_entry_index = _selected_query(
            self.lexical,
            query,
            sense,
        )

        lexical_results = lexical_search(
            self.lexical,
            query,
            top_k=max(top_k, lexical_pool),
            candidate_pool=max(lexical_pool, 250),
            sense=sense,
        )
        lexical_by_entry: dict[int, dict[str, Any]] = {}
        lexical_rank: dict[int, int] = {}
        for rank, item in enumerate(lexical_results, start=1):
            entry_index = self.lexical.word_to_index.get(item["word"])
            if entry_index is None:
                continue
            lexical_by_entry[entry_index] = item
            lexical_rank[entry_index] = rank

        query_vector = self.encoder.encode_query(query_text)
        dense_scores = np.asarray(self.dense.embeddings @ query_vector, dtype=np.float32)
        dense_entries = _top_dense_entries(
            self.lexical,
            dense_scores,
            count=max(top_k, dense_pool),
            exclude_entry=query_entry_index,
        )
        dense_rank: dict[int, int] = {}
        dense_best: dict[int, tuple[float, int]] = {}
        for rank, (entry_index, score, sense_id) in enumerate(dense_entries, start=1):
            dense_rank[entry_index] = rank
            dense_best[entry_index] = (score, sense_id)

        candidate_entries = set(lexical_rank) | set(dense_rank)
        results: list[dict[str, Any]] = []

        for entry_index in candidate_entries:
            entry = self.lexical.entries[entry_index]
            lexical_item = lexical_by_entry.get(entry_index)
            dense_info = dense_best.get(entry_index)
            if dense_info is None:
                sense_ids = self.lexical.entry_to_senses[entry_index]
                best_sense_id = sense_ids[0] if sense_ids else None
                dense_similarity = None
            else:
                dense_similarity, best_sense_id = dense_info

            if lexical_item is not None:
                relation_tier = int(lexical_item.get("relation_tier", 0) or 0)
                lexical_form = lexical_item.get("lexical_form", "standalone")
                relation_hint = lexical_item.get("relation_hint", "definition_similar")
                lexical_score = float(lexical_item.get("score", 0.0))
                definition = lexical_item.get("definition")
                matched_candidate_sense = lexical_item.get("matched_candidate_sense")
                sense_resolution = lexical_item.get("sense_resolution")
            else:
                relation_tier = 0
                lexical_form = "standalone"
                relation_hint = "dense_semantic_similarity"
                lexical_score = 0.0
                sense_resolution = "dense_only"
                if best_sense_id is not None:
                    sense_record = self.lexical.senses[best_sense_id]
                    definition = sense_record["definition"]
                    matched_candidate_sense = {
                        "sense": int(sense_record["sense_index"]),
                        "definition": sense_record["definition"],
                    }
                else:
                    definition = entry.get("definition")
                    matched_candidate_sense = None

            protected_tier = (
                relation_tier
                if relation_tier >= 4 and lexical_form == "standalone"
                else 0
            )
            relation_weight = lexical_relation_weight(relation_tier)
            fusion_score = weighted_rrf(
                lexical_rank=lexical_rank.get(entry_index),
                dense_rank=dense_rank.get(entry_index),
                lexical_weight=lexical_weight * relation_weight,
                dense_weight=dense_weight,
                k=rrf_k,
            )

            results.append(
                {
                    "word": entry["word"],
                    "score": round(float(fusion_score), 8),
                    "protected_relation_tier": protected_tier,
                    "relation_tier": relation_tier,
                    "relation_hint": relation_hint,
                    "lexical_form": lexical_form,
                    "sense_resolution": sense_resolution,
                    "lexical_score": round(lexical_score, 6),
                    "lexical_rank": lexical_rank.get(entry_index),
                    "lexical_fusion_weight": round(
                        lexical_weight * relation_weight,
                        6,
                    ),
                    "dense_similarity": (
                        round(float(dense_similarity), 6)
                        if dense_similarity is not None
                        else None
                    ),
                    "dense_rank": dense_rank.get(entry_index),
                    "query_sense": query_sense,
                    "matched_candidate_sense": matched_candidate_sense,
                    "definition": definition,
                    "sense_count": entry.get("sense_count", 1),
                    "source_ids": entry.get("source_ids", []),
                }
            )

        results.sort(
            key=lambda item: (
                -int(item["protected_relation_tier"]),
                -float(item["score"]),
                -float(item["lexical_score"]),
                -float(item["dense_similarity"] or -1.0),
                item["word"],
            )
        )
        return results[:top_k]
