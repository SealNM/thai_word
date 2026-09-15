from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from thai_dense_v2 import sense_text
from thai_lexical_v1 import SearchArtifacts, normalize_text


def _has_word(model: Any, word: str) -> bool:
    word = normalize_text(word)
    if not word:
        return False
    key_to_index = getattr(model, "key_to_index", None)
    if isinstance(key_to_index, dict):
        return word in key_to_index
    try:
        return word in model
    except TypeError:
        return False


def _get_vector(model: Any, word: str) -> np.ndarray | None:
    word = normalize_text(word)
    if not _has_word(model, word):
        return None
    if hasattr(model, "get_vector"):
        vector = model.get_vector(word)
    else:
        vector = model[word]
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1:
        return None
    return array


def _normalize(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        return None
    return (vector / norm).astype(np.float32, copy=False)


def mean_token_vector(
    text: str,
    *,
    model: Any,
    tokenizer: Callable[[str], Sequence[str]],
) -> tuple[np.ndarray | None, int, int]:
    tokens = [normalize_text(token) for token in tokenizer(normalize_text(text))]
    tokens = [token for token in tokens if token]
    vectors: list[np.ndarray] = []
    for token in tokens:
        vector = _get_vector(model, token)
        if vector is not None:
            vectors.append(vector)
    if not vectors:
        return None, 0, len(tokens)
    mean = np.mean(np.vstack(vectors), axis=0, dtype=np.float32)
    return _normalize(mean), len(vectors), len(tokens)


@dataclass
class WordVectorIndex:
    model_name: str
    vector_size: int
    headword_matrix: np.ndarray
    headword_valid: np.ndarray
    sense_matrix: np.ndarray
    sense_valid: np.ndarray
    headword_coverage: float
    sense_coverage: float


def load_pythainlp_model(model_name: str) -> Any:
    try:
        from pythainlp.word_vector import WordVector
    except ImportError as exc:
        raise RuntimeError(
            "PyThaiNLP word vectors require PyThaiNLP and gensim. "
            "Install: pip install -r requirements-wordvectors.txt"
        ) from exc
    return WordVector(model_name=model_name).get_model()


def build_index(
    artifacts: SearchArtifacts,
    *,
    model_name: str,
    model: Any | None = None,
    tokenizer: Callable[[str], Sequence[str]] | None = None,
) -> WordVectorIndex:
    if model is None:
        model = load_pythainlp_model(model_name)

    if tokenizer is None:
        try:
            from pythainlp.tokenize import word_tokenize
        except ImportError as exc:
            raise RuntimeError("PyThaiNLP tokenization is required.") from exc

        def tokenizer(text: str) -> Sequence[str]:
            return word_tokenize(
                text,
                engine="newmm",
                keep_whitespace=False,
            )

    vector_size = int(getattr(model, "vector_size", 0) or 0)
    if vector_size <= 0:
        sample = None
        for entry in artifacts.entries:
            sample = _get_vector(model, entry["word"])
            if sample is not None:
                break
        if sample is None:
            raise ValueError(f"No dictionary headword exists in model {model_name!r}.")
        vector_size = int(sample.shape[0])

    headword_matrix = np.zeros(
        (len(artifacts.entries), vector_size),
        dtype=np.float32,
    )
    headword_valid = np.zeros(len(artifacts.entries), dtype=bool)

    for entry_index, entry in enumerate(artifacts.entries):
        vector = _normalize(_get_vector(model, entry["word"]))
        if vector is None:
            continue
        headword_matrix[entry_index] = vector
        headword_valid[entry_index] = True

    sense_matrix = np.zeros(
        (len(artifacts.senses), vector_size),
        dtype=np.float32,
    )
    sense_valid = np.zeros(len(artifacts.senses), dtype=bool)

    for sense_id in range(len(artifacts.senses)):
        vector, _, _ = mean_token_vector(
            sense_text(artifacts, sense_id),
            model=model,
            tokenizer=tokenizer,
        )
        if vector is None:
            continue
        sense_matrix[sense_id] = vector
        sense_valid[sense_id] = True

    return WordVectorIndex(
        model_name=model_name,
        vector_size=vector_size,
        headword_matrix=headword_matrix,
        headword_valid=headword_valid,
        sense_matrix=sense_matrix,
        sense_valid=sense_valid,
        headword_coverage=float(headword_valid.mean()) if len(headword_valid) else 0.0,
        sense_coverage=float(sense_valid.mean()) if len(sense_valid) else 0.0,
    )


def _query_entry_and_sense(
    artifacts: SearchArtifacts,
    query: str,
    sense: int | None,
) -> tuple[int | None, int | None]:
    query = normalize_text(query)
    entry_index = artifacts.word_to_index.get(query)
    if entry_index is None:
        return None, None

    sense_ids = artifacts.entry_to_senses[entry_index]
    if not sense_ids:
        return entry_index, None

    if sense is None:
        return entry_index, int(sense_ids[0])

    if sense < 1 or sense > len(sense_ids):
        raise ValueError(
            f"Sense {sense} is out of range for {query!r}; "
            f"available senses: 1-{len(sense_ids)}"
        )
    return entry_index, int(sense_ids[sense - 1])


def _top_entry_results(
    artifacts: SearchArtifacts,
    scores: np.ndarray,
    *,
    valid: np.ndarray,
    top_k: int,
    exclude_entry: int | None,
) -> list[dict[str, Any]]:
    masked = np.asarray(scores, dtype=np.float32).copy()
    masked[~valid] = -np.inf
    if exclude_entry is not None and 0 <= exclude_entry < len(masked):
        masked[exclude_entry] = -np.inf

    count = min(max(0, top_k), int(valid.sum()))
    if count <= 0:
        return []

    order = np.argpartition(-masked, min(count - 1, len(masked) - 1))[:count]
    order = order[np.argsort(-masked[order])]
    return [
        {
            "word": artifacts.entries[int(entry_index)]["word"],
            "score": round(float(masked[int(entry_index)]), 6),
        }
        for entry_index in order
        if np.isfinite(masked[int(entry_index)])
    ]


def search_headwords(
    artifacts: SearchArtifacts,
    index: WordVectorIndex,
    query: str,
    *,
    model: Any,
    tokenizer: Callable[[str], Sequence[str]],
    top_k: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query = normalize_text(query)
    entry_index = artifacts.word_to_index.get(query)
    vector = _normalize(_get_vector(model, query))
    source = "exact"

    token_hits = 0
    token_total = 0
    if vector is None:
        vector, token_hits, token_total = mean_token_vector(
            query,
            model=model,
            tokenizer=tokenizer,
        )
        source = "token_mean"

    if vector is None:
        return [], {
            "available": False,
            "source": source,
            "token_hits": token_hits,
            "token_total": token_total,
        }

    scores = index.headword_matrix @ vector
    return (
        _top_entry_results(
            artifacts,
            scores,
            valid=index.headword_valid,
            top_k=top_k,
            exclude_entry=entry_index,
        ),
        {
            "available": True,
            "source": source,
            "token_hits": token_hits,
            "token_total": token_total,
        },
    )


def search_senses(
    artifacts: SearchArtifacts,
    index: WordVectorIndex,
    query: str,
    *,
    sense: int | None,
    model: Any,
    tokenizer: Callable[[str], Sequence[str]],
    top_k: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entry_index, query_sense_id = _query_entry_and_sense(
        artifacts,
        query,
        sense,
    )

    if query_sense_id is not None:
        text = sense_text(artifacts, query_sense_id)
    else:
        text = normalize_text(query)

    query_vector, token_hits, token_total = mean_token_vector(
        text,
        model=model,
        tokenizer=tokenizer,
    )
    if query_vector is None:
        return [], {
            "available": False,
            "token_hits": token_hits,
            "token_total": token_total,
        }

    scores = index.sense_matrix @ query_vector
    masked = np.asarray(scores, dtype=np.float32).copy()
    masked[~index.sense_valid] = -np.inf

    order = np.argsort(-masked)
    seen: set[int] = set()
    results: list[dict[str, Any]] = []

    for sense_id in order:
        score = float(masked[int(sense_id)])
        if not np.isfinite(score):
            break
        candidate_entry = int(artifacts.senses[int(sense_id)]["entry_index"])
        if candidate_entry == entry_index or candidate_entry in seen:
            continue
        seen.add(candidate_entry)
        candidate_sense = artifacts.senses[int(sense_id)]
        results.append(
            {
                "word": artifacts.entries[candidate_entry]["word"],
                "score": round(score, 6),
                "sense": int(candidate_sense["sense_index"]),
                "definition": candidate_sense["definition"],
            }
        )
        if len(results) >= top_k:
            break

    return results, {
        "available": True,
        "token_hits": token_hits,
        "token_total": token_total,
    }
