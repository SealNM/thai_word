from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


ARTIFACT_VERSION = 1
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
SPACE_RE = re.compile(r"\s+")


class DictionarySchemaError(ValueError):
    pass


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFC", str(value))
    text = ZERO_WIDTH_RE.sub("", text)
    return SPACE_RE.sub(" ", text).strip()


def flatten_definition(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, (int, float, bool)):
        return normalize_text(str(value))
    if isinstance(value, list):
        return normalize_text(" ".join(filter(None, (flatten_definition(v) for v in value))))
    if isinstance(value, dict):
        return normalize_text(" ".join(filter(None, (flatten_definition(v) for v in value.values()))))
    return normalize_text(str(value))


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise DictionarySchemaError("Top-level JSON must be an array of dictionary records.")
    bad = [i for i, row in enumerate(data[:100]) if not isinstance(row, dict)]
    if bad:
        raise DictionarySchemaError(f"Dictionary records must be JSON objects. Invalid sample indices: {bad[:5]}")
    return data


def inspect_records(
    records: list[dict[str, Any]],
    word_field: str = "headword_text",
    definition_field: str = "definition",
) -> dict[str, Any]:
    key_counts: Counter[str] = Counter()
    for row in records:
        key_counts.update(row.keys())

    words = [normalize_text(row.get(word_field, "")) for row in records]
    words = [word for word in words if word]
    definitions = [flatten_definition(row.get(definition_field)) for row in records]
    definitions_present = sum(bool(value) for value in definitions)

    likely_definition_fields = []
    for key, count in key_counts.most_common():
        low = key.lower()
        if any(token in low for token in ("definition", "meaning", "sense", "description", "gloss")):
            likely_definition_fields.append({"field": key, "records": count})

    return {
        "records": len(records),
        "fields": dict(key_counts),
        "word_field": word_field,
        "word_field_records": len(words),
        "unique_words": len(set(words)),
        "duplicate_word_rows": max(0, len(words) - len(set(words))),
        "definition_field": definition_field,
        "definition_records": definitions_present,
        "definition_coverage": (definitions_present / len(records)) if records else 0.0,
        "likely_definition_fields": likely_definition_fields,
    }


def split_tokens(text: str) -> list[str]:
    return text.split()


def _thai_tokenizer(headwords: Iterable[str]):
    try:
        from pythainlp.corpus.common import thai_stopwords
        from pythainlp.tokenize import word_tokenize
        from pythainlp.util import Trie
    except ImportError as exc:
        raise RuntimeError(
            "PyThaiNLP is required. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    custom_dict = Trie(sorted(set(headwords)))
    stopwords = set(thai_stopwords())

    def tokenize(text: str) -> list[str]:
        tokens = word_tokenize(
            normalize_text(text),
            engine="newmm",
            keep_whitespace=False,
            custom_dict=custom_dict,
        )
        cleaned: list[str] = []
        for token in tokens:
            token = normalize_text(token)
            if not token or token.isspace() or token in stopwords:
                continue
            if not any(ch.isalnum() or "\u0e00" <= ch <= "\u0e7f" for ch in token):
                continue
            cleaned.append(token)
        return cleaned

    return tokenize


def prepare_entries(
    records: list[dict[str, Any]],
    word_field: str,
    definition_field: str,
) -> list[dict[str, Any]]:
    if not records:
        raise DictionarySchemaError("Dictionary is empty.")
    if not any(word_field in row for row in records):
        raise DictionarySchemaError(f"Word field {word_field!r} does not exist.")
    if not any(definition_field in row for row in records):
        fields = sorted({key for row in records[:1000] for key in row})
        raise DictionarySchemaError(
            f"Definition field {definition_field!r} does not exist. Available fields include: {fields}"
        )

    merged: dict[str, dict[str, Any]] = {}
    for row in records:
        word = normalize_text(row.get(word_field, ""))
        definition = flatten_definition(row.get(definition_field))
        if not word or not definition:
            continue
        if word not in merged:
            merged[word] = {
                "word": word,
                "definitions": [definition],
                "source_ids": [row.get("headword_ID")],
            }
        else:
            if definition not in merged[word]["definitions"]:
                merged[word]["definitions"].append(definition)
            merged[word]["source_ids"].append(row.get("headword_ID"))

    if not merged:
        raise DictionarySchemaError(
            f"No usable word+definition pairs found using fields {word_field!r} and {definition_field!r}."
        )

    entries = []
    for item in merged.values():
        entries.append(
            {
                "word": item["word"],
                "definition": normalize_text(" ".join(item["definitions"])),
                "source_ids": [value for value in item["source_ids"] if value is not None],
            }
        )
    entries.sort(key=lambda row: row["word"])
    return entries


def build_index(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    word_field: str = "headword_text",
    definition_field: str = "definition",
    min_df: int = 1,
    max_features: int | None = None,
) -> dict[str, Any]:
    records = load_json_records(input_path)
    entries = prepare_entries(records, word_field, definition_field)
    words = [row["word"] for row in entries]
    tokenize = _thai_tokenizer(words)

    token_lists = [tokenize(row["definition"]) for row in entries]
    tokenized_definitions = [" ".join(tokens) for tokens in token_lists]

    vectorizer = TfidfVectorizer(
        tokenizer=split_tokens,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(tokenized_definitions).tocsr()

    word_to_index = {word: i for i, word in enumerate(words)}
    references: list[list[int]] = []
    token_sets: list[set[str]] = []

    for i, tokens in enumerate(token_lists):
        unique_tokens = set(tokens)
        token_sets.append(unique_tokens)
        refs = sorted(
            {
                word_to_index[token]
                for token in unique_tokens
                if token in word_to_index and word_to_index[token] != i
            }
        )
        references.append(refs)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "entries.json").open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, ensure_ascii=False)

    with (out / "word_to_index.json").open("w", encoding="utf-8") as handle:
        json.dump(word_to_index, handle, ensure_ascii=False)

    sparse.save_npz(out / "tfidf_matrix.npz", matrix)
    joblib.dump(vectorizer, out / "vectorizer.joblib", compress=3)
    joblib.dump(references, out / "references.joblib", compress=3)
    joblib.dump(token_sets, out / "token_sets.joblib", compress=3)

    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "word_field": word_field,
        "definition_field": definition_field,
        "records_input": len(records),
        "entries_indexed": len(entries),
        "features": int(matrix.shape[1]),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "matrix_nnz": int(matrix.nnz),
        "matrix_dtype": str(matrix.dtype),
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return metadata


@dataclass
class SearchArtifacts:
    entries: list[dict[str, Any]]
    word_to_index: dict[str, int]
    vectorizer: TfidfVectorizer
    matrix: sparse.csr_matrix
    references: list[list[int]]
    token_sets: list[set[str]]
    tokenize: Any


def load_artifacts(output_dir: str | Path) -> SearchArtifacts:
    out = Path(output_dir)
    with (out / "entries.json").open("r", encoding="utf-8") as handle:
        entries = json.load(handle)
    with (out / "word_to_index.json").open("r", encoding="utf-8") as handle:
        word_to_index = {key: int(value) for key, value in json.load(handle).items()}

    vectorizer = joblib.load(out / "vectorizer.joblib")
    matrix = sparse.load_npz(out / "tfidf_matrix.npz").tocsr()
    references = joblib.load(out / "references.joblib")
    token_sets = joblib.load(out / "token_sets.joblib")
    tokenize = _thai_tokenizer(word_to_index.keys())

    return SearchArtifacts(
        entries=entries,
        word_to_index=word_to_index,
        vectorizer=vectorizer,
        matrix=matrix,
        references=references,
        token_sets=token_sets,
        tokenize=tokenize,
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def search(
    artifacts: SearchArtifacts,
    query: str,
    *,
    top_k: int = 20,
    candidate_pool: int = 250,
) -> list[dict[str, Any]]:
    query = normalize_text(query)
    if not query:
        return []

    query_index = artifacts.word_to_index.get(query)
    if query_index is not None:
        query_vector = artifacts.matrix[query_index]
        query_tokens = artifacts.token_sets[query_index]
        query_refs = set(artifacts.references[query_index])
    else:
        query_tokens_list = artifacts.tokenize(query)
        query_tokens = set(query_tokens_list)
        query_vector = artifacts.vectorizer.transform([" ".join(query_tokens_list)])
        query_refs = set()

    cosine_scores = linear_kernel(query_vector, artifacts.matrix).ravel()
    total = len(cosine_scores)
    pool_size = min(max(top_k * 5, candidate_pool), total)

    if pool_size == total:
        candidate_indices = np.argsort(-cosine_scores)
    else:
        raw = np.argpartition(-cosine_scores, pool_size - 1)[:pool_size]
        candidate_indices = raw[np.argsort(-cosine_scores[raw])]

    results: list[dict[str, Any]] = []
    for idx_value in candidate_indices:
        idx = int(idx_value)
        if query_index is not None and idx == query_index:
            continue

        candidate = artifacts.entries[idx]
        cosine = float(cosine_scores[idx])
        candidate_refs = set(artifacts.references[idx])

        forward_ref = 1.0 if idx in query_refs else 0.0
        reverse_ref = 1.0 if query_index is not None and query_index in candidate_refs else 0.0
        direct_reference = max(forward_ref, reverse_ref)

        shared = _jaccard(query_tokens, artifacts.token_sets[idx])
        word_form = SequenceMatcher(None, query, candidate["word"]).ratio()

        score = (
            0.55 * cosine
            + 0.25 * direct_reference
            + 0.15 * shared
            + 0.05 * word_form
        )

        if forward_ref:
            relation_hint = "definition_mentions_candidate"
        elif reverse_ref:
            relation_hint = "candidate_defined_via_query"
        elif query in candidate["word"] or candidate["word"] in query:
            relation_hint = "compound_or_form_related"
        else:
            relation_hint = "definition_similar"

        results.append(
            {
                "word": candidate["word"],
                "score": round(float(score), 6),
                "relation_hint": relation_hint,
                "signals": {
                    "definition_cosine": round(cosine, 6),
                    "direct_reference": direct_reference,
                    "shared_tokens": round(shared, 6),
                    "word_form": round(float(word_form), 6),
                },
                "definition": candidate["definition"],
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]
