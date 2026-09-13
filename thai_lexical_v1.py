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


ARTIFACT_VERSION = 2
DEFAULT_INPUT_PATH = "thai_dictionary.json"
DEFAULT_ID_FIELD = "word_ID"
DEFAULT_WORD_FIELD = "headword_text"
DEFAULT_DEFINITION_FIELD = "definition_text"
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
    id_field: str = DEFAULT_ID_FIELD,
    word_field: str = DEFAULT_WORD_FIELD,
    definition_field: str = DEFAULT_DEFINITION_FIELD,
) -> dict[str, Any]:
    key_counts: Counter[str] = Counter()
    for row in records:
        key_counts.update(row.keys())

    ids = [row.get(id_field) for row in records if row.get(id_field) is not None]
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
        "id_field": id_field,
        "id_field_records": len(ids),
        "unique_ids": len(set(map(str, ids))),
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
    id_field: str,
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

        source_id = row.get(id_field)
        if word not in merged:
            merged[word] = {
                "word": word,
                "definitions": [definition],
                "source_ids": [source_id] if source_id is not None else [],
            }
        else:
            if definition not in merged[word]["definitions"]:
                merged[word]["definitions"].append(definition)
            if source_id is not None and source_id not in merged[word]["source_ids"]:
                merged[word]["source_ids"].append(source_id)

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
                "definitions": item["definitions"],
                "source_ids": item["source_ids"],
                "sense_count": len(item["definitions"]),
            }
        )
    entries.sort(key=lambda row: row["word"])
    return entries


def build_index(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    id_field: str = DEFAULT_ID_FIELD,
    word_field: str = DEFAULT_WORD_FIELD,
    definition_field: str = DEFAULT_DEFINITION_FIELD,
    min_df: int = 1,
    max_features: int | None = None,
) -> dict[str, Any]:
    records = load_json_records(input_path)
    entries = prepare_entries(records, id_field, word_field, definition_field)
    words = [row["word"] for row in entries]
    tokenize = _thai_tokenizer(words)
    word_to_index = {word: i for i, word in enumerate(words)}

    senses: list[dict[str, Any]] = []
    token_lists: list[list[str]] = []
    entry_to_senses: list[list[int]] = [[] for _ in entries]

    for entry_index, entry in enumerate(entries):
        for sense_index, definition in enumerate(entry["definitions"], start=1):
            sense_id = len(senses)
            tokens = tokenize(definition)
            senses.append(
                {
                    "entry_index": entry_index,
                    "sense_index": sense_index,
                    "definition": definition,
                }
            )
            token_lists.append(tokens)
            entry_to_senses[entry_index].append(sense_id)

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

    references: list[list[int]] = []
    token_sets: list[set[str]] = []
    reverse_reference_senses: list[list[int]] = [[] for _ in entries]

    for sense_id, tokens in enumerate(token_lists):
        unique_tokens = set(tokens)
        token_sets.append(unique_tokens)
        own_entry_index = senses[sense_id]["entry_index"]
        refs = sorted(
            {
                word_to_index[token]
                for token in unique_tokens
                if token in word_to_index and word_to_index[token] != own_entry_index
            }
        )
        references.append(refs)
        for referenced_entry in refs:
            reverse_reference_senses[referenced_entry].append(sense_id)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "entries.json").open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, ensure_ascii=False)

    with (out / "senses.json").open("w", encoding="utf-8") as handle:
        json.dump(senses, handle, ensure_ascii=False)

    with (out / "word_to_index.json").open("w", encoding="utf-8") as handle:
        json.dump(word_to_index, handle, ensure_ascii=False)

    sparse.save_npz(out / "tfidf_matrix.npz", matrix)
    joblib.dump(vectorizer, out / "vectorizer.joblib", compress=3)
    joblib.dump(references, out / "references.joblib", compress=3)
    joblib.dump(reverse_reference_senses, out / "reverse_references.joblib", compress=3)
    joblib.dump(token_sets, out / "token_sets.joblib", compress=3)
    joblib.dump(entry_to_senses, out / "entry_to_senses.joblib", compress=3)

    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "id_field": id_field,
        "word_field": word_field,
        "definition_field": definition_field,
        "records_input": len(records),
        "entries_indexed": len(entries),
        "senses_indexed": len(senses),
        "features": int(matrix.shape[1]),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "matrix_nnz": int(matrix.nnz),
        "matrix_dtype": str(matrix.dtype),
        "representation": "sense_level",
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return metadata


@dataclass
class SearchArtifacts:
    entries: list[dict[str, Any]]
    senses: list[dict[str, Any]]
    word_to_index: dict[str, int]
    vectorizer: TfidfVectorizer
    matrix: sparse.csr_matrix
    references: list[list[int]]
    reverse_references: list[list[int]]
    token_sets: list[set[str]]
    entry_to_senses: list[list[int]]
    tokenize: Any


def load_artifacts(output_dir: str | Path) -> SearchArtifacts:
    out = Path(output_dir)
    with (out / "entries.json").open("r", encoding="utf-8") as handle:
        entries = json.load(handle)
    with (out / "senses.json").open("r", encoding="utf-8") as handle:
        senses = json.load(handle)
    with (out / "word_to_index.json").open("r", encoding="utf-8") as handle:
        word_to_index = {key: int(value) for key, value in json.load(handle).items()}

    vectorizer = joblib.load(out / "vectorizer.joblib")
    matrix = sparse.load_npz(out / "tfidf_matrix.npz").tocsr()
    references = joblib.load(out / "references.joblib")
    reverse_references = joblib.load(out / "reverse_references.joblib")
    token_sets = joblib.load(out / "token_sets.joblib")
    entry_to_senses = joblib.load(out / "entry_to_senses.joblib")
    tokenize = _thai_tokenizer(word_to_index.keys())

    return SearchArtifacts(
        entries=entries,
        senses=senses,
        word_to_index=word_to_index,
        vectorizer=vectorizer,
        matrix=matrix,
        references=references,
        reverse_references=reverse_references,
        token_sets=token_sets,
        entry_to_senses=entry_to_senses,
        tokenize=tokenize,
    )


def list_senses(artifacts: SearchArtifacts, query: str) -> list[dict[str, Any]]:
    query = normalize_text(query)
    entry_index = artifacts.word_to_index.get(query)
    if entry_index is None:
        return []
    result = []
    for sense_id in artifacts.entry_to_senses[entry_index]:
        sense = artifacts.senses[sense_id]
        result.append(
            {
                "sense": int(sense["sense_index"]),
                "definition": sense["definition"],
            }
        )
    return result


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _reference_strength(definition: str, target_word: str) -> float:
    """Estimate whether a headword mention is definitional or merely contextual.

    This intentionally stays dictionary-only. A bare/alternative gloss such as
    "ฝน." or "เมฆ, ฝน." is much stronger than an example such as
    "เช่น เมฆอุ้มฝน" or an associated concept such as "เทวดาแห่งฝน".
    """
    definition = normalize_text(definition)
    target_word = normalize_text(target_word)
    if not definition or not target_word or target_word not in definition:
        return 0.0

    stripped = definition.strip(" \t\r\n.,;:!?()[]{}“”‘’\"'")
    if stripped == target_word:
        return 1.0

    clauses = [
        part.strip(" \t\r\n.,;:!?()[]{}“”‘’\"'")
        for part in re.split(r"[,;]", definition)
    ]
    if any(part == target_word for part in clauses):
        return 0.95

    target_pos = definition.find(target_word)
    prefix = definition[:target_pos]

    contextual_markers = (
        "เช่น",
        "เช่นว่า",
        "ตัวอย่าง",
        "ในคำว่า",
        "ใช้ว่า",
        "อาทิ",
    )
    if any(marker in prefix for marker in contextual_markers):
        return 0.20

    associated_markers = (
        "แห่ง",
        "ไม่มี",
        "ปราศจาก",
        "เกี่ยวกับ",
        "สำหรับ",
    )
    if any(prefix.rstrip().endswith(marker) for marker in associated_markers):
        return 0.15

    first_clause = clauses[0] if clauses else stripped
    if first_clause.startswith(target_word):
        # "ฝนเม็ดใหญ่..." / "ฝนชนิด..." are useful type-of relations,
        # but weaker than a direct gloss "ฝน."
        return 0.65

    if target_word in first_clause:
        return 0.50

    alias_markers = ("เรียกว่า", "ก็เรียก", "หรือเรียก")
    if any(marker in definition for marker in alias_markers):
        return 0.45

    return 0.30


def _forward_reference_strength(definition: str, target_word: str) -> float:
    """Estimate how strongly the query definition presents target_word as a gloss.

    Unlike reverse references, a word mentioned inside the query definition may
    be a true gloss ("ไว เช่น ..."), an alias ("พูดจา ก็ว่า"), an example
    ("กินเร็ว"), or even part of a negated contrast ("ไม่ชักช้า"). These cases
    should not receive the same lexical weight.
    """
    definition = normalize_text(definition)
    target_word = normalize_text(target_word)
    if not definition or not target_word or target_word not in definition:
        return 0.0

    clauses = [
        part.strip(" \t\r\n.,;:!?()[]{}“”‘’\"'")
        for part in re.split(r"[,;]", definition)
    ]
    best = 0.0

    for clause in clauses:
        if target_word not in clause:
            continue

        start = 0
        while True:
            target_pos = clause.find(target_word, start)
            if target_pos < 0:
                break

            before = clause[:target_pos].strip()
            after = clause[target_pos + len(target_word):].strip()

            negation_markers = ("ไม่", "มิ", "มิได้", "หาไม่", "ปราศจาก", "ไร้")
            if any(before.endswith(marker) for marker in negation_markers):
                strength = 0.0
            else:
                example_markers = ("เช่น", "ตัวอย่าง", "ในคำว่า", "อาทิ")
                after_example = any(marker in before for marker in example_markers)

                if after_example:
                    strength = 0.10
                elif not before:
                    alias_markers = ("ก็ว่า", "เรียกว่า", "ใช้ว่า", "หรือว่า")
                    if not after:
                        strength = 1.0
                    elif any(marker in after for marker in alias_markers):
                        strength = 1.0
                    elif after.startswith("เช่น"):
                        strength = 0.95
                    else:
                        # A leading word in a descriptive clause is often the
                        # primary gloss, but is weaker than an explicit alias.
                        strength = 0.70
                else:
                    # Mid-clause mentions normally describe components rather
                    # than interchangeable words.
                    strength = 0.25

            best = max(best, strength)
            start = target_pos + len(target_word)

    return best


def _is_bound_form(word: str) -> bool:
    word = normalize_text(word)
    return word.startswith("-") or word.endswith("-")


def _relation_tier(
    *,
    reverse_strength: float,
    forward_strength: float,
    candidate_word: str,
) -> int:
    """Return a lexical relation tier where higher is more useful for thesaurus search."""
    if reverse_strength >= 0.999:
        tier = 5  # exact gloss / synonym: "ฝน."
    elif reverse_strength >= 0.90:
        tier = 4  # alternative gloss: "เมฆ, ฝน."
    elif reverse_strength >= 0.60:
        tier = 3  # subtype / kind-of: "ฝนเม็ดใหญ่..."
    elif reverse_strength > 0:
        tier = 2  # contextual or associated mention
    elif forward_strength >= 0.90:
        tier = 4  # query gloss / alias: "ไว เช่น ..." or "พูดจา ก็ว่า"
    elif forward_strength >= 0.60:
        tier = 3  # leading descriptive gloss
    elif forward_strength >= 0.20:
        tier = 1  # definition component
    else:
        tier = 0  # example mention or distributional similarity only

    # Dictionary combining forms such as "พรรษ-" are useful metadata but are
    # less directly usable by writers as standalone lexical choices.
    if _is_bound_form(candidate_word) and tier > 0:
        tier -= 1
    return tier


def _hierarchical_score(
    relation_tier: int,
    *,
    cosine: float,
    shared: float,
    word_form: float,
    exact_headword_query: bool,
) -> float:
    semantic_tiebreak = (
        0.55 * max(0.0, min(1.0, cosine))
        + 0.25 * max(0.0, min(1.0, shared))
        + 0.20 * max(0.0, min(1.0, word_form))
    )

    if not exact_headword_query:
        return semantic_tiebreak

    # Tier gaps (0.17) are deliberately wider than the maximum tiebreak
    # contribution (0.15), so a lower relation class can never overtake a
    # stronger lexical relation merely because its definitions share tokens.
    return 0.17 * relation_tier + 0.15 * semantic_tiebreak


def _top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=np.int64)
    count = min(count, len(scores))
    if count == len(scores):
        return np.argsort(-scores)
    raw = np.argpartition(-scores, count - 1)[:count]
    return raw[np.argsort(-scores[raw])]


def search(
    artifacts: SearchArtifacts,
    query: str,
    *,
    top_k: int = 20,
    candidate_pool: int = 250,
    sense: int | None = None,
) -> list[dict[str, Any]]:
    query = normalize_text(query)
    if not query:
        return []

    query_entry_index = artifacts.word_to_index.get(query)

    if query_entry_index is not None:
        available_query_senses = artifacts.entry_to_senses[query_entry_index]
        if not available_query_senses:
            return []
        if sense is None:
            selected_query_senses = [available_query_senses[0]]
        else:
            if sense < 1 or sense > len(available_query_senses):
                raise ValueError(
                    f"Sense {sense} is out of range for {query!r}; "
                    f"available senses: 1-{len(available_query_senses)}"
                )
            selected_query_senses = [available_query_senses[sense - 1]]

        query_matrix = artifacts.matrix[selected_query_senses]
        query_token_sets = [artifacts.token_sets[sense_id] for sense_id in selected_query_senses]
        query_reference_sets = [set(artifacts.references[sense_id]) for sense_id in selected_query_senses]
    else:
        if sense is not None:
            raise ValueError("--sense can only be used when the query exactly matches a dictionary headword.")
        query_tokens_list = artifacts.tokenize(query)
        if not query_tokens_list:
            return []
        query_matrix = artifacts.vectorizer.transform([" ".join(query_tokens_list)])
        selected_query_senses = []
        query_token_sets = [set(query_tokens_list)]
        query_reference_sets = [set()]

    pair_cosine = linear_kernel(query_matrix, artifacts.matrix)
    if pair_cosine.ndim == 1:
        pair_cosine = pair_cosine.reshape(1, -1)
    sense_cosine = np.asarray(pair_cosine).max(axis=0)

    entry_cosine = np.full(len(artifacts.entries), -1.0, dtype=np.float32)
    for sense_id, cosine in enumerate(sense_cosine):
        entry_index = int(artifacts.senses[sense_id]["entry_index"])
        if cosine > entry_cosine[entry_index]:
            entry_cosine[entry_index] = float(cosine)

    pool_size = min(max(top_k * 5, candidate_pool), len(artifacts.entries))
    candidate_entries = set(map(int, _top_indices(entry_cosine, pool_size)))

    if query_entry_index is not None:
        for refs in query_reference_sets:
            candidate_entries.update(refs)
        for sense_id in artifacts.reverse_references[query_entry_index]:
            candidate_entries.add(int(artifacts.senses[sense_id]["entry_index"]))
        candidate_entries.discard(query_entry_index)

    results: list[dict[str, Any]] = []
    for entry_index in candidate_entries:
        candidate = artifacts.entries[entry_index]
        word_form = SequenceMatcher(None, query, candidate["word"]).ratio()

        best: dict[str, Any] | None = None
        for candidate_sense_id in artifacts.entry_to_senses[entry_index]:
            candidate_tokens = artifacts.token_sets[candidate_sense_id]
            candidate_refs = set(artifacts.references[candidate_sense_id])

            for query_position in range(pair_cosine.shape[0]):
                cosine = float(pair_cosine[query_position, candidate_sense_id])
                query_tokens = query_token_sets[query_position]
                query_refs = query_reference_sets[query_position]

                query_sense_id = (
                    selected_query_senses[query_position]
                    if selected_query_senses
                    else None
                )
                query_definition = (
                    artifacts.senses[query_sense_id]["definition"]
                    if query_sense_id is not None
                    else query
                )

                forward_ref = 1.0 if entry_index in query_refs else 0.0
                reverse_ref = (
                    1.0
                    if query_entry_index is not None and query_entry_index in candidate_refs
                    else 0.0
                )
                forward_strength = (
                    _forward_reference_strength(query_definition, candidate["word"])
                    if forward_ref
                    else 0.0
                )
                reverse_strength = (
                    _reference_strength(artifacts.senses[candidate_sense_id]["definition"], query)
                    if reverse_ref
                    else 0.0
                )
                shared = _jaccard(query_tokens, candidate_tokens)
                entry_semantic_cosine = max(0.0, float(entry_cosine[entry_index]))
                sense_reference_ambiguous = bool(
                    query_entry_index is not None
                    and len(available_query_senses) > 1
                    and reverse_strength >= 0.90
                    and entry_semantic_cosine < 0.02
                    and shared == 0.0
                )

                relation_tier = _relation_tier(
                    reverse_strength=reverse_strength,
                    forward_strength=forward_strength,
                    candidate_word=candidate["word"],
                )

                # An explicitly selected sense is strict. A bare gloss such as
                # "รัก." points only to the ambiguous headword and cannot prove
                # which homonym/sense it means. Keep it visible, but do not let
                # it outrank evidence tied to the selected sense.
                if sense is not None and sense_reference_ambiguous:
                    relation_tier = min(relation_tier, 2)

                score = _hierarchical_score(
                    relation_tier,
                    cosine=cosine,
                    shared=shared,
                    word_form=word_form,
                    exact_headword_query=query_entry_index is not None,
                )

                if sense is not None and sense_reference_ambiguous:
                    relation_hint = "ambiguous_headword_reference"
                elif forward_ref and reverse_ref:
                    relation_hint = "mutual_definition_reference"
                elif reverse_ref and reverse_strength >= 0.999:
                    relation_hint = "direct_gloss_or_synonym"
                elif reverse_ref and reverse_strength >= 0.90:
                    relation_hint = "alternative_direct_gloss"
                elif reverse_ref and reverse_strength >= 0.60:
                    relation_hint = "defined_as_kind_of_query"
                elif reverse_ref:
                    relation_hint = "candidate_mentions_query"
                elif forward_strength >= 0.90:
                    relation_hint = "query_gloss_or_alias"
                elif forward_strength >= 0.60:
                    relation_hint = "query_leading_gloss"
                elif forward_strength >= 0.20:
                    relation_hint = "query_definition_component"
                elif forward_ref:
                    relation_hint = "query_example_mentions_candidate"
                elif query in candidate["word"] or candidate["word"] in query:
                    relation_hint = "compound_or_form_related"
                else:
                    relation_hint = "definition_similar"

                if _is_bound_form(candidate["word"]):
                    relation_hint = f"bound_form:{relation_hint}"

                if best is None or score > best["score_raw"]:
                    best = {
                        "score_raw": float(score),
                        "relation_tier": relation_tier,
                        "cosine": cosine,
                        "forward_reference": forward_strength,
                        "reverse_reference": reverse_strength,
                        "shared": shared,
                        "relation_hint": relation_hint,
                        "candidate_sense_id": candidate_sense_id,
                        "query_sense_id": query_sense_id,
                        "sense_reference_ambiguous": sense_reference_ambiguous,
                        "entry_semantic_cosine": entry_semantic_cosine,
                    }

        if best is None:
            continue

        candidate_sense = artifacts.senses[best["candidate_sense_id"]]
        query_sense_record = (
            artifacts.senses[best["query_sense_id"]]
            if best["query_sense_id"] is not None
            else None
        )

        results.append(
            {
                "word": candidate["word"],
                "score": round(best["score_raw"], 6),
                "relation_hint": best["relation_hint"],
                "relation_tier": best["relation_tier"],
                "lexical_form": "bound_form" if _is_bound_form(candidate["word"]) else "standalone",
                "sense_resolution": (
                    "headword_reference_ambiguous"
                    if best["sense_reference_ambiguous"]
                    else "selected_sense_supported"
                ),
                "signals": {
                    "definition_cosine": round(best["cosine"], 6),
                    "reverse_reference": round(best["reverse_reference"], 6),
                    "forward_reference": round(best["forward_reference"], 6),
                    "shared_tokens": round(best["shared"], 6),
                    "word_form": round(float(word_form), 6),
                    "entry_semantic_cosine": round(best["entry_semantic_cosine"], 6),
                },
                "query_sense": (
                    {
                        "sense": int(query_sense_record["sense_index"]),
                        "definition": query_sense_record["definition"],
                    }
                    if query_sense_record is not None
                    else None
                ),
                "matched_candidate_sense": {
                    "sense": int(candidate_sense["sense_index"]),
                    "definition": candidate_sense["definition"],
                },
                "definition": candidate_sense["definition"],
                "sense_count": candidate.get("sense_count", 1),
                "source_ids": candidate.get("source_ids", []),
            }
        )

    results.sort(
        key=lambda item: (
            -item["score"],
            item["word"],
        )
    )
    return results[:top_k]
