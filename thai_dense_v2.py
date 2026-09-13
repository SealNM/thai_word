from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from thai_lexical_v1 import SearchArtifacts, normalize_text


DENSE_ARTIFACT_VERSION = 1

MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "e5-small": {
        "model_id": "intfloat/multilingual-e5-small",
        "trust_remote_code": False,
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    },
    "gte-base": {
        "model_id": "Alibaba-NLP/gte-multilingual-base",
        "trust_remote_code": True,
        "query_prefix": "",
        "document_prefix": "",
    },
}


@dataclass
class DenseArtifacts:
    embeddings: np.ndarray
    metadata: dict[str, Any]


def resolve_model_profile(model: str) -> dict[str, Any]:
    if model in MODEL_PROFILES:
        profile = dict(MODEL_PROFILES[model])
        profile["key"] = model
        return profile
    return {
        "key": model,
        "model_id": model,
        "trust_remote_code": False,
        "query_prefix": "",
        "document_prefix": "",
    }


def model_slug(model_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-")
    return slug or "dense-model"


def sense_text(artifacts: SearchArtifacts, sense_id: int) -> str:
    sense = artifacts.senses[sense_id]
    entry = artifacts.entries[int(sense["entry_index"])]
    return normalize_text(f'{entry["word"]}: {sense["definition"]}')


def _prepare_texts(
    texts: list[str],
    *,
    prefix: str,
) -> list[str]:
    if not prefix:
        return texts
    return [f"{prefix}{text}" for text in texts]


def load_model(profile: dict[str, Any], device: str | None = None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Sentence Transformers is required for V2. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(profile.get("trust_remote_code", False)),
    }
    if device:
        kwargs["device"] = device
    return SentenceTransformer(profile["model_id"], **kwargs)


def build_dense_index(
    artifacts: SearchArtifacts,
    output_dir: str | Path,
    *,
    model: str = "e5-small",
    batch_size: int = 64,
    device: str | None = None,
) -> dict[str, Any]:
    profile = resolve_model_profile(model)
    encoder = load_model(profile, device=device)

    texts = [sense_text(artifacts, sense_id) for sense_id in range(len(artifacts.senses))]
    documents = _prepare_texts(
        texts,
        prefix=str(profile.get("document_prefix", "")),
    )

    embeddings = encoder.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "dense_embeddings.npy", embeddings)

    metadata = {
        "artifact_version": DENSE_ARTIFACT_VERSION,
        "model_key": profile["key"],
        "model_id": profile["model_id"],
        "trust_remote_code": bool(profile.get("trust_remote_code", False)),
        "query_prefix": profile.get("query_prefix", ""),
        "document_prefix": profile.get("document_prefix", ""),
        "normalized": True,
        "rows": int(embeddings.shape[0]),
        "dimensions": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "text_template": "{headword}: {definition}",
    }
    with (out / "dense_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return metadata


def load_dense_artifacts(
    output_dir: str | Path,
    *,
    lexical_artifacts: SearchArtifacts | None = None,
) -> DenseArtifacts:
    out = Path(output_dir)
    with (out / "dense_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    embeddings = np.load(out / "dense_embeddings.npy", mmap_mode="r")

    if embeddings.ndim != 2:
        raise ValueError("Dense embeddings must be a 2D matrix.")
    if int(metadata.get("rows", -1)) != embeddings.shape[0]:
        raise ValueError("Dense metadata row count does not match dense_embeddings.npy.")
    if lexical_artifacts is not None and embeddings.shape[0] != len(lexical_artifacts.senses):
        raise ValueError(
            "Dense index does not match the lexical index: "
            f"{embeddings.shape[0]} dense senses vs {len(lexical_artifacts.senses)} lexical senses."
        )

    return DenseArtifacts(embeddings=embeddings, metadata=metadata)


class DenseEncoder:
    def __init__(self, metadata: dict[str, Any], device: str | None = None) -> None:
        self.metadata = metadata
        profile = {
            "model_id": metadata["model_id"],
            "trust_remote_code": bool(metadata.get("trust_remote_code", False)),
            "query_prefix": metadata.get("query_prefix", ""),
            "document_prefix": metadata.get("document_prefix", ""),
        }
        self.model = load_model(profile, device=device)

    def encode_query(self, text: str) -> np.ndarray:
        text = normalize_text(text)
        prefix = str(self.metadata.get("query_prefix", ""))
        prepared = f"{prefix}{text}" if prefix else text
        vector = self.model.encode(
            [prepared],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vector[0], dtype=np.float32)
