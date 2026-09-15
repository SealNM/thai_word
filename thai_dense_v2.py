from __future__ import annotations

import json
import re
import warnings
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from thai_lexical_v1 import SearchArtifacts, normalize_text


DENSE_ARTIFACT_VERSION = 1

QWEN3_THAI_LEXICAL_TASK = (
    "Given a Thai dictionary query with its selected sense, retrieve Thai dictionary "
    "entries that can substitute for the query while preserving meaning and grammatical role"
)

MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "e5-small": {
        "model_id": "intfloat/multilingual-e5-small",
        "trust_remote_code": False,
        "query_prefix": "query: ",
        "document_prefix": "query: ",
    },
    "e5-base": {
        "model_id": "intfloat/multilingual-e5-base",
        "trust_remote_code": False,
        "query_prefix": "query: ",
        "document_prefix": "query: ",
    },
    "embeddinggemma-300m": {
        "model_id": "google/embeddinggemma-300m",
        "trust_remote_code": False,
        "query_prefix": "",
        "document_prefix": "",
        "query_method": "encode_query",
        "document_method": "encode_document",
        "truncate_dim": None,
    },
    "embeddinggemma-300m-256": {
        "model_id": "google/embeddinggemma-300m",
        "trust_remote_code": False,
        "query_prefix": "",
        "document_prefix": "",
        "query_method": "encode_query",
        "document_method": "encode_document",
        "truncate_dim": 256,
    },
    "qwen3-embedding-0.6b-256": {
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "trust_remote_code": False,
        "query_prefix": f"Instruct: {QWEN3_THAI_LEXICAL_TASK}\nQuery:",
        "document_prefix": "",
        "truncate_dim": 256,
        "max_seq_length": 512,
    },
    "arctic-embed-m-v2-256": {
        "model_id": "Snowflake/snowflake-arctic-embed-m-v2.0",
        "trust_remote_code": True,
        "query_prefix": "query: ",
        "document_prefix": "",
        "truncate_dim": 256,
        "max_seq_length": 512,
        "config_kwargs": {
            "use_memory_efficient_attention": False,
        },
    },
    "gte-base-experimental": {
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
    if model == "gte-base":
        raise ValueError(
            "The 'gte-base' profile was removed from the default V2 benchmark because "
            "Alibaba-NLP/gte-multilingual-base depends on custom remote modeling code "
            "that is not stable across current Colab PyTorch/Transformers stacks. "
            "Use '--model e5-base' for the stable 768-dimensional challenger, or "
            "'--model gte-base-experimental' if you intentionally want to test GTE."
        )
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


def resolve_device(device: str | None) -> str | None:
    """Resolve an explicitly requested accelerator with a safe CPU fallback."""
    if not device:
        return None

    requested = str(device).strip()
    if not requested.lower().startswith("cuda"):
        return requested

    try:
        import torch
    except ImportError:
        warnings.warn(
            f"Requested device {requested!r}, but PyTorch is unavailable; falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"

    if not torch.cuda.is_available():
        build_cuda = getattr(torch.version, "cuda", None)
        reason = (
            "this PyTorch build has no CUDA support"
            if build_cuda is None
            else "CUDA is not available in this runtime"
        )
        warnings.warn(
            f"Requested device {requested!r}, but {reason}; falling back to CPU. "
            "On Colab, enable a GPU runtime if you want CUDA acceleration.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"

    if ":" in requested:
        try:
            index = int(requested.split(":", 1)[1])
        except ValueError:
            index = -1
        if index < 0 or index >= torch.cuda.device_count():
            warnings.warn(
                f"Requested device {requested!r}, but only {torch.cuda.device_count()} "
                "CUDA device(s) are available; falling back to cuda:0.",
                RuntimeWarning,
                stacklevel=2,
            )
            return "cuda:0"

    return requested


def _resolve_hf_token() -> str | None:
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    return get_token()


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

    model_id = str(profile["model_id"])
    hf_token = _resolve_hf_token()
    is_local_model = Path(model_id).exists()
    if model_id.startswith("google/embeddinggemma") and not is_local_model and not hf_token:
        raise RuntimeError(
            "EmbeddingGemma is a gated Hugging Face model, but this runtime is not "
            "authenticated. Add/enable the Colab secret HF_TOKEN, load it into the "
            "runtime (or call huggingface_hub.login), and make sure the same Hugging "
            "Face account has accepted access to google/embeddinggemma-300m."
        )
    if hf_token:
        kwargs["token"] = hf_token

    truncate_dim = profile.get("truncate_dim")
    if truncate_dim is not None:
        kwargs["truncate_dim"] = int(truncate_dim)

    config_kwargs = profile.get("config_kwargs")
    if config_kwargs:
        kwargs["config_kwargs"] = dict(config_kwargs)

    effective_device = resolve_device(device)
    if effective_device:
        kwargs["device"] = effective_device

    encoder = SentenceTransformer(profile["model_id"], **kwargs)
    max_seq_length = profile.get("max_seq_length")
    if max_seq_length is not None:
        encoder.max_seq_length = int(max_seq_length)
    return encoder


def _encode_documents(
    encoder: Any,
    profile: dict[str, Any],
    texts: list[str],
    *,
    batch_size: int,
) -> np.ndarray:
    method_name = str(profile.get("document_method", "encode"))
    if method_name == "encode_document":
        vectors = encoder.encode_document(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    else:
        documents = _prepare_texts(
            texts,
            prefix=str(profile.get("document_prefix", "")),
        )
        vectors = encoder.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    return np.asarray(vectors, dtype=np.float32)


def build_dense_index(
    artifacts: SearchArtifacts,
    output_dir: str | Path,
    *,
    model: str = "e5-small",
    batch_size: int = 64,
    device: str | None = None,
    profile_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = resolve_model_profile(model)
    if profile_overrides:
        profile.update(
            {
                key: value
                for key, value in profile_overrides.items()
                if value is not None
            }
        )
    load_started = perf_counter()
    encoder = load_model(profile, device=device)
    load_seconds = perf_counter() - load_started

    texts = [sense_text(artifacts, sense_id) for sense_id in range(len(artifacts.senses))]

    encode_started = perf_counter()
    embeddings = _encode_documents(
        encoder,
        profile,
        texts,
        batch_size=batch_size,
    )
    encode_seconds = perf_counter() - encode_started

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
        "query_method": profile.get("query_method", "encode"),
        "document_method": profile.get("document_method", "encode"),
        "truncate_dim": profile.get("truncate_dim"),
        "max_seq_length": profile.get("max_seq_length"),
        "config_kwargs": profile.get("config_kwargs"),
        "normalized": True,
        "rows": int(embeddings.shape[0]),
        "dimensions": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "embedding_bytes": int(embeddings.nbytes),
        "embedding_megabytes": round(float(embeddings.nbytes / (1024 ** 2)), 3),
        "model_load_seconds": round(float(load_seconds), 3),
        "encode_seconds": round(float(encode_seconds), 3),
        "requested_device": device or "auto",
        "device": str(getattr(encoder, "device", "auto")),
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
            "query_method": metadata.get("query_method", "encode"),
            "document_method": metadata.get("document_method", "encode"),
            "truncate_dim": metadata.get("truncate_dim"),
            "max_seq_length": metadata.get("max_seq_length"),
        }
        self.model = load_model(profile, device=device)

    def encode_query(self, text: str) -> np.ndarray:
        text = normalize_text(text)
        method_name = str(self.metadata.get("query_method", "encode"))
        if method_name == "encode_query":
            vector = self.model.encode_query(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(vector, dtype=np.float32)

        prefix = str(self.metadata.get("query_prefix", ""))
        prepared = f"{prefix}{text}" if prefix else text
        vector = self.model.encode(
            [prepared],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vector[0], dtype=np.float32)
