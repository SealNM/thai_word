from __future__ import annotations

import json
import os
import time
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

from thai_dense_v2 import DenseArtifacts, load_dense_artifacts
from thai_lexical_v1 import SearchArtifacts, normalize_text


DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
SUPPORTED_MIN_DIMENSION = 128
SUPPORTED_MAX_DIMENSION = 3072
RECOMMENDED_DIMENSIONS = (768, 1536, 3072)


def resolve_gemini_api_key() -> str:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.getenv(name)
        if value:
            return value

    try:
        from google.colab import userdata

        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            try:
                value = userdata.get(name)
            except Exception:
                value = None
            if value:
                return value
    except ImportError:
        pass

    raise RuntimeError(
        "Gemini Embedding 2 requires an API key. Set GEMINI_API_KEY "
        "(preferred) or GOOGLE_API_KEY. In Colab, add the same name to Secrets."
    )


def validate_dimension(dimension: int) -> int:
    dimension = int(dimension)
    if dimension < SUPPORTED_MIN_DIMENSION or dimension > SUPPORTED_MAX_DIMENSION:
        raise ValueError(
            "Gemini Embedding 2 output dimension must be between "
            f"{SUPPORTED_MIN_DIMENSION} and {SUPPORTED_MAX_DIMENSION}; "
            f"got {dimension}."
        )
    return dimension


def prepare_query_text(text: str) -> str:
    text = normalize_text(text)
    return f"task: search result | query: {text}"


def prepare_document_text(title: str, text: str) -> str:
    title = normalize_text(title) or "none"
    text = normalize_text(text)
    return f"title: {title} | text: {text}"


def sense_document_text(artifacts: SearchArtifacts, sense_id: int) -> str:
    sense = artifacts.senses[int(sense_id)]
    entry = artifacts.entries[int(sense["entry_index"])]
    return prepare_document_text(
        str(entry["word"]),
        str(sense["definition"]),
    )


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("Gemini returned a zero-length embedding.")
    return values / norms


def _response_matrix(response: Any, expected: int, dimension: int) -> np.ndarray:
    embeddings = getattr(response, "embeddings", None)
    if not isinstance(embeddings, list) or len(embeddings) != expected:
        raise ValueError(
            "Gemini embed_content returned an unexpected embedding count: "
            f"expected {expected}, got "
            f"{len(embeddings) if isinstance(embeddings, list) else 'unknown'}."
        )

    rows: list[list[float]] = []
    for item in embeddings:
        values = getattr(item, "values", None)
        if values is None:
            raise ValueError("Gemini embedding response is missing values.")
        row = [float(value) for value in values]
        if len(row) != dimension:
            raise ValueError(
                f"Gemini returned {len(row)} dimensions; expected {dimension}."
            )
        rows.append(row)

    return _normalize_rows(np.asarray(rows, dtype=np.float32))


class GeminiEmbeddingEncoder:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_GEMINI_EMBEDDING_MODEL,
        dimension: int = 768,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is required for V2.6. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        self.model_id = str(model_id)
        self.dimension = validate_dimension(dimension)
        self._types = types
        self.client = client or genai.Client(
            api_key=api_key or resolve_gemini_api_key()
        )

    def _contents(self, texts: list[str]) -> list[Any]:
        return [
            self._types.Content(
                parts=[self._types.Part.from_text(text=text)]
            )
            for text in texts
        ]

    def embed_texts(
        self,
        texts: list[str],
        *,
        retries: int = 5,
        initial_delay: float = 2.0,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        attempt = 0
        while True:
            try:
                response = self.client.models.embed_content(
                    model=self.model_id,
                    contents=self._contents(texts),
                    config=self._types.EmbedContentConfig(
                        output_dimensionality=self.dimension,
                    ),
                )
                return _response_matrix(
                    response,
                    expected=len(texts),
                    dimension=self.dimension,
                )
            except Exception as exc:
                attempt += 1
                message = str(exc)
                retryable = any(
                    token in message
                    for token in (
                        "429",
                        "500",
                        "503",
                        "RESOURCE_EXHAUSTED",
                        "UNAVAILABLE",
                        "INTERNAL",
                        "temporarily",
                    )
                )
                if not retryable or attempt > max(0, int(retries)):
                    raise
                time.sleep(float(initial_delay) * (2 ** (attempt - 1)))

    def encode_query(self, text: str) -> np.ndarray:
        prepared = prepare_query_text(text)
        return self.embed_texts([prepared])[0]


def build_gemini_dense_index(
    artifacts: SearchArtifacts,
    output_dir: str | Path,
    *,
    dimension: int,
    model_id: str = DEFAULT_GEMINI_EMBEDDING_MODEL,
    batch_size: int = 50,
    api_key: str | None = None,
    resume: bool = True,
    retries: int = 5,
    progress: Callable[[int, int, float], None] | None = None,
) -> dict[str, Any]:
    dimension = validate_dimension(dimension)
    batch_size = max(1, int(batch_size))
    total = len(artifacts.senses)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    partial_path = out / "dense_embeddings.partial.npy"
    final_path = out / "dense_embeddings.npy"
    progress_path = out / "dense_progress.json"

    encoder = GeminiEmbeddingEncoder(
        model_id=model_id,
        dimension=dimension,
        api_key=api_key,
    )

    start_row = 0
    resumed_from = 0

    if resume and partial_path.exists() and progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        compatible = (
            state.get("model_id") == model_id
            and int(state.get("dimension", -1)) == dimension
            and int(state.get("rows", -1)) == total
        )
        if not compatible:
            raise ValueError(
                "Existing V2.6 partial index is incompatible with this build. "
                "Delete the partial files or choose another output directory."
            )
        start_row = int(state.get("completed_rows", 0))
        resumed_from = start_row
        embeddings = np.lib.format.open_memmap(
            partial_path,
            mode="r+",
            dtype=np.float32,
            shape=(total, dimension),
        )
    else:
        if partial_path.exists():
            partial_path.unlink()
        if progress_path.exists():
            progress_path.unlink()
        embeddings = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.float32,
            shape=(total, dimension),
        )

    build_started = perf_counter()
    request_count = 0

    for begin in range(start_row, total, batch_size):
        end = min(total, begin + batch_size)
        texts = [
            sense_document_text(artifacts, sense_id)
            for sense_id in range(begin, end)
        ]

        batch_started = perf_counter()
        vectors = encoder.embed_texts(
            texts,
            retries=retries,
        )
        embeddings[begin:end] = vectors
        embeddings.flush()
        request_count += 1
        elapsed = perf_counter() - batch_started

        state = {
            "model_id": model_id,
            "dimension": dimension,
            "rows": total,
            "completed_rows": end,
            "batch_size": batch_size,
        }
        progress_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if progress:
            progress(end, total, elapsed)

    del embeddings

    if final_path.exists():
        final_path.unlink()
    partial_path.replace(final_path)
    if progress_path.exists():
        progress_path.unlink()

    build_seconds = perf_counter() - build_started
    metadata = {
        "artifact_version": 2,
        "backend": "gemini-api",
        "model_key": f"gemini-embedding-2-{dimension}",
        "model_id": model_id,
        "dimensions": dimension,
        "output_dimensionality": dimension,
        "normalized": True,
        "rows": total,
        "dtype": "float32",
        "embedding_bytes": int(total * dimension * 4),
        "embedding_megabytes": round(float(total * dimension * 4 / (1024 ** 2)), 3),
        "query_format": "task: search result | query: {content}",
        "document_format": "title: {headword} | text: {definition}",
        "text_template": "title: {headword} | text: {definition}",
        "api_batch_size": batch_size,
        "api_requests_this_run": request_count,
        "resumed_from_row": resumed_from,
        "build_seconds_this_run": round(float(build_seconds), 3),
    }
    (out / "dense_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def load_gemini_dense_index(
    output_dir: str | Path,
    *,
    lexical_artifacts: SearchArtifacts,
    api_key: str | None = None,
) -> tuple[DenseArtifacts, GeminiEmbeddingEncoder]:
    dense = load_dense_artifacts(
        output_dir,
        lexical_artifacts=lexical_artifacts,
    )
    if dense.metadata.get("backend") != "gemini-api":
        raise ValueError(
            f"{output_dir} is not a Gemini API dense index."
        )

    dimension = int(
        dense.metadata.get(
            "output_dimensionality",
            dense.metadata["dimensions"],
        )
    )
    encoder = GeminiEmbeddingEncoder(
        model_id=str(dense.metadata["model_id"]),
        dimension=dimension,
        api_key=api_key,
    )
    return dense, encoder
