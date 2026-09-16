from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

WRITER_ARTIFACT_POLICY_SCHEMA_VERSION = 1
PHASE3_LOCKED_ARTIFACT_ID = "phase3-bge-reranker-v2-m3-locked"
PHASE3_LOCKED_DIR_NAME = "bge-reranker-v2-m3-locked"
REPRODUCTION_DIR_PREFIX = "bge-reranker-v2-m3-reproduction-"
PRODUCTION_DIR_PATTERN = re.compile(r"^bge-reranker-v2-m3-production-v([1-9][0-9]*)$")


class WriterArtifactPolicyError(ValueError):
    pass


def production_artifact_name(version: int) -> str:
    version = int(version)
    if version < 1:
        raise ValueError("Production artifact version must be >= 1.")
    return f"bge-reranker-v2-m3-production-v{version}"


def validate_production_artifact_name(name: str) -> int:
    match = PRODUCTION_DIR_PATTERN.fullmatch(str(name))
    if not match:
        raise WriterArtifactPolicyError(
            "Production writer artifact must use "
            "'bge-reranker-v2-m3-production-v{N}' with N >= 1."
        )
    return int(match.group(1))


def _require_bool(payload: dict[str, Any], key: str, *, context: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise WriterArtifactPolicyError(f"{context}.{key} must be boolean.")
    return value


def _require_nonempty(payload: dict[str, Any], key: str, *, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise WriterArtifactPolicyError(f"{context}.{key} must be a non-empty string.")
    return value


def _require_sha256(payload: dict[str, Any], key: str, *, context: str) -> str:
    value = _require_nonempty(payload, key, context=context).lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise WriterArtifactPolicyError(f"{context}.{key} must be a SHA-256 hex digest.")
    return value


def validate_writer_artifact_registry(payload: dict[str, Any]) -> dict[str, Any]:
    if int(payload.get("schema_version", -1)) != WRITER_ARTIFACT_POLICY_SCHEMA_VERSION:
        raise WriterArtifactPolicyError("Writer artifact registry schema version mismatch.")
    if payload.get("status") != "phase4_wave_i_policy_complete":
        raise WriterArtifactPolicyError("Writer artifact registry is not marked Wave-I complete.")

    benchmark = payload.get("phase3_locked_benchmark")
    reproduction = payload.get("phase4_reproduction")
    production = payload.get("production_policy")
    if not isinstance(benchmark, dict):
        raise WriterArtifactPolicyError("phase3_locked_benchmark must be an object.")
    if not isinstance(reproduction, dict):
        raise WriterArtifactPolicyError("phase4_reproduction must be an object.")
    if not isinstance(production, dict):
        raise WriterArtifactPolicyError("production_policy must be an object.")

    if benchmark.get("role") != "benchmark_locked":
        raise WriterArtifactPolicyError("Phase-3 locked artifact role must be benchmark_locked.")
    if benchmark.get("artifact_id") != PHASE3_LOCKED_ARTIFACT_ID:
        raise WriterArtifactPolicyError("Phase-3 locked artifact id changed.")
    if benchmark.get("canonical_dir_name") != PHASE3_LOCKED_DIR_NAME:
        raise WriterArtifactPolicyError("Phase-3 locked checkpoint directory must stay immutable.")
    _require_nonempty(benchmark, "base_model", context="phase3_locked_benchmark")
    if not _require_bool(
        benchmark,
        "archived_metrics_apply_only_to_this_historical_artifact",
        context="phase3_locked_benchmark",
    ):
        raise WriterArtifactPolicyError(
            "Archived Phase-3 metrics must apply only to the historical locked artifact."
        )
    if _require_bool(
        benchmark,
        "binary_hash_frozen",
        context="phase3_locked_benchmark",
    ):
        _require_sha256(
            benchmark,
            "binary_sha256",
            context="phase3_locked_benchmark",
        )
    elif benchmark.get("binary_sha256") not in (None, ""):
        raise WriterArtifactPolicyError(
            "Phase-3 binary_sha256 must be null when binary_hash_frozen=false."
        )

    if reproduction.get("role") != "reproduction":
        raise WriterArtifactPolicyError("Phase-4 recovered artifact role must be reproduction.")
    reproduction_name = _require_nonempty(
        reproduction,
        "canonical_dir_name",
        context="phase4_reproduction",
    )
    if not reproduction_name.startswith(REPRODUCTION_DIR_PREFIX):
        raise WriterArtifactPolicyError(
            "Reproduced checkpoint needs a distinct reproduction artifact name."
        )
    if reproduction_name == PHASE3_LOCKED_DIR_NAME:
        raise WriterArtifactPolicyError(
            "Reproduced checkpoint must not reuse the locked benchmark identity."
        )
    if _require_bool(
        reproduction,
        "exact_original_binary_verified",
        context="phase4_reproduction",
    ):
        raise WriterArtifactPolicyError(
            "Phase-4 reproduction cannot claim exact original binary identity."
        )
    if _require_bool(
        reproduction,
        "may_claim_phase3_frozen_metrics",
        context="phase4_reproduction",
    ):
        raise WriterArtifactPolicyError(
            "Phase-4 reproduction cannot inherit Phase-3 frozen benchmark metrics."
        )
    if _require_bool(
        reproduction,
        "may_be_presented_as_exact_phase3_checkpoint",
        context="phase4_reproduction",
    ):
        raise WriterArtifactPolicyError(
            "Phase-4 reproduction cannot be presented as the exact Phase-3 checkpoint."
        )
    _require_sha256(
        reproduction,
        "training_dataset_sha256",
        context="phase4_reproduction",
    )
    _require_nonempty(
        reproduction,
        "phase4_evaluation_summary_file",
        context="phase4_reproduction",
    )
    _require_sha256(
        reproduction,
        "phase4_evaluation_summary_sha256",
        context="phase4_reproduction",
    )

    if production.get("status") not in {"not_created", "created"}:
        raise WriterArtifactPolicyError(
            "production_policy.status must be 'not_created' or 'created'."
        )
    if production.get("name_pattern") != "bge-reranker-v2-m3-production-v{N}":
        raise WriterArtifactPolicyError("Production artifact naming pattern changed.")
    for key in (
        "must_use_distinct_artifact_id_and_path",
        "requires_fresh_holdout_for_quality_claim",
        "must_not_use_phase4_consumed_holdout_for_selection",
    ):
        if not _require_bool(production, key, context="production_policy"):
            raise WriterArtifactPolicyError(f"production_policy.{key} must remain true.")
    if _require_bool(
        production,
        "may_claim_phase3_frozen_metrics",
        context="production_policy",
    ):
        raise WriterArtifactPolicyError(
            "Production artifacts cannot inherit Phase-3 frozen benchmark metrics."
        )

    active_name = production.get("active_artifact_name")
    if production.get("status") == "not_created":
        if active_name not in (None, ""):
            raise WriterArtifactPolicyError(
                "No active production artifact may be named while status=not_created."
            )
    else:
        if not isinstance(active_name, str):
            raise WriterArtifactPolicyError(
                "Created production policy needs active_artifact_name."
            )
        validate_production_artifact_name(active_name)
        if active_name in {PHASE3_LOCKED_DIR_NAME, reproduction_name}:
            raise WriterArtifactPolicyError(
                "Production artifact must use a distinct path from benchmark/reproduction."
            )

    return payload


def load_writer_artifact_registry(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise WriterArtifactPolicyError("Writer artifact registry root must be an object.")
    return validate_writer_artifact_registry(payload)
