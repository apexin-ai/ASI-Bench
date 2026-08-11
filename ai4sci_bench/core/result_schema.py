"""Helpers for persisted evaluation-result JSON compatibility."""

from __future__ import annotations

from typing import Any, Mapping

RESULT_SCHEMA_VERSION_FIELD = "result_schema_version"
CURRENT_RESULT_SCHEMA_VERSION = 1
LEGACY_RESULT_SCHEMA_VERSION = 0
SUPPORTED_RESULT_SCHEMA_VERSIONS = {
    LEGACY_RESULT_SCHEMA_VERSION,
    CURRENT_RESULT_SCHEMA_VERSION,
}
REQUIRED_RESULT_KEYS = frozenset({
    "instance_id",
    "task_id",
    "prompt_level",
    "agent_name",
    "gates_passed",
    "gate_results",
    "score_results",
    "final_score",
    "status",
})


def looks_like_eval_result_json(data: Mapping[str, Any]) -> bool:
    """Return whether a JSON blob looks like a persisted eval result."""
    return REQUIRED_RESULT_KEYS.issubset(data.keys())


def read_result_schema_version(data: Mapping[str, Any]) -> int:
    """Read the persisted result schema version, defaulting missing to legacy."""
    raw_version = data.get(RESULT_SCHEMA_VERSION_FIELD, LEGACY_RESULT_SCHEMA_VERSION)
    if isinstance(raw_version, bool) or not isinstance(raw_version, int):
        raise ValueError(
            f"Invalid {RESULT_SCHEMA_VERSION_FIELD}: expected integer, got {raw_version!r}"
        )
    if raw_version < 0:
        raise ValueError(
            f"Invalid {RESULT_SCHEMA_VERSION_FIELD}: expected non-negative integer, got {raw_version!r}"
        )
    return raw_version


def ensure_supported_result_schema(data: Mapping[str, Any]) -> int:
    """Validate and return a supported persisted result schema version."""
    version = read_result_schema_version(data)
    if version not in SUPPORTED_RESULT_SCHEMA_VERSIONS:
        supported = ", ".join(str(v) for v in sorted(SUPPORTED_RESULT_SCHEMA_VERSIONS))
        raise ValueError(
            f"Unsupported {RESULT_SCHEMA_VERSION_FIELD}={version}; supported versions: {supported}"
        )
    return version
