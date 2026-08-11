"""Run metadata recording for reproducibility.

Captures framework version, git state, Python version, and dependency versions
at the start of each benchmark run.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai4sci_bench.core.logger import get_logger

logger = get_logger(__name__)

FRAMEWORK_VERSION = "0.1.2"


def collect_run_metadata(
    agent_config: dict[str, Any] | None = None,
    task_versions: dict[str, str] | None = None,
    sandbox_provenance: dict[str, Any] | None = None,
    task_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect reproducibility metadata for the current run.

    Returns a dict suitable for writing to run_metadata.json.
    """
    metadata = {
        "framework_version": FRAMEWORK_VERSION,
        "git_commit": _get_git_commit(),
        "git_dirty": _is_git_dirty(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": _get_dependency_versions(),
    }

    if agent_config:
        metadata["agent_config"] = agent_config

    if task_versions:
        metadata["task_versions"] = task_versions

    if sandbox_provenance:
        metadata["sandbox"] = sandbox_provenance

    if task_runtime:
        metadata["task_runtime"] = task_runtime

    return metadata


def build_sandbox_provenance(
    sandbox: str,
    *,
    image_identity: str | None = None,
) -> dict[str, Any]:
    """Build normalized sandbox provenance metadata.

    Field naming convention (authoritative — proposal §3 field names are
    superseded by this implementation):
      - ``requested_mode`` / ``effective_mode``: the sandbox mode requested by
        the user and the mode that was actually enforced (may differ on
        degradation).
      - ``enforcement_status``: whether the framework guarantees that the
        sandbox was applied ("fail_closed" means failure = abort, not silent
        fallback).
      - ``verification_status``: how the framework verified that isolation was
        active (e.g. "docker_container", "task_env_active", "linux_namespace").
      - ``image_identity``: Docker image sha256 (only for ``os`` mode).

    These names intentionally differ from the original proposal's
    ``sandbox_mode`` / ``sandbox_verified`` / ``docker_image`` — the current
    names are more precise and already consumed by downstream tooling.
    """
    if sandbox == "none":
        enforcement_status = "not_requested"
        verification_status = "not_applicable"
    elif sandbox == "task":
        enforcement_status = "fail_closed"
        verification_status = "task_env_active"
    elif sandbox == "os":
        enforcement_status = "fail_closed"
        verification_status = "docker_container"
    elif sandbox == "linux_ns":
        enforcement_status = "fail_closed"
        verification_status = "linux_namespace"
    else:
        enforcement_status = "fail_closed"
        verification_status = "runtime_specific"

    return {
        "requested_mode": sandbox,
        "effective_mode": sandbox,
        "enforcement_status": enforcement_status,
        "verification_status": verification_status,
        "image_identity": image_identity,
    }


def build_task_runtime_provenance(
    task_metadata: dict[str, Any],
    *,
    sandbox: str,
    task_env_manager: Any | None = None,
    image_identity: str | None = None,
) -> dict[str, Any]:
    """Build runtime provenance for one task."""
    runtime_python = task_metadata.get("_runtime_python")
    runtime_packages = list(task_metadata.get("_runtime_packages", []))
    task_env_cache_key = None
    resolved_python_version = None
    python_requirement_satisfied = None
    runtime_source = "host_process"
    runtime_interpreter = None
    docker_image_identity = None

    if sandbox == "os":
        runtime_source = "docker_image"
        runtime_interpreter = "/opt/venv/bin/python"
        docker_image_identity = image_identity
    elif sandbox in ("task", "linux_ns") and task_env_manager is not None:
        runtime_source = "task_env_cache"
        task_env_cache_key = task_env_manager.compute_cache_key(task_metadata)
        cached_env = (
            task_env_manager.describe_cached_env(task_metadata)
            if hasattr(task_env_manager, "describe_cached_env")
            else None
        )
        if cached_env is not None:
            resolved_python_version = cached_env.get("resolved_python_version")
            python_requirement_satisfied = cached_env.get("python_requirement_satisfied")

    return {
        "python_requirement": runtime_python,
        "resolved_python_version": resolved_python_version,
        "python_requirement_satisfied": python_requirement_satisfied,
        "packages": runtime_packages,
        "task_env_cache_key": task_env_cache_key,
        "runtime_source": runtime_source,
        "runtime_interpreter": runtime_interpreter,
        "docker_image_identity": docker_image_identity,
    }


def save_run_metadata(output_dir: Path, metadata: dict[str, Any]) -> Path:
    """Save run metadata to output_dir/run_metadata.json.

    Returns the path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_metadata.json"
    path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Run metadata saved to {path}")
    return path


def _get_git_commit() -> str:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _is_git_dirty() -> bool:
    """Check if the git working directory has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except Exception:
        pass
    return False


def _get_dependency_versions() -> dict[str, str]:
    """Get versions of key dependencies."""
    deps: dict[str, str] = {}
    distributions = {
        "numpy": "numpy",
        "scipy": "scipy",
        "click": "click",
        "litellm": "litellm",
        "pyyaml": "PyYAML",
        "openai": "openai",
    }

    for name, distribution in distributions.items():
        try:
            deps[name] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass

    return deps
