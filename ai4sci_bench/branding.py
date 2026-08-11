"""Branding & naming — single source of truth for user-visible names.

Centralizes command name, brand text, HuggingFace instance-repo identifiers,
the user config directory, and environment-variable names, so these are not
scattered as string literals. Import from here rather than hardcoding.

Renaming to `asibench` is done with **old-value fallback** everywhere:
  - config dir: prefer ``~/.asibench/``, fall back to legacy ``~/.ai4sci-bench/``
  - env vars:   read ``ASIBENCH_<NAME>`` first, fall back to ``AI4SCI_<NAME>``
The Python module name ``ai4sci_bench`` is intentionally left unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Branding ────────────────────────────────────────────────────────────────
# Product/brand text shown in docs, banners, CLI help.
BRAND_NAME = "ASI-Bench"
# Primary CLI command (pyproject entry point). `ai4sci-bench` is kept as a
# backwards-compatible alias — see pyproject [project.scripts].
CLI_COMMAND = "asibench"
CLI_COMMAND_ALIAS = "ai4sci-bench"
# The only distribution name published on PyPI. The import name stays unchanged.
DISTRIBUTION_NAME = "asibench"

# ── HuggingFace instance datasets ───────────────────────────────────────────
# Org that hosts the public benchmark and demo instance datasets.
HF_ORG = "Apexintelligence-AI"

# Friendly aliases → public HF dataset repo id. The only official benchmark
# contracts are seed42 and seed31415; demo is a non-official format example.
HF_REPO_ALIASES: dict[str, str] = {
    "demo": f"{HF_ORG}/ASI-Bench-seed2",
    "seed42": f"{HF_ORG}/ASI-Bench-seed42",
    "seed31415": f"{HF_ORG}/ASI-Bench-seed31415",
}
OFFICIAL_HF_REPO_ALIASES: tuple[str, str] = ("seed42", "seed31415")

# Env vars consulted (in order) for an HF token when none is passed explicitly.
HF_TOKEN_ENV_VARS: tuple[str, ...] = (
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)

# Default local dir that `asibench task pull` writes into and `run --instances-dir`
# reads from. Gitignored (demo repo carries reference answers).
DEFAULT_PULL_DIR = "hf_instances/"

# ── Submission / official scoring ───────────────────────────────────────────
# `asibench submit` packages a produce-only run into a self-contained bundle
# and uploads it to the official ASI-Bench website for scoring. The bundle
# carries the agent's outputs + provenance, never private scoring rules.
SUBMIT_ENDPOINT_ENV_SUFFIX = "SUBMIT_ENDPOINT"
SUBMIT_TOKEN_ENV_SUFFIX = "SUBMIT_TOKEN"
DEFAULT_SUBMISSION_DIR = "submissions/"
DEFAULT_SUBMISSION_ENDPOINT = "https://asibench.apexin.ai"

# Default for the separate web-only task-authoring launcher.
DEFAULT_TASK_SUBMISSION_ENDPOINT = "https://asibench.apexin.ai/submit"


def submit_endpoint(explicit: str | None = None) -> str:
    """Resolve the official website used for authenticated result submission.

    Precedence: explicit argument > ``ASIBENCH_SUBMIT_ENDPOINT`` >
    legacy ``AI4SCI_SUBMIT_ENDPOINT`` > the official ASI-Bench website.
    """
    if explicit:
        return explicit
    return (
        env(SUBMIT_ENDPOINT_ENV_SUFFIX, DEFAULT_SUBMISSION_ENDPOINT)
        or DEFAULT_SUBMISSION_ENDPOINT
    )


def submit_token(explicit: str | None = None) -> str | None:
    """Resolve the scoring-service auth token from the arg or env; never hardcoded."""
    if explicit:
        return explicit
    return env(SUBMIT_TOKEN_ENV_SUFFIX)

# ── User config directory ───────────────────────────────────────────────────
CONFIG_DIR_NAME = ".asibench"
LEGACY_CONFIG_DIR_NAME = ".ai4sci-bench"


def config_dir() -> Path:
    """Return the user config dir (``~/.asibench``).

    Old-value fallback: if the new dir does not exist but the legacy
    ``~/.ai4sci-bench`` does, the legacy path is used so existing account pools /
    caches keep working without a forced migration.
    """
    new = Path.home() / CONFIG_DIR_NAME
    legacy = Path.home() / LEGACY_CONFIG_DIR_NAME
    if not new.exists() and legacy.exists():
        return legacy
    return new


def config_path(*parts: str) -> Path:
    """Path to a file/dir under the resolved user config dir."""
    return config_dir().joinpath(*parts)


# ── Environment variables ───────────────────────────────────────────────────
ENV_PREFIX = "ASIBENCH_"
LEGACY_ENV_PREFIX = "AI4SCI_"


def env(name: str, default: str | None = None) -> str | None:
    """Read ``ASIBENCH_<name>``, falling back to legacy ``AI4SCI_<name>``.

    ``name`` is the suffix without prefix, e.g. ``env("LLM_BACKEND")`` reads
    ``ASIBENCH_LLM_BACKEND`` then ``AI4SCI_LLM_BACKEND``. Returns ``default`` if
    neither is set (an empty string counts as unset).
    """
    val = os.environ.get(f"{ENV_PREFIX}{name}")
    if val:
        return val
    val = os.environ.get(f"{LEGACY_ENV_PREFIX}{name}")
    if val:
        return val
    return default


def env_name(name: str) -> str:
    """Canonical (new) env var name for ``name``, e.g. ``ASIBENCH_LLM_BACKEND``."""
    return f"{ENV_PREFIX}{name}"
