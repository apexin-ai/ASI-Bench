"""Helpers for validating sandbox mode support."""

from __future__ import annotations

import platform
import sys

import click


KNOWN_SANDBOX_MODES = ("none", "task", "os", "linux_ns")

SANDBOX_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "none": {
        "label": "No Sandbox",
        "description": (
            "Agent runs directly on the host with full filesystem and network access."
        ),
        "pros": "Zero overhead; fastest startup; works everywhere.",
        "cons": (
            "No isolation — agent can read reference answers, other instances, "
            "host files, and environment variables. Results are NOT suitable for "
            "formal evaluation (see Issue #7: scores inflated from 25 to 100 due "
            "to information leakage)."
        ),
    },
    "task": {
        "label": "Task Environment Sandbox",
        "description": (
            "Isolated Python venv per task (via uv). "
            "Ensures correct package versions declared in task.yaml runtime.packages."
        ),
        "pros": (
            "Cross-platform (no Docker needed); deterministic Python dependencies; "
            "fast cached venv reuse."
        ),
        "cons": (
            "No filesystem or network isolation — agent still has full shell access "
            "and can read host files, reference answers, and other instances."
        ),
    },
    "os": {
        "label": "OS-level Docker Sandbox",
        "description": (
            "Code executes inside a Docker container with task-specific image. "
            "Filesystem, network, and resource isolation via Docker."
        ),
        "pros": (
            "Strong filesystem isolation (agent cannot see host files); "
            "network disabled by default (--network none); "
            "resource limits (memory/CPU); GPU passthrough supported; "
            "task packages pre-installed in image layer."
        ),
        "cons": (
            "Requires Docker; supported by direct_llm, claude_code_cli, "
            "codex_cli, kimi_code_cli, mimo_code_cli, pi_cli, and "
            "opencode_cli adapters; higher startup overhead than task mode."
        ),
    },
    "linux_ns": {
        "label": "Linux Namespace Sandbox",
        "description": (
            "Lightweight isolation using Linux namespaces (unshare + bind mounts). "
            "No Docker required; suitable for HPC clusters and RL training loops."
        ),
        "pros": (
            "No Docker dependency; ~5% disk overhead vs Docker; "
            "fast environment preparation; filesystem isolation via bind mounts."
        ),
        "cons": (
            "Linux only (not supported on macOS/Windows); requires user namespace "
            "privileges; less mature than Docker isolation; no built-in network policy."
        ),
    },
}

SANDBOX_HELP = (
    "Sandbox isolation mode (default: none). "
    "Modes: "
    "none = no isolation (debug only); "
    "task = Python venv per task (cross-platform, no Docker); "
    "os = Docker container isolation (recommended for formal evaluation); "
    "linux_ns = Linux namespace isolation (no Docker, Linux only)."
)


def print_sandbox_banner(sandbox: str) -> None:
    """Print a description of the active sandbox mode at run start."""
    info = SANDBOX_DESCRIPTIONS.get(sandbox)
    if info is None:
        return

    lines = [
        f"Sandbox mode: --sandbox {sandbox}  [{info['label']}]",
        f"  {info['description']}",
        f"  Pros:  {info['pros']}",
        f"  Cons:  {info['cons']}",
    ]

    if sandbox == "none":
        lines.append(
            "  Tip:   Use --sandbox os for isolated evaluation, "
            "or --sandbox task for package isolation without Docker."
        )

    # macOS performance warning for Docker-based sandbox
    if sandbox == "os" and platform.system() == "Darwin":
        lines.append(
            "  Warning: Docker on macOS uses a Linux VM, which adds I/O overhead. "
            "Consider --sandbox task for faster iteration during development."
        )

    click.echo("")
    for line in lines:
        click.echo(line)
    click.echo("")


def validate_sandbox_mode(
    sandbox: str,
    *,
    supported_modes: tuple[str, ...],
    component: str,
) -> None:
    """Validate that a component supports the requested sandbox mode."""
    if sandbox not in KNOWN_SANDBOX_MODES:
        supported_text = ", ".join(KNOWN_SANDBOX_MODES)
        raise ValueError(
            f"Unknown sandbox mode '{sandbox}' for {component}. "
            f"Known modes: {supported_text}."
        )

    if sandbox not in supported_modes:
        supported_text = ", ".join(supported_modes)
        raise ValueError(
            f"{component} does not support sandbox '{sandbox}'. "
            f"Supported modes: {supported_text}."
        )

    # linux_ns requires Linux
    if sandbox == "linux_ns" and sys.platform != "linux":
        raise ValueError(
            f"--sandbox linux_ns requires Linux (current platform: {sys.platform}). "
            "Use --sandbox os (Docker) or --sandbox task instead."
        )
