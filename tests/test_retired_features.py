"""Contracts for internal surfaces retired from the public distribution."""

from pathlib import Path

from ai4sci_bench.cli import cli


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_retired_commands_are_not_public() -> None:
    retired = {
        "batch-run",
        "eval",
        "quickeval",
        "fullrun",
        "rerun-flagged",
        "pipeline",
    }

    assert retired.isdisjoint(cli.commands)


def test_retired_implementation_modules_are_absent() -> None:
    retired_sources = (
        REPO_ROOT / "ai4sci_bench" / "automation",
        REPO_ROOT / "ai4sci_bench" / "pipeline",
        REPO_ROOT / "ai4sci_bench" / "quickeval",
    )

    for source_dir in retired_sources:
        assert not list(source_dir.glob("*.py"))


def test_portal_workflow_does_not_restore_retired_github_automation() -> None:
    retired_files = (
        REPO_ROOT / ".github" / "workflows" / "difficulty-check.yml",
        REPO_ROOT / ".github" / "workflows" / "task-pr-check.yml",
        REPO_ROOT / ".github" / "workflows" / "build-base-image.yml",
        REPO_ROOT / ".github" / "PULL_REQUEST_TEMPLATE" / "task_submission.md",
    )

    assert all(not path.exists() for path in retired_files)
