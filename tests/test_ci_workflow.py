"""Regression checks for the repository's required CI workflow."""

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "ci.yml"
PUBLISH_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "publish.yml"


def test_ci_runs_locked_tests_and_build_checks_on_push_and_pull_requests():
    assert WORKFLOW_PATH.is_file()

    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    triggers = workflow["on"]
    assert "push" in triggers
    assert "pull_request" in triggers
    assert workflow["permissions"] == {"contents": "read"}

    jobs = workflow["jobs"]
    assert {"tests", "build", "required"} <= jobs.keys()
    assert jobs["required"]["needs"] == ["tests", "build"]

    test_commands = "\n".join(
        step.get("run", "") for step in jobs["tests"]["steps"]
    )
    assert "uv sync --locked" in test_commands
    assert "uv run --frozen pytest" in test_commands

    build_commands = "\n".join(
        step.get("run", "") for step in jobs["build"]["steps"]
    )
    assert "uv build" in build_commands
    assert "twine check --strict" in build_commands
    assert "dist/*.whl" in build_commands
    assert "asibench --help" in build_commands


def test_agent_instruction_files_are_regular_and_synchronized():
    agents = ROOT / "AGENTS.md"
    claude = ROOT / "CLAUDE.md"

    assert agents.is_file()
    assert not agents.is_symlink()
    assert agents.read_text(encoding="utf-8") == claude.read_text(encoding="utf-8")


def test_pypi_publish_requires_a_matching_github_release_and_secret():
    assert PUBLISH_WORKFLOW_PATH.is_file()

    workflow_text = PUBLISH_WORKFLOW_PATH.read_text(encoding="utf-8")
    workflow = yaml.safe_load(workflow_text)
    assert workflow["on"] == {"release": {"types": ["published"]}}
    assert workflow["permissions"] == {"contents": "read"}

    publish = workflow["jobs"]["publish"]
    commands = "\n".join(step.get("run", "") for step in publish["steps"])
    assert "github.event.release.tag_name" in workflow_text
    assert "uv version --short" in commands
    assert "uv build" in commands
    assert "twine check --strict" in commands
    assert "pypa/gh-action-pypi-publish" in workflow_text
    assert "secrets.PYPI_API_TOKEN" in workflow_text
    assert re.search(r"pypi-[A-Za-z0-9_-]{40,}", workflow_text) is None
