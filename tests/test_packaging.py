"""Regression tests for the single public PyPI distribution contract."""

from __future__ import annotations

import tomllib
from pathlib import Path

from ai4sci_bench import branding


ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_single_distribution_keeps_legacy_import_and_cli_alias() -> None:
    config = _pyproject()
    project = config["project"]

    assert project["name"] == "asibench"
    assert branding.DISTRIBUTION_NAME == project["name"]
    assert project["scripts"] == {
        "asibench": "ai4sci_bench.cli:main",
        "ai4sci-bench": "ai4sci_bench.cli:main",
    }
    assert config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "ai4sci_bench"
    ]


def test_pypi_metadata_includes_readme_and_license() -> None:
    project = _pyproject()["project"]

    assert project["readme"] == "README.md"
    assert project["license"] == {"file": "LICENSE"}
    assert project["authors"] == [{"name": "Apex Intelligence"}]


def test_default_install_is_lightweight_and_can_pull_hugging_face_tasks() -> None:
    project = _pyproject()["project"]

    assert set(project["dependencies"]) == {
        "click>=8.1.0",
        "huggingface-hub>=0.24.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
    }
    assert set(project["optional-dependencies"]["full"]) == {
        "Pillow>=10.0.0",
        "litellm>=1.0.0",
        "matplotlib>=3.10.8",
        "numpy>=2.0.0",
        "openai>=1.0.0",
        "pandas>=3.0.2",
        "scipy>=1.11.0",
        "taichi>=1.7.4",
    }
    assert set(project["optional-dependencies"]["full"]) <= set(
        _pyproject()["dependency-groups"]["dev"]
    )
