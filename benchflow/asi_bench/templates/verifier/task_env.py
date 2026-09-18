"""Compatibility API for task scorers using the image's existing Python.

Unlike the upstream manager, this module never installs, caches or repairs an
environment. The Docker build is responsible for satisfying task requirements.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


@dataclass(frozen=True)
class TaskEnvironment:
    env_dir: Path
    python_executable: Path

    def build_subprocess_env(self, base_env=None):
        env = dict(os.environ if base_env is None else base_env)
        env["PATH"] = str(self.python_executable.parent) + os.pathsep + env.get("PATH", "")
        return env


class TaskEnvironmentManager:
    """Resolve the current image environment; no native ASI cache protocol."""

    def __init__(self, repo_root, cache_root=None):
        pass

    def ensure_env(self, runtime):
        python = runtime.get("_runtime_python")
        if python and not SpecifierSet(python).contains(".".join(map(str, sys.version_info[:3]))):
            raise ValueError("image Python does not satisfy task runtime")
        for text in runtime.get("_runtime_packages") or []:
            requirement = Requirement(text)
            if requirement.url:
                raise ValueError("task runtime must use index packages")
            if requirement.marker and not requirement.marker.evaluate():
                continue
            if not requirement.specifier.contains(version(requirement.name)):
                raise ValueError(f"image dependency does not satisfy {text}")
        return TaskEnvironment(Path(sys.prefix), Path(sys.executable))
