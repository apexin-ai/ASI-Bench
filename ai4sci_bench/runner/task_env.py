"""Task-level Python environment management backed by uv."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from ai4sci_bench.branding import DISTRIBUTION_NAME, config_path


@dataclass
class TaskEnvironment:
    """Resolved task environment information."""

    env_dir: Path
    python_executable: Path
    bin_dir: Path
    cache_key: str
    python_requirement: str | None
    packages: list[str]
    cache_hit: bool
    resolved_python_version: str | None = None

    def build_subprocess_env(self, base_env: dict[str, str] | None = None) -> dict[str, str]:
        """Return environment variables with this task env activated."""
        env = dict(base_env or os.environ)
        path_parts = [str(self.bin_dir)]
        existing_path = env.get("PATH")
        if existing_path:
            path_parts.append(existing_path)
        env["PATH"] = os.pathsep.join(path_parts)
        env["VIRTUAL_ENV"] = str(self.env_dir)
        return env


class TaskEnvironmentManager:
    """Create and reuse cached task environments."""

    CACHE_SCHEMA_VERSION = 4
    METADATA_FILENAME = "task_env.json"
    LOCK_POLL_INTERVAL_SECONDS = 0.1
    LOCK_TIMEOUT_SECONDS = 1800.0
    STALE_LOCK_SECONDS = 7200.0

    def __init__(self, repo_root: Path, cache_root: Path | None = None):
        self.repo_root = Path(repo_root).resolve()
        self.source_checkout = self._is_installable_project(self.repo_root)
        if cache_root is not None:
            self.cache_root = cache_root
            self.uv_cache_dir = cache_root.parent / "uv-cache"
        elif self.source_checkout:
            self.cache_root = self.repo_root / ".ai4sci-bench" / "task-envs"
            self.uv_cache_dir = self.repo_root / ".ai4sci-bench" / "uv-cache"
        else:
            # A wheel's inferred ``repo_root`` is site-packages.  It is neither
            # an installable project nor a suitable place for mutable caches.
            self.cache_root = config_path("task-envs")
            self.uv_cache_dir = config_path("uv-cache")

    def ensure_env(self, task_metadata: dict[str, Any]) -> TaskEnvironment:
        """Create or reuse a cached environment for one task."""
        runtime_packages = self._normalize_packages(task_metadata.get("_runtime_packages", []))
        python_requirement = task_metadata.get("_runtime_python")
        cache_key = self.compute_cache_key(task_metadata)
        env_dir = self.cache_root / cache_key
        python_executable = self._python_executable(env_dir)
        bin_dir = python_executable.parent

        if self._is_cache_hit(env_dir):
            resolved_python_version = self._read_cached_python_version(env_dir)
            return TaskEnvironment(
                env_dir=env_dir,
                python_executable=python_executable,
                bin_dir=bin_dir,
                cache_key=cache_key,
                python_requirement=python_requirement,
                packages=runtime_packages,
                cache_hit=True,
                resolved_python_version=resolved_python_version,
            )

        self.cache_root.mkdir(parents=True, exist_ok=True)
        with self._acquire_lock(self._lock_path(cache_key)):
            if self._is_cache_hit(env_dir):
                resolved_python_version = self._read_cached_python_version(env_dir)
                return TaskEnvironment(
                    env_dir=env_dir,
                    python_executable=python_executable,
                    bin_dir=bin_dir,
                    cache_key=cache_key,
                    python_requirement=python_requirement,
                    packages=runtime_packages,
                    cache_hit=True,
                    resolved_python_version=resolved_python_version,
                )

            self._cleanup_staging_dirs(cache_key)
            staging_dir = self._staging_env_dir(cache_key)
            try:
                resolved_python_version = self._build_env(
                    env_dir=staging_dir,
                    cache_key=cache_key,
                    runtime_packages=runtime_packages,
                    python_requirement=python_requirement,
                )
                self._publish_staged_env(staging_dir, env_dir)
            except Exception:
                shutil.rmtree(staging_dir, ignore_errors=True)
                raise

        return TaskEnvironment(
            env_dir=env_dir,
            python_executable=python_executable,
            bin_dir=bin_dir,
            cache_key=cache_key,
            python_requirement=python_requirement,
            packages=runtime_packages,
            cache_hit=False,
            resolved_python_version=resolved_python_version,
        )

    def compute_cache_key(self, task_metadata: dict[str, Any]) -> str:
        """Compute a stable cache key for one task runtime spec."""
        payload = {
            "schema_version": self.CACHE_SCHEMA_VERSION,
            "python_requirement": task_metadata.get("_runtime_python"),
            "requested_python": self._select_requested_python(
                task_metadata.get("_runtime_python")
            ),
            "packages": self._normalize_packages(task_metadata.get("_runtime_packages", [])),
            "framework_install": self._framework_install_fingerprint(),
            "system": platform.system(),
            "machine": platform.machine(),
            "python_mm": f"{sys.version_info.major}.{sys.version_info.minor}",
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def _is_cache_hit(self, env_dir: Path) -> bool:
        metadata_path = env_dir / self.METADATA_FILENAME
        return metadata_path.exists() and self._python_executable(env_dir).exists()

    def _lock_path(self, cache_key: str) -> Path:
        return self.cache_root / f"{cache_key}.lock"

    def _staging_env_dir(self, cache_key: str) -> Path:
        return self.cache_root / f"{cache_key}.build-{os.getpid()}-{uuid.uuid4().hex[:8]}"

    @contextmanager
    def _acquire_lock(self, lock_path: Path):
        start = time.monotonic()
        while True:
            try:
                lock_path.mkdir(parents=True, exist_ok=False)
                self._write_lock_metadata(lock_path)
                break
            except FileExistsError:
                if self._lock_is_stale(lock_path):
                    shutil.rmtree(lock_path, ignore_errors=True)
                    continue
                if time.monotonic() - start >= self.LOCK_TIMEOUT_SECONDS:
                    raise TimeoutError(f"Timed out waiting for task env lock: {lock_path}")
                time.sleep(self.LOCK_POLL_INTERVAL_SECONDS)

        try:
            yield
        finally:
            shutil.rmtree(lock_path, ignore_errors=True)

    def _write_lock_metadata(self, lock_path: Path) -> None:
        (lock_path / "owner.json").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "created_at": time.time(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _lock_is_stale(self, lock_path: Path) -> bool:
        owner_path = lock_path / "owner.json"
        created_at = lock_path.stat().st_mtime if lock_path.exists() else 0.0
        hostname = None
        pid = None
        if owner_path.exists():
            try:
                payload = json.loads(owner_path.read_text(encoding="utf-8"))
                created_at = float(payload.get("created_at", created_at))
                hostname = payload.get("hostname")
                pid = payload.get("pid")
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass

        if hostname == socket.gethostname() and isinstance(pid, int):
            return not self._pid_exists(pid)

        return (time.time() - created_at) >= self.STALE_LOCK_SECONDS

    def _pid_exists(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _cleanup_staging_dirs(self, cache_key: str) -> None:
        for staging_dir in self.cache_root.glob(f"{cache_key}.build-*"):
            if staging_dir.is_dir():
                shutil.rmtree(staging_dir, ignore_errors=True)

    def describe_cached_env(self, task_metadata: dict[str, Any]) -> dict[str, Any] | None:
        cache_key = self.compute_cache_key(task_metadata)
        env_dir = self.cache_root / cache_key
        if not self._is_cache_hit(env_dir):
            return None
        return {
            "cache_key": cache_key,
            "resolved_python_version": self._read_cached_python_version(env_dir),
            "python_requirement_satisfied": self._read_cached_python_requirement_satisfied(env_dir),
        }

    def _build_env(
        self,
        *,
        env_dir: Path,
        cache_key: str,
        runtime_packages: list[str],
        python_requirement: str | None,
    ) -> str:
        env_dir.mkdir(parents=True, exist_ok=False)
        python_executable = self._python_executable(env_dir)
        requested_python = self._select_requested_python(python_requirement)
        self._run(
            ["uv", "venv", str(env_dir), "--python", requested_python],
            cwd=self.repo_root,
        )
        resolved_python_version = self._resolve_python_version(python_executable)
        self._validate_python_requirement(
            python_requirement,
            resolved_python_version,
            python_executable,
        )
        self._run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python_executable),
                *self._framework_install_args(),
            ],
            cwd=self.repo_root,
        )
        if runtime_packages:
            self._run(
                ["uv", "pip", "install", "--python", str(python_executable), *runtime_packages],
                cwd=self.repo_root,
            )

        self._write_metadata(
            env_dir,
            {
                "cache_schema_version": self.CACHE_SCHEMA_VERSION,
                "cache_key": cache_key,
                "python_requirement": python_requirement,
                "requested_python": requested_python,
                "resolved_python_version": resolved_python_version,
                "python_requirement_satisfied": None if python_requirement is None else True,
                "packages": runtime_packages,
                "platform": {
                    "system": platform.system(),
                    "machine": platform.machine(),
                    "python_version": platform.python_version(),
                },
            },
        )
        return resolved_python_version

    def _publish_staged_env(self, staging_dir: Path, env_dir: Path) -> None:
        if env_dir.exists():
            shutil.rmtree(env_dir, ignore_errors=True)
        staging_dir.rename(env_dir)

    def _python_executable(self, env_dir: Path) -> Path:
        return env_dir / "Scripts" / "python.exe" if os.name == "nt" else env_dir / "bin" / "python"

    def _select_requested_python(self, python_requirement: str | None) -> str:
        """Choose the interpreter uv should use for the task environment."""
        current_python = sys.executable
        if not python_requirement:
            return current_python
        current_version = platform.python_version()
        if self._version_satisfies(python_requirement, current_version):
            return current_python
        return python_requirement

    def _read_cached_python_version(self, env_dir: Path) -> str | None:
        metadata = self._read_metadata(env_dir)
        resolved = metadata.get("resolved_python_version")
        if isinstance(resolved, str) and resolved.strip():
            return resolved.strip()
        python_executable = self._python_executable(env_dir)
        if python_executable.exists():
            return self._resolve_python_version(python_executable)
        return None

    def _read_cached_python_requirement_satisfied(self, env_dir: Path) -> bool | None:
        metadata = self._read_metadata(env_dir)
        satisfied = metadata.get("python_requirement_satisfied")
        if isinstance(satisfied, bool):
            return satisfied
        requirement = metadata.get("python_requirement")
        resolved = self._read_cached_python_version(env_dir)
        if isinstance(requirement, str) and resolved:
            return self._version_satisfies(requirement, resolved)
        return None

    def _read_metadata(self, env_dir: Path) -> dict[str, Any]:
        metadata_path = env_dir / self.METADATA_FILENAME
        if not metadata_path.exists():
            return {}
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_metadata(self, env_dir: Path, payload: dict[str, Any]) -> None:
        (env_dir / self.METADATA_FILENAME).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _resolve_python_version(self, python_executable: Path) -> str:
        result = subprocess.run(
            [
                str(python_executable),
                "-c",
                "import platform; print(platform.python_version())",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to inspect task env interpreter version: {python_executable}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        return result.stdout.strip()

    def _validate_python_requirement(
        self,
        python_requirement: str | None,
        resolved_python_version: str,
        python_executable: Path,
    ) -> None:
        if not python_requirement:
            return
        if not self._version_satisfies(python_requirement, resolved_python_version):
            raise RuntimeError(
                "Task sandbox setup failed: requested runtime.python "
                f"{python_requirement!r}, but resolved interpreter "
                f"{resolved_python_version!r} at {python_executable} does not satisfy it."
            )

    _SPEC_PATTERN = re.compile(r"^(>=|<=|==|!=|>|<)?\s*([0-9]+(?:\.[0-9]+){0,2})$")

    def _version_satisfies(self, requirement: str, resolved_python_version: str) -> bool:
        resolved = self._parse_version_tuple(resolved_python_version)
        if resolved is None:
            raise RuntimeError(
                f"Unsupported resolved Python version format: {resolved_python_version!r}"
            )
        clauses = [clause.strip() for clause in requirement.split(",") if clause.strip()]
        if not clauses:
            return True
        for clause in clauses:
            match = self._SPEC_PATTERN.match(clause)
            if not match:
                raise RuntimeError(
                    f"Unsupported runtime.python specifier: {requirement!r}. "
                    "Use simple version comparisons like '>=3.11' or '==3.12'."
                )
            operator = match.group(1) or "=="
            required = self._parse_version_tuple(match.group(2))
            if required is None:
                raise RuntimeError(
                    f"Unsupported runtime.python version in specifier: {clause!r}"
                )
            if not self._compare_versions(resolved, required, operator):
                return False
        return True

    def _parse_version_tuple(self, value: str) -> tuple[int, int, int] | None:
        match = re.match(r"^\s*([0-9]+)(?:\.([0-9]+))?(?:\.([0-9]+))?\s*$", value)
        if not match:
            return None
        parts = [int(group) if group is not None else 0 for group in match.groups()]
        return tuple(parts)  # type: ignore[return-value]

    def _compare_versions(
        self,
        resolved: tuple[int, int, int],
        required: tuple[int, int, int],
        operator: str,
    ) -> bool:
        if operator == "==":
            return resolved == required
        if operator == "!=":
            return resolved != required
        if operator == ">=":
            return resolved >= required
        if operator == "<=":
            return resolved <= required
        if operator == ">":
            return resolved > required
        if operator == "<":
            return resolved < required
        raise RuntimeError(f"Unsupported runtime.python operator: {operator!r}")

    def _run(self, command: list[str], cwd: Path) -> None:
        env = dict(os.environ)
        env.setdefault("UV_CACHE_DIR", str(self.uv_cache_dir))
        result = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Task sandbox setup failed: {' '.join(command)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

    def _hash_file(self, path: Path) -> str:
        if not path.exists():
            return "missing"
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _is_installable_project(path: Path) -> bool:
        """Return whether *path* can be passed to a Python package installer."""
        return (path / "pyproject.toml").is_file() or (path / "setup.py").is_file()

    def _installed_requirement(self) -> str:
        """Pin task envs to the framework version that launched the run."""
        try:
            version = importlib.metadata.version(DISTRIBUTION_NAME)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "Task sandbox setup failed: the ASI-Bench source checkout was not "
                "found and the installed asibench distribution version could not be "
                "determined. Reinstall asibench or run from a source checkout."
            ) from exc
        return f"{DISTRIBUTION_NAME}=={version}"

    def _framework_install_args(self) -> list[str]:
        if self.source_checkout:
            return ["-e", str(self.repo_root)]
        return [self._installed_requirement()]

    def _framework_install_fingerprint(self) -> str:
        if self.source_checkout:
            return "editable:" + self._hash_file(self.repo_root / "pyproject.toml")
        return "distribution:" + self._installed_requirement()

    def _normalize_packages(self, packages: list[Any]) -> list[str]:
        normalized = []
        for package in packages:
            pkg = str(package).strip()
            if pkg:
                normalized.append(pkg)
        return sorted(dict.fromkeys(normalized))
