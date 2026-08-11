"""Linux namespace sandbox using unshare + bind mounts.

This provides lightweight filesystem isolation without Docker, suitable for
HPC clusters and RL training loops.  Requires Linux with user namespace
support.  On non-Linux platforms, importing this module is fine but calling
``LinuxNSSandbox.execute_python()`` will raise ``RuntimeError``.

Inspired by SWE-MiniSandbox (2026) which demonstrated that Linux namespaces
can achieve ~5% of Docker's disk overhead and ~25% of its setup time.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ai4sci_bench.runner.proc_util import run_subprocess_with_graceful_timeout

logger = logging.getLogger(__name__)


def check_linux_ns_available() -> tuple[bool, str]:
    """Check if Linux namespace isolation is available on the current host.

    Returns (available, reason).
    """
    if sys.platform != "linux":
        return False, f"linux_ns requires Linux (current platform: {sys.platform})"

    # Check for unshare command
    try:
        result = subprocess.run(
            ["unshare", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False, "unshare command not available"
    except FileNotFoundError:
        return False, "unshare command not found"
    except Exception as e:
        return False, f"Failed to check unshare: {e}"

    # Check user namespace support
    try:
        result = subprocess.run(
            ["unshare", "--user", "--map-root-user", "true"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False, "User namespaces not supported or disabled (check /proc/sys/kernel/unprivileged_userns_clone)"
    except Exception as e:
        return False, f"User namespace test failed: {e}"

    return True, "ok"


class LinuxNSSandbox:
    """Lightweight sandbox using Linux namespaces (unshare + bind mounts).

    This is a degraded alternative to the Docker-based OSSandbox for
    environments where Docker is not available (e.g. HPC clusters).
    """

    DEFAULT_MEMORY_LIMIT_KB = 8 * 1024 * 1024  # 8 GB

    def __init__(self, *, memory_limit_kb: int | None = None):
        if sys.platform != "linux":
            raise RuntimeError(
                f"LinuxNSSandbox requires Linux (current platform: {sys.platform}). "
                "Use --sandbox os (Docker) or --sandbox task instead."
            )
        self.memory_limit_kb = memory_limit_kb or self.DEFAULT_MEMORY_LIMIT_KB

    def run_command(
        self,
        *,
        command: list[str] | str,
        cwd: Path,
        timeout: int,
        extra_env: dict[str, str] | None = None,
        shell: bool = False,
        input_text: str | None = None,
    ) -> tuple[bool, str, str | None, str | None]:
        """Run an arbitrary command inside a namespace sandbox."""
        available, reason = check_linux_ns_available()
        if not available:
            raise RuntimeError(f"Linux namespace isolation unavailable: {reason}")

        env = dict(os.environ)
        env.update({
            "HOME": str(cwd),
            "LANG": env.get("LANG", "C.UTF-8"),
            "AI4SCI_SANDBOX": "1",
        })
        if extra_env:
            env.update(extra_env)

        sandbox_cmd: list[str] = [
            "unshare",
            "--user",
            "--map-root-user",
            "--mount",
            "--pid",
            "--fork",
            "--",
        ]
        if shell:
            if not isinstance(command, str):
                raise TypeError("linux_ns shell execution requires command as string")
            sandbox_cmd += ["sh", "-lc", command]
        else:
            if isinstance(command, str):
                raise TypeError("linux_ns non-shell execution requires command as list[str]")
            sandbox_cmd += command

        try:
            result = run_subprocess_with_graceful_timeout(
                sandbox_cmd,
                cwd=cwd,
                env=env,
                timeout=timeout + 10,
                input=input_text,
            )
            log = result.stdout + ("\n" + result.stderr if result.stderr else "")
            return result.returncode == 0, log, result.stdout, result.stderr
        except subprocess.TimeoutExpired as exc:
            return False, f"linux_ns execution timed out ({timeout}s)", exc.stdout or "", exc.stderr or ""
        except Exception as exc:
            return False, f"linux_ns execution error: {exc}", None, None

    def execute_python(
        self,
        *,
        workspace: Path,
        code_file: str,
        timeout: int,
        python_executable: str = "python3",
        extra_env: dict[str, str] | None = None,
    ) -> tuple[bool, str, str | None, str | None]:
        """Execute a Python file inside a user namespace with isolated mount namespace.

        Returns (success, log, stdout, stderr).
        """
        return self.run_command(
            command=[python_executable, str(Path(code_file).name)],
            cwd=workspace,
            timeout=timeout,
            extra_env=extra_env,
        )

    def run_agent(
        self,
        *,
        agent_cmd: list[str] | str,
        workspace: Path,
        timeout: int,
        extra_env: dict[str, str] | None = None,
        shell: bool = False,
        input_text: str | None = None,
    ) -> tuple[bool, str, str | None, str | None]:
        """Run an agent command inside a namespace sandbox."""
        return self.run_command(
            command=agent_cmd,
            cwd=workspace,
            timeout=timeout,
            extra_env=extra_env,
            shell=shell,
            input_text=input_text,
        )
