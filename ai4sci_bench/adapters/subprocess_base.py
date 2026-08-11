"""Base class for subprocess-based agent adapters.

Extracts the common subprocess execution pattern shared by CLIAgentAdapter,
ClaudeCodeCLIAdapter, and CodexCLIAdapter. Subclasses only need to implement
``_build_command()`` and optionally override hooks for environment, log
parsing, working directory, and shell mode.
"""

from __future__ import annotations

import signal as _signal_module
import subprocess
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any

from ai4sci_bench.core.agent_interface import AgentAdapter
from ai4sci_bench.core.types import AgentOutput, RunStatus, TaskInstance
from ai4sci_bench.runner.linux_ns_sandbox import LinuxNSSandbox
from ai4sci_bench.runner.proc_util import (
    GRACEFUL_SHUTDOWN_SECONDS,
    run_subprocess_with_graceful_timeout,
)
from ai4sci_bench.runner.runtime_root import resolve_runtime_root
from ai4sci_bench.runner.sandbox_support import validate_sandbox_mode
from ai4sci_bench.runner.task_env import TaskEnvironmentManager


# ── Shared helpers (used by both SubprocessAgentAdapter and DirectLLMAdapter) ──


def get_task_environment(
    task_env_manager: TaskEnvironmentManager | None,
    task_instance: TaskInstance,
):
    """Return the task sandbox environment, or None if no manager is set."""
    if task_env_manager is None:
        return None
    return task_env_manager.ensure_env(task_instance.metadata)


def collect_output_files(
    workspace: Path, task_instance: TaskInstance
) -> list[str]:
    """Collect expected output files that exist in the workspace."""
    files = []
    for f in task_instance.metadata.get("output", {}).get("files", []):
        if (workspace / f["name"]).exists():
            files.append(f["name"])
    return files


_SIGNAL_HINTS: dict[int, str] = {
    9: "likely OOM killed by kernel",
    15: "requested termination",
    6: "process aborted (assertion failure)",
    11: "segmentation fault",
    14: "alarm/timeout signal",
}


def _format_exit_error(class_name: str, returncode: int) -> str:
    """Format an exit-code error message, translating signals for negative codes."""
    base = f"{class_name} exited with code {returncode}"
    if returncode >= 0:
        return base
    signum = -returncode
    try:
        sig_name = _signal_module.Signals(signum).name
    except (ValueError, AttributeError):
        return f"{base} (signal {signum})"
    hint = _SIGNAL_HINTS.get(signum, "")
    if hint:
        return f"{base} ({sig_name} — {hint})"
    return f"{base} ({sig_name})"


# ── Base class ────────────────────────────────────────────────────────────────


class SubprocessAgentAdapter(AgentAdapter):
    """Base for agents that run as subprocesses (CLI tools, scripts, etc.).

    Handles:
    - Sandbox environment setup (task env via uv)
    - Subprocess execution with timeout
    - Output file collection
    - Raw log capture

    Subclasses must implement:
    - ``_build_command()``: return the CLI command (list or str)

    Subclasses may override:
    - ``_build_run_env()``: return environment dict
    - ``_parse_log()``: parse raw stdout into human-readable summary
    - ``_raw_stdout_format()``: format hint for raw stdout
    - ``_get_cwd()``: working directory for subprocess
    - ``_use_shell()``: whether to use shell=True
    """

    def __init__(
        self,
        timeout_seconds: int = 10800,
        sandbox: str = "none",
        supported_sandbox_modes: tuple[str, ...] = ("none", "task"),
    ):
        self.timeout_seconds = timeout_seconds
        self.sandbox = sandbox
        self._supported_sandbox_modes = supported_sandbox_modes
        self.repo_root = resolve_runtime_root()
        self.task_env_manager: TaskEnvironmentManager | None = None
        self._linux_ns_sandbox: LinuxNSSandbox | None = None

    def setup(self, config: dict) -> None:
        """Common setup: timeout, sandbox, repo_root."""
        self.timeout_seconds = int(config.get("timeout", self.timeout_seconds))
        self.sandbox = config.get("sandbox", self.sandbox)
        validate_sandbox_mode(
            self.sandbox,
            supported_modes=self._supported_sandbox_modes,
            component=self.__class__.__name__,
        )
        self.repo_root = Path(config.get("repo_root", self.repo_root))
        self.task_env_manager = (
            TaskEnvironmentManager(self.repo_root)
            if self.sandbox in ("task", "linux_ns")
            else None
        )
        self._linux_ns_sandbox = (
            LinuxNSSandbox() if self.sandbox == "linux_ns" else None
        )

    def solve(self, task_instance: TaskInstance) -> AgentOutput:
        """Execute agent subprocess and collect results."""
        eff_timeout = self._get_effective_timeout(task_instance)
        workspace = task_instance.workspace_dir
        task_env = get_task_environment(self.task_env_manager, task_instance)

        cmd = self._build_command(task_instance, task_env)
        env = self._build_run_env(task_instance, task_env)
        cwd = self._get_cwd(task_instance)
        use_shell = self._use_shell()
        stdin_input = self._get_stdin_input(task_instance, task_env)

        t0 = time.time()
        try:
            if self.sandbox == "linux_ns":
                assert self._linux_ns_sandbox is not None
                success, sandbox_log, raw_stdout, raw_stderr = self._linux_ns_sandbox.run_agent(
                    agent_cmd=cmd,
                    workspace=cwd,
                    timeout=eff_timeout,
                    extra_env=env,
                    shell=use_shell,
                    input_text=stdin_input,
                )
                elapsed = time.time() - t0
                produced_files = collect_output_files(workspace, task_instance)
                if "timed out" in sandbox_log:
                    status = RunStatus.TIMEOUT
                elif success:
                    status = RunStatus.COMPLETED
                else:
                    status = RunStatus.FAILED
                if raw_stdout is None and raw_stderr is None:
                    log = sandbox_log
                else:
                    log = self._build_full_log(
                        raw_stdout or "",
                        raw_stderr or "",
                        0 if success else 1,
                    )
                error_message = None if success else sandbox_log
                return AgentOutput(
                    instance_id=task_instance.instance_id,
                    output_dir=workspace,
                    code_files=[f for f in produced_files if f.endswith(".py")],
                    data_files=[f for f in produced_files if not f.endswith(".py")],
                    log=log,
                    execution_time_seconds=elapsed,
                    status=status,
                    error_message=error_message,
                    raw_stdout=raw_stdout,
                    raw_stderr=raw_stderr,
                    raw_stdout_format=(
                        self._raw_stdout_format() if raw_stdout is not None else None
                    ),
                )

            result = run_subprocess_with_graceful_timeout(
                cmd,
                cwd=str(cwd),
                timeout=eff_timeout,
                env=env,
                shell=use_shell,
                input=stdin_input,
            )
            elapsed = time.time() - t0
            raw_stdout = result.stdout
            raw_stderr = result.stderr
            log = self._build_full_log(result.stdout, result.stderr, result.returncode)
            produced_files = collect_output_files(workspace, task_instance)
            status = RunStatus.COMPLETED if result.returncode == 0 else RunStatus.FAILED
            error_message = (
                None
                if result.returncode == 0
                else _format_exit_error(self.__class__.__name__, result.returncode)
            )

        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - t0
            raw_stdout = e.stdout or ""
            raw_stderr = e.stderr or ""
            log = self._build_full_log(e.stdout or "", e.stderr or "", None)
            kill_note = (
                " (force-killed via SIGKILL after grace period)"
                if getattr(e, "forced_kill", False)
                else " (terminated via SIGTERM)"
            )
            timeout_line = (
                f"{self.__class__.__name__} timed out after {eff_timeout}s{kill_note}"
            )
            log = f"{timeout_line}\n\n{log}" if log else timeout_line
            produced_files = collect_output_files(workspace, task_instance)
            status = RunStatus.TIMEOUT
            error_message = timeout_line

        except Exception as e:
            elapsed = time.time() - t0
            raw_stdout = None
            raw_stderr = None
            log = str(e)
            produced_files = []
            status = RunStatus.FAILED
            error_message = str(e)

        return AgentOutput(
            instance_id=task_instance.instance_id,
            output_dir=workspace,
            code_files=[f for f in produced_files if f.endswith(".py")],
            data_files=[f for f in produced_files if not f.endswith(".py")],
            log=log,
            execution_time_seconds=elapsed,
            status=status,
            error_message=error_message,
            raw_stdout=raw_stdout,
            raw_stderr=raw_stderr,
            raw_stdout_format=(
                self._raw_stdout_format() if raw_stdout is not None else None
            ),
        )

    # ── Subclass hooks ──────────────────────────────────────────

    @abstractmethod
    def _build_command(
        self, task_instance: TaskInstance, task_env: Any | None
    ) -> list[str] | str:
        """Return the CLI command to execute.

        Return ``list[str]`` for ``shell=False``, ``str`` for ``shell=True``.
        """
        ...

    def _build_run_env(
        self, task_instance: TaskInstance, task_env: Any | None
    ) -> dict[str, str] | None:
        """Return subprocess environment dict. Default: task_env or inherit."""
        return task_env.build_subprocess_env() if task_env else None

    def _parse_log(self, stdout: str) -> str:
        """Parse raw stdout into human-readable log. Default: return as-is."""
        return stdout

    def _raw_stdout_format(self) -> str:
        """Format hint for raw stdout. Default: 'log'."""
        return "log"

    def _get_cwd(self, task_instance: TaskInstance) -> Path:
        """Working directory for subprocess. Default: workspace (absolute)."""
        return task_instance.workspace_dir.resolve()

    def _use_shell(self) -> bool:
        """Whether to use shell=True. Default: False."""
        return False

    def _get_stdin_input(self, task_instance: TaskInstance | None = None, task_env=None) -> str | None:
        """Return text to feed to subprocess stdin. Default: None (no stdin).

        Subclasses should compute stdin from ``task_instance`` (thread-safe)
        rather than caching in ``self`` attributes.
        """
        return None

    # ── Common helpers ─────────────────────────────────────────

    def _build_full_log(
        self, stdout: str, stderr: str, returncode: int | None
    ) -> str:
        """Build combined log from parsed stdout + stderr + exit code."""
        parts = []
        parsed = self._parse_log(stdout)
        if parsed:
            parts.append(parsed)
        if stderr and stderr.strip():
            parts.append(f"[stderr]\n{stderr.strip()}")
        if returncode not in (None, 0):
            parts.append(f"[exit_code] {returncode}")
        return "\n\n".join(parts)
