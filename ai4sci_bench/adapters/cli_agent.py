"""Generic CLI agent adapter — runs any agent via shell command."""

from __future__ import annotations

import logging
import shlex
from pathlib import Path

from ai4sci_bench.adapters.subprocess_base import SubprocessAgentAdapter
from ai4sci_bench.core.types import AgentOutput, TaskInstance

logger = logging.getLogger(__name__)


class CLIAgentAdapter(SubprocessAgentAdapter):
    """Run any agent system that accepts a workspace directory via CLI.

    Supported placeholders in cmd_template:
      {workspace}     absolute path to the workspace directory
      {prompt_file}   absolute path to workspace/prompt.md
      {prompt_level}  b1, b2, or b3
      {task_id}       e.g. physics.vpfm_leapfrog
      {instance_id}   e.g. physics.vpfm_leapfrog__grid255_seed42

    **Security warning**: ``{instance_id}`` encodes the resolved
    parameters and random seed (e.g. ``physics.vpfm_leapfrog__grid255_cfl1.0_seed42``).
    If the template passes this value as a visible argument to the agent,
    the agent can reverse-engineer the exact parameter set being tested.
    Prefer ``{workspace}`` / ``{prompt_file}`` when possible.  Only use
    ``{instance_id}`` for logging or output-path purposes, never in
    agent-visible prompts.
    """

    def __init__(self, cmd_template: str, timeout: int = 10800, cwd: str | None = None):
        super().__init__(
            timeout_seconds=timeout,
            sandbox="none",
            supported_sandbox_modes=("none", "linux_ns"),
        )
        self.cmd_template = cmd_template
        self.cwd_override = cwd

    def _build_command(self, task_instance: TaskInstance, task_env):
        workspace = task_instance.workspace_dir
        # shlex.quote every substituted value so that paths or ids
        # containing spaces / shell metacharacters cannot break out of
        # the intended argument slot. cmd_template itself is trusted
        # (provided by the framework operator), but the values we
        # substitute include instance/task ids and workspace paths that
        # may be influenced by task authors.
        return self.cmd_template.format(
            workspace=shlex.quote(str(workspace)),
            prompt_file=shlex.quote(str(workspace / "prompt.md")),
            prompt_level=shlex.quote(task_instance.prompt_level.value),
            task_id=shlex.quote(task_instance.task_id),
            instance_id=shlex.quote(task_instance.instance_id),
        )

    def solve(self, task_instance: TaskInstance) -> AgentOutput:
        output = super().solve(task_instance)
        traj_file = task_instance.workspace_dir / "_trajectory.jsonl"
        if traj_file.exists() and output.raw_stdout is None:
            try:
                output.raw_stdout = traj_file.read_text(encoding="utf-8")
                output.raw_stdout_format = "jsonl"
            except Exception as exc:
                logger.warning("Failed to read %s: %s", traj_file, exc)
        return output

    def _parse_log(self, stdout: str) -> str:
        return stdout

    def _get_cwd(self, task_instance: TaskInstance) -> Path:
        return Path(self.cwd_override) if self.cwd_override else task_instance.workspace_dir

    def _use_shell(self) -> bool:
        return True
