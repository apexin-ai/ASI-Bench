"""Task-independent agent CLI invocation.

``solve(TaskInstance)`` is the benchmark contract: it reads ``prompt.md`` from
the task workspace, keys per-run state on the instance, and collects the
task's declared deliverables afterwards. Everything in between -- building the
CLI command line, the API environment, the isolated HOME, running the process
under a graceful timeout, and turning its stream-json into a status and a cost
-- does not depend on the task at all.

``invoke(Invocation)`` exposes that middle layer on its own, so callers that
are not running a benchmark task (for example a task *generator* that hands
the agent a role prompt and a workspace) drive the same CLI code path the
benchmark does. ``solve`` keeps its own task-shaped hooks (``_build_command``,
``_build_run_env``, ``_get_stdin_input``), which subclasses and tests may
override; both paths are built from the same task-independent pieces.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ai4sci_bench.core.types import CostInfo, RunStatus

SystemPromptMode = Literal["append", "replace"]


@dataclass(frozen=True)
class Invocation:
    """One agent CLI run: a prompt, a working directory, and a run identity.

    ``run_key`` names the per-run isolated HOME and the live log files, so two
    invocations that must not share state need distinct keys.

    ``system_prompt`` is added to the CLI's own agent instructions when
    ``system_prompt_mode`` is ``"append"`` (Claude ``--append-system-prompt``,
    Codex ``developer_instructions``) and replaces them when it is
    ``"replace"`` (Claude ``--system-prompt``, Codex
    ``model_instructions_file``). Replacing drops the CLI's built-in guidance on
    how to use its own tools, so prefer ``"append"`` unless the caller's prompt
    was written to stand alone.

    ``allowed_tools`` is Claude's ``--allowedTools`` permission allow-list; it
    matters under a permission mode such as ``dontAsk`` that denies anything
    not listed. Codex has no equivalent and rejects it.

    ``env`` is the base environment for the child process (default: this
    process's environment); the adapter layers its API and HOME settings on
    top.
    """

    prompt: str
    workspace: Path
    run_key: str
    timeout_seconds: int | None = None
    system_prompt: str | None = None
    system_prompt_mode: SystemPromptMode = "append"
    allowed_tools: str | None = None
    env: dict[str, str] | None = None


@dataclass
class InvocationResult:
    """What one CLI run produced, before any task-level interpretation.

    ``launched`` is False when the process could not be started at all (as
    opposed to starting and then failing or timing out).
    """

    status: RunStatus
    returncode: int | None
    raw_stdout: str | None
    raw_stderr: str | None
    log: str
    execution_time_seconds: float
    error_message: str | None = None
    cost: CostInfo | None = None
    launched: bool = True
    raw_stdout_format: str | None = field(default=None)
