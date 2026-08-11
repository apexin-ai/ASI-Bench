"""Docker containerized agent adapter — runs agents inside Docker containers."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from ai4sci_bench.core.agent_interface import AgentAdapter
from ai4sci_bench.core.logger import get_logger
from ai4sci_bench.core.types import AgentOutput, RunStatus, TaskInstance

logger = get_logger(__name__)


class DockerAgentAdapter(AgentAdapter):
    """Run an agent inside a Docker container.

    The container receives the workspace as a mounted volume. The agent
    reads prompt.md and data/ from /workspace, writes output files to
    /workspace, and exits.

    Config example:
        adapter = DockerAgentAdapter(
            image="my-agent:latest",
            cmd_template="python /app/agent.py --workspace /workspace",
        )
    """

    def __init__(
        self,
        image: str,
        cmd_template: str = "python /app/agent.py --workspace /workspace",
        timeout: int = 10800,
        memory_limit: str = "8g",
        cpu_limit: str = "4.0",
        network: bool = False,
        gpu: bool = False,
        env: dict[str, str] | None = None,
        volumes: dict[str, str] | None = None,
    ):
        self.image = image
        self.cmd_template = cmd_template
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.network = network
        self.gpu = gpu
        self.env = env or {}
        self.volumes = volumes or {}

    def solve(self, task_instance: TaskInstance) -> AgentOutput:
        eff_timeout = self._get_effective_timeout(task_instance)
        workspace = task_instance.workspace_dir

        cmd = self._build_docker_cmd(workspace, task_instance)

        t0 = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=eff_timeout + 30,
            )
            elapsed = time.time() - t0
            raw_stdout = result.stdout
            raw_stderr = result.stderr
            log = result.stdout
            if result.stderr:
                log += "\n[stderr]\n" + result.stderr
            status = RunStatus.COMPLETED if result.returncode == 0 else RunStatus.FAILED

        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - t0
            log = f"Docker agent timed out after {eff_timeout}s"
            status = RunStatus.TIMEOUT
            raw_stdout = e.stdout or ""
            raw_stderr = e.stderr or ""

        except Exception as e:
            elapsed = time.time() - t0
            log = f"Docker agent error: {e}"
            status = RunStatus.FAILED
            raw_stdout = None
            raw_stderr = None

        produced_files = self._collect_output_files(workspace, task_instance)

        return AgentOutput(
            instance_id=task_instance.instance_id,
            output_dir=workspace,
            code_files=[f for f in produced_files if f.endswith(".py")],
            data_files=[f for f in produced_files if not f.endswith(".py")],
            log=log,
            execution_time_seconds=elapsed,
            status=status,
            raw_stdout=raw_stdout,
            raw_stderr=raw_stderr,
            raw_stdout_format="log" if raw_stdout is not None else None,
        )

    def _build_docker_cmd(
        self, workspace: Path, task_instance: TaskInstance
    ) -> list[str]:
        """Build the docker run command."""
        cmd = [
            "docker", "run",
            "--rm",
            "-v", f"{workspace.resolve()}:/workspace",
            "-w", "/workspace",
            "--memory", self.memory_limit,
            "--cpus", self.cpu_limit,
        ]

        if not self.network:
            cmd += ["--network", "none"]

        if self.gpu:
            cmd += ["--gpus", "all"]

        for key, value in self.env.items():
            cmd += ["-e", f"{key}={value}"]

        for host_path, container_path in self.volumes.items():
            cmd += ["-v", f"{host_path}:{container_path}"]

        cmd.append(self.image)

        # Format the command template
        agent_cmd = self.cmd_template.format(
            workspace="/workspace",
            prompt_file="/workspace/prompt.md",
            prompt_level=task_instance.prompt_level.value,
            task_id=task_instance.task_id,
            instance_id=task_instance.instance_id,
        )

        cmd += ["bash", "-c", agent_cmd]

        return cmd

    def _collect_output_files(
        self, workspace: Path, task_instance: TaskInstance
    ) -> list[str]:
        """Collect output files from workspace."""
        files = []
        for f in task_instance.metadata.get("output", {}).get("files", []):
            if (workspace / f["name"]).exists():
                files.append(f["name"])
        return files
