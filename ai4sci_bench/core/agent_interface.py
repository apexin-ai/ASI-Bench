"""Agent adapter base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ai4sci_bench.core.types import AgentOutput, TaskInstance


class AgentAdapter(ABC):
    """Base class for agent adapters.

    External agent systems can either implement this interface,
    or use the file-exchange CLI mode without any code changes.
    """

    def _init_network_policy(self, policy: str = "auto") -> None:
        from ai4sci_bench.runner.network_policy import validate_network_policy
        self.network_policy = validate_network_policy(policy)
        self.effective_network_policy = "inherit"

    def _setup_network_policy(self) -> None:
        from ai4sci_bench.runner.network_policy import select_network_policy
        self.effective_network_policy = select_network_policy(
            getattr(self, "network_policy", "auto"), self.sandbox,
            getattr(self, "tool_mode", "restricted"),
        )

    def _run_os_agent(self, *, network_runtime: dict, **kwargs):
        """Single policy boundary for built-in OS harnesses; never fall back."""
        policy = self.effective_network_policy
        network_runtime.update(requested=self.network_policy, effective=policy, enforced=False)
        try:
            if policy == "model-only":
                from ai4sci_bench.runner.network_policy import model_connection
                kwargs["model_connection"] = model_connection(self, kwargs["agent_type"])
            return self._os_sandbox.run_agent(
                **kwargs, network_policy=policy, network_runtime=network_runtime,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            network_runtime.update(startup_check="rejected")
            return False, f"OS sandbox startup rejected: {exc}", None, None, None

    @abstractmethod
    def solve(self, task_instance: TaskInstance) -> AgentOutput:
        """Run the agent on a task instance.

        The framework has already prepared workspace_dir with:
          - prompt.md       (task description at the configured prompt level)
          - task_info.json  (agent-facing machine-readable task metadata)
          - data/           (input data files)

        The agent must write all expected output files into workspace_dir,
        then return an AgentOutput describing what was produced.
        """
        ...

    def _get_effective_timeout(self, task_instance: TaskInstance) -> int:
        """Read the per-instance timeout (thread-safe, no self mutation)."""
        return task_instance.effective_timeout_seconds

    def setup(self, config: dict) -> None:
        """Optional: initialize the agent."""
        pass

    def teardown(self) -> None:
        """Optional: release resources."""
        pass
