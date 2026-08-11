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
