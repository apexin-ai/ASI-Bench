"""Unified trajectory data model for all agent adapters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


VALID_STEP_TYPES = frozenset({
    "llm_request",
    "llm_response",
    "tool_call",
    "tool_result",
    "thinking",
    "code_execution",
    "error",
    "system",
    "rate_limit",
})


@dataclass
class TrajectoryStep:
    """One step in an agent's trajectory."""

    step_index: int
    timestamp_ms: int | None
    step_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrajectoryStep:
        return cls(
            step_index=d["step_index"],
            timestamp_ms=d.get("timestamp_ms"),
            step_type=d["step_type"],
            content=d.get("content", ""),
            metadata=d.get("metadata", {}),
        )


@dataclass
class TrajectorySummary:
    """Aggregated statistics from a trajectory."""

    total_turns: int = 0
    total_tool_calls: int = 0
    tool_call_distribution: dict[str, int] = field(default_factory=dict)
    thinking_total_chars: int = 0
    total_code_executions: int = 0
    code_execution_failures: int = 0
    unique_files_modified: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrajectorySummary:
        return cls(
            total_turns=d.get("total_turns", 0),
            total_tool_calls=d.get("total_tool_calls", 0),
            tool_call_distribution=d.get("tool_call_distribution", {}),
            thinking_total_chars=d.get("thinking_total_chars", 0),
            total_code_executions=d.get("total_code_executions", 0),
            code_execution_failures=d.get("code_execution_failures", 0),
            unique_files_modified=d.get("unique_files_modified", []),
        )

    @classmethod
    def from_steps(cls, steps: list[TrajectoryStep]) -> TrajectorySummary:
        """Compute summary statistics from a list of steps."""
        total_turns = 0
        total_tool_calls = 0
        tool_dist: dict[str, int] = {}
        thinking_chars = 0
        code_execs = 0
        code_failures = 0
        modified_files: set[str] = set()

        for step in steps:
            if step.step_type == "llm_response":
                total_turns += 1
            elif step.step_type == "tool_call":
                total_tool_calls += 1
                tool_name = step.metadata.get("tool_name", "unknown")
                tool_dist[tool_name] = tool_dist.get(tool_name, 0) + 1
                if tool_name in (
                    "Write", "Edit", "NotebookEdit", "write", "edit", "patch",
                ):
                    fp = (step.metadata.get("key_args", {}).get("file_path", "")
                          or step.metadata.get("key_args", {}).get("path", "")
                          or step.metadata.get("key_args", {}).get("filePath", ""))
                    if fp:
                        modified_files.add(fp)
                elif tool_name == "file_change":
                    for change in step.metadata.get("changes", []):
                        if isinstance(change, dict) and change.get("path"):
                            modified_files.add(change["path"])
            elif step.step_type == "thinking":
                thinking_chars += len(step.content)
            elif step.step_type == "tool_result":
                parent_tool = step.metadata.get("parent_tool_name", "")
                if parent_tool in ("Bash", "command_execution", "shell", "bash"):
                    code_execs += 1
                    if step.metadata.get("is_error"):
                        code_failures += 1
            elif step.step_type == "code_execution":
                code_execs += 1
                if step.metadata.get("exit_code", 0) != 0:
                    code_failures += 1

        return cls(
            total_turns=total_turns,
            total_tool_calls=total_tool_calls,
            tool_call_distribution=tool_dist,
            thinking_total_chars=thinking_chars,
            total_code_executions=code_execs,
            code_execution_failures=code_failures,
            unique_files_modified=sorted(modified_files),
        )


@dataclass
class Trajectory:
    """Complete agent trajectory for one instance."""

    instance_id: str
    adapter_type: str
    total_steps: int
    total_duration_ms: int | None
    steps: list[TrajectoryStep] = field(default_factory=list)
    summary: TrajectorySummary = field(default_factory=TrajectorySummary)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "adapter_type": self.adapter_type,
            "total_steps": self.total_steps,
            "total_duration_ms": self.total_duration_ms,
            "steps": [s.to_dict() for s in self.steps],
            "summary": self.summary.to_dict(),
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trajectory:
        steps = [TrajectoryStep.from_dict(s) for s in d.get("steps", [])]
        summary_data = d.get("summary", {})
        return cls(
            instance_id=d["instance_id"],
            adapter_type=d["adapter_type"],
            total_steps=d.get("total_steps", len(steps)),
            total_duration_ms=d.get("total_duration_ms"),
            steps=steps,
            summary=TrajectorySummary.from_dict(summary_data),
        )

    @classmethod
    def from_json(cls, text: str) -> Trajectory:
        return cls.from_dict(json.loads(text))
