"""Shared type definitions for the benchmark framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


DEFAULT_TIMEOUT_SECONDS = 10800


class PromptLevel(Enum):
    """Prompt information gradient levels."""
    B1 = "b1"  # full algorithm description
    B2 = "b2"  # method name + background only
    B3 = "b3"  # goal and output format only
    B4 = "b4"  # B3 + factually correct redundant information


class GenerationMode(Enum):
    """Task instance generation mode."""
    INFINITE = "infinite"  # random sampling from parameter space
    FINITE = "finite"      # fixed enumerated settings list


class TaskLifecycle(Enum):
    """Task definition lifecycle status (used in task.yaml `status` field).

    Distinct from RunStatus — this describes the maturity of the task definition,
    not the outcome of a benchmark run.
    """
    IN_DEVELOPMENT = "in_development"
    TEST = "test"
    SAMPLE = "sample"
    FINAL = "final"
    ABANDONED = "abandoned"


class RunStatus(Enum):
    """Runtime execution status of one benchmark run instance."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ToolMode(Enum):
    """Agent tool isolation level during benchmark runs."""
    RESTRICTED = "restricted"
    SEARCH = "search"
    UNRESTRICTED = "unrestricted"


@dataclass
class TaskInstance:
    """A concrete task instance (task + parameter set).

    **Security note on ``metadata``**:  This field holds the full
    ``task.yaml`` dict (including evaluation config, gates, scoring
    thresholds and reference file names).  Adapters may read
    ``metadata["output"]["files"]`` for output file collection and pass
    the dict to ``TaskEnvironmentManager.ensure_env()`` for runtime
    resolution — but **must never forward the full dict** to the agent
    (via prompts, HTTP payloads, workspace files, environment variables,
    or any other channel).  Doing so would leak evaluation criteria,
    reference file names, scoring thresholds and parameter ranges to the
    agent under test.

    Use :pyattr:`agent_safe_metadata` when only the agent-safe subset is
    needed.
    """

    task_id: str
    instance_id: str
    task_dir: Path
    workspace_dir: Path
    reference_dir: Path
    prompt_level: PromptLevel
    parameters: dict[str, Any]
    metadata: dict[str, Any]
    retained_workspace_dir: Path | None = None
    effective_timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS

    # Keys from metadata that are safe to expose to agents.  Everything
    # else (evaluation, gates, scoring, reference files, …) must stay
    # framework-internal.
    _AGENT_SAFE_METADATA_KEYS = frozenset({"output"})

    @property
    def run_key(self) -> str:
        """Composite key that uniquely identifies an instance + prompt level run."""
        return f"{self.instance_id}__{self.prompt_level.value}"

    @property
    def agent_safe_metadata(self) -> dict[str, Any]:
        """Return the subset of metadata safe to expose to agents.

        Currently only ``output`` (file declarations) is included.
        Evaluation config, gates, scoring thresholds, and reference file
        names are **never** exposed.
        """
        return {
            k: v for k, v in self.metadata.items()
            if k in self._AGENT_SAFE_METADATA_KEYS
        }


@dataclass
class CostInfo:
    """Token usage and cost information for one instance."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class AgentOutput:
    """Output produced by an agent for one task instance."""
    instance_id: str
    output_dir: Path
    code_files: list[str]
    data_files: list[str]
    log: str
    execution_time_seconds: float
    status: RunStatus
    error_message: str | None = None
    cost: CostInfo | None = None
    raw_stdout: str | None = None
    raw_stderr: str | None = None
    raw_stdout_format: str | None = None
    raw_stdout_file: str | None = None
    raw_stderr_file: str | None = None
    raw_model_output: str | None = None
    raw_model_output_format: str | None = None
    raw_model_output_file: str | None = None


@dataclass
class ScoreDetail:
    """Scoring result from a single scorer."""
    scorer_name: str
    score: float
    max_score: float
    passed: bool
    details: dict[str, Any]
    message: str = ""
    severity: str | None = None


@dataclass
class AnalysisReport:
    """Structured error analysis report produced by the analyzer."""
    instance_id: str
    error_category: str
    error_subcategory: str
    root_cause: str
    evidence: list[str]
    fix_suggestions: list[str]
    raw_analysis: str
    confidence: float


@dataclass
class EvalResult:
    """Complete evaluation result for one task instance."""
    instance_id: str
    task_id: str
    prompt_level: PromptLevel
    agent_name: str
    parameters: dict[str, Any]

    gate_results: list[ScoreDetail]
    gates_passed: bool
    score_results: list[ScoreDetail]
    final_score: float
    hard_gates_passed: bool | None = None
    soft_gate_failures: int = 0
    max_possible_score: float = 100.0

    execution_time_seconds: float = 0.0
    status: RunStatus = RunStatus.COMPLETED

    attempt: int = 1

    error_analysis: AnalysisReport | None = None
    agent_output: AgentOutput | None = None
    cost: CostInfo | None = None

    def __post_init__(self) -> None:
        if self.hard_gates_passed is None:
            self.hard_gates_passed = self.gates_passed
        self.gates_passed = self.hard_gates_passed
