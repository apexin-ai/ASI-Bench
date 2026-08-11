"""Tests for the retry mechanism in the orchestrator."""

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ai4sci_bench.core.agent_interface import AgentAdapter
from ai4sci_bench.core.types import (
    AgentOutput,
    EvalResult,
    PromptLevel,
    RunStatus,
    TaskInstance,
)
from ai4sci_bench.runner.orchestrator import BenchmarkOrchestrator, RunConfig


# ---------------------------------------------------------------------------
# Helper agents for testing
# ---------------------------------------------------------------------------


class AlwaysFailAgent(AgentAdapter):
    """Agent that always fails."""

    def __init__(self):
        self.call_count = 0

    def solve(self, task_instance: TaskInstance) -> AgentOutput:
        self.call_count += 1
        return AgentOutput(
            instance_id=task_instance.instance_id,
            output_dir=task_instance.workspace_dir,
            code_files=[],
            data_files=[],
            log=f"fail attempt {self.call_count}",
            execution_time_seconds=0.1,
            status=RunStatus.FAILED,
            error_message="always fails",
        )


class SucceedOnNthAgent(AgentAdapter):
    """Agent that fails N-1 times, then succeeds and writes output."""

    def __init__(self, succeed_on: int = 2, output_data=None):
        self.succeed_on = succeed_on
        self.call_count = 0
        self.output_data = output_data

    def solve(self, task_instance: TaskInstance) -> AgentOutput:
        self.call_count += 1
        if self.call_count >= self.succeed_on:
            if self.output_data:
                for name, data in self.output_data.items():
                    np.save(task_instance.workspace_dir / name, data)
            return AgentOutput(
                instance_id=task_instance.instance_id,
                output_dir=task_instance.workspace_dir,
                code_files=[],
                data_files=list(self.output_data or {}),
                log=f"success on attempt {self.call_count}",
                execution_time_seconds=0.5,
                status=RunStatus.COMPLETED,
            )
        return AgentOutput(
            instance_id=task_instance.instance_id,
            output_dir=task_instance.workspace_dir,
            code_files=[],
            data_files=[],
            log=f"fail attempt {self.call_count}",
            execution_time_seconds=0.1,
            status=RunStatus.FAILED,
            error_message=f"fail {self.call_count}",
        )


class AlwaysSucceedAgent(AgentAdapter):
    """Agent that always succeeds and writes output."""

    def __init__(self, output_data=None):
        self.call_count = 0
        self.output_data = output_data
        self.workspaces_seen: list[Path] = []

    def solve(self, task_instance: TaskInstance) -> AgentOutput:
        self.call_count += 1
        self.workspaces_seen.append(task_instance.workspace_dir)
        if self.output_data:
            for name, data in self.output_data.items():
                np.save(task_instance.workspace_dir / name, data)
        return AgentOutput(
            instance_id=task_instance.instance_id,
            output_dir=task_instance.workspace_dir,
            code_files=[],
            data_files=list(self.output_data or {}),
            log=f"success attempt {self.call_count}",
            execution_time_seconds=0.5,
            status=RunStatus.COMPLETED,
        )


# ---------------------------------------------------------------------------
# RunConfig defaults
# ---------------------------------------------------------------------------


class TestRunConfigRetryDefaults:
    def test_default_retries(self):
        agent = MagicMock(spec=AgentAdapter)
        config = RunConfig(agent=agent)
        assert config.retries == 1

    def test_default_retry_strategy(self):
        agent = MagicMock(spec=AgentAdapter)
        config = RunConfig(agent=agent)
        assert config.retry_strategy == "all"


# ---------------------------------------------------------------------------
# EvalResult attempt field
# ---------------------------------------------------------------------------


class TestEvalResultAttempt:
    def test_default_attempt(self):
        result = EvalResult(
            instance_id="test",
            task_id="test",
            prompt_level=PromptLevel.B2,
            agent_name="test",
            parameters={},
            gate_results=[],
            gates_passed=True,
            score_results=[],
            final_score=50.0,
        )
        assert result.attempt == 1

    def test_custom_attempt(self):
        result = EvalResult(
            instance_id="test",
            task_id="test",
            prompt_level=PromptLevel.B2,
            agent_name="test",
            parameters={},
            gate_results=[],
            gates_passed=True,
            score_results=[],
            final_score=50.0,
            attempt=3,
        )
        assert result.attempt == 3


# ---------------------------------------------------------------------------
# Clone workspace
# ---------------------------------------------------------------------------


class TestCloneWorkspace:
    def test_creates_isolated_workspace(self, sample_task_instance, tmp_dir):
        # Ensure only agent-visible files are copied into retry workspaces
        (sample_task_instance.workspace_dir / "task_info.json").write_text('{"task_id": "test"}')
        (sample_task_instance.reference_dir.parent / "framework_task_info.json").write_text(
            '{"task_id": "test", "instance_id": "test__seed0"}'
        )

        agent = MagicMock(spec=AgentAdapter)
        config = RunConfig(
            agent=agent,
            tasks=["physics.test_task"],
            output_dir=str(tmp_dir / "results"),
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
        )
        orch = BenchmarkOrchestrator(config)

        clone = orch._clone_workspace(sample_task_instance, 2)

        assert clone.workspace_dir != sample_task_instance.workspace_dir
        assert "attempt2" in str(clone.workspace_dir)
        assert (clone.workspace_dir / "prompt.md").exists()
        assert (clone.workspace_dir / "task_info.json").exists()
        assert not (clone.workspace_dir / "framework_task_info.json").exists()

    def test_cloned_workspace_has_data(self, sample_task_instance, tmp_dir):
        # Create data in original workspace
        data_dir = sample_task_instance.workspace_dir / "data"
        data_dir.mkdir(exist_ok=True)
        (data_dir / "input.txt").write_text("test data")

        agent = MagicMock(spec=AgentAdapter)
        config = RunConfig(
            agent=agent,
            tasks=["physics.test_task"],
            output_dir=str(tmp_dir / "results"),
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
        )
        orch = BenchmarkOrchestrator(config)

        clone = orch._clone_workspace(sample_task_instance, 2)

        assert (clone.workspace_dir / "data" / "input.txt").exists()
        assert (clone.workspace_dir / "data" / "input.txt").read_text() == "test data"

    def test_cloned_workspace_shares_reference(self, sample_task_instance, tmp_dir):
        agent = MagicMock(spec=AgentAdapter)
        config = RunConfig(
            agent=agent,
            tasks=["physics.test_task"],
            output_dir=str(tmp_dir / "results"),
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
        )
        orch = BenchmarkOrchestrator(config)

        clone = orch._clone_workspace(sample_task_instance, 2)

        assert clone.reference_dir == sample_task_instance.reference_dir

    def test_cloned_workspace_preserves_metadata(self, sample_task_instance, tmp_dir):
        agent = MagicMock(spec=AgentAdapter)
        config = RunConfig(
            agent=agent,
            tasks=["physics.test_task"],
            output_dir=str(tmp_dir / "results"),
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
        )
        orch = BenchmarkOrchestrator(config)

        clone = orch._clone_workspace(sample_task_instance, 3)

        assert clone.task_id == sample_task_instance.task_id
        assert clone.instance_id == sample_task_instance.instance_id
        assert clone.prompt_level == sample_task_instance.prompt_level
        assert clone.parameters == sample_task_instance.parameters

    def test_agent_output_in_original_not_copied(self, sample_task_instance, tmp_dir):
        # Simulate agent writing output to original workspace
        (sample_task_instance.workspace_dir / "simulation.py").write_text("print('hi')")
        (sample_task_instance.workspace_dir / "output.npy").write_bytes(b"data")

        agent = MagicMock(spec=AgentAdapter)
        config = RunConfig(
            agent=agent,
            tasks=["physics.test_task"],
            output_dir=str(tmp_dir / "results"),
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
        )
        orch = BenchmarkOrchestrator(config)

        clone = orch._clone_workspace(sample_task_instance, 2)

        # Agent output files should NOT be in clone
        assert not (clone.workspace_dir / "simulation.py").exists()
        assert not (clone.workspace_dir / "output.npy").exists()


# ---------------------------------------------------------------------------
# Retry: strategy="all"
# ---------------------------------------------------------------------------


class TestRetryStrategyAll:
    def _make_orchestrator(self, agent, tmp_dir, sample_task_dir, retries=3):
        config = RunConfig(
            agent=agent,
            tasks=["physics.test_task"],
            retries=retries,
            retry_strategy="all",
            sandbox="os",
            output_dir=str(tmp_dir / "results"),
            tasks_dir=str(sample_task_dir.parent.parent),
        )
        return BenchmarkOrchestrator(config)

    def test_runs_n_times_even_if_all_succeed(self, sample_task_instance, tmp_dir):
        output_data = {"output.npy": np.zeros(50, dtype=np.float32)}
        agent = AlwaysSucceedAgent(output_data=output_data)
        orch = self._make_orchestrator(agent, tmp_dir, sample_task_instance.task_dir, retries=3)

        result = orch._run_single_instance(sample_task_instance)

        assert agent.call_count == 3
        assert result.attempt == 1 or result.attempt == 2 or result.attempt == 3

    def test_runs_n_times_even_if_all_fail(self, sample_task_instance, tmp_dir):
        agent = AlwaysFailAgent()
        orch = self._make_orchestrator(agent, tmp_dir, sample_task_instance.task_dir, retries=3)

        result = orch._run_single_instance(sample_task_instance)

        assert agent.call_count == 3
        assert result.status == RunStatus.FAILED

    def test_returns_best_result(self, sample_task_instance, tmp_dir):
        """When some attempts fail and some succeed, return the one with highest score."""
        # Use the reference data as output so the scorer gives a nonzero score
        ref_data = np.load(sample_task_instance.reference_dir / "output_ref.npy")
        agent = SucceedOnNthAgent(succeed_on=2, output_data={"output.npy": ref_data})
        orch = self._make_orchestrator(agent, tmp_dir, sample_task_instance.task_dir, retries=3)

        result = orch._run_single_instance(sample_task_instance)

        assert agent.call_count == 3
        # Best result should be from a successful attempt (score > 0)
        assert result.final_score > 0
        assert result.status == RunStatus.COMPLETED

    def test_saves_all_attempts(self, sample_task_instance, tmp_dir):
        output_data = {"output.npy": np.zeros(50, dtype=np.float32)}
        agent = AlwaysSucceedAgent(output_data=output_data)
        orch = self._make_orchestrator(agent, tmp_dir, sample_task_instance.task_dir, retries=3)

        orch._run_single_instance(sample_task_instance)

        result_dir = tmp_dir / "results" / "physics.test_task"
        result_files = list(result_dir.glob("*.json"))
        assert len(result_files) == 3

    def test_retry_retains_attempt_workspaces(self, sample_task_instance, tmp_dir):
        output_data = {"output.npy": np.zeros(50, dtype=np.float32)}
        sample_task_instance.retained_workspace_dir = (
            tmp_dir
            / "results"
            / "instances"
            / "_workspaces"
            / sample_task_instance.instance_id
            / "workspace_b2"
        )

        agent = AlwaysSucceedAgent(output_data=output_data)
        orch = self._make_orchestrator(agent, tmp_dir, sample_task_instance.task_dir, retries=2)

        orch._run_single_instance(sample_task_instance)

        retained_root = sample_task_instance.retained_workspace_dir.parent
        assert (retained_root / "workspace_b2").exists()
        assert (retained_root / "workspace_b2" / "output.npy").exists()
        assert (retained_root / "workspace_b2_attempt2").exists()
        assert (retained_root / "workspace_b2_attempt2" / "output.npy").exists()

    def test_attempt_2_and_3_have_suffix(self, sample_task_instance, tmp_dir):
        output_data = {"output.npy": np.zeros(50, dtype=np.float32)}
        agent = AlwaysSucceedAgent(output_data=output_data)
        orch = self._make_orchestrator(agent, tmp_dir, sample_task_instance.task_dir, retries=3)

        orch._run_single_instance(sample_task_instance)

        result_dir = tmp_dir / "results" / "physics.test_task"
        filenames = sorted(f.name for f in result_dir.glob("*.json"))
        # One file without suffix (attempt 1), two with __attempt2 and __attempt3
        assert any("__attempt2" in f for f in filenames)
        assert any("__attempt3" in f for f in filenames)
        assert any("__attempt" not in f for f in filenames)

    def test_each_attempt_uses_different_workspace(self, sample_task_instance, tmp_dir):
        agent = AlwaysSucceedAgent()
        orch = self._make_orchestrator(agent, tmp_dir, sample_task_instance.task_dir, retries=3)

        orch._run_single_instance(sample_task_instance)

        workspaces = agent.workspaces_seen
        assert len(workspaces) == 3
        assert len(set(workspaces)) == 3  # All different

    def test_retries_1_means_no_retry(self, sample_task_instance, tmp_dir):
        agent = AlwaysFailAgent()
        orch = self._make_orchestrator(agent, tmp_dir, sample_task_instance.task_dir, retries=1)

        result = orch._run_single_instance(sample_task_instance)

        assert agent.call_count == 1
        assert result.attempt == 1

    def test_saved_json_contains_attempt_field(self, sample_task_instance, tmp_dir):
        output_data = {"output.npy": np.zeros(50, dtype=np.float32)}
        agent = AlwaysSucceedAgent(output_data=output_data)
        orch = self._make_orchestrator(agent, tmp_dir, sample_task_instance.task_dir, retries=2)

        orch._run_single_instance(sample_task_instance)

        result_dir = tmp_dir / "results" / "physics.test_task"
        for json_file in result_dir.glob("*.json"):
            data = json.loads(json_file.read_text())
            assert "attempt" in data
            assert data["attempt"] in (1, 2)


# ---------------------------------------------------------------------------
# Retry: strategy="until_success"
# ---------------------------------------------------------------------------


class TestRetryStrategyUntilSuccess:
    def _make_orchestrator(self, agent, tmp_dir, sample_task_dir, retries=5):
        config = RunConfig(
            agent=agent,
            tasks=["physics.test_task"],
            retries=retries,
            retry_strategy="until_success",
            output_dir=str(tmp_dir / "results"),
            tasks_dir=str(sample_task_dir.parent.parent),
        )
        return BenchmarkOrchestrator(config)

    def test_stops_after_first_success(self, sample_task_instance, tmp_dir):
        output_data = {"output.npy": np.zeros(50, dtype=np.float32)}
        agent = SucceedOnNthAgent(succeed_on=2, output_data=output_data)
        orch = self._make_orchestrator(agent, tmp_dir, sample_task_instance.task_dir, retries=5)

        result = orch._run_single_instance(sample_task_instance)

        # Should stop at attempt 2 (first success), not run all 5
        assert agent.call_count == 2
        assert result.status == RunStatus.COMPLETED

    def test_runs_all_if_never_succeeds(self, sample_task_instance, tmp_dir):
        agent = AlwaysFailAgent()
        orch = self._make_orchestrator(agent, tmp_dir, sample_task_instance.task_dir, retries=3)

        result = orch._run_single_instance(sample_task_instance)

        assert agent.call_count == 3
        assert result.status == RunStatus.FAILED

    def test_saves_only_run_attempts(self, sample_task_instance, tmp_dir):
        output_data = {"output.npy": np.zeros(50, dtype=np.float32)}
        agent = SucceedOnNthAgent(succeed_on=2, output_data=output_data)
        orch = self._make_orchestrator(agent, tmp_dir, sample_task_instance.task_dir, retries=5)

        orch._run_single_instance(sample_task_instance)

        result_dir = tmp_dir / "results" / "physics.test_task"
        result_files = list(result_dir.glob("*.json"))
        # Should only save 2 files (attempt 1 + attempt 2), not 5
        assert len(result_files) == 2

    def test_first_attempt_success_runs_once(self, sample_task_instance, tmp_dir):
        output_data = {"output.npy": np.zeros(50, dtype=np.float32)}
        agent = AlwaysSucceedAgent(output_data=output_data)
        orch = self._make_orchestrator(agent, tmp_dir, sample_task_instance.task_dir, retries=5)

        result = orch._run_single_instance(sample_task_instance)

        assert agent.call_count == 1
        assert result.attempt == 1

    def test_completed_attempt_beats_higher_scoring_failed_attempt(self, sample_task_instance, tmp_dir):
        agent = AlwaysFailAgent()
        orch = self._make_orchestrator(agent, tmp_dir, sample_task_instance.task_dir, retries=2)
        orch.config.retry_strategy = "all"

        def make_result(attempt: int, status: RunStatus, score: float) -> EvalResult:
            output = AgentOutput(
                instance_id=sample_task_instance.instance_id,
                output_dir=sample_task_instance.workspace_dir,
                code_files=[],
                data_files=[],
                log="",
                execution_time_seconds=float(attempt),
                status=status,
            )
            return EvalResult(
                instance_id=sample_task_instance.instance_id,
                task_id=sample_task_instance.task_id,
                prompt_level=sample_task_instance.prompt_level,
                agent_name="MockAgent",
                parameters={},
                gate_results=[],
                gates_passed=True,
                hard_gates_passed=True,
                score_results=[],
                final_score=score,
                execution_time_seconds=float(attempt),
                status=status,
                attempt=attempt,
                agent_output=output,
            )

        failed_high = make_result(1, RunStatus.FAILED, 90.0)
        completed_low = make_result(2, RunStatus.COMPLETED, 10.0)

        with patch.object(orch, "_run_and_evaluate", side_effect=[failed_high, completed_low]):
            result = orch._run_single_instance(sample_task_instance)

        assert result.status == RunStatus.COMPLETED
        assert result.final_score == 10.0
        assert result.attempt == 2


# ---------------------------------------------------------------------------
# Workspace isolation
# ---------------------------------------------------------------------------


class TestWorkspaceIsolation:
    def test_agent_output_does_not_leak_between_attempts(self, sample_task_instance, tmp_dir):
        """Files written by attempt 1 should not appear in attempt 2's workspace."""

        class WriteFileAgent(AgentAdapter):
            def __init__(self):
                self.call_count = 0

            def solve(self, task_instance: TaskInstance) -> AgentOutput:
                self.call_count += 1
                marker = task_instance.workspace_dir / f"marker_{self.call_count}.txt"
                marker.write_text(f"attempt {self.call_count}")

                # Check that previous markers don't exist in this workspace
                for prev in range(1, self.call_count):
                    prev_marker = task_instance.workspace_dir / f"marker_{prev}.txt"
                    assert not prev_marker.exists(), (
                        f"marker_{prev}.txt leaked into attempt {self.call_count}"
                    )

                return AgentOutput(
                    instance_id=task_instance.instance_id,
                    output_dir=task_instance.workspace_dir,
                    code_files=[],
                    data_files=[],
                    log=f"attempt {self.call_count}",
                    execution_time_seconds=0.1,
                    status=RunStatus.COMPLETED,
                )

        agent = WriteFileAgent()
        config = RunConfig(
            agent=agent,
            tasks=["physics.test_task"],
            retries=3,
            retry_strategy="all",
            output_dir=str(tmp_dir / "results"),
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
        )
        orch = BenchmarkOrchestrator(config)

        orch._run_single_instance(sample_task_instance)

        assert agent.call_count == 3

    def test_prompt_identical_across_attempts(self, sample_task_instance, tmp_dir):
        """All attempts should see the same prompt.md content."""
        prompts_seen = []

        class RecordPromptAgent(AgentAdapter):
            def solve(self, task_instance: TaskInstance) -> AgentOutput:
                prompt = (task_instance.workspace_dir / "prompt.md").read_text()
                prompts_seen.append(prompt)
                return AgentOutput(
                    instance_id=task_instance.instance_id,
                    output_dir=task_instance.workspace_dir,
                    code_files=[],
                    data_files=[],
                    log="ok",
                    execution_time_seconds=0.1,
                    status=RunStatus.COMPLETED,
                )

        agent = RecordPromptAgent()
        config = RunConfig(
            agent=agent,
            tasks=["physics.test_task"],
            retries=3,
            retry_strategy="all",
            output_dir=str(tmp_dir / "results"),
            tasks_dir=str(sample_task_instance.task_dir.parent.parent),
        )
        orch = BenchmarkOrchestrator(config)

        orch._run_single_instance(sample_task_instance)

        assert len(prompts_seen) == 3
        assert prompts_seen[0] == prompts_seen[1] == prompts_seen[2]
