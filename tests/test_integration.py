"""End-to-end integration tests: generate → run → evaluate complete chain."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from ai4sci_bench.core.agent_interface import AgentAdapter
from ai4sci_bench.core.types import (
    AgentOutput,
    CostInfo,
    EvalResult,
    PromptLevel,
    RunStatus,
    TaskInstance,
)
from ai4sci_bench.runner.orchestrator import BenchmarkOrchestrator, RunConfig


class PerfectAgent(AgentAdapter):
    """Agent that copies reference files as predictions (always scores 100)."""

    def solve(self, task_instance: TaskInstance) -> AgentOutput:
        import shutil
        workspace = task_instance.workspace_dir
        ref_dir = task_instance.reference_dir

        code_files = []
        data_files = []

        # Copy reference files to workspace (stripping _ref suffix)
        if ref_dir.exists():
            for ref_file in ref_dir.iterdir():
                if ref_file.is_file():
                    name = ref_file.name
                    # Reference files have _ref suffix; agent output doesn't
                    pred_name = name.replace("_ref", "")
                    shutil.copy2(ref_file, workspace / pred_name)
                    if pred_name.endswith(".py"):
                        code_files.append(pred_name)
                    else:
                        data_files.append(pred_name)

        return AgentOutput(
            instance_id=task_instance.instance_id,
            output_dir=workspace,
            code_files=code_files,
            data_files=data_files,
            log="perfect agent: copied reference files",
            execution_time_seconds=0.01,
            status=RunStatus.COMPLETED,
        )


class ZeroAgent(AgentAdapter):
    """Agent that produces no output (always scores 0)."""

    def solve(self, task_instance: TaskInstance) -> AgentOutput:
        return AgentOutput(
            instance_id=task_instance.instance_id,
            output_dir=task_instance.workspace_dir,
            code_files=[],
            data_files=[],
            log="zero agent: no output",
            execution_time_seconds=0.01,
            status=RunStatus.COMPLETED,
        )


class ImageIdentityAgent(PerfectAgent):
    """Perfect agent that also reports an os-sandbox image identity."""

    def solve(self, task_instance: TaskInstance) -> AgentOutput:
        self.sandbox_image_identity = "sha256:test-os-image"
        return super().solve(task_instance)


class CostReportingAgent(ZeroAgent):
    def solve(self, task_instance: TaskInstance) -> AgentOutput:
        output = super().solve(task_instance)
        output.cost = CostInfo(input_tokens=11, output_tokens=7, total_tokens=18)
        return output


class TestEndToEnd:
    """End-to-end integration tests using the test task from conftest."""

    def test_full_pipeline_perfect_agent(self, sample_task_dir, tmp_dir):
        """PerfectAgent should score 100 on the test task."""
        agent = PerfectAgent()
        config = RunConfig(
            agent=agent,
            tasks=["physics.test_task"],
            tasks_dir=str(sample_task_dir.parent.parent),  # tasks/
            output_dir=str(tmp_dir / "results"),
            instances_per_task=1,
            seed=42,
            include_test=True,
            prompt_levels=["b2"],
        )

        orchestrator = BenchmarkOrchestrator(config)
        report = orchestrator.run()

        assert report.n_instances >= 1
        assert report.overall_mean_score == 100.0

    def test_full_pipeline_zero_agent(self, sample_task_dir, tmp_dir):
        """ZeroAgent should score 0 on the test task (gate failure)."""
        agent = ZeroAgent()
        config = RunConfig(
            agent=agent,
            tasks=["physics.test_task"],
            tasks_dir=str(sample_task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
            instances_per_task=1,
            seed=42,
            include_test=True,
            prompt_levels=["b2"],
        )

        orchestrator = BenchmarkOrchestrator(config)
        report = orchestrator.run()

        assert report.n_instances >= 1
        assert report.overall_mean_score == 0.0

    def test_produce_only_result_persists_agent_cost(self, sample_task_dir, tmp_dir):
        results_dir = tmp_dir / "results"
        config = RunConfig(
            agent=CostReportingAgent(),
            tasks=["physics.test_task"],
            tasks_dir=str(sample_task_dir.parent.parent),
            output_dir=str(results_dir),
            instances_per_task=1,
            seed=42,
            include_test=True,
            prompt_levels=["b2"],
            score=False,
        )

        BenchmarkOrchestrator(config).run()

        result_file = next(results_dir.rglob("*__b2.json"))
        result = json.loads(result_file.read_text())
        assert result["cost"] == {
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
            "estimated_cost_usd": 0.0,
        }

    def test_results_saved_to_disk(self, sample_task_dir, tmp_dir):
        """Results are persisted as JSON files."""
        agent = PerfectAgent()
        results_dir = tmp_dir / "results"
        config = RunConfig(
            agent=agent,
            tasks=["physics.test_task"],
            tasks_dir=str(sample_task_dir.parent.parent),
            output_dir=str(results_dir),
            instances_per_task=1,
            seed=42,
            include_test=True,
        )

        orchestrator = BenchmarkOrchestrator(config)
        orchestrator.run()

        # Check result JSON exists
        json_files = list(results_dir.rglob("*.json"))
        assert len(json_files) >= 1

        # At least one should be a valid eval result
        found_result = False
        for jf in json_files:
            data = json.loads(jf.read_text())
            if "instance_id" in data and "final_score" in data:
                found_result = True
                break
        assert found_result

    def test_run_metadata_saved(self, sample_task_dir, tmp_dir):
        """run_metadata.json is created in output dir."""
        agent = PerfectAgent()
        results_dir = tmp_dir / "results"
        config = RunConfig(
            agent=agent,
            tasks=["physics.test_task"],
            tasks_dir=str(sample_task_dir.parent.parent),
            output_dir=str(results_dir),
            instances_per_task=1,
            seed=42,
            include_test=True,
        )

        orchestrator = BenchmarkOrchestrator(config)
        with patch(
            "ai4sci_bench.runner.task_env.TaskEnvironmentManager.ensure_env",
            return_value=None,
        ):
            orchestrator.run()

        metadata_path = results_dir / "run_metadata.json"
        assert metadata_path.exists()
        data = json.loads(metadata_path.read_text())
        assert "framework_version" in data
        assert "timestamp" in data

    def test_run_metadata_refreshes_image_identity_after_run(self, sample_task_dir, tmp_dir):
        agent = ImageIdentityAgent()
        results_dir = tmp_dir / "results"
        config = RunConfig(
            agent=agent,
            tasks=["physics.test_task"],
            tasks_dir=str(sample_task_dir.parent.parent),
            output_dir=str(results_dir),
            instances_per_task=1,
            seed=42,
            include_test=True,
            sandbox="os",
        )

        orchestrator = BenchmarkOrchestrator(config)
        with patch(
            "ai4sci_bench.runner.task_env.TaskEnvironmentManager.ensure_env",
            return_value=None,
        ):
            orchestrator.run()

        data = json.loads((results_dir / "run_metadata.json").read_text())
        assert data["sandbox"]["requested_mode"] == "os"
        assert data["sandbox"]["image_identity"] == "sha256:test-os-image"

    def test_multiple_prompt_levels(self, sample_task_dir, tmp_dir):
        """Running with multiple prompt levels generates instances for each."""
        agent = PerfectAgent()
        config = RunConfig(
            agent=agent,
            tasks=["physics.test_task"],
            tasks_dir=str(sample_task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
            instances_per_task=1,
            seed=42,
            include_test=True,
            prompt_levels=["b1", "b2", "b3"],
        )

        orchestrator = BenchmarkOrchestrator(config)
        report = orchestrator.run()

        # Should have instances for each prompt level
        assert report.n_instances >= 3
        assert len(report.by_prompt_level) == 3

    def test_parallel_execution(self, sample_task_dir, tmp_dir):
        """Parallel execution produces same results as sequential."""
        agent = PerfectAgent()
        config = RunConfig(
            agent=agent,
            tasks=["physics.test_task"],
            tasks_dir=str(sample_task_dir.parent.parent),
            output_dir=str(tmp_dir / "results"),
            instances_per_task=3,
            prompt_levels=["b1"],
            seed=42,
            include_test=True,
            parallel=2,
        )

        orchestrator = BenchmarkOrchestrator(config)
        report = orchestrator.run()

        assert report.n_instances == 3
        assert report.overall_mean_score == 100.0

    def test_resume_skips_completed(self, sample_task_dir, tmp_dir):
        """Resume mode skips already-completed instances."""
        agent = PerfectAgent()
        results_dir = tmp_dir / "results"
        config = RunConfig(
            agent=agent,
            tasks=["physics.test_task"],
            tasks_dir=str(sample_task_dir.parent.parent),
            output_dir=str(results_dir),
            instances_per_task=2,
            prompt_levels=["b1"],
            seed=42,
            include_test=True,
        )

        # First run
        orchestrator = BenchmarkOrchestrator(config)
        report1 = orchestrator.run()
        assert report1.n_instances == 2

        # Resume run (should find completed instances and produce empty new results)
        config2 = RunConfig(
            agent=agent,
            tasks=["physics.test_task"],
            tasks_dir=str(sample_task_dir.parent.parent),
            output_dir=str(results_dir),
            instances_per_task=2,
            prompt_levels=["b1"],
            seed=42,
            include_test=True,
            resume=str(results_dir),
        )
        orchestrator2 = BenchmarkOrchestrator(config2)
        report2 = orchestrator2.run()

        # All instances were already completed, so no new results
        assert report2.n_instances == 0
