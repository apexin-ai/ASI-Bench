"""Tests for parallel execution and new runner components."""

import json
import time
import threading
from pathlib import Path

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
from ai4sci_bench.runner.parallel import ParallelRunner
from ai4sci_bench.runner.metadata import (
    build_task_runtime_provenance,
    collect_run_metadata,
    save_run_metadata,
    FRAMEWORK_VERSION,
)

# ── ParallelRunner Tests ──────────────────────────────────────────────


class TestParallelRunner:
    def _make_instance(self, idx, tmp_dir):
        workspace = tmp_dir / f"workspace_{idx}"
        workspace.mkdir(parents=True, exist_ok=True)
        ref_dir = tmp_dir / f"ref_{idx}"
        ref_dir.mkdir(parents=True, exist_ok=True)
        return TaskInstance(
            task_id="test.task",
            instance_id=f"test__inst_{idx}",
            task_dir=tmp_dir,
            workspace_dir=workspace,
            reference_dir=ref_dir,
            prompt_level=PromptLevel.B2,
            parameters={"idx": idx},
            metadata={},
        )

    def _make_eval_result(self, instance):
        return EvalResult(
            instance_id=instance.instance_id,
            task_id=instance.task_id,
            prompt_level=instance.prompt_level,
            agent_name="TestAgent",
            parameters=instance.parameters,
            gate_results=[],
            gates_passed=True,
            score_results=[],
            final_score=80.0,
        )

    def test_sequential_execution(self, tmp_dir):
        """Sequential mode (max_workers=1) runs all instances."""
        instances = [self._make_instance(i, tmp_dir) for i in range(3)]
        runner = ParallelRunner(max_workers=1)

        def run_fn(inst):
            return self._make_eval_result(inst)

        results = runner.run_instances(instances, run_fn)
        assert len(results) == 3

    def test_parallel_execution(self, tmp_dir):
        """Parallel mode runs instances concurrently."""
        instances = [self._make_instance(i, tmp_dir) for i in range(5)]
        runner = ParallelRunner(max_workers=3)

        call_times = []

        def run_fn(inst):
            call_times.append(time.time())
            time.sleep(0.05)  # short delay
            return self._make_eval_result(inst)

        results = runner.run_instances(instances, run_fn)
        assert len(results) == 5

    def test_skip_completed_ids(self, tmp_dir):
        """Already-completed IDs are skipped."""
        instances = [self._make_instance(i, tmp_dir) for i in range(5)]
        runner = ParallelRunner(max_workers=1)

        completed = {"test__inst_0__b2", "test__inst_2__b2", "test__inst_4__b2"}

        def run_fn(inst):
            return self._make_eval_result(inst)

        results = runner.run_instances(instances, run_fn, completed_ids=completed)
        assert len(results) == 2
        ids = {r.instance_id for r in results}
        assert ids == {"test__inst_1", "test__inst_3"}

    def test_empty_instances(self, tmp_dir):
        """No instances returns empty list."""
        runner = ParallelRunner(max_workers=2)
        results = runner.run_instances([], lambda x: None)
        assert results == []

    def test_all_completed(self, tmp_dir):
        """All instances already completed returns empty list."""
        instances = [self._make_instance(0, tmp_dir)]
        runner = ParallelRunner(max_workers=1)
        results = runner.run_instances(
            instances, lambda x: None, completed_ids={"test__inst_0__b2"}
        )
        assert results == []

    def test_parallel_error_handling(self, tmp_dir):
        """Errors in run_fn are caught and logged (instance skipped)."""
        instances = [self._make_instance(i, tmp_dir) for i in range(3)]
        runner = ParallelRunner(max_workers=2)

        def run_fn(inst):
            if inst.parameters["idx"] == 1:
                raise RuntimeError("test error")
            return self._make_eval_result(inst)

        results = runner.run_instances(instances, run_fn)
        # All 3 produce results; instance 1 gets a failed fallback
        assert len(results) == 3
        failed = [r for r in results if r.status == RunStatus.FAILED]
        assert len(failed) == 1
        assert failed[0].final_score == 0.0

    def test_max_workers_clamped_to_one(self):
        """max_workers < 1 is clamped to 1."""
        runner = ParallelRunner(max_workers=0)
        assert runner.max_workers == 1
        runner = ParallelRunner(max_workers=-5)
        assert runner.max_workers == 1


# ── RunMetadata Tests ─────────────────────────────────────────────────


class TestRunMetadata:
    def test_collect_metadata(self):
        """Metadata includes framework version and Python version."""
        meta = collect_run_metadata()
        assert meta["framework_version"] == FRAMEWORK_VERSION
        assert "python_version" in meta
        assert "timestamp" in meta
        assert "dependencies" in meta
        assert "git_commit" in meta

    def test_collect_with_agent_config(self):
        """Agent config is included when provided."""
        meta = collect_run_metadata(
            agent_config={"type": "ClaudeCodeCLI", "model": "opus"},
        )
        assert meta["agent_config"]["type"] == "ClaudeCodeCLI"

    def test_collect_with_task_versions(self):
        """Task versions are included when provided."""
        meta = collect_run_metadata(
            task_versions={"physics.vic_leapfrog": "1.0"},
        )
        assert meta["task_versions"]["physics.vic_leapfrog"] == "1.0"

    def test_collect_with_sandbox_and_runtime_provenance(self):
        meta = collect_run_metadata(
            sandbox_provenance={"requested_mode": "task", "verification_status": "task_env_active"},
            task_runtime={
                "physics.vic_leapfrog": {
                    "python_requirement": ">=3.11",
                    "resolved_python_version": "3.12.11",
                    "python_requirement_satisfied": True,
                    "packages": ["taichi>=1.7.4"],
                    "task_env_cache_key": "abc123",
                }
            },
        )
        assert meta["sandbox"]["requested_mode"] == "task"
        assert meta["task_runtime"]["physics.vic_leapfrog"]["task_env_cache_key"] == "abc123"
        assert meta["task_runtime"]["physics.vic_leapfrog"]["resolved_python_version"] == "3.12.11"
        assert "task_env_cache_dir" not in meta["task_runtime"]["physics.vic_leapfrog"]

    def test_os_runtime_provenance_uses_docker_runtime_not_task_env_cache(self):
        class FakeTaskEnvManager:
            def compute_cache_key(self, _metadata):
                return "host-cache-key"

            def describe_cached_env(self, _metadata):
                return {
                    "resolved_python_version": "3.13.11",
                    "python_requirement_satisfied": True,
                }

        metadata = {
            "_runtime_python": ">=3.11",
            "_runtime_packages": ["numpy>=2.0"],
        }

        provenance = build_task_runtime_provenance(
            metadata,
            sandbox="os",
            task_env_manager=FakeTaskEnvManager(),
            image_identity="sha256:task-image",
        )

        assert provenance["runtime_source"] == "docker_image"
        assert provenance["runtime_interpreter"] == "/opt/venv/bin/python"
        assert provenance["docker_image_identity"] == "sha256:task-image"
        assert provenance["task_env_cache_key"] is None
        assert provenance["resolved_python_version"] is None

    def test_save_metadata(self, tmp_dir):
        """Metadata is saved to run_metadata.json."""
        meta = collect_run_metadata()
        path = save_run_metadata(tmp_dir, meta)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["framework_version"] == FRAMEWORK_VERSION

    def test_dependencies_detected(self):
        """Key dependencies are detected."""
        meta = collect_run_metadata()
        deps = meta["dependencies"]
        assert "numpy" in deps
        assert "click" in deps


# ── CostInfo Tests ────────────────────────────────────────────────────


class TestCostInfo:
    def test_cost_info_default(self):
        """CostInfo has sensible defaults."""
        cost = CostInfo()
        assert cost.input_tokens == 0
        assert cost.output_tokens == 0
        assert cost.total_tokens == 0
        assert cost.estimated_cost_usd == 0.0

    def test_cost_info_values(self):
        """CostInfo stores values correctly."""
        cost = CostInfo(
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            estimated_cost_usd=0.05,
        )
        assert cost.total_tokens == 1500
        assert cost.estimated_cost_usd == 0.05

    def test_agent_output_with_cost(self, tmp_dir):
        """AgentOutput can include cost info."""
        cost = CostInfo(input_tokens=100, output_tokens=50, total_tokens=150, estimated_cost_usd=0.01)
        output = AgentOutput(
            instance_id="test",
            output_dir=tmp_dir,
            code_files=[],
            data_files=[],
            log="",
            execution_time_seconds=1.0,
            status=RunStatus.COMPLETED,
            cost=cost,
        )
        assert output.cost.total_tokens == 150

    def test_eval_result_with_cost(self):
        """EvalResult can include cost info."""
        cost = CostInfo(estimated_cost_usd=0.10)
        result = EvalResult(
            instance_id="test",
            task_id="test.task",
            prompt_level=PromptLevel.B2,
            agent_name="test",
            parameters={},
            gate_results=[],
            gates_passed=True,
            score_results=[],
            final_score=80.0,
            cost=cost,
        )
        assert result.cost.estimated_cost_usd == 0.10
