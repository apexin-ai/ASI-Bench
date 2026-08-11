"""Tests for fault isolation: scorer crashes, bad YAML, custom scorer errors.

Covers issues #31 (scorer exception aborts run) and #32 (corrupted YAML
in unrelated task crashes run).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from ai4sci_bench.core.types import (
    AgentOutput,
    EvalResult,
    PromptLevel,
    RunStatus,
    ScoreDetail,
    TaskInstance,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_instance(
    task_id: str = "test.task",
    instance_id: str = "test.task__s42_p0",
    tmp_path: Path | None = None,
) -> TaskInstance:
    base = tmp_path or Path("/tmp/test_fault_isolation")
    ws = base / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    ref = base / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    return TaskInstance(
        task_id=task_id,
        instance_id=instance_id,
        task_dir=base / "task",
        workspace_dir=ws,
        reference_dir=ref,
        prompt_level=PromptLevel.B1,
        parameters={},
        metadata={"evaluation": {}},
    )


def _make_agent_output(instance: TaskInstance) -> AgentOutput:
    return AgentOutput(
        instance_id=instance.instance_id,
        output_dir=instance.workspace_dir,
        code_files=[],
        data_files=[],
        log="",
        execution_time_seconds=1.0,
        status=RunStatus.COMPLETED,
    )


# ── Issue #31: Scorer exception isolation ────────────────────────────────────


class TestScorerExceptionIsolation:
    """A scorer crash should produce a zero-score result, not abort the run."""

    def test_gate_scorer_crash_produces_zero_score(self, tmp_path):
        """Gate scorer raising ValueError yields score=0, passed=False."""
        from ai4sci_bench.runner.orchestrator import _evaluate_gates_and_scores

        evaluation = {
            "gates": [{"scorer": "crashing_scorer", "severity": "hard", "config": {}}],
            "scoring": [{"scorer": "good_scorer", "weight": 50.0, "config": {}}],
        }

        good_result = ScoreDetail(
            scorer_name="good_scorer",
            score=50.0, max_score=50.0, passed=True, details={},
        )

        def mock_score(pred, ref, cfg):
            raise ValueError("shape mismatch (256,) vs (2,2)")

        mock_crashing = MagicMock()
        mock_crashing.score = mock_score

        mock_good = MagicMock()
        mock_good.score = MagicMock(return_value=good_result)

        def get_scorer_side_effect(name):
            if name == "crashing_scorer":
                return mock_crashing
            return mock_good

        with patch("ai4sci_bench.runner.orchestrator.get_scorer", side_effect=get_scorer_side_effect):
            gate_results, hard_passed, soft_fails, score_results, final_score = (
                _evaluate_gates_and_scores(evaluation, tmp_path, tmp_path, {})
            )

        assert len(gate_results) == 1
        assert gate_results[0].passed is False
        assert gate_results[0].score == 0.0
        assert gate_results[0].details.get("scorer_internal_error") is True
        assert "ValueError" in gate_results[0].message
        assert hard_passed is False
        # Scoring should not run because hard gate failed
        assert len(score_results) == 0
        assert final_score == 0.0

    def test_scoring_scorer_crash_produces_zero_score(self, tmp_path):
        """Scoring scorer crash yields score=0 for that scorer, others proceed."""
        from ai4sci_bench.runner.orchestrator import _evaluate_gates_and_scores

        evaluation = {
            "gates": [],
            "scoring": [
                {"scorer": "ok_scorer", "weight": 50.0, "config": {}},
                {"scorer": "bad_scorer", "weight": 30.0, "config": {}},
                {"scorer": "ok_scorer2", "weight": 20.0, "config": {}},
            ],
        }

        ok1 = ScoreDetail(scorer_name="ok_scorer", score=50.0, max_score=50.0, passed=True, details={})
        ok2 = ScoreDetail(scorer_name="ok_scorer2", score=20.0, max_score=20.0, passed=True, details={})

        call_idx = [0]
        ok_results = [ok1, ok2]

        def get_scorer_side_effect(name):
            mock = MagicMock()
            if name == "bad_scorer":
                mock.score = MagicMock(side_effect=ZeroDivisionError("divide by zero"))
            else:
                result = ok_results[call_idx[0]]
                call_idx[0] += 1
                mock.score = MagicMock(return_value=result)
            return mock

        with patch("ai4sci_bench.runner.orchestrator.get_scorer", side_effect=get_scorer_side_effect):
            gate_results, hard_passed, soft_fails, score_results, final_score = (
                _evaluate_gates_and_scores(evaluation, tmp_path, tmp_path, {})
            )

        assert hard_passed is True
        assert len(score_results) == 3
        # First scorer OK
        assert score_results[0].score == 50.0
        # Second scorer crashed → 0
        assert score_results[1].score == 0.0
        assert score_results[1].details.get("scorer_internal_error") is True
        assert "ZeroDivisionError" in score_results[1].message
        # Third scorer still ran
        assert score_results[2].score == 20.0
        # Total = 50 + 0 + 20 = 70
        assert final_score == 70.0

    def test_soft_gate_crash_does_not_block_scoring(self, tmp_path):
        """A soft gate crash is treated as soft failure, scoring still runs."""
        from ai4sci_bench.runner.orchestrator import _evaluate_gates_and_scores

        evaluation = {
            "gates": [{"scorer": "crashing_gate", "severity": "soft", "config": {}}],
            "scoring": [{"scorer": "good_scorer", "weight": 100.0, "config": {}}],
        }

        good_result = ScoreDetail(
            scorer_name="good_scorer", score=80.0, max_score=100.0, passed=True, details={},
        )

        def get_scorer_side_effect(name):
            mock = MagicMock()
            if name == "crashing_gate":
                mock.score = MagicMock(side_effect=RuntimeError("boom"))
            else:
                mock.score = MagicMock(return_value=good_result)
            return mock

        with patch("ai4sci_bench.runner.orchestrator.get_scorer", side_effect=get_scorer_side_effect):
            gate_results, hard_passed, soft_fails, score_results, final_score = (
                _evaluate_gates_and_scores(evaluation, tmp_path, tmp_path, {})
            )

        assert hard_passed is True
        assert soft_fails == 1
        assert len(score_results) == 1
        assert final_score == 80.0


class TestCustomScorerLoadFailure:
    """Custom scorer load failure should not crash evaluation."""

    def test_bad_custom_scorer_returns_zero_eval(self, tmp_path):
        """If load_custom_scorer raises, _evaluate returns a zero-score result."""
        from ai4sci_bench.runner.orchestrator import BenchmarkOrchestrator

        instance = _make_instance(tmp_path=tmp_path)
        agent_output = _make_agent_output(instance)

        orchestrator = MagicMock(spec=BenchmarkOrchestrator)
        orchestrator.agent = MagicMock()
        orchestrator.agent.__class__.__name__ = "TestAgent"

        with patch(
            "ai4sci_bench.scorers.custom.load_custom_scorer",
            side_effect=ImportError("missing numpy"),
        ):
            result = BenchmarkOrchestrator._evaluate(orchestrator, instance, agent_output)

        assert result.final_score == 0.0
        assert result.gates_passed is False
        assert len(result.score_results) == 1
        assert result.score_results[0].details.get("scorer_internal_error") is True
        assert "ImportError" in result.score_results[0].message


class TestEvaluationCrashIsolation:
    """Analysis failures must not abort produce-only execution."""

    def test_analyzer_crash_does_not_abort(self, tmp_path):
        """If error analyzer crashes, the eval result is still returned."""
        from ai4sci_bench.runner.orchestrator import BenchmarkOrchestrator

        instance = _make_instance(tmp_path=tmp_path)
        agent_output = _make_agent_output(instance)
        eval_result = EvalResult(
            instance_id=instance.instance_id,
            task_id=instance.task_id,
            prompt_level=PromptLevel.B1,
            agent_name="TestAgent",
            parameters={},
            gate_results=[],
            gates_passed=True,
            score_results=[],
            final_score=50.0,
            max_possible_score=100.0,
            execution_time_seconds=1.0,
            status=RunStatus.COMPLETED,
            agent_output=agent_output,
        )

        orchestrator = MagicMock(spec=BenchmarkOrchestrator)
        orchestrator.agent = MagicMock()
        orchestrator.agent.__class__.__name__ = "TestAgent"
        orchestrator._run_agent = MagicMock(return_value=agent_output)
        orchestrator._evaluate = MagicMock(return_value=eval_result)
        orchestrator.analyzer = MagicMock()
        orchestrator.analyzer.enabled = True
        orchestrator.config = MagicMock()
        orchestrator.config.analyze_only_failed = False
        orchestrator._load_ref_specs = MagicMock(return_value=None)
        orchestrator.analyzer.analyze = MagicMock(
            side_effect=RuntimeError("analyzer broke"),
        )

        result = BenchmarkOrchestrator._run_and_evaluate(orchestrator, instance, attempt=1)
        assert result.final_score == 50.0
        assert result.error_analysis is None


# ── Issue #32: Bad YAML isolation ────────────────────────────────────────────


class TestBadYAMLIsolation:
    """A corrupted task.yaml in an unrelated task must not crash operations."""

    def _create_tasks_dir(self, tmp_path: Path) -> Path:
        """Create a tasks dir with a good task and a bad task."""
        tasks_dir = tmp_path / "tasks"

        # Good task
        good_dir = tasks_dir / "physics" / "good_task"
        good_dir.mkdir(parents=True)
        (good_dir / "task.yaml").write_text(yaml.dump({
            "id": "physics.good_task",
            "name": "Good Task",
            "status": "in_development",
        }))

        # Bad task (corrupted YAML)
        bad_dir = tasks_dir / "math" / "bad_task"
        bad_dir.mkdir(parents=True)
        (bad_dir / "task.yaml").write_text(
            "id: math.bad_task\n  broken: {invalid yaml\n  unclosed: [bracket"
        )

        return tasks_dir

    def test_discover_tasks_skips_bad_yaml(self, tmp_path):
        """discover_tasks skips corrupted YAML files and returns valid tasks."""
        from ai4sci_bench.core.task import TaskLoader

        tasks_dir = self._create_tasks_dir(tmp_path)
        loader = TaskLoader(tasks_dir)
        tasks = loader.discover_tasks(include_dev=True)

        assert len(tasks) == 1
        assert tasks[0]["id"] == "physics.good_task"

    def test_load_task_by_id_skips_bad_yaml(self, tmp_path):
        """load_task_by_id succeeds even when unrelated task.yaml is corrupt."""
        from ai4sci_bench.core.task import TaskLoader

        tasks_dir = self._create_tasks_dir(tmp_path)
        loader = TaskLoader(tasks_dir)
        metadata = loader.load_task_by_id("physics.good_task")
        assert metadata["id"] == "physics.good_task"

    def test_load_task_by_id_direct_path_optimization(self, tmp_path):
        """load_task_by_id uses direct path lookup, avoiding full scan."""
        from ai4sci_bench.core.task import TaskLoader

        tasks_dir = self._create_tasks_dir(tmp_path)
        loader = TaskLoader(tasks_dir)

        # The direct path should find it without scanning bad_task
        with patch.object(loader, "load_task_metadata", wraps=loader.load_task_metadata) as mock_load:
            metadata = loader.load_task_by_id("physics.good_task")
            assert metadata["id"] == "physics.good_task"
            # Should have been called exactly once (direct path hit)
            assert mock_load.call_count == 1

    def test_load_task_by_id_not_found_despite_bad_yaml(self, tmp_path):
        """load_task_by_id raises ValueError when task not found (bad YAML skipped)."""
        from ai4sci_bench.core.task import TaskLoader

        tasks_dir = self._create_tasks_dir(tmp_path)
        loader = TaskLoader(tasks_dir)
        with pytest.raises(ValueError, match="not found"):
            loader.load_task_by_id("physics.nonexistent")

    def test_provenance_survives_bad_yaml(self, tmp_path):
        """_build_result_provenance degrades gracefully on YAML errors."""
        from ai4sci_bench.runner.orchestrator import BenchmarkOrchestrator

        tasks_dir = self._create_tasks_dir(tmp_path)
        # Make load_task_by_id raise yaml.YAMLError
        eval_result = MagicMock()
        eval_result.task_id = "math.bad_task"

        orchestrator = MagicMock(spec=BenchmarkOrchestrator)
        from ai4sci_bench.core.task import TaskLoader
        orchestrator.task_loader = TaskLoader(tasks_dir)
        orchestrator.config = MagicMock()
        orchestrator.config.sandbox = "none"
        orchestrator.agent = MagicMock()
        orchestrator.agent.sandbox_image_identity = None
        orchestrator.instance_generator = MagicMock()
        orchestrator._build_agent_provenance = MagicMock(return_value={"agent": "test"})

        provenance = BenchmarkOrchestrator._build_result_provenance(orchestrator, eval_result)
        assert "agent" in provenance
        assert "runtime" not in provenance


# ── Sequential runner fault isolation ────────────────────────────────────────


class TestSequentialRunnerFaultIsolation:
    """Sequential runner should not abort on individual instance failures."""

    def test_sequential_continues_after_crash(self, tmp_path):
        """Sequential run continues when one instance crashes."""
        from ai4sci_bench.runner.parallel import ParallelRunner

        instances = [
            TaskInstance(
                task_id="t",
                instance_id=f"inst_{i}",
                task_dir=tmp_path,
                workspace_dir=tmp_path,
                reference_dir=tmp_path,
                prompt_level=PromptLevel.B1,
                parameters={"idx": i},
                metadata={},
            )
            for i in range(3)
        ]

        def run_fn(inst):
            if inst.parameters["idx"] == 1:
                raise RuntimeError("sequential crash")
            return EvalResult(
                instance_id=inst.instance_id,
                task_id="t",
                prompt_level=PromptLevel.B1,
                agent_name="test",
                parameters=inst.parameters,
                gate_results=[],
                gates_passed=True,
                score_results=[],
                final_score=100.0,
                status=RunStatus.COMPLETED,
            )

        runner = ParallelRunner(max_workers=1)
        results = runner.run_instances(instances, run_fn)

        assert len(results) == 3
        ok_results = [r for r in results if r.status == RunStatus.COMPLETED]
        failed_results = [r for r in results if r.status == RunStatus.FAILED]
        assert len(ok_results) == 2
        assert len(failed_results) == 1
        assert failed_results[0].instance_id == "inst_1"
        assert failed_results[0].final_score == 0.0
