"""Tests for core type definitions."""

from ai4sci_bench.core.types import (
    AgentOutput,
    AnalysisReport,
    EvalResult,
    PromptLevel,
    RunStatus,
    ScoreDetail,
    TaskInstance,
    TaskLifecycle,
)
from pathlib import Path


class TestPromptLevel:
    def test_values(self):
        assert PromptLevel.B1.value == "b1"
        assert PromptLevel.B2.value == "b2"
        assert PromptLevel.B3.value == "b3"

    def test_from_string(self):
        assert PromptLevel("b1") == PromptLevel.B1
        assert PromptLevel("b2") == PromptLevel.B2
        assert PromptLevel("b3") == PromptLevel.B3


class TestTaskLifecycle:
    def test_values(self):
        assert TaskLifecycle.IN_DEVELOPMENT.value == "in_development"
        assert TaskLifecycle.TEST.value == "test"
        assert TaskLifecycle.SAMPLE.value == "sample"
        assert TaskLifecycle.FINAL.value == "final"
        assert TaskLifecycle.ABANDONED.value == "abandoned"

    def test_from_string(self):
        assert TaskLifecycle("abandoned") == TaskLifecycle.ABANDONED
        assert TaskLifecycle("sample") == TaskLifecycle.SAMPLE


class TestRunStatus:
    def test_values(self):
        assert RunStatus.PENDING.value == "pending"
        assert RunStatus.RUNNING.value == "running"
        assert RunStatus.COMPLETED.value == "completed"
        assert RunStatus.FAILED.value == "failed"
        assert RunStatus.TIMEOUT.value == "timeout"


class TestScoreDetail:
    def test_creation(self):
        sd = ScoreDetail(
            scorer_name="test_scorer",
            score=85.0,
            max_score=100.0,
            passed=True,
            details={"metric": "l2"},
        )
        assert sd.scorer_name == "test_scorer"
        assert sd.score == 85.0
        assert sd.passed is True
        assert sd.message == ""

    def test_with_message(self):
        sd = ScoreDetail(
            scorer_name="test",
            score=0.0,
            max_score=1.0,
            passed=False,
            details={},
            message="Failed check",
        )
        assert sd.message == "Failed check"


class TestEvalResult:
    def test_creation(self):
        er = EvalResult(
            instance_id="test__id",
            task_id="test.task",
            prompt_level=PromptLevel.B2,
            agent_name="test_agent",
            parameters={"size": 10},
            gate_results=[],
            gates_passed=True,
            score_results=[],
            final_score=75.0,
        )
        assert er.final_score == 75.0
        assert er.gates_passed is True
        assert er.max_possible_score == 100.0
        assert er.status == RunStatus.COMPLETED
        assert er.error_analysis is None

    def test_with_error_analysis(self):
        report = AnalysisReport(
            instance_id="test",
            error_category="algorithm_error",
            error_subcategory="wrong_formula",
            root_cause="test",
            evidence=[],
            fix_suggestions=[],
            raw_analysis="test",
            confidence=0.9,
        )
        er = EvalResult(
            instance_id="test",
            task_id="test.task",
            prompt_level=PromptLevel.B1,
            agent_name="agent",
            parameters={},
            gate_results=[],
            gates_passed=False,
            score_results=[],
            final_score=0.0,
            error_analysis=report,
        )
        assert er.error_analysis.error_category == "algorithm_error"


class TestAgentOutput:
    def test_creation(self):
        ao = AgentOutput(
            instance_id="test",
            output_dir=Path("/tmp/test"),
            code_files=["sim.py"],
            data_files=["output.npy"],
            log="test log",
            execution_time_seconds=10.5,
            status=RunStatus.COMPLETED,
        )
        assert ao.status == RunStatus.COMPLETED
        assert ao.error_message is None

    def test_failed(self):
        ao = AgentOutput(
            instance_id="test",
            output_dir=Path("/tmp"),
            code_files=[],
            data_files=[],
            log="error",
            execution_time_seconds=1.0,
            status=RunStatus.FAILED,
            error_message="ImportError",
        )
        assert ao.error_message == "ImportError"


class TestTaskInstanceRunKey:
    """Tests for TaskInstance.run_key composite key."""

    def test_run_key_includes_prompt_level(self):
        ti = TaskInstance(
            task_id="physics.test",
            instance_id="physics.test__size10_seed42",
            task_dir=Path("."),
            workspace_dir=Path("."),
            reference_dir=Path("."),
            prompt_level=PromptLevel.B1,
            parameters={"size": 10},
            metadata={},
        )
        assert ti.run_key == "physics.test__size10_seed42__b1"

    def test_run_key_different_for_different_levels(self):
        base = dict(
            task_id="physics.test",
            instance_id="physics.test__size10_seed42",
            task_dir=Path("."),
            workspace_dir=Path("."),
            reference_dir=Path("."),
            parameters={"size": 10},
            metadata={},
        )
        ti_b1 = TaskInstance(**base, prompt_level=PromptLevel.B1)
        ti_b2 = TaskInstance(**base, prompt_level=PromptLevel.B2)
        assert ti_b1.run_key != ti_b2.run_key
        assert "__b1" in ti_b1.run_key
        assert "__b2" in ti_b2.run_key
