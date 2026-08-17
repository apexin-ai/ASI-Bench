"""Tests for reporting and aggregation."""

import os
from pathlib import Path

from ai4sci_bench.core.types import (
    AgentOutput,
    AnalysisReport,
    EvalResult,
    PromptLevel,
    RunStatus,
    ScoreDetail,
)
from ai4sci_bench.reporting.aggregator import aggregate_results
from ai4sci_bench.reporting.results import RunReport
from ai4sci_bench.analysis.report_generator import ReportGenerator


def _make_result(task_id, score, prompt_level="b2", error_cat=None, soft_gate_failures=0):
    error_analysis = None
    if error_cat:
        error_analysis = AnalysisReport(
            instance_id=f"{task_id}_0",
            error_category=error_cat,
            error_subcategory="test",
            root_cause="test",
            evidence=[],
            fix_suggestions=[],
            raw_analysis="",
            confidence=0.8,
        )
    return EvalResult(
        instance_id=f"{task_id}_0",
        task_id=task_id,
        prompt_level=PromptLevel(prompt_level),
        agent_name="TestAgent",
        parameters={},
        gate_results=[],
        gates_passed=score > 0,
        hard_gates_passed=score > 0,
        soft_gate_failures=soft_gate_failures,
        score_results=[
            ScoreDetail(scorer_name="test", score=score, max_score=100.0, passed=True, details={})
        ],
        final_score=score,
        execution_time_seconds=10.0,
        error_analysis=error_analysis,
    )


def _make_unscored_result(task_id, prompt_level="b2"):
    result = _make_result(task_id, 0.0, prompt_level)
    result.gates_passed = True
    result.hard_gates_passed = True
    result.score_results = [
        ScoreDetail(
            scorer_name="unscored_submission",
            score=0.0,
            max_score=0.0,
            passed=True,
            details={"unscored_submission": True},
        )
    ]
    return result


class TestResultDiscovery:
    def test_prunes_non_result_directories_before_descent(self, tmp_path, monkeypatch):
        from ai4sci_bench.reporting import result_loader

        results_root = tmp_path / "results"
        result_dir = results_root / "physics.test_task"
        result_dir.mkdir(parents=True)
        (result_dir / "result.json").write_text("{}", encoding="utf-8")

        skipped_dir_names = {
            "batch_records",
            "_workspaces",
            "reference",
            "__pycache__",
            "agent_api_cassettes",
            "scorer_api_cassettes",
            "recovery_archive",
            "reproducibility",
        }
        for dirname in skipped_dir_names:
            nested = results_root / dirname / "nested"
            nested.mkdir(parents=True)
            (nested / "noise.json").write_text("{}", encoding="utf-8")

        visited_dirs = []
        real_walk = os.walk

        def recording_walk(*args, **kwargs):
            for dirpath, dirnames, filenames in real_walk(*args, **kwargs):
                visited_dirs.append(Path(dirpath).relative_to(results_root))
                yield dirpath, dirnames, filenames

        monkeypatch.setattr(result_loader.os, "walk", recording_walk)

        groups = result_loader.load_grouped_results(results_root)

        assert groups == []
        assert Path("physics.test_task") in visited_dirs
        visited_parts = {part for path in visited_dirs for part in path.parts}
        assert skipped_dir_names.isdisjoint(visited_parts)


class TestAggregation:
    def test_empty_results(self):
        report = aggregate_results([])
        assert report.n_instances == 0
        assert report.overall_mean_score == 0.0

    def test_single_result(self):
        results = [_make_result("physics.task1", 85.0)]
        report = aggregate_results(results)
        assert report.n_instances == 1
        assert report.n_tasks == 1
        assert report.overall_mean_score == 85.0

    def test_result_loader_completed_beats_higher_scoring_failed_attempt(self):
        from ai4sci_bench.reporting.result_loader import is_better_result

        failed_high = _make_result("physics.task1", 90.0)
        failed_high.status = RunStatus.FAILED
        completed_low = _make_result("physics.task1", 10.0)
        completed_low.status = RunStatus.COMPLETED

        assert is_better_result(completed_low, failed_high)
        assert not is_better_result(failed_high, completed_low)

    def test_multiple_tasks(self):
        results = [
            _make_result("physics.task1", 80.0),
            _make_result("physics.task1", 90.0),
            _make_result("chemistry.task2", 60.0),
        ]
        report = aggregate_results(results)
        assert report.n_tasks == 2
        assert report.n_instances == 3
        assert report.overall_mean_score == (80 + 90 + 60) / 3

    def test_outcome_breakdown_counts(self):
        results = [
            _make_result("physics.task1", 80.0),
            _make_result("physics.task2", 0.0),
        ]
        results[1].gates_passed = False
        report = aggregate_results(results)
        assert report.gate_failed_instances == 1
        assert report.soft_gate_warning_instances == 0
        assert report.zero_score_after_gates_instances == 0
        assert report.nonzero_score_instances == 1

    def test_soft_gate_warning_counts(self):
        results = [
            _make_result("physics.task1", 80.0, soft_gate_failures=1),
            _make_result("physics.task2", 70.0),
        ]
        report = aggregate_results(results)
        assert report.gate_failed_instances == 0
        assert report.soft_gate_warning_instances == 1
        assert report.zero_score_after_gates_instances == 0
        assert report.nonzero_score_instances == 1

    def test_by_prompt_level(self):
        results = [
            _make_result("t.a", 90.0, "b1"),
            _make_result("t.a", 70.0, "b2"),
            _make_result("t.a", 50.0, "b3"),
        ]
        report = aggregate_results(results)
        assert report.by_prompt_level["b1"] == 90.0
        assert report.by_prompt_level["b2"] == 70.0
        assert report.by_prompt_level["b3"] == 50.0

    def test_by_domain(self):
        results = [
            _make_result("physics.a", 80.0),
            _make_result("physics.b", 70.0),
            _make_result("chemistry.c", 60.0),
        ]
        report = aggregate_results(results)
        assert len(report.by_domain) == 2
        domains = {d.domain: d.mean_score for d in report.by_domain}
        assert domains["physics"] == 75.0  # (80+70)/2
        assert domains["chemistry"] == 60.0

    def test_error_distribution(self):
        results = [
            _make_result("t.a", 0.0, error_cat="algorithm_error"),
            _make_result("t.b", 0.0, error_cat="algorithm_error"),
            _make_result("t.c", 0.0, error_cat="implementation_bug"),
            _make_result("t.d", 100.0),
        ]
        report = aggregate_results(results)
        assert report.error_distribution["algorithm_error"] == 2
        assert report.error_distribution["implementation_bug"] == 1

    def test_format_summary(self):
        results = [
            _make_result("physics.a", 85.0, "b1"),
            _make_result("physics.a", 65.0, "b2"),
        ]
        report = aggregate_results(results)
        summary = report.format_summary()
        assert "ASI-Bench" in summary
        assert "75.0" in summary  # mean of 85 and 65
        assert "Outcome Breakdown" in summary
        assert "Hard gate failures" in summary

    def test_format_task_table_hidden_when_all_results_are_unscored(self):
        report = aggregate_results([
            _make_unscored_result("physics.a", "b1"),
            _make_unscored_result("physics.a", "b2"),
        ])

        assert report.format_task_table() == ""

    def test_format_task_table_uses_na_for_unscored_levels_in_mixed_report(self):
        report = aggregate_results([
            _make_unscored_result("physics.a", "b1"),
            _make_result("physics.a", 80.0, "b2"),
        ])

        table = report.format_task_table()

        assert "B1" in table
        assert "B2" in table
        assert "N/A" in table
        assert "80.0" in table
        assert "40.0" not in table
        assert report.overall_mean_score == 80.0

    def test_format_summary_includes_low_score_reasons(self):
        low = _make_result("physics.a", 10.0, "b2")
        low.gates_passed = True
        low.score_results = [
            ScoreDetail(
                scorer_name="numerical:mean_relative_l2",
                score=0.0,
                max_score=30.0,
                passed=True,
                details={"mean_relative_l2": 0.9},
                message="mean_relative_l2=0.900000; score=0.00/30.00",
            )
        ]
        report = aggregate_results([low])
        summary = report.format_summary()
        assert "Low-Score Instances" in summary
        assert "numerical:mean_relative_l2" in summary

    def test_format_summary_builds_reason_from_legacy_details(self):
        low = _make_result("physics.a", 10.0, "b2")
        low.gates_passed = True
        low.score_results = [
            ScoreDetail(
                scorer_name="numerical:per_frame_relative_l2",
                score=10.0,
                max_score=50.0,
                passed=True,
                details={
                    "frame_0": {"rel_err": 0.0, "pts": 10},
                    "frame_1": {"rel_err": 0.617494, "pts": 0.0},
                    "frame_2": {"rel_err": 1.261864, "pts": 0.0},
                },
                message="",
            ),
            ScoreDetail(
                scorer_name="numerical:mean_relative_l2",
                score=0.0,
                max_score=30.0,
                passed=True,
                details={"mean_relative_l2": 0.9462474163},
                message="",
            ),
        ]
        report = aggregate_results([low])
        summary = report.format_summary()
        assert "1/3 frames awarded" in summary
        assert "worst frame_2 rel_err=1.261864" in summary
        assert "mean_relative_l2=0.946247" in summary

    def test_format_summary_includes_agent_failure_reason(self):
        failed = _make_result("physics.a", 0.0, "b2")
        failed.gates_passed = False
        failed.hard_gates_passed = False
        failed.agent_output = AgentOutput(
            instance_id=failed.instance_id,
            output_dir=Path("."),
            code_files=[],
            data_files=[],
            log="traceback",
            execution_time_seconds=1.0,
            status=RunStatus.FAILED,
            error_message="command failed",
        )
        failed.gate_results = [
            ScoreDetail(
                scorer_name="file_match",
                score=0.0,
                max_score=1.0,
                passed=False,
                details={},
                message="File not found: output.npy",
            )
        ]
        report = aggregate_results([failed])
        summary = report.format_summary()
        assert "agent_status: failed" in summary
        assert "agent_error: command failed" in summary

    def test_format_summary_includes_soft_gate_warning_reason(self):
        warned = _make_result("physics.a", 100.0, "b2", soft_gate_failures=1)
        warned.gate_results = [
            ScoreDetail(
                scorer_name="code_analysis",
                score=0.0,
                max_score=1.0,
                passed=False,
                details={},
                message="Required pattern not found: scipy",
                severity="soft",
            )
        ]
        report = aggregate_results([warned])
        summary = report.format_summary()
        assert "soft warning; score 100.0/100" in summary
        assert "code_analysis (soft): Required pattern not found: scipy" in summary


class TestReportGenerator:
    def test_instance_report(self):
        result = _make_result("physics.test", 82.0)
        gen = ReportGenerator()
        report = gen.generate_instance_report(result)
        assert "physics.test" in report
        assert "82.0" in report

    def test_summary_report(self):
        results = [
            _make_result("physics.a", 80.0),
            _make_result("physics.b", 60.0),
        ]
        gen = ReportGenerator()
        report = gen.generate_summary_report(results)
        assert "Mean score: 70.0" in report

    def test_empty_summary(self):
        gen = ReportGenerator()
        report = gen.generate_summary_report([])
        assert "No results" in report


# ── TODO-7: Report behavior statistics tests ────────────────────────────


class TestBehaviorSummary:
    def test_aggregate_computes_behavior_summary(self):
        from ai4sci_bench.reporting.results import BehaviorSummary
        results = []
        for i, turns in enumerate([10, 12, 14]):
            r = _make_result(f"task.{i}", 50.0)
            r.agent_output = AgentOutput(
                instance_id=r.instance_id,
                output_dir=Path("/tmp"),
                code_files=[],
                data_files=[],
                log="",
                execution_time_seconds=1.0,
                status=RunStatus.COMPLETED,
            )
            r.agent_output._trajectory_summary = {
                "total_turns": turns,
                "total_tool_calls": turns * 2,
                "tool_call_distribution": {"Write": turns, "Bash": turns},
                "thinking_total_chars": turns * 100,
                "code_execution_failures": 1,
            }
            results.append(r)

        report = aggregate_results(results)
        assert report.behavior_summary is not None
        assert abs(report.behavior_summary.avg_turns - 12.0) < 0.01
        assert report.behavior_summary.tool_call_distribution["Write"] == 36  # 10+12+14

    def test_aggregate_behavior_summary_absent_when_no_trajectory(self):
        results = [_make_result("task.a", 80.0), _make_result("task.b", 60.0)]
        report = aggregate_results(results)
        assert report.behavior_summary is None

    def test_format_summary_includes_behavior_section(self):
        from ai4sci_bench.reporting.results import BehaviorSummary
        report = RunReport(
            agent_name="test",
            n_tasks=1,
            n_instances=1,
            overall_mean_score=50.0,
            behavior_summary=BehaviorSummary(
                avg_turns=12.3,
                avg_tool_calls=25.1,
                tool_call_distribution={"Write": 4, "Bash": 8},
                avg_thinking_chars=15200,
            ),
        )
        output = report.format_summary()
        assert "Agent Behavior" in output
        assert "12.3" in output

    def test_format_summary_omits_behavior_when_absent(self):
        report = RunReport(
            agent_name="test",
            n_tasks=1,
            n_instances=1,
            overall_mean_score=50.0,
        )
        output = report.format_summary()
        assert "Agent Behavior" not in output

    def test_low_score_section_shows_trajectory_context(self):
        r = _make_result("task.x", 30.0)
        r.max_possible_score = 100.0
        r.agent_output = AgentOutput(
            instance_id=r.instance_id,
            output_dir=Path("/tmp"),
            code_files=[],
            data_files=[],
            log="",
            execution_time_seconds=1.0,
            status=RunStatus.COMPLETED,
        )
        r.agent_output._trajectory_summary = {
            "total_code_executions": 8,
            "code_execution_failures": 5,
        }
        report = RunReport(
            agent_name="test",
            n_tasks=1,
            n_instances=1,
            overall_mean_score=30.0,
            results=[r],
        )
        lines = report._format_low_score_section()
        combined = "\n".join(lines)
        assert "8 code executions (5 failures)" in combined


# ── TODO-8: Retry trajectory diff tests ─────────────────────────────────


class TestRetryDiff:
    def test_retry_diff_shows_attempt_comparison(self):
        r1 = _make_result("task.a", 0.0)
        r1.attempt = 1
        r1.max_possible_score = 100.0
        r2 = _make_result("task.a", 80.0)
        r2.instance_id = r1.instance_id  # same instance
        r2.attempt = 2
        r2.max_possible_score = 100.0

        report = RunReport(
            agent_name="test",
            n_tasks=1,
            n_instances=2,
            overall_mean_score=40.0,
            results=[r1, r2],
        )
        lines = report._format_retry_diff_section()
        combined = "\n".join(lines)
        assert "Retry Comparison" in combined
        assert "Attempt 1" in combined
        assert "Attempt 2" in combined

    def test_retry_diff_single_attempt_no_diff(self):
        r = _make_result("task.a", 80.0)
        r.attempt = 1
        report = RunReport(
            agent_name="test",
            n_tasks=1,
            n_instances=1,
            overall_mean_score=80.0,
            results=[r],
        )
        lines = report._format_retry_diff_section()
        assert lines == []


# ── Bugfix regression: result_loader restores _trajectory_summary ──────

class TestResultLoaderTrajectoryRestore:
    """Bug: trajectory_summary wasn't restored from JSON when loading results."""

    def test_loaded_result_has_trajectory_summary(self, tmp_path):
        import json
        result_data = {
            "result_schema_version": 1,
            "instance_id": "test__seed42",
            "task_id": "physics.test",
            "attempt": 1,
            "prompt_level": "b2",
            "agent_name": "TestAgent",
            "parameters": {},
            "gates_passed": True,
            "hard_gates_passed": True,
            "soft_gate_failures": 0,
            "gate_results": [],
            "score_results": [],
            "final_score": 80.0,
            "max_possible_score": 100.0,
            "execution_time_seconds": 1.0,
            "status": "completed",
            "provenance": {},
            "agent_output": {
                "code_files": [],
                "data_files": [],
                "log": "",
                "error_message": None,
                "status": "completed",
                "trajectory_summary": {
                    "total_turns": 12,
                    "total_tool_calls": 25,
                    "tool_call_distribution": {"Write": 4, "Bash": 8},
                    "thinking_total_chars": 15000,
                    "total_code_executions": 8,
                    "code_execution_failures": 3,
                },
            },
        }
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps(result_data))

        from ai4sci_bench.reporting.result_loader import parse_eval_result
        eval_result = parse_eval_result(result_data)
        assert eval_result.agent_output is not None
        ts = getattr(eval_result.agent_output, "_trajectory_summary", None)
        assert ts is not None
        assert ts["total_turns"] == 12
        assert ts["tool_call_distribution"]["Write"] == 4

    def test_loaded_result_without_trajectory_summary_is_none(self):
        result_data = {
            "result_schema_version": 1,
            "instance_id": "test__seed42",
            "task_id": "physics.test",
            "attempt": 1,
            "prompt_level": "b2",
            "agent_name": "TestAgent",
            "parameters": {},
            "gates_passed": True,
            "hard_gates_passed": True,
            "soft_gate_failures": 0,
            "gate_results": [],
            "score_results": [],
            "final_score": 80.0,
            "max_possible_score": 100.0,
            "execution_time_seconds": 1.0,
            "status": "completed",
            "provenance": {},
            "agent_output": {
                "code_files": [],
                "data_files": [],
                "log": "",
                "error_message": None,
                "status": "completed",
            },
        }
        from ai4sci_bench.reporting.result_loader import parse_eval_result
        eval_result = parse_eval_result(result_data)
        assert eval_result.agent_output is not None
        ts = getattr(eval_result.agent_output, "_trajectory_summary", None)
        assert ts is None
