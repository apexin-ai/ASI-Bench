"""Supplementary edge-case tests — covers gaps identified from architecture_design.md.

Covers:
  - BenchmarkOrchestrator: _load_completed_ids, _load_ref_specs, _save_result, _evaluate gate-fail
  - aggregate_results: single instance std, domain extraction without dot, cost=None
  - RunReport.format_summary: all conditional sections
  - Leaderboard: equal scores ranking, load with missing keys, empty format_table
  - ResultVisualizer: save_charts without matplotlib, save_charts with matplotlib
  - HTTPAgentAdapter: solve success, URLError, generic exception
  - DockerAgentAdapter: solve success, timeout, _collect_output_files
  - ContaminationGuard: canary_registry property, protect_prompt with canary disabled,
    perturb_prompt with empty text, multiple synonym occurrences
  - CLI helpers: _parse_eval_result with missing fields, _build_agent factory variants
  - metadata.py: _get_git_commit failure, _is_git_dirty, _get_dependency_versions missing pkg
  - ParallelRunner: zero/negative max_workers, error in parallel run_fn
  - Logger: get_logger empty name, setup_logging with invalid level
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Orchestrator edge cases
# ---------------------------------------------------------------------------

class TestOrchestratorEdgeCases:
    """Tests for BenchmarkOrchestrator private helper methods."""

    def _make_orchestrator(self, tmp_path, **kwargs):
        from ai4sci_bench.runner.orchestrator import BenchmarkOrchestrator, RunConfig
        from ai4sci_bench.core.agent_interface import AgentAdapter
        from ai4sci_bench.core.types import AgentOutput, RunStatus, TaskInstance

        class DummyAgent(AgentAdapter):
            def solve(self, task_instance: TaskInstance) -> AgentOutput:
                return AgentOutput(
                    instance_id=task_instance.instance_id,
                    output_dir=task_instance.workspace_dir,
                    code_files=[], data_files=[], log="ok",
                    execution_time_seconds=1.0,
                    status=RunStatus.COMPLETED,
                )

        config = RunConfig(
            agent=DummyAgent(),
            output_dir=str(tmp_path / "results"),
            tasks_dir=str(tmp_path / "tasks"),
            **kwargs,
        )
        (tmp_path / "tasks").mkdir(exist_ok=True)
        return BenchmarkOrchestrator(config)

    def test_load_completed_ids_valid_json(self, tmp_path):
        """_load_completed_ids loads instance_id from valid JSON files."""
        from ai4sci_bench.core.result_schema import (
            CURRENT_RESULT_SCHEMA_VERSION,
            RESULT_SCHEMA_VERSION_FIELD,
        )

        orch = self._make_orchestrator(tmp_path, resume=str(tmp_path / "results"))
        result_dir = tmp_path / "results" / "physics.test"
        result_dir.mkdir(parents=True)
        for instance_id in ("inst_1", "inst_2"):
            (result_dir / f"{instance_id}__b2.json").write_text(json.dumps({
                RESULT_SCHEMA_VERSION_FIELD: CURRENT_RESULT_SCHEMA_VERSION,
                "instance_id": instance_id,
                "task_id": "physics.test",
                "prompt_level": "b2",
                "agent_name": "DummyAgent",
                "gates_passed": True,
                "gate_results": [],
                "score_results": [],
                "final_score": 100.0,
                "status": "completed",
            }))

        ids = orch._load_completed_ids()
        assert ids == {"inst_1__b2", "inst_2__b2"}

    def test_load_completed_ids_malformed_json(self, tmp_path):
        """_load_completed_ids silently ignores malformed JSON."""
        from ai4sci_bench.core.result_schema import (
            CURRENT_RESULT_SCHEMA_VERSION,
            RESULT_SCHEMA_VERSION_FIELD,
        )

        orch = self._make_orchestrator(tmp_path, resume=str(tmp_path / "results"))
        result_dir = tmp_path / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "bad.json").write_text("not json at all")
        (result_dir / "no_id.json").write_text(json.dumps({"foo": "bar"}))
        (result_dir / "good.json").write_text(json.dumps({
            RESULT_SCHEMA_VERSION_FIELD: CURRENT_RESULT_SCHEMA_VERSION,
            "instance_id": "good",
            "task_id": "physics.test_task",
            "prompt_level": "b2",
            "agent_name": "DummyAgent",
            "gates_passed": True,
            "gate_results": [],
            "score_results": [],
            "final_score": 100.0,
            "status": "completed",
        }))

        ids = orch._load_completed_ids()
        assert ids == {"good__b2"}

    def test_load_completed_ids_no_resume(self, tmp_path):
        """_load_completed_ids defaults to output_dir when resume is None."""
        from ai4sci_bench.core.result_schema import (
            CURRENT_RESULT_SCHEMA_VERSION,
            RESULT_SCHEMA_VERSION_FIELD,
        )

        orch = self._make_orchestrator(tmp_path)
        # resume is None, so it uses output_dir
        result_dir = tmp_path / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "r.json").write_text(json.dumps({
            RESULT_SCHEMA_VERSION_FIELD: CURRENT_RESULT_SCHEMA_VERSION,
            "instance_id": "r1",
            "task_id": "physics.test_task",
            "prompt_level": "b2",
            "agent_name": "DummyAgent",
            "gates_passed": True,
            "gate_results": [],
            "score_results": [],
            "final_score": 100.0,
            "status": "completed",
        }))

        ids = orch._load_completed_ids()
        assert "r1__b2" in ids

    def test_load_ref_specs_exists(self, tmp_path):
        """_load_ref_specs returns content when file exists."""
        from ai4sci_bench.core.types import TaskInstance, PromptLevel

        orch = self._make_orchestrator(tmp_path)
        task_dir = tmp_path / "my_task"
        task_dir.mkdir()
        (task_dir / "reference_specs.md").write_text("# Specs\nSome content")

        instance = TaskInstance(
            task_id="test", instance_id="test_1",
            task_dir=task_dir, workspace_dir=tmp_path, reference_dir=tmp_path,
            prompt_level=PromptLevel.B2, parameters={}, metadata={},
        )
        result = orch._load_ref_specs(instance)
        assert result == "# Specs\nSome content"

    def test_load_ref_specs_missing(self, tmp_path):
        """_load_ref_specs returns None when file doesn't exist."""
        from ai4sci_bench.core.types import TaskInstance, PromptLevel

        orch = self._make_orchestrator(tmp_path)
        task_dir = tmp_path / "my_task"
        task_dir.mkdir()

        instance = TaskInstance(
            task_id="test", instance_id="test_1",
            task_dir=task_dir, workspace_dir=tmp_path, reference_dir=tmp_path,
            prompt_level=PromptLevel.B2, parameters={}, metadata={},
        )
        assert orch._load_ref_specs(instance) is None

    def test_save_result_with_cost(self, tmp_path):
        """_save_result includes cost info when present."""
        from ai4sci_bench.core.types import EvalResult, PromptLevel, RunStatus, CostInfo

        orch = self._make_orchestrator(tmp_path)
        eval_result = EvalResult(
            instance_id="inst_1", task_id="physics.test",
            prompt_level=PromptLevel.B2, agent_name="TestAgent",
            parameters={}, gate_results=[], gates_passed=True,
            score_results=[], final_score=85.0,
            execution_time_seconds=10.0, status=RunStatus.COMPLETED,
        )
        eval_result.cost = CostInfo(
            input_tokens=100, output_tokens=50,
            total_tokens=150, estimated_cost_usd=0.01,
        )
        orch._save_result(eval_result)

        saved = json.loads(
            (tmp_path / "results" / "physics.test" / "inst_1__b2.json").read_text()
        )
        assert saved["cost"]["input_tokens"] == 100
        assert saved["cost"]["estimated_cost_usd"] == 0.01

    def test_save_result_with_error_analysis(self, tmp_path):
        """_save_result includes error_analysis when present."""
        from ai4sci_bench.core.types import EvalResult, PromptLevel, RunStatus
        from ai4sci_bench.analysis.error_analyzer import AnalysisReport

        orch = self._make_orchestrator(tmp_path)
        eval_result = EvalResult(
            instance_id="inst_2", task_id="chem.test",
            prompt_level=PromptLevel.B1, agent_name="TestAgent",
            parameters={}, gate_results=[], gates_passed=False,
            score_results=[], final_score=0.0,
            execution_time_seconds=5.0, status=RunStatus.FAILED,
        )
        eval_result.error_analysis = AnalysisReport(
            instance_id="inst_2",
            error_category="algorithm_error",
            error_subcategory="wrong_formula",
            root_cause="Used wrong equation",
            evidence=["line 42"],
            fix_suggestions=["Use Euler"],
            raw_analysis="...",
            confidence=0.8,
        )
        orch._save_result(eval_result)

        saved = json.loads(
            (tmp_path / "results" / "chem.test" / "inst_2__b1.json").read_text()
        )
        assert saved["error_analysis"]["error_category"] == "algorithm_error"
        assert saved["error_analysis"]["confidence"] == 0.8

    def test_evaluate_gates_fail_skips_scoring(self, tmp_path):
        """When gates fail, scoring is skipped and final_score=0."""
        from ai4sci_bench.core.types import (
            AgentOutput, EvalResult, PromptLevel, RunStatus, TaskInstance,
        )

        orch = self._make_orchestrator(tmp_path)

        # Create workspace with a code file that won't pass the gate
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "simulation.py").write_text("print('hello')")

        ref_dir = tmp_path / "reference"
        ref_dir.mkdir()

        instance = TaskInstance(
            task_id="test.task", instance_id="test_inst",
            task_dir=tmp_path, workspace_dir=workspace, reference_dir=ref_dir,
            prompt_level=PromptLevel.B2, parameters={},
            metadata={
                "evaluation": {
                    "gates": [
                        {
                            "scorer": "code_analysis",
                            "config": {
                                "checks": [{"pattern": "import taichi", "required": True}],
                                "target_file": "simulation.py",
                            },
                        }
                    ],
                    "scoring": [
                        {
                            "scorer": "numerical",
                            "weight": 100,
                            "config": {"metric": "relative_l2", "pred_file": "out.npy", "ref_file": "ref.npy"},
                        }
                    ],
                },
            },
        )

        agent_output = AgentOutput(
            instance_id="test_inst", output_dir=workspace,
            code_files=["simulation.py"], data_files=[], log="",
            execution_time_seconds=1.0, status=RunStatus.COMPLETED,
        )

        result = orch._evaluate(instance, agent_output)
        assert result.gates_passed is False
        assert result.final_score == 0.0
        assert len(result.score_results) == 0
        assert len(result.gate_results) == 1


# ---------------------------------------------------------------------------
# Aggregator edge cases
# ---------------------------------------------------------------------------

class TestAggregatorEdgeCases:

    def _make_eval_result(self, task_id="physics.test", score=80.0,
                          prompt_level="b2", error_cat=None, cost=None):
        from ai4sci_bench.core.types import EvalResult, PromptLevel, RunStatus, CostInfo
        from ai4sci_bench.analysis.error_analyzer import AnalysisReport

        er = EvalResult(
            instance_id=f"{task_id}_inst",
            task_id=task_id,
            prompt_level=PromptLevel(prompt_level),
            agent_name="TestAgent",
            parameters={},
            gate_results=[], gates_passed=True,
            score_results=[], final_score=score,
            execution_time_seconds=10.0,
            status=RunStatus.COMPLETED,
        )
        if error_cat:
            er.error_analysis = AnalysisReport(
                instance_id=er.instance_id,
                error_category=error_cat,
                error_subcategory="sub",
                root_cause="rc", evidence=[], fix_suggestions=[],
                raw_analysis="", confidence=0.5,
            )
        if cost is not None:
            er.cost = CostInfo(
                input_tokens=100, output_tokens=50,
                total_tokens=150, estimated_cost_usd=cost,
            )
        return er

    def test_single_instance_std_is_zero(self):
        """Standard deviation of a single instance should be 0."""
        from ai4sci_bench.reporting.aggregator import aggregate_results

        results = [self._make_eval_result(score=75.0)]
        report = aggregate_results(results)
        assert report.by_task[0].std_score == 0.0

    def test_domain_extraction_without_dot(self):
        """Task IDs without '.' get domain='unknown'."""
        from ai4sci_bench.reporting.aggregator import aggregate_results

        results = [self._make_eval_result(task_id="nodomain", score=60.0)]
        report = aggregate_results(results)
        assert report.by_domain[0].domain == "unknown"

    def test_cost_aggregation_with_none_costs(self):
        """Cost aggregation skips results without cost info."""
        from ai4sci_bench.reporting.aggregator import aggregate_results

        results = [
            self._make_eval_result(score=80.0, cost=0.05),
            self._make_eval_result(score=70.0, cost=None),  # no cost
        ]
        report = aggregate_results(results)
        assert report.total_cost_usd == 0.05
        assert report.total_tokens == 150  # only first result has tokens

    def test_error_distribution_aggregation(self):
        """Error distribution counts by category."""
        from ai4sci_bench.reporting.aggregator import aggregate_results

        results = [
            self._make_eval_result(score=30, error_cat="algorithm_error"),
            self._make_eval_result(score=20, error_cat="algorithm_error"),
            self._make_eval_result(score=10, error_cat="implementation_bug"),
        ]
        report = aggregate_results(results)
        assert report.error_distribution["algorithm_error"] == 2
        assert report.error_distribution["implementation_bug"] == 1

    def test_multiple_prompt_levels(self):
        """Aggregation correctly computes per-level means."""
        from ai4sci_bench.reporting.aggregator import aggregate_results

        results = [
            self._make_eval_result(score=90, prompt_level="b1"),
            self._make_eval_result(score=70, prompt_level="b2"),
            self._make_eval_result(score=50, prompt_level="b3"),
        ]
        report = aggregate_results(results)
        assert report.by_prompt_level["b1"] == 90.0
        assert report.by_prompt_level["b2"] == 70.0
        assert report.by_prompt_level["b3"] == 50.0


# ---------------------------------------------------------------------------
# RunReport.format_summary edge cases
# ---------------------------------------------------------------------------

class TestFormatSummary:

    def test_format_summary_all_sections(self):
        """format_summary includes all sections when data is present."""
        from ai4sci_bench.reporting.results import RunReport, DomainSummary, TaskSummary

        report = RunReport(
            agent_name="TestAgent", n_tasks=1, n_instances=2,
            overall_mean_score=75.0,
            by_prompt_level={"b1": 90.0, "b2": 60.0},
            by_domain=[DomainSummary(domain="physics", n_tasks=1, mean_score=75.0)],
            error_distribution={"algorithm_error": 3, "implementation_bug": 1},
            total_cost_usd=1.50, total_tokens=5000,
            total_execution_time=120.0,
        )
        summary = report.format_summary()
        assert "B1" in summary
        assert "B2" in summary
        assert "physics" in summary
        assert "algorithm_error" in summary
        assert "$1.50" in summary
        assert "5,000" in summary
        assert "Outcome Breakdown" in summary

    def test_format_summary_empty_sections(self):
        """format_summary omits sections when data is empty."""
        from ai4sci_bench.reporting.results import RunReport

        report = RunReport(
            agent_name="TestAgent", n_tasks=0, n_instances=0,
            overall_mean_score=0.0,
        )
        summary = report.format_summary()
        assert "By Prompt Level" not in summary
        assert "By Domain" not in summary
        assert "Error Distribution" not in summary
        assert "Total cost" not in summary
        assert "Outcome Breakdown" not in summary

    def test_format_summary_zero_cost_hidden(self):
        """Cost section is hidden when total_cost_usd == 0."""
        from ai4sci_bench.reporting.results import RunReport

        report = RunReport(
            agent_name="Agent", n_tasks=1, n_instances=1,
            overall_mean_score=50.0,
            total_cost_usd=0.0, total_tokens=0,
        )
        assert "Total cost" not in report.format_summary()


# ---------------------------------------------------------------------------
# Leaderboard edge cases
# ---------------------------------------------------------------------------

class TestLeaderboardEdgeCases:

    def _make_report(self, agent_name, score, n_tasks=1, n_instances=1):
        from ai4sci_bench.reporting.results import RunReport, TaskSummary

        return RunReport(
            agent_name=agent_name, n_tasks=n_tasks, n_instances=n_instances,
            overall_mean_score=score,
            by_task=[TaskSummary(
                task_id="test", n_instances=n_instances,
                mean_score=score, min_score=score, max_score=score,
                std_score=0, gates_pass_rate=1.0,
            )],
        )

    def test_equal_scores_stable_order(self):
        """Entries with equal scores maintain insertion order after sorting."""
        from ai4sci_bench.reporting.leaderboard import Leaderboard

        lb = Leaderboard()
        lb.add_from_report(self._make_report("A", 80.0))
        lb.add_from_report(self._make_report("B", 80.0))

        # Both should be ranked, scores are equal
        assert len(lb.entries) == 2
        assert lb.entries[0].rank == 1
        assert lb.entries[1].rank == 2
        scores = [e.overall_score for e in lb.entries]
        assert scores == [80.0, 80.0]

    def test_format_table_empty(self):
        """format_table with no entries returns a placeholder."""
        from ai4sci_bench.reporting.leaderboard import Leaderboard

        lb = Leaderboard()
        assert lb.format_table() == "No entries in leaderboard."

    def test_load_with_missing_optional_keys(self, tmp_path):
        """load handles JSON with missing optional keys using defaults."""
        from ai4sci_bench.reporting.leaderboard import Leaderboard

        data = [
            {
                "rank": 1,
                "agent_name": "Agent",
                "overall_score": 90.0,
                "n_tasks": 1,
                "n_instances": 5,
                # Missing: scores_by_domain, scores_by_level, gates_pass_rate, etc.
            }
        ]
        p = tmp_path / "lb.json"
        p.write_text(json.dumps(data))
        lb = Leaderboard.load(p)
        assert lb.entries[0].scores_by_domain == {}
        assert lb.entries[0].gates_pass_rate == 0.0
        assert lb.entries[0].total_cost == 0.0

    def test_add_from_report_empty_by_task(self):
        """add_from_report handles empty by_task (gates_pass_rate=0)."""
        from ai4sci_bench.reporting.leaderboard import Leaderboard
        from ai4sci_bench.reporting.results import RunReport

        report = RunReport(
            agent_name="Agent", n_tasks=0, n_instances=0,
            overall_mean_score=0.0,
        )
        lb = Leaderboard()
        lb.add_from_report(report)
        assert lb.entries[0].gates_pass_rate == 0.0

    def test_format_table_with_domain_and_level(self):
        """format_table includes domain and level breakdowns."""
        from ai4sci_bench.reporting.leaderboard import Leaderboard
        from ai4sci_bench.reporting.results import RunReport, TaskSummary, DomainSummary

        lb = Leaderboard()
        report = RunReport(
            agent_name="Agent", n_tasks=1, n_instances=2,
            overall_mean_score=85.0,
            by_task=[TaskSummary(
                task_id="test", n_instances=2, mean_score=85.0,
                min_score=80.0, max_score=90.0, std_score=5.0,
                gates_pass_rate=1.0,
            )],
            by_domain=[DomainSummary(domain="physics", n_tasks=1, mean_score=85.0)],
            by_prompt_level={"b1": 90.0, "b2": 80.0},
        )
        lb.add_from_report(report)
        table = lb.format_table()
        assert "Agent" in table
        assert "By Domain" in table
        assert "By Prompt Level" in table


# ---------------------------------------------------------------------------
# Visualizer edge cases
# ---------------------------------------------------------------------------

class TestVisualizerEdgeCases:

    def test_save_charts_without_matplotlib(self, tmp_path):
        """save_charts returns empty list when matplotlib is not available."""
        from ai4sci_bench.reporting.visualizer import ResultVisualizer
        from ai4sci_bench.reporting.results import RunReport, DomainSummary

        report = RunReport(
            agent_name="Agent", n_tasks=1, n_instances=1,
            overall_mean_score=50.0,
            by_domain=[DomainSummary(domain="physics", n_tasks=1, mean_score=50.0)],
        )
        viz = ResultVisualizer(report)

        with mock.patch.dict(sys.modules, {"matplotlib": None}):
            saved = viz.save_charts(tmp_path / "charts")
        assert saved == []

    def test_save_charts_with_matplotlib(self, tmp_path):
        """save_charts creates PNG files when matplotlib is available."""
        try:
            import matplotlib
        except ImportError:
            pytest.skip("matplotlib not installed")

        from ai4sci_bench.reporting.visualizer import ResultVisualizer
        from ai4sci_bench.reporting.results import RunReport, DomainSummary, TaskSummary

        report = RunReport(
            agent_name="Agent", n_tasks=1, n_instances=2,
            overall_mean_score=75.0,
            by_task=[TaskSummary(
                task_id="t", n_instances=2, mean_score=75.0,
                min_score=70.0, max_score=80.0, std_score=5.0,
                gates_pass_rate=1.0,
            )],
            by_domain=[DomainSummary(domain="physics", n_tasks=1, mean_score=75.0)],
            by_prompt_level={"b1": 80.0, "b2": 70.0},
            error_distribution={"algorithm_error": 2},
        )
        viz = ResultVisualizer(report)
        charts_dir = tmp_path / "charts"
        saved = viz.save_charts(charts_dir)
        assert len(saved) == 4
        for p in saved:
            assert p.exists()
            assert p.suffix == ".png"

    def test_full_report_without_error_distribution(self):
        """full_report omits error distribution section if empty."""
        from ai4sci_bench.reporting.visualizer import ResultVisualizer
        from ai4sci_bench.reporting.results import RunReport

        report = RunReport(
            agent_name="Agent", n_tasks=0, n_instances=0,
            overall_mean_score=0.0,
        )
        viz = ResultVisualizer(report)
        text = viz.full_report()
        assert "Error Distribution" not in text


# ---------------------------------------------------------------------------
# HTTPAgentAdapter edge cases
# ---------------------------------------------------------------------------

class TestHTTPAgentAdapterEdgeCases:

    def _make_instance(self, tmp_path):
        from ai4sci_bench.core.types import TaskInstance, PromptLevel

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "prompt.md").write_text("Solve this task")
        data_dir = workspace / "data"
        data_dir.mkdir()
        np.save(data_dir / "input.npy", np.array([1.0, 2.0]))

        return TaskInstance(
            task_id="test.task", instance_id="test_inst",
            task_dir=tmp_path, workspace_dir=workspace, reference_dir=tmp_path,
            prompt_level=PromptLevel.B2, parameters={},
            metadata={"output": {"files": [{"name": "out.py"}]}},
        )

    def test_solve_success(self, tmp_path):
        """solve writes output files and returns COMPLETED on success."""
        from ai4sci_bench.adapters.http_agent import HTTPAgentAdapter
        from ai4sci_bench.core.types import RunStatus
        import base64

        adapter = HTTPAgentAdapter(base_url="http://localhost:9999")
        instance = self._make_instance(tmp_path)

        response = {
            "status": "completed",
            "code_files": {"solution.py": "print('hello')"},
            "data_files": {"result.npy": base64.b64encode(b"\x00\x01").decode()},
            "log": "done",
        }

        with mock.patch.object(adapter, "_post", return_value=response):
            output = adapter.solve(instance)

        assert output.status == RunStatus.COMPLETED
        assert "solution.py" in output.code_files
        assert "result.npy" in output.data_files
        assert (instance.workspace_dir / "solution.py").read_text() == "print('hello')"

    def test_solve_success_preserves_raw_log_fields(self, tmp_path):
        """solve should preserve raw log payloads returned by the server."""
        from ai4sci_bench.adapters.http_agent import HTTPAgentAdapter

        adapter = HTTPAgentAdapter(base_url="http://localhost:9999")
        instance = self._make_instance(tmp_path)
        response = {
            "status": "completed",
            "code_files": {},
            "data_files": {},
            "log": "done",
            "raw_stdout": "{\"event\":\"tool_use\"}\n",
            "raw_stderr": "warning",
            "raw_stdout_format": "jsonl",
        }

        with mock.patch.object(adapter, "_post", return_value=response):
            output = adapter.solve(instance)

        assert output.raw_stdout == "{\"event\":\"tool_use\"}\n"
        assert output.raw_stderr == "warning"
        assert output.raw_stdout_format == "jsonl"

    def test_solve_url_error(self, tmp_path):
        """solve returns FAILED on URLError."""
        import urllib.error
        from ai4sci_bench.adapters.http_agent import HTTPAgentAdapter
        from ai4sci_bench.core.types import RunStatus

        adapter = HTTPAgentAdapter(base_url="http://localhost:9999")
        instance = self._make_instance(tmp_path)

        with mock.patch.object(
            adapter, "_post", side_effect=urllib.error.URLError("Connection refused")
        ):
            output = adapter.solve(instance)

        assert output.status == RunStatus.FAILED
        assert "HTTP request failed" in output.log

    def test_solve_generic_exception(self, tmp_path):
        """solve returns FAILED on generic Exception."""
        from ai4sci_bench.adapters.http_agent import HTTPAgentAdapter
        from ai4sci_bench.core.types import RunStatus

        adapter = HTTPAgentAdapter(base_url="http://localhost:9999")
        instance = self._make_instance(tmp_path)

        with mock.patch.object(
            adapter, "_post", side_effect=RuntimeError("unexpected")
        ):
            output = adapter.solve(instance)

        assert output.status == RunStatus.FAILED
        assert "HTTP agent error" in output.log

    def test_solve_failed_status(self, tmp_path):
        """solve returns FAILED when server reports failure."""
        from ai4sci_bench.adapters.http_agent import HTTPAgentAdapter
        from ai4sci_bench.core.types import RunStatus

        adapter = HTTPAgentAdapter(base_url="http://localhost:9999")
        instance = self._make_instance(tmp_path)

        response = {
            "status": "failed",
            "code_files": {},
            "data_files": {},
            "log": "error occurred",
            "error_message": "runtime error",
        }
        with mock.patch.object(adapter, "_post", return_value=response):
            output = adapter.solve(instance)

        assert output.status == RunStatus.FAILED
        assert output.error_message == "runtime error"


# ---------------------------------------------------------------------------
# DockerAgentAdapter edge cases
# ---------------------------------------------------------------------------

class TestDockerAgentAdapterEdgeCases:

    def _make_instance(self, tmp_path):
        from ai4sci_bench.core.types import TaskInstance, PromptLevel

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "prompt.md").write_text("Task")
        (workspace / "simulation.py").write_text("print('hi')")

        return TaskInstance(
            task_id="physics.test", instance_id="phys_inst",
            task_dir=tmp_path, workspace_dir=workspace, reference_dir=tmp_path,
            prompt_level=PromptLevel.B2, parameters={},
            metadata={"output": {"files": [
                {"name": "simulation.py"},
                {"name": "output.npy"},
            ]}},
        )

    def test_solve_success(self, tmp_path):
        """solve returns COMPLETED when docker run succeeds."""
        from ai4sci_bench.adapters.docker_agent import DockerAgentAdapter
        from ai4sci_bench.core.types import RunStatus

        adapter = DockerAgentAdapter(image="test:latest")
        instance = self._make_instance(tmp_path)

        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = "output log"
        mock_result.stderr = ""

        with mock.patch("subprocess.run", return_value=mock_result):
            output = adapter.solve(instance)

        assert output.status == RunStatus.COMPLETED
        assert "simulation.py" in output.code_files
        assert output.log == "output log"
        assert output.raw_stdout == "output log"
        assert output.raw_stderr == ""
        assert output.raw_stdout_format == "log"

    def test_solve_timeout(self, tmp_path):
        """solve returns TIMEOUT when subprocess times out."""
        import subprocess as sp
        from ai4sci_bench.adapters.docker_agent import DockerAgentAdapter
        from ai4sci_bench.core.types import RunStatus

        adapter = DockerAgentAdapter(image="test:latest", timeout=10)
        instance = self._make_instance(tmp_path)

        with mock.patch(
            "subprocess.run",
            side_effect=sp.TimeoutExpired(
                cmd="docker",
                timeout=10,
                output="partial docker stdout",
                stderr="partial docker stderr",
            ),
        ):
            output = adapter.solve(instance)

        assert output.status == RunStatus.TIMEOUT
        assert "timed out" in output.log
        assert output.raw_stdout == "partial docker stdout"
        assert output.raw_stderr == "partial docker stderr"
        assert output.raw_stdout_format == "log"

    def test_solve_generic_error(self, tmp_path):
        """solve returns FAILED on generic exception."""
        from ai4sci_bench.adapters.docker_agent import DockerAgentAdapter
        from ai4sci_bench.core.types import RunStatus

        adapter = DockerAgentAdapter(image="test:latest")
        instance = self._make_instance(tmp_path)

        with mock.patch("subprocess.run", side_effect=OSError("Docker not found")):
            output = adapter.solve(instance)

        assert output.status == RunStatus.FAILED
        assert "Docker not found" in output.log

    def test_collect_output_files(self, tmp_path):
        """_collect_output_files returns only existing files."""
        from ai4sci_bench.adapters.docker_agent import DockerAgentAdapter

        adapter = DockerAgentAdapter(image="test:latest")
        instance = self._make_instance(tmp_path)

        # simulation.py exists, output.npy does not
        files = adapter._collect_output_files(instance.workspace_dir, instance)
        assert "simulation.py" in files
        assert "output.npy" not in files

    def test_build_docker_cmd_with_gpu_and_env(self, tmp_path):
        """_build_docker_cmd includes GPU, env, and volume flags."""
        from ai4sci_bench.adapters.docker_agent import DockerAgentAdapter

        adapter = DockerAgentAdapter(
            image="test:latest",
            gpu=True,
            network=True,
            env={"KEY": "val"},
            volumes={"/host/path": "/container/path"},
        )
        instance = self._make_instance(tmp_path)
        cmd = adapter._build_docker_cmd(instance.workspace_dir, instance)

        assert "--gpus" in cmd
        assert "all" in cmd
        assert "--network" not in cmd  # network=True means no --network none
        assert "-e" in cmd
        assert "KEY=val" in cmd
        assert "/host/path:/container/path" in cmd


# ---------------------------------------------------------------------------
# ContaminationGuard edge cases
# ---------------------------------------------------------------------------

class TestContaminationGuardEdgeCases:

    def test_canary_registry_returns_copy(self):
        """canary_registry returns a copy, not the internal dict."""
        from ai4sci_bench.generators.contamination import ContaminationGuard

        guard = ContaminationGuard()
        guard.protect_prompt("Test prompt", "inst_1")
        reg = guard.canary_registry
        reg["injected"] = "extra"
        assert "injected" not in guard.canary_registry

    def test_protect_prompt_canary_disabled(self):
        """protect_prompt with enable_canary=False doesn't inject canary."""
        from ai4sci_bench.generators.contamination import ContaminationGuard

        guard = ContaminationGuard(enable_canary=False, enable_perturbation=False)
        result = guard.protect_prompt("Hello world", "inst_1")
        assert "CANARY" not in result
        assert result == "Hello world"

    def test_protect_prompt_perturbation_only(self):
        """protect_prompt with only perturbation enabled."""
        from ai4sci_bench.generators.contamination import ContaminationGuard

        guard = ContaminationGuard(enable_canary=False, enable_perturbation=True)
        # Use a seed that triggers at least one perturbation
        result = guard.protect_prompt("implement the solution in the current directory", "inst_1", seed=42)
        assert "CANARY" not in result

    def test_perturb_prompt_empty_text(self):
        """perturb_prompt handles empty text."""
        from ai4sci_bench.generators.contamination import perturb_prompt

        result = perturb_prompt("", seed=0)
        assert result == ""

    def test_detect_canary_no_match(self):
        """detect_canary returns empty list for text without canaries."""
        from ai4sci_bench.generators.contamination import detect_canary

        assert detect_canary("Just some regular text") == []

    def test_check_contamination_returns_instance_ids(self):
        """check_contamination returns instance IDs whose canaries are found."""
        from ai4sci_bench.generators.contamination import ContaminationGuard

        guard = ContaminationGuard(enable_perturbation=False)
        prompt1 = guard.protect_prompt("Prompt 1", "inst_A")
        prompt2 = guard.protect_prompt("Prompt 2", "inst_B")

        # Text containing canary from prompt1 but not prompt2
        contaminated = guard.check_contamination(prompt1)
        assert "inst_A" in contaminated


# ---------------------------------------------------------------------------
# CLI helper edge cases
# ---------------------------------------------------------------------------

class TestCLIHelperEdgeCases:

    def test_parse_eval_result_minimal(self):
        """_parse_eval_result handles minimal data with defaults."""
        from ai4sci_bench.cli import _parse_eval_result

        data = {
            "instance_id": "test_1",
            "task_id": "physics.test",
        }
        result = _parse_eval_result(data)
        assert result.instance_id == "test_1"
        assert result.prompt_level.value == "b2"  # default
        assert result.agent_name == "unknown"
        assert result.gates_passed is False  # default
        assert result.final_score == 0.0
        assert result.error_analysis is None

    def test_parse_eval_result_with_error_analysis(self):
        """_parse_eval_result parses error_analysis with missing optional fields."""
        from ai4sci_bench.cli import _parse_eval_result

        data = {
            "instance_id": "test_2",
            "task_id": "chem.test",
            "prompt_level": "b1",
            "error_analysis": {
                "error_category": "algorithm_error",
                # missing: error_subcategory, root_cause, etc.
            },
        }
        result = _parse_eval_result(data)
        assert result.error_analysis is not None
        assert result.error_analysis.error_category == "algorithm_error"
        assert result.error_analysis.error_subcategory == "unknown"  # default
        assert result.error_analysis.evidence == []

    def test_parse_eval_result_with_score_details(self):
        """_parse_eval_result correctly parses gate_results and score_results."""
        from ai4sci_bench.cli import _parse_eval_result

        data = {
            "instance_id": "test_3",
            "task_id": "bio.test",
            "gate_results": [
                {"scorer_name": "file_match", "score": 1.0, "max_score": 1.0,
                 "passed": True},
            ],
            "score_results": [
                {"scorer_name": "numerical", "score": 50.0, "max_score": 100.0,
                 "passed": True, "details": {"metric": "l2"}},
            ],
            "final_score": 50.0,
            "gates_passed": True,
        }
        result = _parse_eval_result(data)
        assert len(result.gate_results) == 1
        assert result.gate_results[0].scorer_name == "file_match"
        assert len(result.score_results) == 1
        assert result.score_results[0].details == {"metric": "l2"}

    def test_parse_eval_result_with_agent_output_raw_log_metadata(self):
        """_parse_eval_result preserves raw-log metadata fields on agent_output."""
        from ai4sci_bench.cli import _parse_eval_result

        data = {
            "instance_id": "test_4",
            "task_id": "physics.test",
            "agent_output": {
                "code_files": [],
                "data_files": [],
                "log": "summary",
                "status": "failed",
                "error_message": "boom",
                "raw_stdout_file": "test_4__b2.agent_stdout.jsonl",
                "raw_stdout_format": "jsonl",
                "raw_stderr_file": "test_4__b2.agent_stderr.log",
                "raw_model_output_file": "test_4__b2.agent_model_output.md",
                "raw_model_output_format": "markdown",
            },
        }
        result = _parse_eval_result(data)
        assert result.agent_output is not None
        assert result.agent_output.raw_stdout_format == "jsonl"
        assert result.agent_output.raw_stdout_file == "test_4__b2.agent_stdout.jsonl"
        assert result.agent_output.raw_stderr_file == "test_4__b2.agent_stderr.log"
        assert result.agent_output.raw_model_output_file == "test_4__b2.agent_model_output.md"
        assert result.agent_output.raw_model_output_format == "markdown"

    def test_build_agent_with_agent_cmd(self):
        """_build_agent returns CLIAgentAdapter when agent_cmd is provided."""
        from ai4sci_bench.cli import _build_agent
        from ai4sci_bench.adapters.cli_agent import CLIAgentAdapter

        agent = _build_agent("echo hello", None, {})
        assert isinstance(agent, CLIAgentAdapter)

    def test_build_agent_claude_code_cli(self):
        """_build_agent returns ClaudeCodeCLIAdapter for 'claude_code_cli'."""
        from ai4sci_bench.cli import _build_agent
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter

        agent = _build_agent(None, "claude_code_cli", {})
        assert isinstance(agent, ClaudeCodeCLIAdapter)

    def test_build_agent_direct_llm(self):
        """_build_agent returns DirectLLMAdapter for 'direct_llm'."""
        from ai4sci_bench.cli import _build_agent
        from ai4sci_bench.adapters.direct_llm import DirectLLMAdapter

        agent = _build_agent(None, "direct_llm", {"model": "test/model"})
        assert isinstance(agent, DirectLLMAdapter)
        assert agent.model == "test/model"

    def test_build_agent_codex_cli(self):
        """_build_agent returns CodexCLIAdapter for 'codex_cli'."""
        from ai4sci_bench.cli import _build_agent
        from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter

        agent = _build_agent(None, "codex_cli", {})
        assert isinstance(agent, CodexCLIAdapter)
        assert agent.model == "gpt-5.5"

    def test_build_agent_codex_cli_with_config(self):
        """_build_agent passes config to CodexCLIAdapter."""
        from ai4sci_bench.cli import _build_agent
        from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter

        agent = _build_agent(None, "codex_cli", {"model": "o3", "full_auto": False})
        assert isinstance(agent, CodexCLIAdapter)
        assert agent.model == "o3"
        assert agent.full_auto is False

    def test_build_agent_default(self):
        """_build_agent returns a default CLIAgentAdapter when no agent specified."""
        from ai4sci_bench.cli import _build_agent
        from ai4sci_bench.adapters.cli_agent import CLIAgentAdapter

        agent = _build_agent(None, None, {})
        assert isinstance(agent, CLIAgentAdapter)


# ---------------------------------------------------------------------------
# Metadata edge cases
# ---------------------------------------------------------------------------

class TestMetadataEdgeCases:

    def test_get_git_commit_failure(self):
        """_get_git_commit returns 'unknown' when git is not available."""
        from ai4sci_bench.runner.metadata import _get_git_commit

        with mock.patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
            assert _get_git_commit() == "unknown"

    def test_get_git_commit_nonzero_return(self):
        """_get_git_commit returns 'unknown' when git returns non-zero."""
        from ai4sci_bench.runner.metadata import _get_git_commit

        mock_result = mock.Mock()
        mock_result.returncode = 128
        with mock.patch("subprocess.run", return_value=mock_result):
            assert _get_git_commit() == "unknown"

    def test_is_git_dirty_no_changes(self):
        """_is_git_dirty returns False for clean working directory."""
        from ai4sci_bench.runner.metadata import _is_git_dirty

        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        with mock.patch("subprocess.run", return_value=mock_result):
            assert _is_git_dirty() is False

    def test_is_git_dirty_with_changes(self):
        """_is_git_dirty returns True when there are changes."""
        from ai4sci_bench.runner.metadata import _is_git_dirty

        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = " M file.py\n"
        with mock.patch("subprocess.run", return_value=mock_result):
            assert _is_git_dirty() is True

    def test_is_git_dirty_failure(self):
        """_is_git_dirty returns False when git fails."""
        from ai4sci_bench.runner.metadata import _is_git_dirty

        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            assert _is_git_dirty() is False

    def test_get_dependency_versions_missing_package(self):
        """_get_dependency_versions skips packages that aren't installed."""
        import importlib.metadata

        from ai4sci_bench.runner.metadata import _get_dependency_versions

        original_version = importlib.metadata.version

        def fake_version(name):
            if name == "scipy":
                raise importlib.metadata.PackageNotFoundError(name)
            return original_version(name)

        with mock.patch("importlib.metadata.version", side_effect=fake_version):
            versions = _get_dependency_versions()
        assert "scipy" not in versions

    def test_save_run_metadata(self, tmp_path):
        """save_run_metadata writes JSON and returns path."""
        from ai4sci_bench.runner.metadata import save_run_metadata

        meta = {"framework_version": "0.1.0", "test": True}
        path = save_run_metadata(tmp_path / "output", meta)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["test"] is True

    def test_collect_run_metadata_includes_all_fields(self):
        """collect_run_metadata includes all standard fields."""
        from ai4sci_bench.runner.metadata import collect_run_metadata

        meta = collect_run_metadata(
            agent_config={"type": "TestAgent"},
            task_versions={"task1": "1.0"},
        )
        assert "framework_version" in meta
        assert "python_version" in meta
        assert "platform" in meta
        assert "timestamp" in meta
        assert "dependencies" in meta
        assert meta["agent_config"]["type"] == "TestAgent"
        assert meta["task_versions"]["task1"] == "1.0"


# ---------------------------------------------------------------------------
# ParallelRunner edge cases
# ---------------------------------------------------------------------------

class TestParallelRunnerEdgeCases:

    def test_zero_max_workers_clamped_to_one(self):
        """max_workers=0 is clamped to 1."""
        from ai4sci_bench.runner.parallel import ParallelRunner

        runner = ParallelRunner(max_workers=0)
        assert runner.max_workers == 1

    def test_negative_max_workers_clamped_to_one(self):
        """max_workers=-5 is clamped to 1."""
        from ai4sci_bench.runner.parallel import ParallelRunner

        runner = ParallelRunner(max_workers=-5)
        assert runner.max_workers == 1

    def test_run_instances_empty_after_filter(self):
        """run_instances returns [] when all instances are in completed_ids."""
        from ai4sci_bench.runner.parallel import ParallelRunner
        from ai4sci_bench.core.types import TaskInstance, PromptLevel

        runner = ParallelRunner(max_workers=1)

        instance = TaskInstance(
            task_id="t", instance_id="already_done",
            task_dir=Path("."), workspace_dir=Path("."), reference_dir=Path("."),
            prompt_level=PromptLevel.B2, parameters={}, metadata={},
        )

        results = runner.run_instances(
            [instance],
            run_fn=lambda i: None,
            completed_ids={"already_done__b2"},
        )
        assert results == []

    def test_parallel_error_handling_continues(self):
        """Parallel runner continues despite errors in individual run_fn calls."""
        from ai4sci_bench.runner.parallel import ParallelRunner
        from ai4sci_bench.core.types import (
            TaskInstance, PromptLevel, EvalResult, RunStatus,
        )

        runner = ParallelRunner(max_workers=2)

        instances = [
            TaskInstance(
                task_id="t", instance_id=f"inst_{i}",
                task_dir=Path("."), workspace_dir=Path("."), reference_dir=Path("."),
                prompt_level=PromptLevel.B2, parameters={}, metadata={},
            )
            for i in range(3)
        ]

        call_count = 0

        def run_fn(inst):
            nonlocal call_count
            call_count += 1
            if inst.instance_id == "inst_1":
                raise RuntimeError("boom")
            return EvalResult(
                instance_id=inst.instance_id, task_id="t",
                prompt_level=PromptLevel.B2, agent_name="test",
                parameters={}, gate_results=[], gates_passed=True,
                score_results=[], final_score=100.0,
                status=RunStatus.COMPLETED,
            )

        results = runner.run_instances(instances, run_fn=run_fn)
        # All 3 instances should produce results; inst_1 gets a failed fallback
        assert len(results) == 3
        failed = [r for r in results if r.status == RunStatus.FAILED]
        assert len(failed) == 1
        assert failed[0].instance_id == "inst_1"


# ---------------------------------------------------------------------------
# Logger edge cases
# ---------------------------------------------------------------------------

class TestLoggerEdgeCases:

    def test_get_logger_prefixed(self):
        """get_logger already prefixed name is returned as-is."""
        from ai4sci_bench.core.logger import get_logger

        log = get_logger("ai4sci_bench.test.module")
        assert log.name == "ai4sci_bench.test.module"

    def test_get_logger_unprefixed(self):
        """get_logger adds ai4sci_bench prefix."""
        from ai4sci_bench.core.logger import get_logger

        log = get_logger("mymodule")
        assert log.name == "ai4sci_bench.mymodule"

    def test_setup_logging_string_level(self):
        """setup_logging accepts string level names."""
        from ai4sci_bench.core.logger import setup_logging
        import logging

        # Reset handlers to allow re-initialization
        root = logging.getLogger("ai4sci_bench")
        root.handlers.clear()

        setup_logging(level="DEBUG")
        assert root.level == logging.DEBUG

        # Clean up
        root.handlers.clear()

    def test_setup_logging_with_file(self, tmp_path):
        """setup_logging creates a file handler when log_file is specified."""
        from ai4sci_bench.core.logger import setup_logging
        import logging

        root = logging.getLogger("ai4sci_bench")
        root.handlers.clear()

        log_file = str(tmp_path / "test.log")
        setup_logging(level="INFO", log_file=log_file)

        has_file_handler = any(
            isinstance(h, logging.FileHandler) for h in root.handlers
        )
        assert has_file_handler

        # Clean up
        root.handlers.clear()
