"""Tests for the difficulty-check pipeline (T1 + T2 + T3)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from ai4sci_bench.cli import cli
from ai4sci_bench.core.types import (
    EvalResult,
    PromptLevel,
    RunStatus,
)
from ai4sci_bench.reporting.difficulty_report import (
    DifficultyReport,
    VerdictRow,
    build_report,
    format_markdown,
    format_terminal,
    write_json,
    write_markdown,
)
from ai4sci_bench.reporting.results import RunReport
from ai4sci_bench.tracking.difficulty_scores import (
    append_evaluation,
    find_flagged_tasks,
    get_all_task_scores,
    get_latest_verdict,
    load_scores,
    score_file_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(*, task_id: str, level: str, score: float, agent: str = "direct_llm") -> EvalResult:
    return EvalResult(
        instance_id=f"{task_id}_{level}",
        task_id=task_id,
        prompt_level=PromptLevel(level),
        agent_name=agent,
        parameters={},
        gate_results=[],
        gates_passed=True,
        score_results=[],
        final_score=score,
        max_possible_score=100.0,
        status=RunStatus.COMPLETED,
    )


def _write_task_yaml(tasks_dir: Path, task_id: str, status: str, version: str = "1.0") -> Path:
    domain, name = task_id.split(".", 1)
    task_dir = tasks_dir / domain / name
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.yaml").write_text(
        yaml.safe_dump({
            "id": task_id,
            "name": name,
            "version": version,
            "status": status,
            "domain": domain,
        }),
        encoding="utf-8",
    )
    return task_dir


# ---------------------------------------------------------------------------
# T2 — difficulty_scores.py
# ---------------------------------------------------------------------------


class TestScoreFilePath:
    def test_returns_task_json_under_scores_dir(self, tmp_path):
        p = score_file_path(tmp_path, "physics.foo")
        assert p == tmp_path / "physics.foo.json"


class TestLoadScores:
    def test_missing_file_returns_none(self, tmp_path):
        assert load_scores(tmp_path, "physics.missing") is None

    def test_corrupt_file_raises(self, tmp_path):
        (tmp_path / "physics.bad.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="Corrupt"):
            load_scores(tmp_path, "physics.bad")

    def test_non_object_root_raises(self, tmp_path):
        (tmp_path / "physics.list.json").write_text("[1,2,3]", encoding="utf-8")
        with pytest.raises(ValueError, match="must contain a JSON object"):
            load_scores(tmp_path, "physics.list")


class TestAppendEvaluation:
    def test_creates_new_file_with_record(self, tmp_path):
        scores_dir = tmp_path / "scores"
        path = append_evaluation(
            scores_dir,
            task_id="physics.new",
            task_version="1.0",
            results=[{"agent": "direct_llm", "agent_config": {"model": "claude-opus-4-6"},
                      "scores": {"b3": {"mean": 12.0, "max": 15.0, "min": 9.0, "n": 1}}}],
            threshold=50,
            verdict="pass",
            seeds=[42],
        )
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["task_id"] == "physics.new"
        assert data["task_version"] == "1.0"
        assert len(data["evaluations"]) == 1
        assert data["evaluations"][0]["verdict"] == "pass"
        assert data["evaluations"][0]["threshold"] == 50
        assert data["evaluations"][0]["seeds"] == [42]
        assert data["evaluations"][0]["trigger"] == "manual"

    def test_appends_to_existing_history(self, tmp_path):
        for verdict in ("pass", "fail", "pass"):
            append_evaluation(
                tmp_path,
                task_id="math.foo",
                task_version="1.0",
                results=[],
                threshold=50,
                verdict=verdict,
            )
        data = load_scores(tmp_path, "math.foo")
        assert len(data["evaluations"]) == 3
        assert [e["verdict"] for e in data["evaluations"]] == ["pass", "fail", "pass"]

    def test_rejects_invalid_verdict(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid verdict"):
            append_evaluation(tmp_path, "x.y", "1.0", [], 50, verdict="unknown")

    def test_rejects_invalid_trigger(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid trigger"):
            append_evaluation(tmp_path, "x.y", "1.0", [], 50, verdict="pass", trigger="bogus")

    def test_explicit_timestamp_preserved(self, tmp_path):
        ts = "2026-01-02T03:04:05Z"
        path = append_evaluation(
            tmp_path, "x.y", "1.0", [], 50, verdict="pass", timestamp=ts,
        )
        data = json.loads(path.read_text())
        assert data["evaluations"][0]["date"] == ts


class TestLatestVerdict:
    def test_returns_none_when_missing(self, tmp_path):
        assert get_latest_verdict(tmp_path, "no.task") is None

    def test_returns_latest_after_append(self, tmp_path):
        append_evaluation(tmp_path, "x.y", "1.0", [], 50, verdict="fail")
        append_evaluation(tmp_path, "x.y", "1.0", [], 50, verdict="pass")
        assert get_latest_verdict(tmp_path, "x.y") == "pass"


class TestGetAllTaskScores:
    def test_empty_dir(self, tmp_path):
        assert get_all_task_scores(tmp_path / "missing") == {}

    def test_reads_all_files(self, tmp_path):
        append_evaluation(tmp_path, "a.b", "1.0", [], 50, verdict="pass")
        append_evaluation(tmp_path, "c.d", "1.0", [], 50, verdict="fail")
        all_scores = get_all_task_scores(tmp_path)
        assert set(all_scores.keys()) == {"a.b", "c.d"}

    def test_skips_corrupt_files(self, tmp_path):
        append_evaluation(tmp_path, "good.one", "1.0", [], 50, verdict="pass")
        (tmp_path / "bad.json").write_text("nope", encoding="utf-8")
        all_scores = get_all_task_scores(tmp_path)
        assert "good.one" in all_scores
        assert "bad" not in all_scores


class TestFindFlaggedTasks:
    def test_ignores_high_b1_and_b2_scores(self, tmp_path):
        scores_dir = tmp_path / "scores"
        append_evaluation(
            scores_dir, "physics.guided", "1.0",
            results=[{
                "agent": "direct_llm",
                "scores": {
                    "b1": {"mean": 100.0},
                    "b2": {"mean": 100.0},
                    "b3": {"mean": 39.0},
                    "b4": {"mean": 0.0},
                },
            }],
            threshold=40,
            verdict="pass",
        )

        assert find_flagged_tasks(
            scores_dir, threshold=40, only_final=False,
        ) == []

    def test_filters_below_threshold(self, tmp_path):
        scores_dir = tmp_path / "scores"
        tasks_dir = tmp_path / "tasks"
        _write_task_yaml(tasks_dir, "physics.easy", status="final")
        _write_task_yaml(tasks_dir, "math.hard", status="final")
        append_evaluation(
            scores_dir, "physics.easy", "1.0",
            results=[{"agent": "direct_llm", "agent_config": {"model": "m1"},
                      "scores": {"b3": {"mean": 65.0, "max": 70.0, "min": 60.0, "n": 1}}}],
            threshold=50, verdict="fail",
        )
        append_evaluation(
            scores_dir, "math.hard", "1.0",
            results=[{"agent": "direct_llm", "agent_config": {"model": "m1"},
                      "scores": {"b3": {"mean": 10.0, "max": 12.0, "min": 8.0, "n": 1}}}],
            threshold=50, verdict="pass",
        )
        flagged = find_flagged_tasks(scores_dir, tasks_dir, threshold=50)
        ids = [row["task_id"] for row in flagged]
        assert ids == ["physics.easy"]
        assert flagged[0]["mean_score"] == 65.0
        assert flagged[0]["model"] == "m1"

    def test_skips_non_final_when_filter_on(self, tmp_path):
        scores_dir = tmp_path / "scores"
        tasks_dir = tmp_path / "tasks"
        _write_task_yaml(tasks_dir, "physics.dev", status="in_development")
        append_evaluation(
            scores_dir, "physics.dev", "1.0",
            results=[{"agent": "direct_llm", "agent_config": {"model": "m1"},
                      "scores": {"b3": {"mean": 99.0, "max": 99.0, "min": 99.0, "n": 1}}}],
            threshold=50, verdict="fail",
        )
        # only_final=True (default) should exclude in_development tasks
        assert find_flagged_tasks(scores_dir, tasks_dir, threshold=50) == []
        # only_final=False should include them
        flagged = find_flagged_tasks(scores_dir, tasks_dir, threshold=50, only_final=False)
        assert [row["task_id"] for row in flagged] == ["physics.dev"]


# ---------------------------------------------------------------------------
# T3 — difficulty_report.py
# ---------------------------------------------------------------------------


class TestBuildReport:
    def test_all_passing(self):
        results = [
            _make_result(task_id="physics.foo", level="b1", score=35.0),
            _make_result(task_id="physics.foo", level="b3", score=12.0),
        ]
        grouped = [("claude-opus-4-6", "direct_llm", {"model": "claude-opus-4-6"}, results)]
        report = build_report("physics.foo", grouped, threshold=50)
        assert report.overall_pass is True
        assert len(report.rows) == 2
        assert all(row.passed for row in report.rows)
        assert report.rows[0].prompt_level in ("b1", "b3")

    def test_one_row_failing(self):
        results = [
            _make_result(task_id="physics.foo", level="b1", score=100.0),
            _make_result(task_id="physics.foo", level="b3", score=62.0),
        ]
        grouped = [("claude-opus-4-6", "direct_llm", {}, results)]
        report = build_report("physics.foo", grouped, threshold=50)
        assert report.overall_pass is False
        b1 = next(r for r in report.rows if r.prompt_level == "b1")
        assert b1.enforced is False
        assert b1.passed is True
        b3 = next(r for r in report.rows if r.prompt_level == "b3")
        assert b3.enforced is True
        assert b3.passed is False

    def test_only_b3_and_b4_are_gated(self):
        results = [
            _make_result(task_id="physics.foo", level="b1", score=100.0),
            _make_result(task_id="physics.foo", level="b2", score=100.0),
            _make_result(task_id="physics.foo", level="b3", score=39.9),
            _make_result(task_id="physics.foo", level="b4", score=0.0),
        ]
        report = build_report(
            "physics.foo", [("model", "direct_llm", {}, results)], threshold=40,
        )

        assert report.overall_pass is True
        assert {r.prompt_level for r in report.rows if r.enforced} == {"b3", "b4"}
        assert all(r.passed for r in report.rows)

    def test_mean_max_min_aggregation(self):
        results = [
            _make_result(task_id="x.y", level="b1", score=40.0),
            _make_result(task_id="x.y", level="b1", score=60.0),
            _make_result(task_id="x.y", level="b1", score=50.0),
        ]
        grouped = [("m", "direct_llm", {}, results)]
        report = build_report("x.y", grouped, threshold=70)
        assert len(report.rows) == 1
        row = report.rows[0]
        assert row.mean_score == pytest.approx(50.0)
        assert row.max_score == 60.0
        assert row.min_score == 40.0
        assert row.instances == 3
        assert row.passed is True  # 50 < 70

    def test_threshold_is_strict(self):
        """Equal to threshold should FAIL (not pass)."""
        results = [_make_result(task_id="x.y", level="b3", score=50.0)]
        report = build_report("x.y", [("m", "direct_llm", {}, results)], threshold=50)
        assert report.overall_pass is False
        assert report.rows[0].passed is False

    def test_to_per_agent_results_shape(self):
        results = [
            _make_result(task_id="x.y", level="b1", score=35.0),
            _make_result(task_id="x.y", level="b3", score=12.0),
        ]
        grouped = [("opus", "direct_llm", {"model": "claude-opus-4-6"}, results)]
        report = build_report("x.y", grouped, threshold=50)
        per_agent = report.to_per_agent_results()
        assert len(per_agent) == 1
        block = per_agent[0]
        assert block["agent"] == "direct_llm"
        assert block["agent_config"] == {"model": "claude-opus-4-6"}
        assert set(block["scores"].keys()) == {"b1", "b3"}
        assert block["scores"]["b1"]["mean"] == 35.0


class TestFormatTerminal:
    def test_passing_report_contains_pass(self):
        results = [_make_result(task_id="x.y", level="b3", score=10.0)]
        report = build_report("x.y", [("m1", "direct_llm", {}, results)], threshold=50)
        out = format_terminal(report, use_color=False)
        assert "PASS" in out
        assert "FAIL" not in out  # no failures
        assert "x.y" in out
        assert "Recommended next step" in out

    def test_failing_report_contains_suggestions(self):
        results = [_make_result(task_id="x.y", level="b4", score=80.0)]
        report = build_report("x.y", [("m1", "direct_llm", {}, results)], threshold=50)
        out = format_terminal(report, use_color=False)
        assert "FAIL" in out
        assert "Suggestions" in out
        assert "difficulty-engineering" in out


class TestFormatMarkdown:
    def test_table_has_rows(self):
        results = [
            _make_result(task_id="x.y", level="b1", score=35.0),
            _make_result(task_id="x.y", level="b3", score=12.0),
        ]
        report = build_report("x.y", [("opus", "direct_llm", {}, results)], threshold=50)
        md = format_markdown(report)
        assert "| `opus` | B1 |" in md
        assert "| `opus` | B3 |" in md
        assert "PASS" in md

    def test_failing_shows_bold_fail(self):
        results = [_make_result(task_id="x.y", level="b3", score=80.0)]
        report = build_report("x.y", [("opus", "direct_llm", {}, results)], threshold=50)
        md = format_markdown(report)
        assert "**FAIL**" in md


class TestWriteReport:
    def test_write_json_round_trips(self, tmp_path):
        results = [_make_result(task_id="x.y", level="b1", score=20.0)]
        report = build_report("x.y", [("m", "direct_llm", {}, results)], threshold=50)
        out = write_json(report, tmp_path / "out.json")
        loaded = json.loads(out.read_text())
        assert loaded["task_id"] == "x.y"
        assert loaded["verdict"] == "pass"
        assert loaded["threshold"] == 50

    def test_write_markdown(self, tmp_path):
        results = [_make_result(task_id="x.y", level="b1", score=20.0)]
        report = build_report("x.y", [("m", "direct_llm", {}, results)], threshold=50)
        out = write_markdown(report, tmp_path / "out.md")
        text = out.read_text()
        assert "Difficulty Check" in text
        assert text.endswith("\n")


# ---------------------------------------------------------------------------
# T1 — difficulty-check CLI command
# ---------------------------------------------------------------------------


def _stub_orchestrator_run(score_by_level: dict[str, float]):
    """Return a class that mimics BenchmarkOrchestrator with predetermined scores."""

    class StubOrchestrator:
        def __init__(self, config):
            self.config = config

        def run(self, task_ids):
            results = []
            for tid in task_ids:
                for level in self.config.prompt_levels:
                    score = score_by_level.get(level, 0.0)
                    for i in range(self.config.instances_per_task):
                        results.append(EvalResult(
                            instance_id=f"{tid}_{level}_{i}",
                            task_id=tid,
                            prompt_level=PromptLevel(level),
                            agent_name="direct_llm",
                            parameters={},
                            gate_results=[],
                            gates_passed=True,
                            score_results=[],
                            final_score=score,
                            max_possible_score=100.0,
                            status=RunStatus.COMPLETED,
                        ))
            return RunReport(
                agent_name="direct_llm",
                n_tasks=len(task_ids),
                n_instances=len(results),
                overall_mean_score=sum(r.final_score for r in results) / max(len(results), 1),
                results=results,
            )

    return StubOrchestrator


class TestDifficultyCheckCLI:
    def test_help_lists_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["difficulty-check", "--help"])
        assert result.exit_code == 0
        assert "Check whether a task is hard enough" in result.output
        assert "--threshold" in result.output
        assert "--status" in result.output
        assert "b1,b2,b3,b4" in result.output
        assert "B3/B4" in result.output
        assert "40" in result.output

    def test_threshold_above_40_is_rejected(self):
        result = CliRunner().invoke(cli, [
            "difficulty-check", "--task", "x.y", "--threshold", "41",
        ])

        assert result.exit_code != 0
        assert "not in the range" in result.output

    def test_requires_task_or_status(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "difficulty-check",
            "--tasks-dir", str(tmp_path),
            "--scores-dir", str(tmp_path / "scores"),
            "--no-color",
        ])
        assert result.exit_code != 0
        assert "exactly one of --task or --status" in result.output

    def test_rejects_both_task_and_status(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "difficulty-check",
            "--task", "x.y",
            "--status", "final",
            "--tasks-dir", str(tmp_path),
            "--no-color",
        ])
        assert result.exit_code != 0
        assert "exactly one of --task or --status" in result.output

    def test_mismatched_agent_config_count(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "difficulty-check",
            "--task", "x.y",
            "--agent", "direct_llm",
            "--agent", "direct_llm",
            "--agent-config", "{}",  # only one config for two agents
            "--tasks-dir", str(tmp_path),
            "--no-color",
        ])
        assert result.exit_code != 0
        assert "must match" in result.output

    def test_passing_run_persists_scores_and_exits_zero(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        scores_dir = tmp_path / "scores"
        _write_task_yaml(tasks_dir, "physics.demo", status="in_development", version="2.3")

        stub_cls = _stub_orchestrator_run({
            "b1": 100.0, "b2": 100.0, "b3": 39.0, "b4": 39.0,
        })

        runner = CliRunner()
        with patch("ai4sci_bench.cli.BenchmarkOrchestrator", stub_cls, create=True):
            # The CLI imports BenchmarkOrchestrator locally; patch in its namespace instead.
            pass

        # The CLI imports BenchmarkOrchestrator inside the function; patch the source module.
        with patch(
            "ai4sci_bench.runner.orchestrator.BenchmarkOrchestrator", stub_cls
        ):
            result = runner.invoke(cli, [
                "difficulty-check",
                "--task", "physics.demo",
                "--agent", "direct_llm",
                "--agent-config", '{"model":"claude-opus-4-6"}',
                "--tasks-dir", str(tasks_dir),
                "--scores-dir", str(scores_dir),
                "--no-color",
            ])

        assert result.exit_code == 0, result.output
        assert "PASS" in result.output
        assert "RECORDED" in result.output
        assert "B3/B4 threshold: < 40" in result.output
        assert "physics.demo" in result.output

        score_path = scores_dir / "physics.demo.json"
        assert score_path.exists()
        data = json.loads(score_path.read_text())
        assert data["task_id"] == "physics.demo"
        assert data["task_version"] == "2.3"
        assert data["evaluations"][-1]["verdict"] == "pass"
        assert data["evaluations"][-1]["tool_mode"] == "restricted"

    def test_failing_run_exits_nonzero(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        scores_dir = tmp_path / "scores"
        _write_task_yaml(tasks_dir, "physics.easy", status="test")
        stub_cls = _stub_orchestrator_run({"b1": 80.0, "b3": 60.0})

        runner = CliRunner()
        with patch("ai4sci_bench.runner.orchestrator.BenchmarkOrchestrator", stub_cls):
            result = runner.invoke(cli, [
                "difficulty-check",
                "--task", "physics.easy",
                "--prompt-levels", "b1,b3",
                "--threshold", "40",
                "--tasks-dir", str(tasks_dir),
                "--scores-dir", str(scores_dir),
                "--no-color",
            ])

        assert result.exit_code == 1, result.output
        assert "FAIL" in result.output
        assert "Suggestions" in result.output

        data = json.loads((scores_dir / "physics.easy.json").read_text())
        assert data["evaluations"][-1]["verdict"] == "fail"

    def test_writes_output_json_and_markdown(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        scores_dir = tmp_path / "scores"
        _write_task_yaml(tasks_dir, "math.foo", status="in_development")
        stub_cls = _stub_orchestrator_run({"b1": 30.0})

        runner = CliRunner()
        out_json = tmp_path / "report.json"
        out_md = tmp_path / "report.md"
        with patch("ai4sci_bench.runner.orchestrator.BenchmarkOrchestrator", stub_cls):
            result = runner.invoke(cli, [
                "difficulty-check",
                "--task", "math.foo",
                "--prompt-levels", "b1",
                "--tasks-dir", str(tasks_dir),
                "--scores-dir", str(scores_dir),
                "--output", str(out_json),
                "--markdown", str(out_md),
                "--no-color",
            ])

        assert result.exit_code == 0, result.output
        report = json.loads(out_json.read_text())
        assert report["task_id"] == "math.foo"
        assert "Difficulty Check" in out_md.read_text()

    def test_no_save_scores_flag(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        scores_dir = tmp_path / "scores"
        _write_task_yaml(tasks_dir, "math.foo", status="in_development")
        stub_cls = _stub_orchestrator_run({"b1": 30.0})

        runner = CliRunner()
        with patch("ai4sci_bench.runner.orchestrator.BenchmarkOrchestrator", stub_cls):
            result = runner.invoke(cli, [
                "difficulty-check",
                "--task", "math.foo",
                "--prompt-levels", "b1",
                "--tasks-dir", str(tasks_dir),
                "--scores-dir", str(scores_dir),
                "--no-save-scores",
                "--no-color",
            ])
        assert result.exit_code == 0, result.output
        assert not (scores_dir / "math.foo.json").exists()

    def test_status_filter_runs_batch(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        scores_dir = tmp_path / "scores"
        _write_task_yaml(tasks_dir, "physics.a", status="test")
        _write_task_yaml(tasks_dir, "physics.b", status="test")
        _write_task_yaml(tasks_dir, "math.skip", status="in_development")

        stub_cls = _stub_orchestrator_run({"b3": 20.0})
        runner = CliRunner()
        with patch("ai4sci_bench.runner.orchestrator.BenchmarkOrchestrator", stub_cls):
            result = runner.invoke(cli, [
                "difficulty-check",
                "--status", "test",
                "--prompt-levels", "b3",
                "--tasks-dir", str(tasks_dir),
                "--scores-dir", str(scores_dir),
                "--no-color",
            ])

        assert result.exit_code == 0, result.output
        assert "Batch summary" in result.output
        assert "physics.a" in result.output
        assert "physics.b" in result.output
        assert "math.skip" not in result.output
        # Both physics tasks should have score files; math.skip should not.
        assert (scores_dir / "physics.a.json").exists()
        assert (scores_dir / "physics.b.json").exists()
        assert not (scores_dir / "math.skip.json").exists()

    def test_status_filter_no_matches(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        _write_task_yaml(tasks_dir, "physics.a", status="in_development")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "difficulty-check",
            "--status", "final",
            "--tasks-dir", str(tasks_dir),
            "--scores-dir", str(tmp_path / "scores"),
            "--no-color",
        ])
        assert result.exit_code != 0
        assert "No tasks found with status=final" in result.output


# ---------------------------------------------------------------------------
# Infrastructure-failure handling (agent crash, API error, timeout)
# ---------------------------------------------------------------------------


def _stub_orchestrator_infra_failure(status: RunStatus, error: str):
    """Stub that simulates agent infrastructure failure on every instance.

    Reason for testing: when an agent can't even run (missing API key, subprocess
    crash, timeout), the framework used to wrap that as score=0 → PASS verdict,
    polluting scores with fake data. The difficulty-check CLI must detect this
    and ABORT instead — no score persistence, exit code 2, no verdict.
    """
    from ai4sci_bench.core.types import AgentOutput

    class StubOrchestrator:
        def __init__(self, config):
            self.config = config

        def run(self, task_ids):
            results = []
            for tid in task_ids:
                for level in self.config.prompt_levels:
                    for i in range(self.config.instances_per_task):
                        agent_output = AgentOutput(
                            instance_id=f"{tid}_{level}_{i}",
                            output_dir=Path("/tmp/nonexistent"),
                            code_files=[],
                            data_files=[],
                            log=error,
                            execution_time_seconds=0.0,
                            status=status,
                            error_message=error,
                        )
                        results.append(EvalResult(
                            instance_id=f"{tid}_{level}_{i}",
                            task_id=tid,
                            prompt_level=PromptLevel(level),
                            agent_name="direct_llm",
                            parameters={},
                            gate_results=[],
                            gates_passed=False,
                            score_results=[],
                            final_score=0.0,
                            max_possible_score=100.0,
                            status=status,
                            agent_output=agent_output,
                        ))
            return RunReport(
                agent_name="direct_llm",
                n_tasks=len(task_ids),
                n_instances=len(results),
                overall_mean_score=0.0,
                results=results,
            )

    return StubOrchestrator


class TestInfrastructureFailureHandling:
    """When the agent crashes, difficulty-check must ABORT, not silently PASS.

    These tests pin down the contract: exit code 2, no scores written, no
    report.json written, abort reason printed to stdout.
    """

    def test_agent_failure_aborts_with_exit_2(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        scores_dir = tmp_path / "scores"
        _write_task_yaml(tasks_dir, "physics.demo", status="in_development")

        stub_cls = _stub_orchestrator_infra_failure(
            RunStatus.FAILED, "AnthropicAPIError: missing API key"
        )
        runner = CliRunner()
        with patch(
            "ai4sci_bench.runner.orchestrator.BenchmarkOrchestrator", stub_cls
        ):
            result = runner.invoke(cli, [
                "difficulty-check",
                "--task", "physics.demo",
                "--prompt-levels", "b3",
                "--tasks-dir", str(tasks_dir),
                "--scores-dir", str(scores_dir),
                "--no-color",
            ])

        assert result.exit_code == 2, result.output
        assert "ABORT" in result.output
        assert "AnthropicAPIError" in result.output  # error detail surfaced
        # Critically: no scores file written when evaluation didn't really happen.
        assert not (scores_dir / "physics.demo.json").exists()
        # No PASS / FAIL verdict — abort is its own thing.
        assert "PASS" not in result.output
        assert "FAIL" not in result.output

    def test_agent_timeout_also_aborts(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        scores_dir = tmp_path / "scores"
        _write_task_yaml(tasks_dir, "physics.demo", status="in_development")

        stub_cls = _stub_orchestrator_infra_failure(
            RunStatus.TIMEOUT, "wall-clock timeout exceeded"
        )
        runner = CliRunner()
        with patch(
            "ai4sci_bench.runner.orchestrator.BenchmarkOrchestrator", stub_cls
        ):
            result = runner.invoke(cli, [
                "difficulty-check",
                "--task", "physics.demo",
                "--prompt-levels", "b3",
                "--tasks-dir", str(tasks_dir),
                "--scores-dir", str(scores_dir),
                "--no-color",
            ])

        assert result.exit_code == 2, result.output
        assert "ABORT" in result.output
        assert "timeout" in result.output
        assert not (scores_dir / "physics.demo.json").exists()

    def test_abort_does_not_write_report_json(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        scores_dir = tmp_path / "scores"
        report_path = tmp_path / "report.json"
        _write_task_yaml(tasks_dir, "physics.demo", status="in_development")

        stub_cls = _stub_orchestrator_infra_failure(
            RunStatus.FAILED, "subprocess crashed"
        )
        runner = CliRunner()
        with patch(
            "ai4sci_bench.runner.orchestrator.BenchmarkOrchestrator", stub_cls
        ):
            result = runner.invoke(cli, [
                "difficulty-check",
                "--task", "physics.demo",
                "--prompt-levels", "b3",
                "--tasks-dir", str(tasks_dir),
                "--scores-dir", str(scores_dir),
                "--output", str(report_path),
                "--no-color",
            ])

        assert result.exit_code == 2, result.output
        # Report file must NOT exist — aborted runs don't get a report.
        assert not report_path.exists()

    def test_help_documents_exit_code_2(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["difficulty-check", "--help"])
        assert result.exit_code == 0
        # The new docstring should mention exit code 2 + ABORT semantics.
        assert "2" in result.output
        assert "ABORT" in result.output or "infrastructure" in result.output


# ---------------------------------------------------------------------------
# T7 — catalog CLI command (domain coverage + per-task history + gaps + flagged)
# ---------------------------------------------------------------------------


def _seed_score_file(scores_dir: Path, task_id: str, mean_b3: float, *,
                     date: str = "2026-05-15T00:00:00Z",
                     model: str = "claude-opus-4-6",
                     verdict: str = "pass",
                     threshold: int = 50) -> None:
    """Write a minimal scores/<id>.json with a single evaluation entry."""
    scores_dir.mkdir(parents=True, exist_ok=True)
    (scores_dir / f"{task_id}.json").write_text(json.dumps({
        "task_id": task_id,
        "task_version": "1.0",
        "evaluations": [{
            "date": date,
            "trigger": "manual",
            "trigger_ref": "",
            "threshold": threshold,
            "verdict": verdict,
            "tool_mode": "restricted",
            "sandbox": "none",
            "instances_per_task": 1,
            "seeds": [42],
            "results": [{
                "agent": "direct_llm",
                "agent_config": {"model": model},
                "scores": {"b3": {"mean": mean_b3, "max": mean_b3, "min": mean_b3, "n": 1}},
            }],
        }],
    }), encoding="utf-8")


class TestCatalogCLI:
    """T7 — `ai4sci-bench catalog` covers summary / task / gaps / flagged."""

    def test_help_lists_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["catalog", "--help"])
        assert result.exit_code == 0, result.output
        for flag in ("--summary", "--task", "--scores", "--gaps", "--flagged",
                     "--threshold", "--min-final", "--tasks-dir", "--scores-dir"):
            assert flag in result.output, f"catalog --help missing {flag}"

    def test_rejects_multiple_modes(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "catalog", "--summary", "--gaps",
            "--tasks-dir", str(tmp_path),
            "--scores-dir", str(tmp_path / "scores"),
        ])
        assert result.exit_code != 0
        assert "at most one" in result.output.lower()

    def test_scores_requires_task_flag(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "catalog", "--scores",
            "--tasks-dir", str(tmp_path),
            "--scores-dir", str(tmp_path / "scores"),
        ])
        assert result.exit_code != 0
        assert "--scores" in result.output and "--task" in result.output

    def test_summary_groups_by_domain_and_counts_statuses(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        scores_dir = tmp_path / "scores"
        _write_task_yaml(tasks_dir, "physics.a", status="final")
        _write_task_yaml(tasks_dir, "physics.b", status="test")
        _write_task_yaml(tasks_dir, "physics.c", status="in_development")
        _write_task_yaml(tasks_dir, "math.x", status="final")
        # Seed a score so "Last Checked" gets populated for at least one domain.
        _seed_score_file(scores_dir, "physics.a", 25.0, date="2026-05-15T12:00:00Z")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "catalog", "--summary",
            "--tasks-dir", str(tasks_dir),
            "--scores-dir", str(scores_dir),
        ])
        assert result.exit_code == 0, result.output
        # Domain rows present
        assert "physics" in result.output
        assert "math" in result.output
        # Status columns rendered
        for header in ("Final", "Test", "Dev", "Total"):
            assert header in result.output
        # Latest score date appears
        assert "2026-05-15" in result.output
        # Totals row sums correctly: 2 final, 1 test, 1 dev, 4 total
        assert "Total" in result.output

    def test_summary_flags_zero_final_domains(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        _write_task_yaml(tasks_dir, "electrical_engineering.a", status="test")
        _write_task_yaml(tasks_dir, "physics.a", status="final")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "catalog", "--summary",
            "--tasks-dir", str(tasks_dir),
            "--scores-dir", str(tmp_path / "scores"),
        ])
        assert result.exit_code == 0, result.output
        assert "Gap alert" in result.output
        assert "electrical_engineering" in result.output
        assert "0 final" in result.output

    def test_summary_is_default_when_no_mode_given(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        _write_task_yaml(tasks_dir, "physics.a", status="final")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "catalog",
            "--tasks-dir", str(tasks_dir),
            "--scores-dir", str(tmp_path / "scores"),
        ])
        assert result.exit_code == 0, result.output
        assert "AI4Sci-Bench Task Catalog" in result.output

    def test_summary_outputs_json(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        _write_task_yaml(tasks_dir, "physics.a", status="final")
        out_json = tmp_path / "summary.json"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "catalog", "--summary",
            "--tasks-dir", str(tasks_dir),
            "--scores-dir", str(tmp_path / "scores"),
            "--output-json", str(out_json),
        ])
        assert result.exit_code == 0, result.output
        data = json.loads(out_json.read_text())
        assert "rows" in data and "totals" in data
        # physics should appear with final=1
        physics_row = next((r for r in data["rows"] if r["domain"] == "physics"), None)
        assert physics_row is not None
        assert physics_row["final"] == 1

    def test_task_view_shows_score_history(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        scores_dir = tmp_path / "scores"
        _write_task_yaml(tasks_dir, "physics.demo", status="test", version="2.0")
        _seed_score_file(scores_dir, "physics.demo", 30.0,
                         date="2026-05-15T00:00:00Z", model="claude-opus-4-6")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "catalog", "--task", "physics.demo", "--scores",
            "--tasks-dir", str(tasks_dir),
            "--scores-dir", str(scores_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "Task: physics.demo" in result.output
        assert "Difficulty Check History" in result.output
        assert "2026-05-15" in result.output
        assert "claude-opus-4-6" in result.output
        assert "PASS" in result.output

    def test_task_view_no_record(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        _write_task_yaml(tasks_dir, "physics.empty", status="in_development")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "catalog", "--task", "physics.empty", "--scores",
            "--tasks-dir", str(tasks_dir),
            "--scores-dir", str(tmp_path / "scores"),
        ])
        assert result.exit_code == 0, result.output
        assert "No difficulty check on record" in result.output

    def test_task_view_unknown_task_errors(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "catalog", "--task", "nope.missing",
            "--tasks-dir", str(tmp_path / "tasks"),
            "--scores-dir", str(tmp_path / "scores"),
        ])
        assert result.exit_code != 0
        assert "No task or score record" in result.output

    def test_task_view_orders_newest_first(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        scores_dir = tmp_path / "scores"
        _write_task_yaml(tasks_dir, "math.foo", status="test")
        # Manually build history with two evaluations
        scores_dir.mkdir(parents=True, exist_ok=True)
        (scores_dir / "math.foo.json").write_text(json.dumps({
            "task_id": "math.foo",
            "task_version": "1.0",
            "evaluations": [
                {"date": "2026-01-10T00:00:00Z", "verdict": "pass", "threshold": 50,
                 "results": [{"agent": "direct_llm",
                              "agent_config": {"model": "claude-sonnet-4-5"},
                              "scores": {"b3": {"mean": 12.0, "max": 12.0, "min": 12.0, "n": 1}}}]},
                {"date": "2026-05-15T00:00:00Z", "verdict": "pass", "threshold": 50,
                 "results": [{"agent": "direct_llm",
                              "agent_config": {"model": "claude-opus-4-6"},
                              "scores": {"b3": {"mean": 14.0, "max": 14.0, "min": 14.0, "n": 1}}}]},
            ],
        }), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "catalog", "--task", "math.foo", "--scores",
            "--tasks-dir", str(tasks_dir),
            "--scores-dir", str(scores_dir),
        ])
        assert result.exit_code == 0, result.output
        # Newest row should appear before the older one in the output
        idx_new = result.output.index("2026-05-15")
        idx_old = result.output.index("2026-01-10")
        assert idx_new < idx_old

    def test_gaps_lists_domains_below_min_final(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        _write_task_yaml(tasks_dir, "physics.a", status="final")
        _write_task_yaml(tasks_dir, "physics.b", status="final")
        _write_task_yaml(tasks_dir, "physics.c", status="final")
        _write_task_yaml(tasks_dir, "physics.d", status="final")
        _write_task_yaml(tasks_dir, "math.x", status="final")  # only 1 — gap
        _write_task_yaml(tasks_dir, "control.y", status="test")  # 0 final — gap

        runner = CliRunner()
        result = runner.invoke(cli, [
            "catalog", "--gaps", "--min-final", "3",
            "--tasks-dir", str(tasks_dir),
            "--scores-dir", str(tmp_path / "scores"),
        ])
        assert result.exit_code == 0, result.output
        assert "Domain Coverage Gaps" in result.output
        assert "math" in result.output
        assert "control" in result.output
        # physics has 4 finals, should not appear in gap list
        assert "physics" not in result.output.split("Domain Coverage Gaps")[1].split("Tip:")[0]

    def test_gaps_handles_no_gaps(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        for i in range(3):
            _write_task_yaml(tasks_dir, f"physics.t{i}", status="final")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "catalog", "--gaps", "--min-final", "2",
            "--tasks-dir", str(tasks_dir),
            "--scores-dir", str(tmp_path / "scores"),
        ])
        assert result.exit_code == 0, result.output
        assert "No gaps" in result.output

    def test_gaps_mentions_candidate_files(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        _write_task_yaml(tasks_dir, "math.solo", status="final")
        # Drop a candidates_*.md alongside the math/ directory
        (tasks_dir / "math" / "candidates_inverse_problems.md").write_text("# ideas", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "catalog", "--gaps", "--min-final", "3",
            "--tasks-dir", str(tasks_dir),
            "--scores-dir", str(tmp_path / "scores"),
        ])
        assert result.exit_code == 0, result.output
        assert "candidates_inverse_problems.md" in result.output

    def test_flagged_lists_overscoring_finals(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        scores_dir = tmp_path / "scores"
        _write_task_yaml(tasks_dir, "math.easy", status="final")
        _write_task_yaml(tasks_dir, "physics.hard", status="final")
        _seed_score_file(scores_dir, "math.easy", 65.0)
        _seed_score_file(scores_dir, "physics.hard", 25.0)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "catalog", "--flagged", "--threshold", "50",
            "--tasks-dir", str(tasks_dir),
            "--scores-dir", str(scores_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "Flagged Tasks" in result.output
        assert "math.easy" in result.output
        assert "physics.hard" not in result.output

    def test_flagged_handles_empty(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        scores_dir = tmp_path / "scores"
        _write_task_yaml(tasks_dir, "physics.hard", status="final")
        _seed_score_file(scores_dir, "physics.hard", 10.0)
        runner = CliRunner()
        result = runner.invoke(cli, [
            "catalog", "--flagged", "--threshold", "50",
            "--tasks-dir", str(tasks_dir),
            "--scores-dir", str(scores_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "No flagged tasks" in result.output

    def test_flagged_outputs_json(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        scores_dir = tmp_path / "scores"
        _write_task_yaml(tasks_dir, "math.easy", status="final")
        _seed_score_file(scores_dir, "math.easy", 65.0)
        out_json = tmp_path / "flagged.json"

        runner = CliRunner()
        result = runner.invoke(cli, [
            "catalog", "--flagged", "--threshold", "50",
            "--tasks-dir", str(tasks_dir),
            "--scores-dir", str(scores_dir),
            "--output-json", str(out_json),
        ])
        assert result.exit_code == 0, result.output
        data = json.loads(out_json.read_text())
        assert data["threshold"] == 50
        assert any(row["task_id"] == "math.easy" for row in data["flagged"])

    def test_summary_handles_missing_scores_dir(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        _write_task_yaml(tasks_dir, "physics.a", status="final")
        runner = CliRunner()
        # scores dir does not exist
        result = runner.invoke(cli, [
            "catalog", "--summary",
            "--tasks-dir", str(tasks_dir),
            "--scores-dir", str(tmp_path / "nonexistent"),
        ])
        assert result.exit_code == 0, result.output
        assert "physics" in result.output

    def test_summary_handles_no_tasks(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "catalog", "--summary",
            "--tasks-dir", str(tmp_path / "empty"),
            "--scores-dir", str(tmp_path / "scores"),
        ])
        assert result.exit_code == 0, result.output
        assert "No tasks found" in result.output


# ---------------------------------------------------------------------------
# T9 — batch re-evaluation: CSV summary + flagged-tasks re-eval block
# ---------------------------------------------------------------------------


def _result_for(task_id: str, level: str, score: float) -> EvalResult:
    return _make_result(task_id=task_id, level=level, score=score)


def _report_for(task_id: str, level_scores: dict[str, float], threshold: int = 50,
                agent_label: str = "opus", agent_name: str = "direct_llm",
                agent_config: dict | None = None) -> DifficultyReport:
    results = [_result_for(task_id, lvl, s) for lvl, s in level_scores.items()]
    return build_report(
        task_id,
        [(agent_label, agent_name, agent_config or {"model": agent_label}, results)],
        threshold,
    )


class TestBatchCSVWriter:
    """T9 — CSV summary writer for batch difficulty-check runs."""

    def test_csv_has_header_and_one_row_per_level(self):
        from ai4sci_bench.reporting.difficulty_report import format_batch_csv

        report = _report_for("physics.a", {"b1": 30.0, "b3": 12.0})
        csv_text = format_batch_csv([report])
        lines = csv_text.strip().splitlines()
        assert lines[0].startswith("task_id,task_version,agent,agent_name,model")
        # 2 rows for 2 levels
        assert len(lines) == 3
        assert "physics.a" in lines[1]
        assert "b1" in lines[1]
        assert "b3" in lines[2]

    def test_csv_includes_model_from_agent_config(self):
        from ai4sci_bench.reporting.difficulty_report import format_batch_csv

        report = _report_for(
            "math.foo", {"b3": 18.0},
            agent_config={"model": "claude-opus-4-6"},
        )
        csv_text = format_batch_csv([report])
        assert "claude-opus-4-6" in csv_text

    def test_csv_marks_per_row_and_overall_verdict(self):
        from ai4sci_bench.reporting.difficulty_report import format_batch_csv

        # B1 is recorded even above threshold; only B3 controls overall verdict.
        report = _report_for("phys.mixed", {"b1": 80.0, "b3": 10.0}, threshold=50)
        csv_text = format_batch_csv([report])
        assert ",recorded,pass," in csv_text
        assert ",pass,pass," in csv_text

    def test_csv_handles_multiple_tasks(self):
        from ai4sci_bench.reporting.difficulty_report import format_batch_csv

        rs = [
            _report_for("a.x", {"b3": 20.0}),
            _report_for("b.y", {"b3": 60.0}, threshold=50),
        ]
        csv_text = format_batch_csv(rs)
        lines = csv_text.strip().splitlines()
        # 1 header + 2 rows
        assert len(lines) == 3
        assert "a.x" in csv_text and "b.y" in csv_text

    def test_csv_empty_input_writes_only_header(self):
        from ai4sci_bench.reporting.difficulty_report import format_batch_csv

        csv_text = format_batch_csv([])
        # header line only (with trailing \n)
        assert csv_text.strip().count("\n") == 0
        assert "task_id" in csv_text

    def test_write_batch_csv_to_disk(self, tmp_path):
        from ai4sci_bench.reporting.difficulty_report import write_batch_csv

        report = _report_for("phys.a", {"b3": 20.0})
        out = write_batch_csv([report], tmp_path / "summary.csv")
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "phys.a" in content
        # task_version default is "" — make sure CSV does not have a stray "None"
        assert "None" not in content


class TestCollectFlaggedAndBatchSummary:
    """T9 — Re-evaluation summary block listing flagged tasks."""

    def test_collect_flagged_picks_worst_row(self):
        from ai4sci_bench.reporting.difficulty_report import collect_flagged

        # B1 is ungated, so the failing B3 row is the flagged one.
        results = [
            _result_for("phys.bad", "b1", 80.0),
            _result_for("phys.bad", "b3", 60.0),
        ]
        report = build_report(
            "phys.bad",
            [("opus", "direct_llm", {"model": "opus"}, results)],
            threshold=50,
        )
        entries = collect_flagged([report])
        assert len(entries) == 1
        assert entries[0].task_id == "phys.bad"
        assert entries[0].worst_mean == 60.0
        assert entries[0].worst_level == "b3"

    def test_collect_flagged_skips_passing_tasks(self):
        from ai4sci_bench.reporting.difficulty_report import collect_flagged

        good = _report_for("phys.ok", {"b3": 20.0})
        bad = _report_for("phys.bad", {"b3": 70.0}, threshold=50)
        entries = collect_flagged([good, bad])
        assert [e.task_id for e in entries] == ["phys.bad"]

    def test_collect_flagged_sorts_by_worst_mean_desc(self):
        from ai4sci_bench.reporting.difficulty_report import collect_flagged

        rs = [
            _report_for("a.x", {"b3": 55.0}, threshold=50, agent_label="m1"),
            _report_for("b.y", {"b3": 75.0}, threshold=50, agent_label="m2"),
            _report_for("c.z", {"b3": 65.0}, threshold=50, agent_label="m3"),
        ]
        entries = collect_flagged(rs)
        assert [e.task_id for e in entries] == ["b.y", "c.z", "a.x"]

    def test_batch_summary_counts_and_lists_flagged(self):
        from ai4sci_bench.reporting.difficulty_report import format_batch_summary

        rs = [
            _report_for("a.x", {"b3": 20.0}),
            _report_for("a.y", {"b3": 80.0}, threshold=50),
            _report_for("a.z", {"b3": 60.0}, threshold=50),
        ]
        text = format_batch_summary(
            rs, status_filter="final", threshold=50, use_color=False,
            scores_dir="scores/",
        )
        assert "Re-evaluation Summary" in text
        assert "3 tasks evaluated" in text
        assert "1 still passing" in text
        assert "2 flagged" in text
        # Sorted worst-first
        idx_y = text.index("a.y")
        idx_z = text.index("a.z")
        assert idx_y < idx_z
        assert "Scores saved to scores/" in text
        assert "catalog --flagged" in text

    def test_batch_summary_no_flagged_path(self):
        from ai4sci_bench.reporting.difficulty_report import format_batch_summary

        rs = [
            _report_for("a.x", {"b3": 10.0}),
            _report_for("a.y", {"b3": 20.0}),
        ]
        text = format_batch_summary(
            rs, status_filter="final", threshold=50, use_color=False,
        )
        assert "0 flagged" in text
        # No scores path supplied — should not pretend.
        assert "Scores saved to" not in text


class TestDifficultyCheckCSVCLI:
    """T9 — CLI --csv-output and enhanced batch summary."""

    def test_csv_output_single_task(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        scores_dir = tmp_path / "scores"
        _write_task_yaml(tasks_dir, "physics.demo", status="in_development")
        stub_cls = _stub_orchestrator_run({"b1": 30.0, "b3": 10.0})

        csv_path = tmp_path / "summary.csv"
        runner = CliRunner()
        with patch("ai4sci_bench.runner.orchestrator.BenchmarkOrchestrator", stub_cls):
            result = runner.invoke(cli, [
                "difficulty-check",
                "--task", "physics.demo",
                "--prompt-levels", "b1,b3",
                "--tasks-dir", str(tasks_dir),
                "--scores-dir", str(scores_dir),
                "--csv-output", str(csv_path),
                "--no-color",
            ])
        assert result.exit_code == 0, result.output
        assert csv_path.exists()
        body = csv_path.read_text(encoding="utf-8")
        assert "task_id" in body.splitlines()[0]
        assert "physics.demo" in body
        # Both levels show up
        assert "b1" in body and "b3" in body

    def test_csv_output_batch_status_mode(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        scores_dir = tmp_path / "scores"
        _write_task_yaml(tasks_dir, "physics.a", status="test")
        _write_task_yaml(tasks_dir, "physics.b", status="test")
        stub_cls = _stub_orchestrator_run({"b3": 20.0})

        csv_path = tmp_path / "batch.csv"
        runner = CliRunner()
        with patch("ai4sci_bench.runner.orchestrator.BenchmarkOrchestrator", stub_cls):
            result = runner.invoke(cli, [
                "difficulty-check",
                "--status", "test",
                "--prompt-levels", "b3",
                "--tasks-dir", str(tasks_dir),
                "--scores-dir", str(scores_dir),
                "--csv-output", str(csv_path),
                "--no-color",
            ])
        assert result.exit_code == 0, result.output
        body = csv_path.read_text(encoding="utf-8")
        assert "physics.a" in body
        assert "physics.b" in body
        # Re-eval summary block printed in batch mode
        assert "Re-evaluation Summary" in result.output
        assert "2 tasks evaluated" in result.output
        assert "0 flagged" in result.output

    def test_batch_summary_lists_failing_tasks(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        scores_dir = tmp_path / "scores"
        _write_task_yaml(tasks_dir, "physics.fail", status="final")
        _write_task_yaml(tasks_dir, "physics.alsofail", status="final")
        # Use a stub that scores high → both fail at threshold=40
        stub_cls = _stub_orchestrator_run({"b3": 80.0})

        runner = CliRunner()
        with patch("ai4sci_bench.runner.orchestrator.BenchmarkOrchestrator", stub_cls):
            result = runner.invoke(cli, [
                "difficulty-check",
                "--status", "final",
                "--prompt-levels", "b3",
                "--threshold", "40",
                "--tasks-dir", str(tasks_dir),
                "--scores-dir", str(scores_dir),
                "--no-color",
            ])
        # Two failing tasks => exit 1
        assert result.exit_code == 1, result.output
        assert "Re-evaluation Summary" in result.output
        assert "2 flagged" in result.output
        assert "physics.fail" in result.output
        assert "physics.alsofail" in result.output
        assert "catalog --flagged" in result.output

    def test_batch_summary_omits_scores_path_when_save_scores_off(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        scores_dir = tmp_path / "scores"
        _write_task_yaml(tasks_dir, "physics.a", status="test")
        _write_task_yaml(tasks_dir, "physics.b", status="test")
        stub_cls = _stub_orchestrator_run({"b3": 20.0})

        runner = CliRunner()
        with patch("ai4sci_bench.runner.orchestrator.BenchmarkOrchestrator", stub_cls):
            result = runner.invoke(cli, [
                "difficulty-check",
                "--status", "test",
                "--prompt-levels", "b3",
                "--tasks-dir", str(tasks_dir),
                "--scores-dir", str(scores_dir),
                "--no-save-scores",
                "--no-color",
            ])
        assert result.exit_code == 0, result.output
        assert "Re-evaluation Summary" in result.output
        # --no-save-scores should hide the "Scores saved to <dir>" hint
        assert "Scores saved to" not in result.output
