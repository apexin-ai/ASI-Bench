import csv
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from ai4sci_bench.cli import cli
from ai4sci_bench.reporting.batch_records import _write_text_atomic, write_batch_records


def _make_eval_result_json(
    instance_id="inst1",
    task_id="physics.test_task",
    prompt_level="b1",
    agent_name="TestAgent",
    final_score=50.0,
    max_possible_score=100.0,
    attempt=1,
    status="completed",
    execution_time_seconds=10.0,
    **overrides,
):
    data = {
        "instance_id": instance_id,
        "task_id": task_id,
        "attempt": attempt,
        "prompt_level": prompt_level,
        "agent_name": agent_name,
        "parameters": {},
        "gates_passed": True,
        "hard_gates_passed": True,
        "soft_gate_failures": 0,
        "gate_results": [],
        "score_results": [],
        "final_score": final_score,
        "max_possible_score": max_possible_score,
        "execution_time_seconds": execution_time_seconds,
        "status": status,
    }
    data.update(overrides)
    return data


def _as_string_row(row: dict[str, object]) -> dict[str, str]:
    return {
        key: value if isinstance(value, str) else str(value)
        for key, value in row.items()
    }


class TestBatchRecords:
    def test_run_can_write_batch_records_to_shared_root(self, sample_task_dir, tmp_dir):
        tasks_dir = sample_task_dir.parent.parent
        shared_root = tmp_dir / "shared_results"
        agent_output = shared_root / "codex_cli"
        runner = CliRunner()

        result = runner.invoke(cli, [
            "run",
            "--tasks", "physics.test_task",
            "--tasks-dir", str(tasks_dir),
            "--output-dir", str(agent_output),
            "--instances-per-task", "1",
            "--seed", "42",
            "--prompt-levels", "b2",
            "--include-test",
            "--write-batch-records",
            "--batch-records-root", str(shared_root),
        ])

        assert result.exit_code == 0
        assert "Batch record artifacts:" in result.output

        batch_records_dir = shared_root / "batch_records"
        overview_path = batch_records_dir / "batch_overview.csv"
        overview_json_path = batch_records_dir / "batch_overview.json"
        scoreboard_path = batch_records_dir / "task_scoreboard.csv"
        scoreboard_json_path = batch_records_dir / "task_scoreboard.json"
        task_level_long_path = batch_records_dir / "task_level_long.csv"
        task_level_long_json_path = batch_records_dir / "task_level_long.json"
        markdown_path = batch_records_dir / "batch_overview.md"
        assert overview_path.exists()
        assert overview_json_path.exists()
        assert scoreboard_path.exists()
        assert scoreboard_json_path.exists()
        assert task_level_long_path.exists()
        assert task_level_long_json_path.exists()
        assert markdown_path.exists()

        with overview_path.open(newline="", encoding="utf-8") as handle:
            overview_rows = list(csv.DictReader(handle))
        assert len(overview_rows) == 1
        assert overview_rows[0]["agent_label"] == "codex_cli"
        assert {"agent_name", "adapter_class", "model_name", "method_group"} <= set(overview_rows[0])
        assert overview_rows[0]["b1"] == ""
        assert overview_rows[0]["b2"] == overview_rows[0]["overall_mean_score"]
        assert overview_rows[0]["overall_mean_score"] == ""

        with scoreboard_path.open(newline="", encoding="utf-8") as handle:
            scoreboard_rows = list(csv.DictReader(handle))
        assert len(scoreboard_rows) == 1
        assert scoreboard_rows[0]["mean_score"] == ""
        assert scoreboard_rows[0]["min_score"] == ""
        assert scoreboard_rows[0]["max_score"] == ""

        with task_level_long_path.open(newline="", encoding="utf-8") as handle:
            level_rows = list(csv.DictReader(handle))
        assert len(level_rows) == 1
        assert level_rows[0]["mean_score"] == ""
        assert level_rows[0]["gates_pass_rate"] == ""

    def test_run_rejects_batch_records_root_without_flag(self, sample_task_dir, tmp_dir):
        tasks_dir = sample_task_dir.parent.parent
        shared_root = tmp_dir / "shared_results"
        runner = CliRunner()

        result = runner.invoke(cli, [
            "run",
            "--tasks", "physics.test_task",
            "--tasks-dir", str(tasks_dir),
            "--output-dir", str(shared_root / "codex_cli"),
            "--instances-per-task", "1",
            "--seed", "42",
            "--include-test",
            "--batch-records-root", str(shared_root),
        ])

        assert result.exit_code != 0
        assert "--batch-records-root requires --write-batch-records" in result.output

    def test_write_batch_records_reuses_report_dedupe_ignores_workspace_noise_and_tracks_levels(self, tmp_dir):
        results_dir = tmp_dir / "results"
        task_dir = results_dir / "gpt-5.4" / "physics.test_task"
        task_dir.mkdir(parents=True)
        (results_dir / "gpt-5.4" / "run_metadata.json").write_text(json.dumps({
            "agent_config": {
                "agent_name": "direct_llm",
                "adapter_class": "DirectLLMAdapter",
                "config": {"model": "openai/gpt-5.4"},
            }
        }), encoding="utf-8")

        (task_dir / "inst1__b1.json").write_text(json.dumps(
            _make_eval_result_json(
                agent_name="DirectLLMAdapter",
                final_score=20.0,
            )
        ))
        (task_dir / "inst1__b1__attempt2.json").write_text(json.dumps(
            _make_eval_result_json(
                agent_name="DirectLLMAdapter",
                attempt=2,
                final_score=80.0,
            )
        ))
        (task_dir / "inst1__b2.json").write_text(json.dumps(
            _make_eval_result_json(
                agent_name="DirectLLMAdapter",
                prompt_level="b2",
                final_score=60.0,
            )
        ))

        workspace_dir = results_dir / "gpt-5.4" / "instances" / "inst1" / "workspace_b1"
        workspace_dir.mkdir(parents=True)
        (workspace_dir / "task_info.json").write_text(json.dumps({
            "instance_id": "inst1",
            "prompt_level": "b1",
        }))

        paths = write_batch_records(results_dir)
        assert len(paths) == 7

        overview_path = results_dir / "batch_records" / "batch_overview.csv"
        overview_json_path = results_dir / "batch_records" / "batch_overview.json"
        with overview_path.open(newline="", encoding="utf-8") as handle:
            overview_rows = list(csv.DictReader(handle))
        assert len(overview_rows) == 1
        assert overview_rows[0]["agent_label"] == "gpt-5.4"
        assert overview_rows[0]["agent_name"] == "direct_llm"
        assert overview_rows[0]["adapter_class"] == "DirectLLMAdapter"
        assert overview_rows[0]["model_name"] == "openai/gpt-5.4"
        assert overview_rows[0]["method_group"] == "direct_llm"
        assert overview_rows[0]["n_instances"] == "2"
        assert overview_rows[0]["overall_mean_score"] == "70.0"
        assert overview_rows[0]["b1"] == "80.0"
        assert overview_rows[0]["b2"] == "60.0"
        assert overview_rows[0]["b3"] == ""
        overview_json_rows = json.loads(overview_json_path.read_text(encoding="utf-8"))
        assert [_as_string_row(row) for row in overview_json_rows] == overview_rows

        scoreboard_path = results_dir / "batch_records" / "task_scoreboard.csv"
        scoreboard_json_path = results_dir / "batch_records" / "task_scoreboard.json"
        with scoreboard_path.open(newline="", encoding="utf-8") as handle:
            scoreboard_rows = list(csv.DictReader(handle))
        assert len(scoreboard_rows) == 1
        assert scoreboard_rows[0]["method_group"] == "direct_llm"
        assert scoreboard_rows[0]["task_id"] == "physics.test_task"
        assert scoreboard_rows[0]["mean_score"] == "70.0"
        assert scoreboard_rows[0]["b1"] == "80.0"
        assert scoreboard_rows[0]["b2"] == "60.0"
        scoreboard_json_rows = json.loads(scoreboard_json_path.read_text(encoding="utf-8"))
        assert [_as_string_row(row) for row in scoreboard_json_rows] == scoreboard_rows

        task_level_long_path = results_dir / "batch_records" / "task_level_long.csv"
        task_level_long_json_path = results_dir / "batch_records" / "task_level_long.json"
        with task_level_long_path.open(newline="", encoding="utf-8") as handle:
            long_rows = list(csv.DictReader(handle))
        assert len(long_rows) == 2
        assert {(row["prompt_level"], row["mean_score"]) for row in long_rows} == {
            ("b1", "80.0"),
            ("b2", "60.0"),
        }
        assert {row["model_name"] for row in long_rows} == {"openai/gpt-5.4"}
        level_json_rows = json.loads(task_level_long_json_path.read_text(encoding="utf-8"))
        assert [_as_string_row(row) for row in level_json_rows] == long_rows

    def test_write_batch_records_falls_back_to_result_provenance_for_flat_results(self, tmp_dir):
        results_dir = tmp_dir / "results"
        task_dir = results_dir / "physics.test_task"
        task_dir.mkdir(parents=True)

        (task_dir / "inst1__b1_gpt.json").write_text(json.dumps(
            _make_eval_result_json(
                instance_id="inst1",
                agent_name="DirectLLMAdapter",
                final_score=40.0,
                provenance={
                    "agent": {
                        "agent_name": "direct_llm",
                        "adapter_class": "DirectLLMAdapter",
                        "config": {"model": "openai/gpt-5.4"},
                    }
                },
            )
        ))
        (task_dir / "inst1__b1_codex.json").write_text(json.dumps(
            _make_eval_result_json(
                instance_id="inst2",
                agent_name="CodexCLIAdapter",
                final_score=75.0,
                provenance={
                    "agent": {
                        "agent_name": "codex_cli",
                        "adapter_class": "CodexCLIAdapter",
                        "config": {"model": "gpt-5.4"},
                    }
                },
            )
        ))

        write_batch_records(results_dir)

        overview_path = results_dir / "batch_records" / "batch_overview.csv"
        with overview_path.open(newline="", encoding="utf-8") as handle:
            overview_rows = list(csv.DictReader(handle))

        assert len(overview_rows) == 2
        rows_by_label = {row["agent_label"]: row for row in overview_rows}
        assert rows_by_label["gpt-5.4"]["agent_name"] == "direct_llm"
        assert rows_by_label["gpt-5.4"]["method_group"] == "direct_llm"
        assert rows_by_label["gpt-5.4"]["model_name"] == "openai/gpt-5.4"
        assert rows_by_label["codex_cli_gpt-5.4"]["agent_name"] == "codex_cli"
        assert rows_by_label["codex_cli_gpt-5.4"]["adapter_class"] == "CodexCLIAdapter"
        assert rows_by_label["codex_cli_gpt-5.4"]["method_group"] == "codex_cli"
        assert rows_by_label["codex_cli_gpt-5.4"]["model_name"] == "gpt-5.4"

    def test_json_mirrors_are_machine_readable_row_mirrors(self, tmp_dir):
        results_dir = tmp_dir / "results"
        task_dir = results_dir / "gpt-5.4" / "physics.test_task"
        task_dir.mkdir(parents=True)
        (results_dir / "gpt-5.4" / "run_metadata.json").write_text(json.dumps({
            "agent_config": {
                "agent_name": "direct_llm",
                "adapter_class": "DirectLLMAdapter",
                "config": {"model": "openai/gpt-5.4"},
            }
        }), encoding="utf-8")
        (task_dir / "inst1__b2.json").write_text(json.dumps(
            _make_eval_result_json(
                agent_name="DirectLLMAdapter",
                prompt_level="b2",
                final_score=88.0,
            )
        ))

        write_batch_records(results_dir)

        batch_dir = results_dir / "batch_records"
        overview_json_rows = json.loads((batch_dir / "batch_overview.json").read_text(encoding="utf-8"))
        scoreboard_json_rows = json.loads((batch_dir / "task_scoreboard.json").read_text(encoding="utf-8"))
        level_json_rows = json.loads((batch_dir / "task_level_long.json").read_text(encoding="utf-8"))

        assert isinstance(overview_json_rows, list)
        assert isinstance(scoreboard_json_rows, list)
        assert isinstance(level_json_rows, list)
        assert overview_json_rows[0]["agent_label"] == "gpt-5.4"
        assert scoreboard_json_rows[0]["task_id"] == "physics.test_task"
        assert level_json_rows[0]["prompt_level"] == "b2"

    def test_atomic_write_preserves_existing_file_on_replace_failure(self, tmp_dir, monkeypatch):
        target = tmp_dir / "batch_records" / "batch_overview.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("old-content\n", encoding="utf-8")
        original_replace = Path.replace

        def _boom_replace(self, target_path):
            if self.name.startswith(".batch_overview.csv.tmp-"):
                raise OSError("simulated replace failure")
            return original_replace(self, target_path)

        monkeypatch.setattr(Path, "replace", _boom_replace)

        with pytest.raises(OSError, match="replace failure"):
            _write_text_atomic(target, "new-content\n")

        assert target.read_text(encoding="utf-8") == "old-content\n"
        assert sorted(path.name for path in target.parent.iterdir()) == ["batch_overview.csv"]
