"""Tests for the static task validator and CLI --static/--pre-submit modes."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from ai4sci_bench.validators.static_validator import (
    ValidationResult,
    detect_changed_tasks,
    validate_task_static,
)


@pytest.fixture
def task_scaffold(tmp_dir):
    """Create a valid task scaffold for testing."""
    task_dir = tmp_dir / "tasks" / "math" / "test_validator_task"
    task_dir.mkdir(parents=True)

    metadata = {
        "id": "math.test_validator_task",
        "name": "Test Validator Task",
        "version": "1.0",
        "status": "test",
        "domain": "math",
        "subdomain": "testing",
        "prompts": {
            "b1": "prompt_b1.md",
            "b2": "prompt_b2.md",
            "b3": "prompt_b3.md",
        },
        "evaluation": {
            "gates": [
                {
                    "scorer": "file_match",
                    "severity": "hard",
                    "config": {"checks": [{"file": "result.npy"}]},
                }
            ],
            "scoring": [
                {
                    "scorer": "numerical",
                    "weight": 100,
                    "config": {
                        "metric": "relative_l2",
                        "pred_file": "result.npy",
                        "ref_file": "result_ref.npy",
                    },
                }
            ],
        },
        "generation": {
            "script": "generate_gt.py",
            "mode": "infinite",
            "parameters": {"seed": {"type": "int", "range": [0, 999999], "default": 0}},
            "timeout_seconds": 60,
        },
    }

    (task_dir / "task.yaml").write_text(yaml.dump(metadata))
    (task_dir / "prompt_b1.md").write_text("Full algorithm description.")
    (task_dir / "prompt_b2.md").write_text("Method name and background.")
    (task_dir / "prompt_b3.md").write_text("Goal only.")
    (task_dir / "generate_gt.py").write_text("# placeholder\n")

    return task_dir, tmp_dir / "tasks"


def _split_task_yaml(task_dir: Path) -> None:
    """Convert a legacy test task to the scaffolded meta/eval layout."""
    task_yaml = task_dir / "task.yaml"
    metadata = yaml.safe_load(task_yaml.read_text())
    eval_data = {
        "task_id": metadata["id"],
        "evaluation": metadata.pop("evaluation"),
        "generation": metadata.pop("generation"),
    }
    (task_dir / "task_meta.yaml").write_text(yaml.dump(metadata))
    (task_dir / "task_eval.yaml").write_text(yaml.dump(eval_data))
    task_yaml.unlink()


class TestValidationResult:
    def test_passed_when_no_errors(self):
        r = ValidationResult(task_dir="/tmp/test")
        assert r.passed is True

    def test_failed_when_errors(self):
        r = ValidationResult(task_dir="/tmp/test", errors=["something wrong"])
        assert r.passed is False

    def test_to_dict(self):
        r = ValidationResult(
            task_dir="/tmp/test",
            errors=["err1"],
            warnings=["warn1"],
        )
        d = r.to_dict()
        assert d["task_dir"] == "/tmp/test"
        assert d["passed"] is False
        assert d["errors"] == ["err1"]
        assert d["warnings"] == ["warn1"]


class TestStaticValidation:
    def test_valid_task_passes(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert result.passed, f"Expected pass but got errors: {result.errors}"

    def test_missing_task_yaml(self, tmp_dir):
        empty_dir = tmp_dir / "empty"
        empty_dir.mkdir()
        result = validate_task_static(empty_dir)
        assert not result.passed
        assert any("task.yaml not found" in e for e in result.errors)

    def test_valid_split_task_passes(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        _split_task_yaml(task_dir)

        result = validate_task_static(task_dir, tasks_root=tasks_root)

        assert result.passed, f"Expected pass but got errors: {result.errors}"

    def test_invalid_split_eval_yaml_reports_source(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        _split_task_yaml(task_dir)
        (task_dir / "task_eval.yaml").write_text("{{invalid yaml")

        result = validate_task_static(task_dir, tasks_root=tasks_root)

        assert not result.passed
        assert any("task_eval.yaml is not valid YAML" in e for e in result.errors)

    def test_invalid_yaml(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        (task_dir / "task.yaml").write_text("{{invalid yaml")
        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert not result.passed
        assert any("not valid YAML" in e for e in result.errors)

    def test_missing_required_fields(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        metadata = yaml.safe_load((task_dir / "task.yaml").read_text())
        del metadata["id"]
        del metadata["name"]
        (task_dir / "task.yaml").write_text(yaml.dump(metadata))

        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert not result.passed
        assert any("id" in e for e in result.errors)
        assert any("name" in e for e in result.errors)

    def test_missing_version(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        metadata = yaml.safe_load((task_dir / "task.yaml").read_text())
        del metadata["version"]
        (task_dir / "task.yaml").write_text(yaml.dump(metadata))

        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert not result.passed
        assert any("version" in e for e in result.errors)

    def test_invalid_version_format(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        metadata = yaml.safe_load((task_dir / "task.yaml").read_text())
        metadata["version"] = "abc"
        (task_dir / "task.yaml").write_text(yaml.dump(metadata))

        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert not result.passed
        assert any("Invalid version format" in e for e in result.errors)

    def test_invalid_status(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        metadata = yaml.safe_load((task_dir / "task.yaml").read_text())
        metadata["status"] = "bogus"
        (task_dir / "task.yaml").write_text(yaml.dump(metadata))

        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert not result.passed
        assert any("Invalid status" in e for e in result.errors)

    def test_sample_status_is_valid(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        metadata = yaml.safe_load((task_dir / "task.yaml").read_text())
        metadata["status"] = "sample"
        (task_dir / "task.yaml").write_text(yaml.dump(metadata))

        result = validate_task_static(task_dir, tasks_root=tasks_root)

        assert result.passed, result.errors

    def test_abandoned_without_reason_warns(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        metadata = yaml.safe_load((task_dir / "task.yaml").read_text())
        metadata["status"] = "abandoned"
        (task_dir / "task.yaml").write_text(yaml.dump(metadata))

        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert result.passed  # warnings don't fail
        assert any("abandoned" in w and "abandoned_reason" in w for w in result.warnings)

    def test_abandoned_skips_prompt_checks(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        metadata = yaml.safe_load((task_dir / "task.yaml").read_text())
        metadata["status"] = "abandoned"
        metadata["abandoned_reason"] = "no longer needed"
        (task_dir / "task.yaml").write_text(yaml.dump(metadata))
        # Delete prompts — should NOT cause errors for abandoned task
        for f in task_dir.glob("prompt_*.md"):
            f.unlink()

        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert result.passed

    def test_missing_prompt_file(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        (task_dir / "prompt_b2.md").unlink()

        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert not result.passed
        assert any("prompt_b2.md" in e for e in result.errors)

    def test_empty_prompt_file(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        (task_dir / "prompt_b1.md").write_text("")

        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert not result.passed
        assert any("empty" in e.lower() and "prompt_b1" in e for e in result.errors)

    def test_prompt_file_with_nul_byte_fails(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        (task_dir / "prompt_b1.md").write_bytes(b"valid prefix\x00valid suffix")

        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert not result.passed
        assert any("NUL byte" in e and "prompt_b1.md" in e for e in result.errors)

    def test_prompt_file_with_non_utf8_bytes_fails(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        (task_dir / "prompt_b2.md").write_bytes(b"\xff\xfe not utf-8")

        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert not result.passed
        assert any("not valid UTF-8" in e and "prompt_b2.md" in e for e in result.errors)

    def test_missing_generation_script(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        (task_dir / "generate_gt.py").unlink()

        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert not result.passed
        assert any("generate_gt.py" in e for e in result.errors)

    def test_invalid_generation_mode(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        metadata = yaml.safe_load((task_dir / "task.yaml").read_text())
        metadata["generation"]["mode"] = "bogus_mode"
        (task_dir / "task.yaml").write_text(yaml.dump(metadata))

        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert not result.passed
        assert any("Invalid generation mode" in e for e in result.errors)

    def test_missing_evaluation_block(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        metadata = yaml.safe_load((task_dir / "task.yaml").read_text())
        del metadata["evaluation"]
        (task_dir / "task.yaml").write_text(yaml.dump(metadata))

        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert not result.passed
        assert any("evaluation" in e for e in result.errors)

    def test_scoring_entry_missing_scorer(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        metadata = yaml.safe_load((task_dir / "task.yaml").read_text())
        metadata["evaluation"]["scoring"] = [{"weight": 100, "config": {}}]
        (task_dir / "task.yaml").write_text(yaml.dump(metadata))

        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert not result.passed
        assert any("scorer" in e for e in result.errors)

    def test_scoring_entry_missing_weight(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        metadata = yaml.safe_load((task_dir / "task.yaml").read_text())
        metadata["evaluation"]["scoring"] = [{"scorer": "numerical", "config": {}}]
        (task_dir / "task.yaml").write_text(yaml.dump(metadata))

        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert not result.passed
        assert any("weight" in e for e in result.errors)

    def test_custom_scorer_script_missing(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        metadata = yaml.safe_load((task_dir / "task.yaml").read_text())
        metadata["evaluation"]["scoring"] = [
            {
                "scorer": "custom",
                "weight": 100,
                "config": {"script": "custom_scorer.py"},
            }
        ]
        (task_dir / "task.yaml").write_text(yaml.dump(metadata))

        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert not result.passed
        assert any("custom_scorer.py" in e for e in result.errors)

    def test_custom_scorer_script_exists(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        metadata = yaml.safe_load((task_dir / "task.yaml").read_text())
        metadata["evaluation"]["scoring"] = [
            {
                "scorer": "custom",
                "weight": 100,
                "config": {"script": "custom_scorer.py"},
            }
        ]
        (task_dir / "task.yaml").write_text(yaml.dump(metadata))
        (task_dir / "custom_scorer.py").write_text("# scorer\n")

        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert result.passed

    def test_path_convention_domain_mismatch(self, tmp_dir):
        task_dir = tmp_dir / "tasks" / "physics" / "some_task"
        task_dir.mkdir(parents=True)
        tasks_root = tmp_dir / "tasks"

        metadata = {
            "id": "chemistry.some_task",  # id says chemistry, but dir says physics
            "name": "Some Task",
            "version": "1.0",
            "status": "test",
            "domain": "chemistry",
            "evaluation": {"gates": [], "scoring": []},
            "generation": {"script": "generate_gt.py"},
        }
        (task_dir / "task.yaml").write_text(yaml.dump(metadata))
        (task_dir / "prompt_b1.md").write_text("b1")
        (task_dir / "prompt_b2.md").write_text("b2")
        (task_dir / "prompt_b3.md").write_text("b3")
        (task_dir / "generate_gt.py").write_text("# gt\n")

        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert not result.passed
        assert any("domain" in e.lower() and "does not match" in e.lower() for e in result.errors)

    def test_task_id_domain_vs_declared_domain(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        metadata = yaml.safe_load((task_dir / "task.yaml").read_text())
        metadata["domain"] = "physics"  # id prefix is "math"
        (task_dir / "task.yaml").write_text(yaml.dump(metadata))

        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert not result.passed
        assert any("domain" in e for e in result.errors)

    def test_task_info_json_leakage(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        leaked = {
            "task_id": "math.test_validator_task",
            "instance_id": "math.test_validator_task__seed42",
            "parameters": {"seed": 42},
        }
        (task_dir / "task_info.json").write_text(json.dumps(leaked))

        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert not result.passed
        assert any("instance_id" in e for e in result.errors)
        assert any("parameters" in e for e in result.errors)

    def test_non_standard_domain_warns(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        metadata = yaml.safe_load((task_dir / "task.yaml").read_text())
        metadata["id"] = "philosophy.test_task"
        metadata["domain"] = "philosophy"
        (task_dir / "task.yaml").write_text(yaml.dump(metadata))

        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert any("philosophy" in w and "standard" in w for w in result.warnings)

    def test_b4_declared_but_missing(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        metadata = yaml.safe_load((task_dir / "task.yaml").read_text())
        metadata["prompts"]["b4"] = "prompt_b4.md"
        (task_dir / "task.yaml").write_text(yaml.dump(metadata))

        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert not result.passed
        assert any("prompt_b4.md" in e for e in result.errors)

    def test_b4_declared_and_exists(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        metadata = yaml.safe_load((task_dir / "task.yaml").read_text())
        metadata["prompts"]["b4"] = "prompt_b4.md"
        (task_dir / "task.yaml").write_text(yaml.dump(metadata))
        (task_dir / "prompt_b4.md").write_text("B4 content.")

        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert result.passed


def _set_final_status(task_dir: Path) -> str:
    """Promote the task in ``task_scaffold`` to ``status: final`` and return its id."""
    metadata = yaml.safe_load((task_dir / "task.yaml").read_text())
    metadata["status"] = "final"
    (task_dir / "task.yaml").write_text(yaml.dump(metadata))
    return metadata["id"]


class TestStatusScoreConsistency:
    """T8: status: final tasks must have a passing difficulty-check record."""

    def test_no_scores_dir_means_no_check(self, task_scaffold):
        # Backwards-compat: if caller doesn't pass scores_dir, the new
        # check is a no-op even for final tasks.
        task_dir, tasks_root = task_scaffold
        _set_final_status(task_dir)
        result = validate_task_static(task_dir, tasks_root=tasks_root)
        assert result.passed
        assert not any("difficulty check" in w for w in result.warnings)

    def test_non_final_task_skips_check(self, task_scaffold, tmp_dir):
        task_dir, tasks_root = task_scaffold
        scores_dir = tmp_dir / "scores_empty"
        scores_dir.mkdir()
        # Task is still 'test' from the scaffold default — no warning expected.
        result = validate_task_static(task_dir, tasks_root=tasks_root, scores_dir=scores_dir)
        assert result.passed
        assert not any("difficulty check" in w for w in result.warnings)

    def test_final_without_score_file_warns(self, task_scaffold, tmp_dir):
        task_dir, tasks_root = task_scaffold
        _set_final_status(task_dir)
        scores_dir = tmp_dir / "scores"
        scores_dir.mkdir()

        result = validate_task_static(task_dir, tasks_root=tasks_root, scores_dir=scores_dir)
        # Warnings don't fail the validation result.
        assert result.passed
        assert any(
            "final" in w and "no difficulty check record" in w for w in result.warnings
        )

    def test_final_with_empty_evaluations_warns(self, task_scaffold, tmp_dir):
        task_dir, tasks_root = task_scaffold
        task_id = _set_final_status(task_dir)
        scores_dir = tmp_dir / "scores"
        scores_dir.mkdir()
        (scores_dir / f"{task_id}.json").write_text(json.dumps({
            "task_id": task_id,
            "task_version": "1.0",
            "evaluations": [],
        }))

        result = validate_task_static(task_dir, tasks_root=tasks_root, scores_dir=scores_dir)
        assert result.passed
        assert any("empty" in w for w in result.warnings)

    def test_final_with_only_fail_verdicts_warns(self, task_scaffold, tmp_dir):
        task_dir, tasks_root = task_scaffold
        task_id = _set_final_status(task_dir)
        scores_dir = tmp_dir / "scores"
        scores_dir.mkdir()
        (scores_dir / f"{task_id}.json").write_text(json.dumps({
            "task_id": task_id,
            "task_version": "1.0",
            "evaluations": [
                {"date": "2026-05-01T00:00:00Z", "verdict": "fail", "threshold": 50},
                {"date": "2026-05-15T00:00:00Z", "verdict": "fail", "threshold": 50},
            ],
        }))

        result = validate_task_static(task_dir, tasks_root=tasks_root, scores_dir=scores_dir)
        assert result.passed
        assert any(
            "no difficulty check has ever verdict='pass'" in w for w in result.warnings
        )

    def test_final_with_passing_record_no_warning(self, task_scaffold, tmp_dir):
        task_dir, tasks_root = task_scaffold
        task_id = _set_final_status(task_dir)
        scores_dir = tmp_dir / "scores"
        scores_dir.mkdir()
        (scores_dir / f"{task_id}.json").write_text(json.dumps({
            "task_id": task_id,
            "task_version": "1.0",
            "evaluations": [
                {"date": "2026-05-15T00:00:00Z", "verdict": "pass", "threshold": 50},
            ],
        }))

        result = validate_task_static(task_dir, tasks_root=tasks_root, scores_dir=scores_dir)
        assert result.passed
        assert not any("difficulty check" in w for w in result.warnings)

    def test_final_with_any_historical_pass_no_warning(self, task_scaffold, tmp_dir):
        # T8: spec says "at least one record with verdict: pass" — so a single
        # historical pass clears the check even if the latest run regressed.
        task_dir, tasks_root = task_scaffold
        task_id = _set_final_status(task_dir)
        scores_dir = tmp_dir / "scores"
        scores_dir.mkdir()
        (scores_dir / f"{task_id}.json").write_text(json.dumps({
            "task_id": task_id,
            "task_version": "1.0",
            "evaluations": [
                {"date": "2026-03-10T00:00:00Z", "verdict": "pass", "threshold": 50},
                {"date": "2026-05-15T00:00:00Z", "verdict": "fail", "threshold": 50},
            ],
        }))

        result = validate_task_static(task_dir, tasks_root=tasks_root, scores_dir=scores_dir)
        assert result.passed
        assert not any("difficulty check" in w for w in result.warnings)

    def test_final_with_corrupt_score_file_warns(self, task_scaffold, tmp_dir):
        task_dir, tasks_root = task_scaffold
        task_id = _set_final_status(task_dir)
        scores_dir = tmp_dir / "scores"
        scores_dir.mkdir()
        (scores_dir / f"{task_id}.json").write_text("{not valid json")

        result = validate_task_static(task_dir, tasks_root=tasks_root, scores_dir=scores_dir)
        assert result.passed
        assert any("could not be parsed" in w for w in result.warnings)

    def test_cli_validate_propagates_scores_dir(self, task_scaffold, tmp_dir):
        task_dir, tasks_root = task_scaffold
        _set_final_status(task_dir)
        scores_dir = tmp_dir / "scores"
        scores_dir.mkdir()

        from ai4sci_bench.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--static", str(task_dir),
            "--tasks-dir", str(tasks_root),
            "--scores-dir", str(scores_dir),
        ])
        assert result.exit_code == 0, f"Expected pass:\n{result.output}"
        assert "Static checks: PASSED" in result.output
        # Warning must surface in the CLI output so reviewers see it.
        assert "no difficulty check record" in result.output


class TestCLIStaticMode:
    def test_static_on_valid_task(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        from ai4sci_bench.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--static", str(task_dir),
            "--tasks-dir", str(tasks_root),
        ])
        assert result.exit_code == 0, f"Expected pass:\n{result.output}"
        assert "Static checks: PASSED" in result.output

    def test_static_on_invalid_task(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        (task_dir / "prompt_b1.md").unlink()

        from ai4sci_bench.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--static", str(task_dir),
            "--tasks-dir", str(tasks_root),
        ])
        assert result.exit_code != 0
        assert "FAIL" in result.output

    def test_static_missing_task_yaml(self, tmp_dir):
        empty_dir = tmp_dir / "empty"
        empty_dir.mkdir()

        from ai4sci_bench.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--static", str(empty_dir),
        ])
        assert result.exit_code != 0
        assert "task.yaml" in result.output

    def test_static_split_task_passes(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        _split_task_yaml(task_dir)

        from ai4sci_bench.cli import cli

        result = CliRunner().invoke(cli, [
            "validate",
            "--static", str(task_dir),
            "--tasks-dir", str(tasks_root),
        ])

        assert result.exit_code == 0
        assert "Static checks: PASSED" in result.output

    def test_task_required_when_no_static_or_presubmit(self):
        from ai4sci_bench.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["validate"])
        assert result.exit_code != 0
        assert "--task is required" in result.output

    def test_original_task_mode_still_works(self, sample_task_dir):
        from ai4sci_bench.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--task", "physics.test_task",
            "--tasks-dir", str(sample_task_dir.parent.parent),
        ])
        assert result.exit_code == 0
        assert "VALIDATION PASSED" in result.output


class TestCLIPreSubmitMode:
    def test_pre_submit_static_failure_aborts(self, task_scaffold):
        task_dir, tasks_root = task_scaffold
        (task_dir / "prompt_b1.md").unlink()

        from ai4sci_bench.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--pre-submit", str(task_dir),
            "--tasks-dir", str(tasks_root),
        ])
        assert result.exit_code != 0
        assert "Static checks: FAILED" in result.output
        assert "Skipping GT generation" in result.output

    def test_pre_submit_runs_gt_generation(self, sample_task_dir):
        from ai4sci_bench.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate",
            "--pre-submit", str(sample_task_dir),
            "--tasks-dir", str(sample_task_dir.parent.parent),
        ])
        assert "GT Generation" in result.output
        assert "seed=" in result.output

    def test_pre_submit_split_task_runs_gt_generation(self, sample_task_dir):
        _split_task_yaml(sample_task_dir)

        from ai4sci_bench.cli import cli

        result = CliRunner().invoke(cli, [
            "validate",
            "--pre-submit", str(sample_task_dir),
            "--tasks-dir", str(sample_task_dir.parent.parent),
        ])

        assert "Static checks: PASSED" in result.output
        assert "GT Generation" in result.output
        assert "seed=" in result.output

    def test_pre_submit_infers_custom_tasks_root(self, sample_task_dir):
        """A task path outside ./tasks must work without a duplicate flag."""
        from ai4sci_bench.cli import cli

        result = CliRunner().invoke(cli, [
            "validate",
            "--pre-submit", str(sample_task_dir),
        ])

        assert result.exit_code == 0, result.output
        assert "GT Self-check: PASSED" in result.output
        assert "Pre-submit validation: PASSED for physics.test_task" in result.output


class TestDetectChangedTasks:
    def test_no_git_repo_returns_empty(self, tmp_dir):
        result = detect_changed_tasks(tmp_dir)
        assert result == []

    @pytest.mark.parametrize("metadata_file", ["task.yaml", "task_meta.yaml"])
    def test_returns_unique_task_dirs(self, tmp_dir, metadata_file):
        import subprocess

        subprocess.run(["git", "init"], cwd=tmp_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", "init"],
            cwd=tmp_dir,
            capture_output=True,
            env={"GIT_AUTHOR_NAME": "test", "GIT_COMMITTER_NAME": "test",
                 "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_EMAIL": "t@t",
                 "HOME": str(tmp_dir)},
        )

        task_dir = tmp_dir / "tasks" / "math" / "my_task"
        task_dir.mkdir(parents=True)
        (task_dir / metadata_file).write_text("id: math.my_task\n")
        (task_dir / "generate_gt.py").write_text("# gt\n")

        subprocess.run(["git", "add", "."], cwd=tmp_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "add task"],
            cwd=tmp_dir,
            capture_output=True,
            env={"GIT_AUTHOR_NAME": "test", "GIT_COMMITTER_NAME": "test",
                 "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_EMAIL": "t@t",
                 "HOME": str(tmp_dir)},
        )

        result = detect_changed_tasks(tmp_dir, base_ref="HEAD~1")
        assert len(result) == 1
        assert result[0].name == "my_task"
