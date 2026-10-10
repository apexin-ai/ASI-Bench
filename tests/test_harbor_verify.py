"""Harbor verifier uses the public seed31415 local scoring contract."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from ai4sci_bench.cli import cli
from ai4sci_bench.local_scoring import score_seed31415_results
from tests.test_local_scoring import _enable_task_scoring_runtime, _write_fixture


def _paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    tasks, instances, results = _write_fixture(tmp_path)
    task = tasks / "physics" / "example"
    instance = instances / "physics.example__seed31415"
    outputs = next(results.glob("*/*.outputs"))
    return task, instance, outputs, tmp_path / "verifier"


def _run(task: Path, instance: Path, outputs: Path, out: Path):
    return CliRunner().invoke(
        cli,
        [
            "harbor-verify", "--task-dir", str(task),
            "--instance-dir", str(instance), "--outputs-dir", str(outputs),
            "--prompt-level", "b1", "--out", str(out),
        ],
    )


def test_harbor_verify_matches_local_score_for_same_prediction(tmp_path):
    task, instance, outputs, out = _paths(tmp_path)
    local, _ = score_seed31415_results(tmp_path / "results", tmp_path / "instances", tmp_path / "tasks")

    result = _run(task, instance, outputs, out)

    assert result.exit_code == 0, result.output
    detail = json.loads((out / "score_detail.json").read_text())
    scored = local["results"][0]
    for key in ("final_score", "max_score", "hard_gates_passed", "gate_results", "score_results"):
        assert detail[key] == scored[key]
    assert detail["official"] is False
    assert detail["provenance"]["framework_version"]
    assert detail["artifact_sha256"]
    assert detail["task_bundle_sha256"]
    assert json.loads((out / "reward.json").read_text()) == {"reward": scored["final_score"] / 100}


@pytest.mark.parametrize("directory_missing", [False, True])
def test_harbor_verify_empty_submission_is_valid_zero(tmp_path, directory_missing):
    task, instance, outputs, out = _paths(tmp_path)
    if directory_missing:
        shutil.rmtree(outputs)
    else:
        (outputs / "output.npy").unlink()

    result = _run(task, instance, outputs, out)

    assert result.exit_code == 0, result.output
    detail = json.loads((out / "score_detail.json").read_text())
    assert detail["evaluation_status"] == "completed"
    assert detail["final_score"] == 0
    assert detail["scorer_internal_error"] is False
    assert json.loads((out / "reward.json").read_text()) == {"reward": 0.0}


def test_harbor_verify_evaluator_failure_removes_stale_reward(tmp_path, monkeypatch):
    task, instance, outputs, out = _paths(tmp_path)
    out.mkdir()
    (out / "reward.json").write_text('{"reward": 1}')

    def fail(*_args, **_kwargs):
        raise RuntimeError("broken evaluator")

    monkeypatch.setattr("ai4sci_bench.harbor_verify._evaluate_score_job", fail)
    result = _run(task, instance, outputs, out)

    assert result.exit_code != 0
    assert not (out / "reward.json").exists()
    detail = json.loads((out / "score_detail.json").read_text())
    assert detail["evaluation_status"] == "evaluation_invalid"
    assert detail["final_score"] is None
    assert detail["scorer_internal_error"] is True
    assert detail["failure_kind"] == "evaluator_runtime_error"


def test_harbor_verify_scorer_internal_error_has_no_reward(tmp_path, monkeypatch):
    task, instance, outputs, out = _paths(tmp_path)

    def invalid(*_args, **_kwargs):
        return {
            "evaluation_status": "evaluation_invalid",
            "final_score": None,
            "max_score": 100.0,
            "scorer_internal_error": True,
            "failure_kind": "evaluator_runtime_error",
        }

    monkeypatch.setattr("ai4sci_bench.harbor_verify._evaluate_score_job", invalid)
    result = _run(task, instance, outputs, out)

    assert result.exit_code != 0
    assert not (out / "reward.json").exists()
    assert json.loads((out / "score_detail.json").read_text())["final_score"] is None


def test_harbor_verify_task_runtime_failure_is_evaluator_unavailable(tmp_path, monkeypatch):
    task, instance, outputs, out = _paths(tmp_path)
    _enable_task_scoring_runtime(tmp_path / "tasks")

    def unavailable(*_args, **_kwargs):
        raise FileNotFoundError("runtime installer unavailable")

    monkeypatch.setattr(
        "ai4sci_bench.runner.task_env.TaskEnvironmentManager.ensure_env",
        unavailable,
    )
    result = _run(task, instance, outputs, out)

    assert result.exit_code != 0
    assert not (out / "reward.json").exists()
    detail = json.loads((out / "score_detail.json").read_text())
    assert detail["failure_kind"] == "evaluator_unavailable"


def test_harbor_verify_preinstalled_runtime_bypasses_dynamic_install(tmp_path, monkeypatch):
    task, instance, outputs, out = _paths(tmp_path)
    _enable_task_scoring_runtime(tmp_path / "tasks")
    monkeypatch.setenv("ASIBENCH_HARBOR_TASK_RUNTIME_PREINSTALLED", "1")

    def unexpected_install(*_args, **_kwargs):
        raise AssertionError("runtime installer should not run inside the verifier")

    monkeypatch.setattr(
        "ai4sci_bench.harbor_verify._prepare_score_runtimes", unexpected_install,
    )
    result = CliRunner().invoke(
        cli,
        ["harbor-verify", "--task-dir", str(task), "--instance-dir", str(instance),
         "--outputs-dir", str(outputs), "--prompt-level", "b1", "--out", str(out),
         "--task-runtime-preinstalled"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads((out / "score_detail.json").read_text())["evaluation_status"] == "completed"


def test_harbor_verify_preinstalled_runtime_requires_dedicated_image(tmp_path):
    task, instance, outputs, out = _paths(tmp_path)
    _enable_task_scoring_runtime(tmp_path / "tasks")
    result = CliRunner().invoke(
        cli,
        ["harbor-verify", "--task-dir", str(task), "--instance-dir", str(instance),
         "--outputs-dir", str(outputs), "--prompt-level", "b1", "--out", str(out),
         "--task-runtime-preinstalled"],
        env={"ASIBENCH_HARBOR_TASK_RUNTIME_PREINSTALLED": ""},
    )
    assert result.exit_code != 0
    assert not (out / "reward.json").exists()
    detail = json.loads((out / "score_detail.json").read_text())
    assert detail["evaluation_status"] == "evaluation_invalid"


def test_harbor_verify_missing_scorer_dependency_is_unavailable(tmp_path, monkeypatch):
    task, instance, outputs, out = _paths(tmp_path)

    def missing(*_args, **_kwargs):
        raise ModuleNotFoundError("scientific scorer package missing")

    monkeypatch.setattr("ai4sci_bench.harbor_verify._evaluate_score_job", missing)
    result = _run(task, instance, outputs, out)

    assert result.exit_code != 0
    assert not (out / "reward.json").exists()
    detail = json.loads((out / "score_detail.json").read_text())
    assert detail["failure_kind"] == "evaluator_unavailable"


@pytest.mark.parametrize("unsafe", ["symlink", "overwrite-data"])
def test_harbor_verify_rejects_unsafe_outputs_without_reward(tmp_path, unsafe):
    task, instance, outputs, out = _paths(tmp_path)
    if unsafe == "symlink":
        (outputs / "link").symlink_to(instance / "reference", target_is_directory=True)
    else:
        (instance / "data").mkdir()
        (instance / "data" / "input.txt").write_text("trusted")
        (outputs / "data").mkdir()
        (outputs / "data" / "input.txt").write_text("overwritten")

    result = _run(task, instance, outputs, out)

    assert result.exit_code != 0
    assert not (out / "reward.json").exists()
    detail = json.loads((out / "score_detail.json").read_text())
    assert detail["failure_kind"] == "missing_evaluator_input"


def test_harbor_verify_rejects_seed42(tmp_path):
    task, instance, outputs, out = _paths(tmp_path)
    other = instance.with_name("physics.example__seed42")
    instance.rename(other)

    result = _run(task, other, outputs, out)

    assert result.exit_code != 0
    assert not (out / "reward.json").exists()
    assert "seed31415" in result.output


def test_harbor_verify_missing_reference_is_evaluator_error(tmp_path):
    task, instance, outputs, out = _paths(tmp_path)
    (instance / "reference" / "output_ref.npy").unlink()

    result = _run(task, instance, outputs, out)

    assert result.exit_code != 0
    assert not (out / "reward.json").exists()
    detail = json.loads((out / "score_detail.json").read_text())
    assert detail["failure_kind"] == "missing_evaluator_input"
    assert detail["final_score"] is None


def test_harbor_verify_never_writes_inside_outputs(tmp_path):
    task, instance, outputs, _ = _paths(tmp_path)
    result = _run(task, instance, outputs, outputs)
    assert result.exit_code != 0
    assert not (outputs / "reward.json").exists()
    assert "out must be separate" in result.output


def test_harbor_verify_rejects_symlinked_output_destination(tmp_path):
    task, instance, outputs, out = _paths(tmp_path)
    out.symlink_to(outputs, target_is_directory=True)

    result = _run(task, instance, outputs, out)

    assert result.exit_code != 0
    assert not (outputs / "reward.json").exists()


def test_harbor_verify_invalid_judge_selector_removes_stale_reward(tmp_path, monkeypatch):
    task, instance, outputs, out = _paths(tmp_path)
    out.mkdir()
    (out / "reward.json").write_text('{"reward": 1}')
    monkeypatch.delenv("ASIBENCH_HARBOR_MISSING_JUDGE_KEY", raising=False)

    result = CliRunner().invoke(
        cli,
        [
            "harbor-verify", "--task-dir", str(task),
            "--instance-dir", str(instance), "--outputs-dir", str(outputs),
            "--prompt-level", "b1", "--out", str(out),
            "--judge-api-key-env", "ASIBENCH_HARBOR_MISSING_JUDGE_KEY",
        ],
    )

    assert result.exit_code != 0
    assert not (out / "reward.json").exists()
