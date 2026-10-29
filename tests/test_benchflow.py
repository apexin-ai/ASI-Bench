import json
from pathlib import Path

import pytest

from ai4sci_bench.benchflow import BenchFlowScoringError, _sha256_tree, score_seed31415_manifest
from ai4sci_bench.core.judge_api import (
    JudgeAPIOverride,
    get_judge_api_override,
    use_judge_api_override,
)


def _manifest(tmp_path: Path, *, seed=31415, instance_id="demo__seed31415"):
    instance_dir = tmp_path / instance_id
    (instance_dir / "reference").mkdir(parents=True)
    (instance_dir / "reference" / "answer.txt").write_text("reference", encoding="utf-8")
    prediction_dir = tmp_path / "prediction"
    prediction_dir.mkdir()
    (prediction_dir / "answer.txt").write_text("prediction", encoding="utf-8")
    run_result = tmp_path / "run_result.json"
    run_result.write_text(json.dumps({
        "result_schema_version": 1,
        "task_id": "demo.task",
        "instance_id": instance_id,
        "prompt_level": "b1",
        "status": "completed",
        "agent_output": {
            "status": "completed",
            "persisted_outputs": {
                "dir": prediction_dir.name,
                "files": [{"path": "answer.txt"}],
            },
        },
    }), encoding="utf-8")
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    return {
        "schema_version": 2,
        "benchmark": "ASI-Bench",
        "seed": seed,
        "task_id": "demo.task",
        "instance_id": instance_id,
        "prediction_dir": str(prediction_dir),
        "run_result": str(run_result),
        "instance_dir": str(instance_dir),
        "tasks_dir": str(tasks_dir),
    }


def test_benchflow_rejects_non_seed31415_before_path_access(tmp_path):
    manifest = _manifest(tmp_path, seed=42)
    manifest["prediction_dir"] = str(tmp_path / "does-not-exist")
    with pytest.raises(BenchFlowScoringError, match="only supports seed31415"):
        score_seed31415_manifest(manifest)


def test_benchflow_rejects_mismatched_instance_directory(tmp_path):
    manifest = _manifest(tmp_path, instance_id="reported__seed31415")
    other = tmp_path / "other__seed31415"
    (other / "reference").mkdir(parents=True)
    manifest["instance_dir"] = str(other)
    with pytest.raises(BenchFlowScoringError, match="basename must match"):
        score_seed31415_manifest(manifest)


def test_benchflow_requires_run_result_json(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.pop("run_result")

    with pytest.raises(BenchFlowScoringError, match="manifest.run_result"):
        score_seed31415_manifest(manifest)


def test_prediction_artifact_digest_is_order_independent(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a.txt").write_text("a", encoding="utf-8")
    (first / "b.txt").write_text("b", encoding="utf-8")
    (second / "b.txt").write_text("b", encoding="utf-8")
    (second / "a.txt").write_text("a", encoding="utf-8")
    assert _sha256_tree(first) == _sha256_tree(second)


def test_cli_manifest_schema_is_json_serializable(tmp_path):
    manifest = _manifest(tmp_path)
    assert json.loads(json.dumps(manifest))["seed"] == 31415


def test_benchflow_emits_stable_score_details(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path)
    task_dir = Path(manifest["tasks_dir"]) / "demo"
    task_dir.mkdir()

    class FakeLoader:
        def __init__(self, _tasks_dir):
            pass

        def load_task_by_id(self, _task_id):
            return {"_task_dir": str(task_dir), "evaluation": {"scoring": [{"weight": 10}]}}

    monkeypatch.setattr("ai4sci_bench.benchflow.TaskLoader", FakeLoader)
    monkeypatch.setattr(
        "ai4sci_bench.scorers.custom.load_custom_scorer",
        lambda _task_dir: None,
    )
    monkeypatch.setattr(
        "ai4sci_bench.runner.orchestrator._evaluate_gates_and_scores",
        lambda *_args, **_kwargs: ([], True, 0, [], 7.5),
    )

    result = __import__("ai4sci_bench.benchflow", fromlist=["score_seed31415_manifest"]).score_seed31415_manifest(manifest)
    assert result["status"] == "completed"
    assert result["evaluation_status"] == "completed"
    assert result["attempt_status"] == "completed"
    assert result["score"] == 7.5
    assert result["max_score"] == 10.0
    assert len(result["artifact_sha256"]) == 64
    assert result["seed"] == 31415


def test_benchflow_applies_task_score_divisor(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path)
    task_dir = Path(manifest["tasks_dir"]) / "demo"
    task_dir.mkdir()

    class FakeLoader:
        def __init__(self, _tasks_dir):
            pass

        def load_task_by_id(self, _task_id):
            return {
                "_task_dir": str(task_dir),
                "evaluation": {"score_divisor": 1.05, "scoring": [{"weight": 105}]},
            }

    monkeypatch.setattr("ai4sci_bench.benchflow.TaskLoader", FakeLoader)
    monkeypatch.setattr("ai4sci_bench.scorers.custom.load_custom_scorer", lambda _task_dir: None)
    monkeypatch.setattr(
        "ai4sci_bench.runner.orchestrator._evaluate_gates_and_scores",
        lambda *_args, **_kwargs: ([], True, 0, [], 105.0),
    )

    result = score_seed31415_manifest(manifest)
    assert result["score"] == 100.0
    assert result["max_score"] == 100.0
    assert result["score_divisor"] == 1.05


def test_benchflow_reads_failed_run_result_instead_of_trusting_runner_exit(
    monkeypatch, tmp_path
):
    manifest = _manifest(tmp_path)
    run_result = Path(manifest["run_result"])
    payload = json.loads(run_result.read_text())
    payload["status"] = "failed"
    payload["agent_output"]["status"] = "failed"
    run_result.write_text(json.dumps(payload), encoding="utf-8")
    task_dir = Path(manifest["tasks_dir"]) / "demo"
    task_dir.mkdir()

    class FakeLoader:
        def __init__(self, _tasks_dir):
            pass

        def load_task_by_id(self, _task_id):
            return {"_task_dir": str(task_dir), "evaluation": {"scoring": [{"weight": 10}]}}

    monkeypatch.setattr("ai4sci_bench.benchflow.TaskLoader", FakeLoader)
    monkeypatch.setattr("ai4sci_bench.scorers.custom.load_custom_scorer", lambda _task_dir: None)
    monkeypatch.setattr(
        "ai4sci_bench.runner.orchestrator._evaluate_gates_and_scores",
        lambda *_args, **_kwargs: ([], True, 0, [], 0.0),
    )

    result = score_seed31415_manifest(manifest)

    assert result["status"] == "attempt_failed"
    assert result["evaluation_status"] == "completed"
    assert result["attempt_status"] == "execution_failed"
    assert result["retryable"] is True


@pytest.mark.parametrize(
    ("raw_status", "expected_status"),
    [
        ("timeout", "execution_timeout"),
        ("running", "execution_incomplete"),
    ],
)
def test_benchflow_normalizes_noncompleted_attempt_statuses(
    monkeypatch, tmp_path, raw_status, expected_status
):
    manifest = _manifest(tmp_path)
    run_result = Path(manifest["run_result"])
    payload = json.loads(run_result.read_text())
    payload["status"] = raw_status
    payload["agent_output"]["status"] = raw_status
    run_result.write_text(json.dumps(payload), encoding="utf-8")
    task_dir = Path(manifest["tasks_dir"]) / "demo"
    task_dir.mkdir()

    class FakeLoader:
        def __init__(self, _tasks_dir):
            pass

        def load_task_by_id(self, _task_id):
            return {"_task_dir": str(task_dir), "evaluation": {"scoring": []}}

    monkeypatch.setattr("ai4sci_bench.benchflow.TaskLoader", FakeLoader)
    monkeypatch.setattr("ai4sci_bench.scorers.custom.load_custom_scorer", lambda _task_dir: None)
    monkeypatch.setattr(
        "ai4sci_bench.runner.orchestrator._evaluate_gates_and_scores",
        lambda *_args, **_kwargs: ([], True, 0, [], 0.0),
    )

    result = score_seed31415_manifest(manifest)

    assert result["status"] == "attempt_failed"
    assert result["attempt_status"] == expected_status
    assert result["evaluation_status"] == "completed"


def test_benchflow_rejects_prediction_dir_not_owned_by_run_result(tmp_path):
    manifest = _manifest(tmp_path)
    other = tmp_path / "other-prediction"
    other.mkdir()
    manifest["prediction_dir"] = str(other)

    with pytest.raises(BenchFlowScoringError, match="persisted output directory"):
        score_seed31415_manifest(manifest)


def test_benchflow_scopes_runtime_judge_override_to_evaluation(monkeypatch, tmp_path):
    """BenchFlow must expose the operator override only while scorers run."""
    manifest = _manifest(tmp_path)
    task_dir = Path(manifest["tasks_dir"]) / "demo"
    task_dir.mkdir()

    class FakeLoader:
        def __init__(self, _tasks_dir):
            pass

        def load_task_by_id(self, _task_id):
            return {"_task_dir": str(task_dir), "evaluation": {"scoring": [{"weight": 10}]}}

    override = JudgeAPIOverride(
        api_base="https://api.tokenrouter.com/v1",
        api_key_env="TOKENROUTER_API_KEY",
        api_protocol="openai",
    )
    monkeypatch.setenv("TOKENROUTER_API_KEY", "test-only-judge-secret")
    outer_override = JudgeAPIOverride(api_protocol="native")
    seen: list[JudgeAPIOverride | None] = []

    monkeypatch.setattr("ai4sci_bench.benchflow.TaskLoader", FakeLoader)
    monkeypatch.setattr("ai4sci_bench.scorers.custom.load_custom_scorer", lambda _task_dir: None)

    def fake_evaluate(*_args, **_kwargs):
        seen.append(get_judge_api_override())
        return ([], True, 0, [], 7.5)

    monkeypatch.setattr(
        "ai4sci_bench.runner.orchestrator._evaluate_gates_and_scores",
        fake_evaluate,
    )

    with use_judge_api_override(outer_override):
        assert get_judge_api_override() is outer_override
        result = score_seed31415_manifest(manifest, judge_api_override=override)
        assert get_judge_api_override() is outer_override

    assert seen == [override]
    assert result["score"] == 7.5


def test_benchflow_score_manifest_file_accepts_runtime_override(monkeypatch, tmp_path):
    """The file wrapper must pass the override through to manifest scoring."""
    from ai4sci_bench.benchflow import score_manifest_file

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"sentinel": True}), encoding="utf-8")
    override = JudgeAPIOverride(
        api_base="https://api.example.test/v1",
        api_key_env="JUDGE_KEY",
        api_protocol="openai",
    )
    seen: list[JudgeAPIOverride | None] = []

    def fake_score(manifest, *, judge_api_override=None):
        assert manifest == {"sentinel": True}
        seen.append(judge_api_override)
        return {"status": "completed"}

    monkeypatch.setattr("ai4sci_bench.benchflow.score_seed31415_manifest", fake_score)
    destination = score_manifest_file(
        manifest_path,
        judge_api_override=override,
    )

    assert seen == [override]
    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "completed"}


def test_benchflow_without_argument_preserves_outer_judge_scope(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path)
    task_dir = Path(manifest["tasks_dir"]) / "demo"
    task_dir.mkdir()

    class FakeLoader:
        def __init__(self, _tasks_dir):
            pass

        def load_task_by_id(self, _task_id):
            return {"_task_dir": str(task_dir), "evaluation": {"scoring": []}}

    outer_override = JudgeAPIOverride(api_protocol="native")
    seen: list[JudgeAPIOverride | None] = []
    monkeypatch.setattr("ai4sci_bench.benchflow.TaskLoader", FakeLoader)
    monkeypatch.setattr("ai4sci_bench.scorers.custom.load_custom_scorer", lambda _task_dir: None)

    def fake_evaluate(*_args, **_kwargs):
        seen.append(get_judge_api_override())
        return ([], True, 0, [], 0.0)

    monkeypatch.setattr(
        "ai4sci_bench.runner.orchestrator._evaluate_gates_and_scores",
        fake_evaluate,
    )
    with use_judge_api_override(outer_override):
        score_seed31415_manifest(manifest)

    assert seen == [outer_override]


def _data_manifest(tmp_path):
    """A real task whose numerical score depends on immutable instance data."""
    from tests.test_local_scoring import _write_fixture

    tasks, instances, results = _write_fixture(tmp_path)
    task = tasks / "physics" / "example"
    meta = task / "task_meta.yaml"
    meta.write_text(meta.read_text().replace("files: []", "files:\n    - name: input.npy"))
    instance = instances / "physics.example__seed31415"
    (instance / "data").mkdir()
    import shutil
    shutil.copy2(instance / "reference" / "output_ref.npy", instance / "data" / "input.npy")
    evaluation = task / "task_eval.yaml"
    # numerical joins pred_file to pred_dir, reproducing the missing data failure.
    evaluation.write_text(evaluation.read_text().replace("pred_file: output.npy", "pred_file: data/input.npy"))
    run_result = next(results.glob("*/*.json"))
    payload = json.loads(run_result.read_text())
    payload["agent_output"]["status"] = "completed"
    payload["agent_output"]["persisted_outputs"]["dir"] = run_result.stem + ".outputs"
    run_result.write_text(json.dumps(payload))
    return {
        "schema_version": 2, "seed": 31415,
        "task_id": "physics.example", "instance_id": instance.name,
        "prompt_level": "b1", "run_result": str(run_result),
        "prediction_dir": str(run_result.with_suffix(".outputs")),
        "instance_dir": str(instance), "tasks_dir": str(tasks),
    }


def test_benchflow_stages_instance_data_and_matches_standard_score(tmp_path):
    from ai4sci_bench.local_scoring import score_seed31415_results

    manifest = _data_manifest(tmp_path)
    predictions = Path(manifest["prediction_dir"])
    before = _sha256_tree(predictions)
    local, _ = score_seed31415_results(tmp_path / "results", tmp_path / "instances", tmp_path / "tasks")
    result = score_seed31415_manifest(manifest)
    scored = local["results"][0]
    assert result["score"] == scored["final_score"] == 100.0
    assert result["score_details"] == scored["score_results"]
    assert result["gate_results"] == scored["gate_results"]
    assert result["artifact_sha256"] == _sha256_tree(predictions) == before
    assert not (predictions / "data").exists()


@pytest.mark.parametrize("problem", ["missing", "symlink", "overwrite"])
def test_benchflow_invalid_instance_inputs_are_not_model_zero(tmp_path, problem):
    import shutil

    manifest = _data_manifest(tmp_path)
    data = Path(manifest["instance_dir"]) / "data"
    if problem == "missing":
        shutil.rmtree(data)
    elif problem == "symlink":
        (data / "link.npy").symlink_to(data / "input.npy")
    else:
        forged = Path(manifest["prediction_dir"]) / "data"
        forged.mkdir()
        (forged / "input.npy").write_text("agent replacement")
    result = score_seed31415_manifest(manifest)
    assert result["evaluation_status"] == "evaluation_invalid"
    assert result["score"] is None
    assert result["scorer_internal_error"] is True
    assert result["failure_kind"] == "missing_evaluator_input"


def test_benchflow_runtime_setup_failure_is_invalid(tmp_path, monkeypatch):
    from tests.test_local_scoring import _enable_task_scoring_runtime
    from ai4sci_bench.runner.task_env import TaskEnvironmentManager

    manifest = _data_manifest(tmp_path)
    _enable_task_scoring_runtime(Path(manifest["tasks_dir"]))

    def unavailable(*_args):
        raise RuntimeError("test installer unavailable")

    monkeypatch.setattr(TaskEnvironmentManager, "ensure_env", unavailable)
    result = score_seed31415_manifest(manifest)
    assert result["evaluation_status"] == "evaluation_invalid"
    assert result["score"] is None
    assert result["failure_kind"] == "evaluator_unavailable"


def test_benchflow_task_runtime_loads_scorer_in_fresh_worker(tmp_path, monkeypatch):
    from tests.test_local_scoring import _enable_task_scoring_runtime, _fake_task_environment
    from ai4sci_bench.runner.task_env import TaskEnvironmentManager

    manifest = _data_manifest(tmp_path)
    tasks = Path(manifest["tasks_dir"])
    _enable_task_scoring_runtime(tasks)
    # This import can only succeed after the runtime is activated.
    (tasks / "physics" / "example" / "custom_scorer.py").write_text(
        "import fake_evaluator_dependency\n"
    )
    environment = _fake_task_environment(tmp_path)
    monkeypatch.setattr(TaskEnvironmentManager, "ensure_env", lambda *_args: environment)
    result = score_seed31415_manifest(manifest)
    assert result["evaluation_status"] == "completed"
    assert result["score"] == 100.0
    import sys
    assert "fake_evaluator_dependency" not in sys.modules


def test_benchflow_missing_prediction_remains_valid_zero(tmp_path, monkeypatch):
    from ai4sci_bench.runner.task_env import TaskEnvironmentManager

    manifest = _data_manifest(tmp_path)
    def unexpected(*_args):
        raise AssertionError("ordinary task must not build a runtime")
    monkeypatch.setattr(TaskEnvironmentManager, "ensure_env", unexpected)
    (Path(manifest["prediction_dir"]) / "output.npy").unlink()
    result = score_seed31415_manifest(manifest)
    assert result["evaluation_status"] == "completed"
    assert result["score"] == 0.0
    assert result["scorer_internal_error"] is False


def test_benchflow_scorer_import_error_is_invalid_and_redacted(tmp_path, monkeypatch):
    manifest = _data_manifest(tmp_path)
    task = Path(manifest["tasks_dir"]) / "physics" / "example"
    (task / "custom_scorer.py").write_text("raise RuntimeError('test-private-key')\n")
    monkeypatch.setenv("TEST_BENCHFLOW_JUDGE_KEY", "test-private-key")
    result = score_seed31415_manifest(manifest, judge_api_override=JudgeAPIOverride(
        api_key_env="TEST_BENCHFLOW_JUDGE_KEY",
    ))
    assert result["evaluation_status"] == "evaluation_invalid"
    assert result["score"] is None
    assert result["failure_kind"] == "evaluator_runtime_error"
    assert "test-private-key" not in json.dumps(result)
    assert "<redacted>" in result["score_details"][0]["message"]


def test_benchflow_rejects_symlink_prediction_root(tmp_path):
    manifest = _data_manifest(tmp_path)
    alias = tmp_path / "aliased-outputs"
    alias.symlink_to(manifest["prediction_dir"], target_is_directory=True)
    manifest["prediction_dir"] = str(alias)
    with pytest.raises(BenchFlowScoringError, match="prediction_dir must be a real directory"):
        score_seed31415_manifest(manifest)


def test_benchflow_cli_writes_invalid_score_as_json_null(tmp_path):
    import shutil
    from click.testing import CliRunner
    from ai4sci_bench.cli import cli

    manifest = _data_manifest(tmp_path)
    shutil.rmtree(Path(manifest["instance_dir"]) / "data")
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(manifest))
    destination = tmp_path / "benchflow_score.json"
    invocation = CliRunner().invoke(cli, [
        "benchflow-score", "--manifest", str(source), "--output", str(destination),
    ])
    assert invocation.exit_code == 0, invocation.output
    result = json.loads(destination.read_text())
    assert result["status"] == result["evaluation_status"] == "evaluation_invalid"
    assert result["score"] is None
    assert result["failure_kind"] == "missing_evaluator_input"
