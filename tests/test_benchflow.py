import json
from pathlib import Path

import pytest

from ai4sci_bench.benchflow import BenchFlowScoringError, _sha256_tree, score_seed31415_manifest


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
    assert result["attempt_status"] == "failed"
    assert result["retryable"] is True


def test_benchflow_rejects_prediction_dir_not_owned_by_run_result(tmp_path):
    manifest = _manifest(tmp_path)
    other = tmp_path / "other-prediction"
    other.mkdir()
    manifest["prediction_dir"] = str(other)

    with pytest.raises(BenchFlowScoringError, match="persisted output directory"):
        score_seed31415_manifest(manifest)
