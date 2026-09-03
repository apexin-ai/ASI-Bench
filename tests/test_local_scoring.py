"""Public seed31415 local scoring and private seed42 boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from ai4sci_bench.cli import cli
from ai4sci_bench.core.judge_api import (
    JudgeAPIOverride,
    get_judge_api_override,
    use_judge_api_override,
)
from ai4sci_bench.local_scoring import score_seed31415_results


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    tasks_dir = root / "tasks"
    task_dir = tasks_dir / "physics" / "example"
    task_dir.mkdir(parents=True)
    (task_dir / "task_meta.yaml").write_text(
        """
id: physics.example
name: Example
version: '1.0'
status: final
domain: physics
runtime:
  python: '>=3.11'
  packages: []
prompts:
  b1: prompt_b1.md
  b2: prompt_b2.md
  b3: prompt_b3.md
  b4: prompt_b4.md
input:
  files: []
output:
  files:
    - name: output.npy
      type: data
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (task_dir / "task_eval.yaml").write_text(
        """
task_id: physics.example
evaluation:
  gates:
    - scorer: file_match
      severity: hard
      config:
        checks:
          - file: output.npy
            shape: [2]
            dtype: float64
  scoring:
    - scorer: numerical
      weight: 100
      config:
        metric: relative_l2
        pred_file: output.npy
        ref_file: output_ref.npy
        threshold: 0.01
""".strip()
        + "\n",
        encoding="utf-8",
    )

    instance_id = "physics.example__seed31415"
    instances_dir = root / "instances"
    instance_dir = instances_dir / instance_id
    (instance_dir / "reference").mkdir(parents=True)
    expected = np.array([1.0, 2.0], dtype=np.float64)
    np.save(instance_dir / "reference" / "output_ref.npy", expected)
    (instance_dir / "instance_meta.json").write_text(
        json.dumps({"params_used": {}}), encoding="utf-8"
    )

    results_dir = root / "results"
    result_task_dir = results_dir / "physics.example"
    outputs_dir = result_task_dir / f"{instance_id}__b1.outputs"
    outputs_dir.mkdir(parents=True)
    np.save(outputs_dir / "output.npy", expected)
    result_json = {
        "instance_id": instance_id,
        "task_id": "physics.example",
        "prompt_level": "b1",
        "parameters": {},
        "status": "completed",
        "final_score": 0.0,
        "agent_output": {
            "persisted_outputs": {
                "files": [{"path": "output.npy", "missing": False}]
            }
        },
    }
    (result_task_dir / f"{instance_id}__b1.json").write_text(
        json.dumps(result_json), encoding="utf-8"
    )
    return tasks_dir, instances_dir, results_dir


def test_seed31415_local_score_uses_public_reference(tmp_path):
    tasks_dir, instances_dir, results_dir = _write_fixture(tmp_path)
    report_path = tmp_path / "score.json"

    result = CliRunner().invoke(
        cli,
        [
            "score",
            "--repo",
            "seed31415",
            "--results-dir",
            str(results_dir),
            "--instances-dir",
            str(instances_dir),
            "--tasks-dir",
            str(tasks_dir),
            "--output",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "100.00 / 100.00" in result.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["repo"] == "seed31415"
    assert report["official"] is False
    assert report["instance_count"] == 1
    scored = report["results"][0]
    assert scored["hard_gates_passed"] is True
    assert scored["final_score"] == 100.0
    assert scored["max_score"] == 100.0
    # Local scoring must not rewrite the original produce-only result.
    original = json.loads(next(results_dir.glob("*/*.json")).read_text())
    assert original["final_score"] == 0.0


def test_seed42_local_score_is_rejected_before_path_validation(tmp_path):
    report_path = tmp_path / "should-not-exist.json"
    result = CliRunner().invoke(
        cli,
        [
            "score",
            "--repo",
            "seed42",
            "--results-dir",
            str(tmp_path / "missing-results"),
            "--instances-dir",
            str(tmp_path / "missing-instances"),
            "--output",
            str(report_path),
        ],
    )

    assert result.exit_code != 0
    assert "seed42" in result.output
    assert "asibench submit" in result.output
    assert not report_path.exists()


def test_seed31415_missing_reference_fails_clearly(tmp_path):
    tasks_dir, instances_dir, results_dir = _write_fixture(tmp_path)
    reference = instances_dir / "physics.example__seed31415" / "reference"
    for path in reference.iterdir():
        path.unlink()
    reference.rmdir()

    result = CliRunner().invoke(
        cli,
        [
            "score",
            "--repo",
            "seed31415",
            "--results-dir",
            str(results_dir),
            "--instances-dir",
            str(instances_dir),
            "--tasks-dir",
            str(tasks_dir),
        ],
    )

    assert result.exit_code != 0
    assert "reference" in result.output.lower()


def test_local_scoring_scopes_runtime_judge_override(monkeypatch, tmp_path):
    tasks_dir, instances_dir, results_dir = _write_fixture(tmp_path)
    override = JudgeAPIOverride(
        api_base="https://api.tokenrouter.com/v1",
        api_key_env="TOKENROUTER_API_KEY",
        api_protocol="openai",
    )
    outer_override = JudgeAPIOverride(api_protocol="native")
    seen: list[JudgeAPIOverride | None] = []

    def fake_evaluate(*_args, **_kwargs):
        seen.append(get_judge_api_override())
        return ([], True, 0, [], 7.5)

    monkeypatch.setattr(
        "ai4sci_bench.runner.orchestrator._evaluate_gates_and_scores",
        fake_evaluate,
    )
    with use_judge_api_override(outer_override):
        assert get_judge_api_override() is outer_override
        report, _destination = score_seed31415_results(
            results_dir,
            instances_dir,
            tasks_dir,
            judge_api_override=override,
        )
        assert get_judge_api_override() is outer_override

    assert seen == [override]
    assert report["total_score"] == 7.5


def test_local_scoring_without_argument_preserves_outer_judge_scope(monkeypatch, tmp_path):
    tasks_dir, instances_dir, results_dir = _write_fixture(tmp_path)
    outer_override = JudgeAPIOverride(api_protocol="native")
    seen: list[JudgeAPIOverride | None] = []

    def fake_evaluate(*_args, **_kwargs):
        seen.append(get_judge_api_override())
        return ([], True, 0, [], 7.5)

    monkeypatch.setattr(
        "ai4sci_bench.runner.orchestrator._evaluate_gates_and_scores",
        fake_evaluate,
    )
    with use_judge_api_override(outer_override):
        score_seed31415_results(results_dir, instances_dir, tasks_dir)

    assert seen == [outer_override]
