"""Public seed31415 local scoring and private seed42 boundaries."""

from __future__ import annotations

import json
import os
import platform
import time
from multiprocessing.process import BaseProcess
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from ai4sci_bench.cli import cli
from ai4sci_bench.core.judge_api import (
    JudgeAPIOverride,
    get_judge_api_override,
    use_judge_api_override,
)
from ai4sci_bench.core.scorer import (
    failure_score_metadata,
    normalize_task_score,
)
from ai4sci_bench.core.types import ScoreDetail
from ai4sci_bench.local_scoring import (
    LocalScoringError,
    score_seed31415_results,
)
from ai4sci_bench.runner.task_env import TaskEnvironment, TaskEnvironmentManager


def test_task_score_divisor_normalizes_score_and_max_score():
    score, maximum = normalize_task_score(
        {"score_divisor": 1.05},
        105.0,
        105.0,
    )
    assert score == pytest.approx(100.0)
    assert maximum == pytest.approx(100.0)


def test_task_score_divisor_defaults_to_one():
    score, maximum = normalize_task_score({}, 42.0, 100.0)
    assert score == 42.0
    assert maximum == 100.0


@pytest.mark.parametrize("divisor", [False, 0, -1, float("inf"), "invalid"])
def test_invalid_task_score_divisor_is_rejected(divisor):
    with pytest.raises(ValueError, match="finite positive number"):
        normalize_task_score({"score_divisor": divisor}, 42.0, 100.0)


def test_worker_failure_preserves_raw_max_for_invalid_score_divisor():
    assert failure_score_metadata({"score_divisor": 0}, 105.0) == (105.0, None)


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


def _add_result_level(results_dir: Path, level: str) -> None:
    task_dir = results_dir / "physics.example"
    instance_id = "physics.example__seed31415"
    outputs_dir = task_dir / f"{instance_id}__{level}.outputs"
    outputs_dir.mkdir(parents=True)
    np.save(outputs_dir / "output.npy", np.array([1.0, 2.0], dtype=np.float64))
    result_json = {
        "instance_id": instance_id,
        "task_id": "physics.example",
        "prompt_level": level,
        "parameters": {},
        "status": "completed",
        "final_score": 0.0,
    }
    (task_dir / f"{instance_id}__{level}.json").write_text(
        json.dumps(result_json), encoding="utf-8"
    )


def _enable_task_scoring_runtime(tasks_dir: Path) -> None:
    task_dir = tasks_dir / "physics" / "example"
    eval_path = task_dir / "task_eval.yaml"
    eval_path.write_text(
        eval_path.read_text(encoding="utf-8").replace(
            "evaluation:\n",
            "evaluation:\n  runtime: task\n",
            1,
        ),
        encoding="utf-8",
    )
    meta_path = task_dir / "task_meta.yaml"
    meta_path.write_text(
        meta_path.read_text(encoding="utf-8").replace(
            "  packages: []",
            "  packages:\n    - fake-evaluator-dependency==1.0",
            1,
        ),
        encoding="utf-8",
    )


def _fake_task_environment(root: Path) -> TaskEnvironment:
    env_dir = root / "task-env"
    site_packages = env_dir / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    (site_packages / "fake_evaluator_dependency.py").write_text(
        "AVAILABLE = True\n",
        encoding="utf-8",
    )
    return TaskEnvironment(
        env_dir=env_dir,
        python_executable=env_dir / "bin" / "python",
        bin_dir=env_dir / "bin",
        cache_key="fake-runtime",
        python_requirement=">=3.11",
        packages=["fake-evaluator-dependency==1.0"],
        cache_hit=True,
        resolved_python_version=platform.python_version(),
    )


def _write_parallel_test_scorer(
    tasks_dir: Path,
    state_dir: Path,
    *,
    crash_level: str | None = None,
) -> None:
    task_dir = tasks_dir / "physics" / "example"
    scorer_source = f'''
import os
from pathlib import Path
import sys
import time

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail


@register_scorer("parallel_test_scorer")
class ParallelTestScorer(Scorer):
    def score(self, pred_dir, ref_dir, config):
        level = config.get("prompt_level", "")
        state_dir = Path(config["state_dir"])
        state_dir.mkdir(parents=True, exist_ok=True)
        marker = state_dir / f"{{level}}.started"
        marker.write_text(str(os.getpid()))
        active_count = 1
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            active_count = max(active_count, len(list(state_dir.glob("*.started"))))
            if active_count >= 2:
                break
            time.sleep(0.01)
        (state_dir / f"{{level}}.ready").write_text(str(os.getpid()))
        deadline = time.monotonic() + 3.0
        while len(list(state_dir.glob("*.ready"))) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        if level == {crash_level!r}:
            os._exit(23)
        if level == "b1":
            deadline = time.monotonic() + 3.0
            while not (state_dir / "b3.started").exists() and time.monotonic() < deadline:
                time.sleep(0.01)

        leak_path = str(state_dir / "sys-path-leak")
        leak_module = "asibench_parallel_scorer_leak"
        isolation_clean = (
            "ASIBENCH_SCORER_LEAK" not in os.environ
            and leak_path not in sys.path
            and leak_module not in sys.modules
        )
        os.environ["ASIBENCH_SCORER_LEAK"] = level
        sys.path.append(leak_path)
        sys.modules[leak_module] = object()
        os.chdir(pred_dir)

        marker.unlink(missing_ok=True)
        concurrent = active_count >= 2
        weight = float(config.get("weight", 1.0))
        return ScoreDetail(
            scorer_name="parallel_test_scorer",
            score=weight if concurrent else 0.0,
            max_score=weight,
            passed=concurrent,
            details={{
                "active_count": active_count,
                "concurrent": concurrent,
                "isolation_clean": isolation_clean,
                "pid": os.getpid(),
            }},
        )
'''.lstrip()
    (task_dir / "custom_scorer.py").write_text(scorer_source, encoding="utf-8")
    state_json = json.dumps(str(state_dir))
    (task_dir / "task_eval.yaml").write_text(
        f"""
task_id: physics.example
evaluation:
  gates: []
  scoring:
    - scorer: parallel_test_scorer
      weight: 100
      config:
        state_dir: {state_json}
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_judge_test_scorer(tasks_dir: Path, *, raise_with_secret: bool) -> None:
    task_dir = tasks_dir / "physics" / "example"
    scorer_source = f'''
from ai4sci_bench.core.judge_api import get_judge_api_override
from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail


@register_scorer("judge_override_test_scorer")
class JudgeOverrideTestScorer(Scorer):
    def score(self, pred_dir, ref_dir, config):
        override = get_judge_api_override()
        secret = override.resolve_api_key() if override is not None else None
        if {raise_with_secret!r}:
            raise RuntimeError(f"provider rejected credential {{secret}}")
        weight = float(config.get("weight", 1.0))
        return ScoreDetail(
            scorer_name="judge_override_test_scorer",
            score=weight,
            max_score=weight,
            passed=True,
            details={{
                "override": override.public_metadata() if override else None,
                "secret_resolved": bool(secret),
            }},
        )
'''.lstrip()
    (task_dir / "custom_scorer.py").write_text(scorer_source, encoding="utf-8")
    (task_dir / "task_eval.yaml").write_text(
        """
task_id: physics.example
evaluation:
  gates: []
  scoring:
    - scorer: judge_override_test_scorer
      weight: 100
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_instance_data_test_scorer(
    tasks_dir: Path,
    instances_dir: Path,
    *,
    declared_name: str = "data/input.txt",
) -> None:
    task_dir = tasks_dir / "physics" / "example"
    task_meta = task_dir / "task_meta.yaml"
    task_meta.write_text(
        task_meta.read_text(encoding="utf-8").replace(
            "input:\n  files: []",
            f"input:\n  files:\n    - name: {declared_name}\n      type: data",
        ),
        encoding="utf-8",
    )
    data_dir = instances_dir / "physics.example__seed31415" / "data"
    data_dir.mkdir()
    (data_dir / "input.txt").write_text("immutable-input", encoding="utf-8")
    (task_dir / "custom_scorer.py").write_text(
        '''
from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail


@register_scorer("instance_data_test_scorer")
class InstanceDataTestScorer(Scorer):
    def score(self, pred_dir, ref_dir, config):
        data = (pred_dir / "data/input.txt").read_text(encoding="utf-8")
        output_exists = (pred_dir / "output.npy").is_file()
        weight = float(config.get("weight", 1.0))
        passed = data == "immutable-input" and output_exists
        return ScoreDetail(
            scorer_name="instance_data_test_scorer",
            score=weight if passed else 0.0,
            max_score=weight,
            passed=passed,
            details={"data": data, "output_exists": output_exists},
        )
'''.lstrip(),
        encoding="utf-8",
    )
    (task_dir / "task_eval.yaml").write_text(
        """
task_id: physics.example
evaluation:
  gates: []
  scoring:
    - scorer: instance_data_test_scorer
      weight: 100
""".strip()
        + "\n",
        encoding="utf-8",
    )


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
    assert "Scoring [1/1] physics.example b1" in result.output
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


def test_internal_scorer_error_is_unscored_and_excluded_from_totals(
    monkeypatch, tmp_path
):
    tasks_dir, instances_dir, results_dir = _write_fixture(tmp_path)
    internal_error = ScoreDetail(
        scorer_name="numerical",
        score=0.0,
        max_score=100.0,
        passed=False,
        details={
            "failure_kind": "scorer_internal_error",
            "scorer_internal_error": True,
        },
        message="reference bundle is incomplete",
    )
    monkeypatch.setattr(
        "ai4sci_bench.runner.orchestrator._evaluate_gates_and_scores",
        lambda *_args, **_kwargs: ([], True, 0, [internal_error], 0.0),
    )

    report, _destination = score_seed31415_results(
        results_dir,
        instances_dir,
        tasks_dir,
    )

    assert report["schema_version"] == 2
    assert report["instance_count"] == 1
    assert report["scored_instance_count"] == 0
    assert report["scorer_error_count"] == 1
    assert report["total_score"] == 0.0
    assert report["total_max_score"] == 0.0
    assert report["mean_percent"] is None
    scored = report["results"][0]
    assert scored["evaluation_status"] == "evaluation_invalid"
    assert scored["final_score"] is None
    assert scored["max_score"] == 100.0
    assert scored["failure_kind"] == "evaluator_runtime_error"


def test_evaluator_unavailable_is_unscored_and_classified(monkeypatch, tmp_path):
    from ai4sci_bench.scorers._judge_common import evaluator_unavailable_result

    tasks_dir, instances_dir, results_dir = _write_fixture(tmp_path)
    unavailable = evaluator_unavailable_result(
        scorer_name="llm_judge",
        weight=100.0,
        error="provider unavailable",
    )
    monkeypatch.setattr(
        "ai4sci_bench.runner.orchestrator._evaluate_gates_and_scores",
        lambda *_args, **_kwargs: ([], True, 0, [unavailable], 0.0),
    )

    report, _destination = score_seed31415_results(
        results_dir,
        instances_dir,
        tasks_dir,
    )

    scored = report["results"][0]
    assert scored["evaluation_status"] == "evaluation_invalid"
    assert scored["final_score"] is None
    assert scored["scorer_internal_error"] is True
    assert scored["failure_kind"] == "evaluator_unavailable"
    assert report["scored_instance_count"] == 0
    assert report["scorer_error_count"] == 1
    assert report["total_max_score"] == 0.0


@pytest.mark.parametrize(
    ("parallel", "declared_name"),
    [
        (1, "data/input.txt"),
        (2, "input.txt"),
        (1, "pairs/<ii>/source.npy"),
    ],
)
def test_local_scoring_stages_instance_data_without_mutating_outputs(
    tmp_path, parallel, declared_name
):
    tasks_dir, instances_dir, results_dir = _write_fixture(tmp_path)
    _write_instance_data_test_scorer(
        tasks_dir,
        instances_dir,
        declared_name=declared_name,
    )
    outputs_dir = next(results_dir.glob("*/*.outputs"))

    report, _destination = score_seed31415_results(
        results_dir,
        instances_dir,
        tasks_dir,
        parallel=parallel,
    )

    scored = report["results"][0]
    assert scored["evaluation_status"] == "completed"
    assert scored["final_score"] == 100.0
    assert scored["score_results"][0]["details"] == {
        "data": "immutable-input",
        "output_exists": True,
    }
    assert not (outputs_dir / "data").exists()


@pytest.mark.parametrize("parallel", [1, 2])
def test_missing_instance_data_is_invalid_and_excluded(tmp_path, parallel):
    tasks_dir, instances_dir, results_dir = _write_fixture(tmp_path)
    _write_instance_data_test_scorer(tasks_dir, instances_dir)
    data_file = (
        instances_dir
        / "physics.example__seed31415"
        / "data"
        / "input.txt"
    )
    data_file.unlink()
    data_file.parent.rmdir()

    report, _destination = score_seed31415_results(
        results_dir,
        instances_dir,
        tasks_dir,
        parallel=parallel,
    )

    scored = report["results"][0]
    assert scored["evaluation_status"] == "evaluation_invalid"
    assert scored["final_score"] is None
    assert scored["scorer_internal_error"] is True
    assert scored["failure_kind"] == "missing_evaluator_input"
    assert scored["score_results"][0]["details"]["failure_kind"] == (
        "missing_evaluator_input"
    )
    assert report["scored_instance_count"] == 0
    assert report["scorer_error_count"] == 1
    assert report["total_max_score"] == 0.0


@pytest.mark.parametrize("unsafe_input", ["symlink", "output-overwrite"])
def test_local_scoring_rejects_unsafe_staging_inputs(tmp_path, unsafe_input):
    tasks_dir, instances_dir, results_dir = _write_fixture(tmp_path)
    _write_instance_data_test_scorer(tasks_dir, instances_dir)
    data_file = (
        instances_dir
        / "physics.example__seed31415"
        / "data"
        / "input.txt"
    )
    if unsafe_input == "symlink":
        external = tmp_path / "external-input.txt"
        external.write_text("immutable-input", encoding="utf-8")
        data_file.unlink()
        data_file.symlink_to(external)
    else:
        output_data = next(results_dir.glob("*/*.outputs")) / "data"
        output_data.mkdir()
        (output_data / "input.txt").write_text("agent-overwrite", encoding="utf-8")

    report, _destination = score_seed31415_results(
        results_dir,
        instances_dir,
        tasks_dir,
    )

    scored = report["results"][0]
    assert scored["evaluation_status"] == "evaluation_invalid"
    assert scored["failure_kind"] == "missing_evaluator_input"
    assert scored["final_score"] is None


def test_legitimate_zero_remains_a_scored_result(monkeypatch, tmp_path):
    tasks_dir, instances_dir, results_dir = _write_fixture(tmp_path)
    scored_zero = ScoreDetail(
        scorer_name="numerical",
        score=0.0,
        max_score=100.0,
        passed=False,
        details={
            "relative_l2": 10.0,
            "failure_kind": "submission_error",
            "scorer_internal_error": False,
        },
        message="prediction is outside tolerance",
    )
    monkeypatch.setattr(
        "ai4sci_bench.runner.orchestrator._evaluate_gates_and_scores",
        lambda *_args, **_kwargs: ([], True, 0, [scored_zero], 0.0),
    )

    report, _destination = score_seed31415_results(
        results_dir,
        instances_dir,
        tasks_dir,
    )

    assert report["scored_instance_count"] == 1
    assert report["scorer_error_count"] == 0
    assert report["total_score"] == 0.0
    assert report["total_max_score"] == 100.0
    assert report["mean_percent"] == 0.0
    scored = report["results"][0]
    assert scored["evaluation_status"] == "completed"
    assert scored["final_score"] == 0.0
    assert "failure_kind" not in scored


def test_score_cli_displays_internal_error_as_not_scored(monkeypatch, tmp_path):
    report_path = tmp_path / "score.json"
    monkeypatch.setattr(
        "ai4sci_bench.local_scoring.score_seed31415_results",
        lambda *_args, **_kwargs: (
            {
                "results": [
                    {
                        "instance_id": "physics.example__seed31415",
                        "prompt_level": "b1",
                        "evaluation_status": "evaluation_invalid",
                        "final_score": None,
                        "max_score": 100.0,
                    }
                ],
                "scored_instance_count": 0,
                "scorer_error_count": 1,
                "total_score": 0.0,
                "total_max_score": 0.0,
                "mean_percent": None,
            },
            report_path,
        ),
    )

    result = CliRunner().invoke(
        cli,
        [
            "score",
            "--repo",
            "seed31415",
            "--results-dir",
            str(tmp_path / "results"),
            "--instances-dir",
            str(tmp_path / "instances"),
        ],
    )

    assert result.exit_code != 0
    assert "NOT SCORED (internal scorer error)" in result.output
    assert "0.00 / 100.00" not in result.output
    assert "Total: no valid scores" in result.output


def test_local_scoring_scopes_runtime_judge_override(monkeypatch, tmp_path):
    tasks_dir, instances_dir, results_dir = _write_fixture(tmp_path)
    monkeypatch.setenv("TOKENROUTER_API_KEY", "test-secret")
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


def test_regular_local_scoring_does_not_prepare_a_task_runtime(monkeypatch, tmp_path):
    tasks_dir, instances_dir, results_dir = _write_fixture(tmp_path)

    def unexpected_runtime(*_args, **_kwargs):
        raise AssertionError("ordinary local scoring must not prepare a task runtime")

    monkeypatch.setattr(TaskEnvironmentManager, "ensure_env", unexpected_runtime)

    report, _destination = score_seed31415_results(
        results_dir,
        instances_dir,
        tasks_dir,
    )

    assert report["scored_instance_count"] == 1
    assert report["results"][0]["final_score"] == 100.0


def test_opted_in_local_scoring_prepares_one_runtime_and_activates_it(
    monkeypatch, tmp_path
):
    tasks_dir, instances_dir, results_dir = _write_fixture(tmp_path)
    _add_result_level(results_dir, "b2")
    _add_result_level(results_dir, "b3")
    _enable_task_scoring_runtime(tasks_dir)
    task_dir = tasks_dir / "physics" / "example"
    (task_dir / "custom_scorer.py").write_text(
        "import fake_evaluator_dependency\n",
        encoding="utf-8",
    )
    environment = _fake_task_environment(tmp_path)
    calls: list[dict[str, object]] = []

    def prepare_runtime(_manager, spec):
        calls.append(spec)
        return environment

    monkeypatch.setattr(TaskEnvironmentManager, "ensure_env", prepare_runtime)

    report, _destination = score_seed31415_results(
        results_dir,
        instances_dir,
        tasks_dir,
        parallel=1,
    )

    assert len(calls) == 1
    assert calls[0]["_runtime_packages"] == [
        "fake-evaluator-dependency==1.0"
    ]
    assert report["scored_instance_count"] == 3
    assert [item["final_score"] for item in report["results"]] == [100.0] * 3


def test_opted_in_runtime_setup_failure_is_evaluator_unavailable(
    monkeypatch, tmp_path
):
    tasks_dir, instances_dir, results_dir = _write_fixture(tmp_path)
    _enable_task_scoring_runtime(tasks_dir)

    def unavailable_runtime(_manager, _spec):
        raise FileNotFoundError("runtime installer is unavailable")

    monkeypatch.setattr(TaskEnvironmentManager, "ensure_env", unavailable_runtime)

    report, _destination = score_seed31415_results(
        results_dir,
        instances_dir,
        tasks_dir,
    )

    scored = report["results"][0]
    assert scored["evaluation_status"] == "evaluation_invalid"
    assert scored["final_score"] is None
    assert scored["scorer_internal_error"] is True
    assert scored["failure_kind"] == "evaluator_unavailable"


def test_opted_in_runtime_rejects_a_different_python_minor(monkeypatch, tmp_path):
    tasks_dir, instances_dir, results_dir = _write_fixture(tmp_path)
    _enable_task_scoring_runtime(tasks_dir)
    environment = _fake_task_environment(tmp_path)
    environment.resolved_python_version = "99.1.0"
    monkeypatch.setattr(
        TaskEnvironmentManager,
        "ensure_env",
        lambda _manager, _spec: environment,
    )

    report, _destination = score_seed31415_results(
        results_dir,
        instances_dir,
        tasks_dir,
    )

    scored = report["results"][0]
    assert scored["evaluation_status"] == "evaluation_invalid"
    assert scored["failure_kind"] == "evaluator_unavailable"
    assert "current Python major/minor" in scored["score_results"][0]["message"]


def test_parallel_local_scoring_is_bounded_isolated_and_ordered(tmp_path):
    tasks_dir, instances_dir, results_dir = _write_fixture(tmp_path)
    _add_result_level(results_dir, "b2")
    _add_result_level(results_dir, "b3")
    _write_parallel_test_scorer(tasks_dir, tmp_path / "parallel-state")
    progress: list[tuple[int, int, str]] = []

    report, _destination = score_seed31415_results(
        results_dir,
        instances_dir,
        tasks_dir,
        parallel=2,
        progress_callback=lambda completed, total, item: progress.append(
            (completed, total, item["prompt_level"])
        ),
    )

    assert [item["prompt_level"] for item in report["results"]] == ["b1", "b2", "b3"]
    assert [item["final_score"] for item in report["results"]] == [100.0] * 3
    worker_pids = {
        item["score_results"][0]["details"]["pid"] for item in report["results"]
    }
    assert len(worker_pids) == 3
    assert os.getpid() not in worker_pids
    details = [item["score_results"][0]["details"] for item in report["results"]]
    assert all(item["isolation_clean"] for item in details)
    assert max(item["active_count"] for item in details) == 2
    assert "ASIBENCH_SCORER_LEAK" not in os.environ
    assert [item[0] for item in progress] == [1, 2, 3]
    assert all(item[1] == 3 for item in progress)
    assert {item[2] for item in progress} == {"b1", "b2", "b3"}


def test_parallel_worker_crash_is_an_invalid_evaluation(tmp_path):
    tasks_dir, instances_dir, results_dir = _write_fixture(tmp_path)
    _add_result_level(results_dir, "b2")
    _write_parallel_test_scorer(
        tasks_dir,
        tmp_path / "parallel-state",
        crash_level="b1",
    )

    report, _destination = score_seed31415_results(
        results_dir,
        instances_dir,
        tasks_dir,
        parallel=2,
    )

    assert report["instance_count"] == 2
    assert report["scored_instance_count"] == 1
    assert report["scorer_error_count"] == 1
    scored = report["results"][0]
    assert scored["evaluation_status"] == "evaluation_invalid"
    assert scored["final_score"] is None
    assert scored["scorer_internal_error"] is True
    assert scored["failure_kind"] == "evaluator_runtime_error"
    assert scored["score_results"][0]["details"]["scorer_internal_error"] is True
    assert scored["score_results"][0]["details"]["failure_kind"] == (
        "evaluator_runtime_error"
    )
    assert report["results"][1]["evaluation_status"] == "completed"


@pytest.mark.parametrize("parallel", [0, -1, 1.5, True, "2"])
def test_local_scoring_rejects_invalid_library_parallel_values(tmp_path, parallel):
    with pytest.raises(LocalScoringError, match="parallel"):
        score_seed31415_results(
            tmp_path / "missing-results",
            tmp_path / "missing-instances",
            parallel=parallel,
        )


def test_preflight_validates_later_jobs_before_loading_or_running_scorer(
    monkeypatch, tmp_path
):
    tasks_dir, instances_dir, results_dir = _write_fixture(tmp_path)
    _add_result_level(results_dir, "b2")
    late_outputs = (
        results_dir
        / "physics.example"
        / "physics.example__seed31415__b2.outputs"
    )
    (late_outputs / "output.npy").unlink()
    late_outputs.rmdir()
    import_marker = tmp_path / "custom-scorer-imported"
    (tasks_dir / "physics" / "example" / "custom_scorer.py").write_text(
        f"from pathlib import Path\nPath({str(import_marker)!r}).write_text('loaded')\n",
        encoding="utf-8",
    )
    evaluate_calls = []
    monkeypatch.setattr(
        "ai4sci_bench.runner.orchestrator._evaluate_gates_and_scores",
        lambda *_args, **_kwargs: evaluate_calls.append(True),
    )

    with pytest.raises(LocalScoringError, match="Persisted output directory"):
        score_seed31415_results(results_dir, instances_dir, tasks_dir)

    assert evaluate_calls == []
    assert not import_marker.exists()


def test_parallel_worker_start_failure_does_not_drop_other_results(
    monkeypatch, tmp_path
):
    tasks_dir, instances_dir, results_dir = _write_fixture(tmp_path)
    _add_result_level(results_dir, "b2")
    original_start = BaseProcess.start

    def fail_first_worker(process):
        if process.name == "asibench-score-0":
            raise OSError("worker capacity unavailable")
        return original_start(process)

    monkeypatch.setattr(BaseProcess, "start", fail_first_worker)
    report, _destination = score_seed31415_results(
        results_dir,
        instances_dir,
        tasks_dir,
        parallel=2,
    )

    assert report["instance_count"] == 2
    assert report["scorer_error_count"] == 1
    assert report["results"][0]["evaluation_status"] == "evaluation_invalid"
    assert report["results"][0]["final_score"] is None
    assert report["results"][1]["evaluation_status"] == "completed"
    assert "WorkerStartError" in report["results"][0]["score_results"][0]["message"]


@pytest.mark.parametrize("parallel", [1, 2])
def test_local_scoring_receives_judge_override_without_persisting_secret(
    monkeypatch, tmp_path, parallel
):
    tasks_dir, instances_dir, results_dir = _write_fixture(tmp_path)
    _write_judge_test_scorer(tasks_dir, raise_with_secret=False)
    secret = "parallel-judge-secret"
    monkeypatch.setenv("TEST_JUDGE_KEY", secret)
    override = JudgeAPIOverride(
        api_base="https://api.example.test/v1",
        api_key_env="TEST_JUDGE_KEY",
        api_protocol="openai",
    )

    report, _destination = score_seed31415_results(
        results_dir,
        instances_dir,
        tasks_dir,
        judge_api_override=override,
        parallel=parallel,
    )

    details = report["results"][0]["score_results"][0]["details"]
    assert details["override"] == override.public_metadata()
    assert details["secret_resolved"] is True
    assert secret not in json.dumps(report)


@pytest.mark.parametrize("parallel", [1, 2])
def test_local_scoring_redacts_judge_secret_from_scorer_failure(
    monkeypatch, tmp_path, parallel
):
    tasks_dir, instances_dir, results_dir = _write_fixture(tmp_path)
    _write_judge_test_scorer(tasks_dir, raise_with_secret=True)
    secret = "parallel-secret-in-error"
    monkeypatch.setenv("TEST_JUDGE_KEY", secret)
    override = JudgeAPIOverride(
        api_base="https://api.example.test/v1",
        api_key_env="TEST_JUDGE_KEY",
        api_protocol="openai",
    )

    report, _destination = score_seed31415_results(
        results_dir,
        instances_dir,
        tasks_dir,
        judge_api_override=override,
        parallel=parallel,
    )

    serialized = json.dumps(report)
    assert report["results"][0]["evaluation_status"] == "evaluation_invalid"
    assert secret not in serialized
    assert "<redacted>" in serialized


def test_serial_and_parallel_local_scoring_have_equivalent_score_content(tmp_path):
    serial_paths = _write_fixture(tmp_path / "serial")
    parallel_paths = _write_fixture(tmp_path / "parallel")
    _add_result_level(serial_paths[2], "b2")
    _add_result_level(parallel_paths[2], "b2")

    serial_report, _ = score_seed31415_results(
        serial_paths[2], serial_paths[1], serial_paths[0], parallel=1
    )
    parallel_report, _ = score_seed31415_results(
        parallel_paths[2], parallel_paths[1], parallel_paths[0], parallel=2
    )

    assert serial_report["results"] == parallel_report["results"]
    for field in (
        "instance_count",
        "scored_instance_count",
        "scorer_error_count",
        "total_score",
        "total_max_score",
        "mean_percent",
    ):
        assert serial_report[field] == parallel_report[field]


def test_atomic_report_replace_failure_preserves_existing_report(monkeypatch, tmp_path):
    tasks_dir, instances_dir, results_dir = _write_fixture(tmp_path)
    report_path = tmp_path / "score.json"
    report_path.write_text("previous-report\n", encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        score_seed31415_results(
            results_dir,
            instances_dir,
            tasks_dir,
            output_path=report_path,
        )

    assert report_path.read_text(encoding="utf-8") == "previous-report\n"
    assert list(tmp_path.glob(".score.json.tmp-*")) == []
