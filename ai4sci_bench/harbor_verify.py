"""Score one materialized public ASI-Bench task inside a Harbor verifier."""

from __future__ import annotations

import json
import math
import platform
from dataclasses import replace
from pathlib import Path
from typing import Any

from ai4sci_bench import __version__
from ai4sci_bench.benchflow import _git_revision, _sha256_tree
from ai4sci_bench.core.judge_api import JudgeAPIOverride
from ai4sci_bench.core.scorer import scoring_max_score
from ai4sci_bench.core.task import TaskLoader
from ai4sci_bench.local_scoring import (
    _ScoreJob,
    _MissingEvaluatorInputError,
    _evaluate_score_job,
    _json_default,
    _load_parameters,
    _prepare_score_runtimes,
    _redact_secret,
    _redact_scoring_logs,
    _score_jobs_parallel,
    _worker_failure_result,
)


class HarborScoringError(ValueError):
    """The Harbor task bundle is invalid or its evaluator could not score."""


def _require_directory(label: str, value: str | Path) -> Path:
    path = Path(value).absolute()
    if path.is_symlink() or not path.is_dir():
        raise HarborScoringError(f"{label} must be a real directory: {path}")
    return path


def _score_job(
    task_dir: Path,
    instance_dir: Path,
    outputs_dir: Path,
    prompt_level: str,
) -> _ScoreJob:
    if prompt_level not in {"b1", "b2", "b3", "b4"}:
        raise HarborScoringError("prompt_level must be b1, b2, b3, or b4")
    if not (task_dir / "task_meta.yaml").is_file() or not (
        task_dir / "task_eval.yaml"
    ).is_file():
        raise HarborScoringError("task_dir needs public task_meta.yaml and task_eval.yaml")
    try:
        metadata = TaskLoader(task_dir.parent).load_task_metadata(
            task_dir / "task_meta.yaml"
        )
    except Exception as exc:
        raise HarborScoringError(f"Invalid public task bundle: {exc}") from exc
    task_id = metadata.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise HarborScoringError("task_meta.yaml needs a task id")
    instance_id = f"{task_id}__seed31415"
    if instance_dir.name != instance_id:
        raise HarborScoringError(
            f"Only the matching public seed31415 instance is accepted: {instance_id}"
        )
    reference_dir = instance_dir / "reference"
    evaluation = metadata.get("evaluation")
    if not isinstance(evaluation, dict):
        raise HarborScoringError("task_eval.yaml needs an evaluation contract")
    runtime_mode = evaluation.get("runtime")
    if runtime_mode not in (None, "task"):
        raise HarborScoringError(f"Unsupported evaluation.runtime: {runtime_mode!r}")
    input_config = metadata.get("input", {})
    input_files = input_config.get("files", []) if isinstance(input_config, dict) else []
    task_runtime = None
    if runtime_mode == "task":
        task_runtime = {
            "_runtime_python": metadata.get("_runtime_python"),
            "_runtime_packages": list(metadata.get("_runtime_packages", [])),
        }
    return _ScoreJob(
        index=0,
        source_result="harbor",
        task_id=task_id,
        instance_id=instance_id,
        prompt_level=prompt_level,
        attempt=1,
        task_dir=str(task_dir.resolve()),
        output_dir=str(outputs_dir),
        reference_dir=str(reference_dir),
        data_dir=str(instance_dir / "data"),
        requires_instance_data=bool(input_files),
        evaluation=evaluation,
        parameters=_load_parameters({}, instance_dir),
        max_score=scoring_max_score(evaluation),
        task_runtime=task_runtime,
        outputs_recorded_empty=not outputs_dir.exists() and not outputs_dir.is_symlink(),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_harbor_task(
    task_dir: str | Path,
    instance_dir: str | Path,
    outputs_dir: str | Path,
    prompt_level: str,
    out: str | Path,
    *,
    judge_api_override: JudgeAPIOverride | None = None,
) -> dict[str, Any]:
    """Write Harbor reward plus full non-official diagnostics for one attempt.

    A failed evaluator writes diagnostics but deliberately leaves reward absent.
    Missing agent outputs are an ordinary zero-score submission.
    """
    destination = Path(out).absolute()
    outputs = Path(outputs_dir).absolute()
    for input_path in (Path(task_dir).absolute(), Path(instance_dir).absolute(), outputs):
        resolved_input = input_path.resolve()
        resolved_out = destination.resolve()
        if (
            resolved_out == resolved_input
            or resolved_out.is_relative_to(resolved_input)
            or resolved_input.is_relative_to(resolved_out)
        ):
            raise HarborScoringError(
                "out must be separate from task, instance, and outputs directories"
            )
    destination.mkdir(parents=True, exist_ok=True)
    reward_path = destination / "reward.json"
    reward_path.unlink(missing_ok=True)
    detail_path = destination / "score_detail.json"
    detail_path.unlink(missing_ok=True)

    task = _require_directory("task_dir", task_dir)
    instance = _require_directory("instance_dir", instance_dir)
    job = _score_job(task, instance, outputs, prompt_level)
    secret = judge_api_override.resolve_api_key() if judge_api_override else None
    try:
        reference_dir = Path(job.reference_dir)
        if (
            reference_dir.is_symlink()
            or not reference_dir.is_dir()
            or not any(reference_dir.iterdir())
        ):
            raise _MissingEvaluatorInputError(
                f"Public seed31415 reference directory is missing or empty: {reference_dir}"
            )
        job = _prepare_score_runtimes([job], tasks_root=task.parent)[0]
        if job.task_runtime is not None and job.runtime_error is None:
            result = _score_jobs_parallel(
                [replace(job, index=0)],
                parallel=1,
                judge_api_override=judge_api_override,
                progress_callback=None,
            )[0]
        else:
            with _redact_scoring_logs(secret):
                result = _evaluate_score_job(job, judge_api_override)
    except Exception as exc:
        result = _worker_failure_result(
            job,
            error_type=type(exc).__name__,
            error=_redact_secret(str(exc), secret),
            failure_kind=getattr(
                exc,
                "failure_kind",
                "evaluator_unavailable"
                if isinstance(exc, ModuleNotFoundError)
                else "evaluator_runtime_error",
            ),
            scorer_name="_harbor_scoring_runtime",
        )
    result = _redact_secret(result, secret)
    result.update(
        {
            "schema_version": 1,
            "benchmark": "ASI-Bench",
            "repo": "seed31415",
            "official": False,
            "artifact_sha256": (
                _sha256_tree(outputs)
                if outputs.is_dir() and not outputs.is_symlink()
                else None
            ),
            "task_bundle_sha256": _sha256_tree(task),
            "task_bundle_revision": _git_revision(task),
            "provenance": {
                "framework_version": __version__,
                "python_version": platform.python_version(),
            },
        }
    )
    score = result.get("final_score")
    if result["evaluation_status"] == "completed":
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            result = _worker_failure_result(
                job,
                error_type="InvalidScore",
                error="Evaluator returned a non-finite final score",
                scorer_name="_harbor_scoring_runtime",
            ) | {
                key: result[key]
                for key in (
                    "schema_version", "benchmark", "repo", "official",
                    "artifact_sha256", "task_bundle_sha256",
                    "task_bundle_revision",
                    "provenance",
                )
            }
    _write_json(detail_path, result)
    if result["evaluation_status"] != "completed":
        raise HarborScoringError(
            f"Evaluator failed ({result.get('failure_kind', 'evaluator_runtime_error')}); "
            f"inspect {detail_path}"
        )
    _write_json(reward_path, {"reward": float(result["final_score"]) / 100.0})
    return result
