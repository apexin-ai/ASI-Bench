"""Local scoring for the public-reference seed31415 benchmark contract.

The seed42 contract is intentionally excluded: its references stay on the
ASI-Bench website and results must be submitted for scoring there.
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai4sci_bench.core.task import TaskLoader
from ai4sci_bench.core.types import ScoreDetail
from ai4sci_bench.core.judge_api import JudgeAPIOverride, use_judge_api_override

PUBLIC_LOCAL_SCORING_REPO = "seed31415"
PRIVATE_SCORING_REPO = "seed42"
DEFAULT_LOCAL_SCORE_REPORT = "local_score_seed31415.json"


class LocalScoringError(RuntimeError):
    """Raised when a public local-scoring input is incomplete or inconsistent."""


def _iter_result_files(results_dir: Path):
    for path in sorted(results_dir.glob("*/*.json")):
        if ".trajectory." in path.name or ".agent_model_output." in path.name:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and data.get("task_id") and data.get("instance_id"):
            yield path, data


def _load_parameters(result: dict[str, Any], instance_dir: Path) -> dict[str, Any]:
    parameters = result.get("parameters")
    if isinstance(parameters, dict):
        return parameters
    meta_path = instance_dir / "instance_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {}
        params_used = meta.get("params_used")
        if isinstance(params_used, dict):
            return params_used
    return {}


def _detail_dict(detail: ScoreDetail) -> dict[str, Any]:
    return asdict(detail)


def _has_internal_error(details: list[ScoreDetail]) -> bool:
    return any(
        isinstance(detail.details, dict)
        and detail.details.get("scorer_internal_error") is True
        for detail in details
    )


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def score_seed31415_results(
    results_dir: str | Path,
    instances_dir: str | Path,
    tasks_dir: str | Path = "tasks",
    *,
    output_path: str | Path | None = None,
    judge_api_override: JudgeAPIOverride | None = None,
) -> tuple[dict[str, Any], Path]:
    """Score produce-only results against public seed31415 references.

    The source result JSON files are never modified. A separate report records
    local, non-official scores and full scorer details.
    """
    results_root = Path(results_dir)
    instances_root = Path(instances_dir)
    tasks_root = Path(tasks_dir)
    for label, path in (
        ("results", results_root),
        ("instances", instances_root),
        ("tasks", tasks_root),
    ):
        if not path.is_dir():
            raise LocalScoringError(f"{label} directory not found: {path}")

    result_files = list(_iter_result_files(results_root))
    if not result_files:
        raise LocalScoringError(
            f"No per-instance result JSON found under {results_root}. "
            "Run `asibench run ...` first."
        )

    # Register framework scorers once; custom task scorers are loaded per task.
    # Scientific dependencies intentionally stay in the ``full`` extra.
    try:
        import ai4sci_bench.scorers  # noqa: F401
        from ai4sci_bench.runner.orchestrator import _evaluate_gates_and_scores
        from ai4sci_bench.scorers.custom import load_custom_scorer
    except ModuleNotFoundError as exc:
        raise LocalScoringError(
            "Local scoring requires scientific dependencies. Install "
            "`asibench[full]` and retry."
        ) from exc

    loader = TaskLoader(tasks_root)
    scored_results: list[dict[str, Any]] = []
    scorer_error_count = 0
    for result_path, source in result_files:
        task_id = str(source["task_id"])
        instance_id = str(source["instance_id"])
        if not instance_id.endswith("__seed31415") and os.environ.get("ASIBENCH_ALLOW_NON_SEED_SCORE") != "1":
            raise LocalScoringError(
                f"Result {result_path} is not a seed31415 instance: {instance_id}"
            )

        instance_dir = instances_root / instance_id
        reference_dir = instance_dir / "reference"
        if not reference_dir.is_dir() or not any(reference_dir.iterdir()):
            raise LocalScoringError(
                f"Public seed31415 reference directory is missing or empty: "
                f"{reference_dir}. Re-run `asibench task pull --repo seed31415`."
            )

        output_dir = result_path.parent / f"{result_path.stem}.outputs"
        if not output_dir.is_dir():
            raise LocalScoringError(
                f"Persisted output directory not found for {result_path.name}: {output_dir}"
            )

        try:
            metadata = loader.load_task_by_id(task_id)
        except ValueError as exc:
            raise LocalScoringError(str(exc)) from exc
        evaluation = metadata.get("evaluation")
        if not isinstance(evaluation, dict):
            raise LocalScoringError(
                f"Task {task_id} has no public evaluation contract under {tasks_root}"
            )
        task_dir = Path(metadata["_task_dir"])
        try:
            load_custom_scorer(task_dir)
        except Exception as exc:
            raise LocalScoringError(
                f"Could not load public scorer for {task_id}: {type(exc).__name__}: {exc}"
            ) from exc

        prompt_level = str(source.get("prompt_level") or "")
        parameters = _load_parameters(source, instance_dir)
        # ``None`` means the caller did not supply an override; preserve an
        # outer library scope or the documented ASIBENCH_JUDGE_* environment
        # fallback.  Callers that need to suppress ambient settings can use an
        # explicit ``use_judge_api_override(None)`` scope.
        judge_scope = (
            use_judge_api_override(judge_api_override)
            if judge_api_override is not None
            else nullcontext()
        )
        with judge_scope:
            gates, hard_ok, soft_failures, scores, final_score = (
                _evaluate_gates_and_scores(
                    evaluation,
                    output_dir,
                    reference_dir,
                    parameters,
                    prompt_level=prompt_level or None,
                )
            )
        all_details = [*gates, *scores]
        internal_error = _has_internal_error(all_details)
        scorer_error_count += int(internal_error)
        max_score = float(
            sum(float(config.get("weight", 1.0)) for config in evaluation.get("scoring", []))
        )
        scored_results.append(
            {
                "source_result": str(result_path.relative_to(results_root)),
                "task_id": task_id,
                "instance_id": instance_id,
                "prompt_level": prompt_level,
                "attempt": int(source.get("attempt", 1)),
                "hard_gates_passed": hard_ok,
                "soft_gate_failures": soft_failures,
                "gate_results": [_detail_dict(item) for item in gates],
                "score_results": [_detail_dict(item) for item in scores],
                "final_score": float(final_score),
                "max_score": max_score,
                "scorer_internal_error": internal_error,
            }
        )

    total_score = sum(item["final_score"] for item in scored_results)
    total_max = sum(item["max_score"] for item in scored_results)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": PUBLIC_LOCAL_SCORING_REPO,
        "official": False,
        "score_scope": "public_local",
        "instance_count": len(scored_results),
        "scorer_error_count": scorer_error_count,
        "total_score": total_score,
        "total_max_score": total_max,
        "mean_percent": (100.0 * total_score / total_max) if total_max else 0.0,
        "results": scored_results,
    }
    destination = (
        Path(output_path)
        if output_path is not None
        else results_root / DEFAULT_LOCAL_SCORE_REPORT
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return report, destination
