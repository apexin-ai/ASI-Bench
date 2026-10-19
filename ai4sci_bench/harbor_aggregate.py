"""Aggregate Harbor verifier details using ASI-Bench evaluation semantics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


class HarborAggregateError(ValueError):
    """A Harbor score detail is malformed or no details were provided."""


def aggregate_harbor_score_details(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Exclude evaluator-invalid trials while retaining completed zero scores."""
    results: list[dict[str, Any]] = []
    total_score = 0.0
    total_max_score = 0.0
    valid_count = 0
    scorer_error_count = 0
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path in seen:
            continue
        seen.add(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HarborAggregateError(f"Cannot read score detail {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise HarborAggregateError(f"Score detail must be a JSON object: {path}")
        status = payload.get("evaluation_status")
        internal_error = payload.get("scorer_internal_error") is True
        score = payload.get("final_score")
        maximum = payload.get("max_score")
        is_valid = (
            status == "completed"
            and not internal_error
            and isinstance(score, (int, float))
            and math.isfinite(score)
            and isinstance(maximum, (int, float))
            and math.isfinite(maximum)
            and maximum >= 0
        )
        if not is_valid:
            scorer_error_count += 1
            results.append({
                "path": str(path),
                "task_id": payload.get("task_id"),
                "instance_id": payload.get("instance_id"),
                "prompt_level": payload.get("prompt_level"),
                "evaluation_status": status,
                "failure_kind": payload.get("failure_kind"),
                "included": False,
            })
            continue
        valid_count += 1
        total_score += float(score)
        total_max_score += float(maximum)
        results.append({
            "path": str(path),
            "task_id": payload.get("task_id"),
            "instance_id": payload.get("instance_id"),
            "prompt_level": payload.get("prompt_level"),
            "evaluation_status": status,
            "final_score": float(score),
            "max_score": float(maximum),
            "included": True,
        })
    if not results:
        raise HarborAggregateError("No score_detail.json files were provided")
    return {
        "schema_version": 1,
        "benchmark": "ASI-Bench",
        "repo": "seed31415",
        "official": False,
        "aggregation": "sum valid scores / sum valid max scores; evaluation_invalid excluded",
        "trial_count": len(results),
        "valid_count": valid_count,
        "scorer_error_count": scorer_error_count,
        "total_score": total_score,
        "total_max_score": total_max_score,
        "mean_percent": 100.0 * total_score / total_max_score if total_max_score else None,
        "results": results,
    }


def find_harbor_score_details(root: str | Path) -> list[Path]:
    directory = Path(root).absolute()
    if directory.is_symlink() or not directory.is_dir():
        raise HarborAggregateError(f"details_dir must be a real directory: {directory}")
    return sorted(path for path in directory.rglob("score_detail.json") if not path.is_symlink())
