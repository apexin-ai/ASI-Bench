from __future__ import annotations

import json

import pytest

from ai4sci_bench.harbor_aggregate import (
    HarborAggregateError,
    aggregate_harbor_score_details,
    find_harbor_score_details,
)


def _write(path, detail):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(detail), encoding="utf-8")
    return path


def test_aggregate_includes_valid_zero_and_excludes_evaluator_failure(tmp_path):
    zero = _write(tmp_path / "zero/score_detail.json", {
        "task_id": "math.zero", "instance_id": "math.zero__seed31415",
        "prompt_level": "b1", "evaluation_status": "completed",
        "scorer_internal_error": False, "final_score": 0.0, "max_score": 100.0,
    })
    score = _write(tmp_path / "score/score_detail.json", {
        "task_id": "math.score", "instance_id": "math.score__seed31415",
        "prompt_level": "b1", "evaluation_status": "completed",
        "scorer_internal_error": False, "final_score": 75.0, "max_score": 100.0,
    })
    invalid = _write(tmp_path / "invalid/score_detail.json", {
        "task_id": "math.invalid", "instance_id": "math.invalid__seed31415",
        "prompt_level": "b1", "evaluation_status": "evaluation_invalid",
        "scorer_internal_error": True, "final_score": None, "max_score": 100.0,
        "failure_kind": "evaluator_runtime_error",
    })

    report = aggregate_harbor_score_details([zero, score, invalid])

    assert report["trial_count"] == 3
    assert report["valid_count"] == 2
    assert report["scorer_error_count"] == 1
    assert report["total_score"] == 75.0
    assert report["total_max_score"] == 200.0
    assert report["mean_percent"] == 37.5
    assert [item["included"] for item in report["results"]] == [True, True, False]
    assert find_harbor_score_details(tmp_path) == sorted([zero, score, invalid])


def test_aggregate_rejects_empty_details(tmp_path):
    with pytest.raises(HarborAggregateError, match="No score_detail"):
        aggregate_harbor_score_details([])
