"""Validate trajectory data against the expected schema."""

from __future__ import annotations

from typing import Any

from ai4sci_bench.core.trajectory import VALID_STEP_TYPES


def validate_trajectory(data: list[dict[str, Any]]) -> list[str]:
    """Validate a list of trajectory step dicts.

    Returns a list of error/warning strings. Empty list means valid.
    Errors are prefixed with "error:" and warnings with "warning:".
    """
    issues: list[str] = []

    if not isinstance(data, list):
        issues.append("error: trajectory must be a list of step dicts")
        return issues

    prev_ts: int | None = None
    for i, step in enumerate(data):
        if not isinstance(step, dict):
            issues.append(f"error: step {i} is not a dict")
            continue

        step_type = step.get("step_type")
        if step_type is None:
            issues.append(f"error: step {i} missing step_type")
        elif step_type not in VALID_STEP_TYPES:
            issues.append(f"error: step {i} has invalid step_type '{step_type}'")

        ts = step.get("timestamp_ms")
        if ts is not None:
            if not isinstance(ts, (int, float)):
                issues.append(f"warning: step {i} timestamp_ms is not numeric")
            elif prev_ts is not None and ts < prev_ts:
                issues.append(
                    f"warning: step {i} timestamp_ms {ts} < previous {prev_ts} (non-monotonic)"
                )
            prev_ts = ts

    return issues
