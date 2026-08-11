"""Generic fallback extractor — wraps raw output as a single trajectory step."""

from __future__ import annotations

from ai4sci_bench.core.trajectory import Trajectory, TrajectoryStep, TrajectorySummary


def extract_from_raw(text: str, instance_id: str = "", adapter_type: str = "generic") -> Trajectory:
    """Wrap raw text output as a single-step trajectory."""
    steps = [TrajectoryStep(
        step_index=0,
        timestamp_ms=None,
        step_type="llm_response",
        content=text,
    )]
    return Trajectory(
        instance_id=instance_id,
        adapter_type=adapter_type,
        total_steps=1,
        total_duration_ms=None,
        steps=steps,
        summary=TrajectorySummary.from_steps(steps),
    )
