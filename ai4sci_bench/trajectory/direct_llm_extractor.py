"""Extract structured trajectory from Direct LLM adapter structured log."""

from __future__ import annotations

import json

from ai4sci_bench.core.trajectory import Trajectory, TrajectoryStep, TrajectorySummary


def extract_from_structured_log(log_text: str, instance_id: str = "") -> Trajectory:
    """Parse Direct LLM structured log (JSON) into a Trajectory object.

    The structured log is a JSON array of step objects produced by
    DirectLLMAdapter.solve() when TODO-2 is implemented.
    """
    steps: list[TrajectoryStep] = []
    step_idx = 0

    try:
        structured_log = json.loads(log_text)
    except json.JSONDecodeError:
        return _wrap_raw_text(log_text, instance_id)

    if not isinstance(structured_log, list):
        return _wrap_raw_text(log_text, instance_id)

    for entry in structured_log:
        if not isinstance(entry, dict):
            continue
        step_name = entry.get("step", "unknown")

        if step_name == "prompt":
            steps.append(TrajectoryStep(
                step_index=step_idx,
                timestamp_ms=entry.get("timestamp_ms"),
                step_type="llm_request",
                content=entry.get("user_prompt", ""),
                metadata={
                    "system_prompt": entry.get("system_prompt", ""),
                    "model": entry.get("model", ""),
                },
            ))
            step_idx += 1

        elif step_name == "llm_response":
            steps.append(TrajectoryStep(
                step_index=step_idx,
                timestamp_ms=entry.get("timestamp_ms"),
                step_type="llm_response",
                content=entry.get("content", ""),
                metadata={
                    "content_length": entry.get("content_length", 0),
                    "usage": entry.get("usage", {}),
                    "latency_ms": entry.get("latency_ms"),
                },
            ))
            step_idx += 1

        elif step_name == "code_extraction":
            steps.append(TrajectoryStep(
                step_index=step_idx,
                timestamp_ms=entry.get("timestamp_ms"),
                step_type="system",
                content=f"Extracted code: {entry.get('num_candidates', 0)} candidates",
                metadata={
                    "num_candidates": entry.get("num_candidates", 0),
                    "candidate_scores": entry.get("candidate_scores", []),
                    "selected_index": entry.get("selected_index"),
                    "selected_code_length": entry.get("selected_code_length", 0),
                },
            ))
            step_idx += 1

        elif step_name == "code_execution":
            exit_code = entry.get("exit_code", -1)
            steps.append(TrajectoryStep(
                step_index=step_idx,
                timestamp_ms=entry.get("timestamp_ms"),
                step_type="code_execution",
                content=f"Executed {entry.get('code_file', 'unknown')}",
                metadata={
                    "code_file": entry.get("code_file", ""),
                    "exit_code": exit_code,
                    "duration_ms": entry.get("duration_ms"),
                    "stdout_length": entry.get("stdout_length", 0),
                    "stderr_length": entry.get("stderr_length", 0),
                },
            ))
            step_idx += 1

        elif step_name == "error":
            steps.append(TrajectoryStep(
                step_index=step_idx,
                timestamp_ms=entry.get("timestamp_ms"),
                step_type="error",
                content=entry.get("error", ""),
                metadata={"error_type": entry.get("error_type", "")},
            ))
            step_idx += 1

    summary = TrajectorySummary.from_steps(steps)
    return Trajectory(
        instance_id=instance_id,
        adapter_type="direct_llm",
        total_steps=len(steps),
        total_duration_ms=None,
        steps=steps,
        summary=summary,
    )


def _wrap_raw_text(text: str, instance_id: str) -> Trajectory:
    """Wrap raw text as a single-step trajectory (fallback)."""
    steps = [TrajectoryStep(
        step_index=0,
        timestamp_ms=None,
        step_type="llm_response",
        content=text,
    )]
    return Trajectory(
        instance_id=instance_id,
        adapter_type="direct_llm",
        total_steps=1,
        total_duration_ms=None,
        steps=steps,
        summary=TrajectorySummary.from_steps(steps),
    )
