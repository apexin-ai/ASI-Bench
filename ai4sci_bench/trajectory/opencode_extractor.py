"""Extract structured trajectory from opencode ``run --format json`` streams.

opencode (verified v1.17.15) emits one JSON object per line, each of the
shape ``{"type": <name>, "timestamp": <ms>, "sessionID": <id>, ...payload}``:

- ``text`` / ``reasoning`` — completed assistant parts (payload ``part``);
- ``tool_use`` — a tool part in its final state (``part.tool``,
  ``part.state.status`` in ``running``/``completed``/``error``, with
  ``input``/``output``/``error``);
- ``step_start`` / ``step_finish`` — model step boundaries; ``step_finish``
  parts carry ``tokens`` (``input``/``output``/``reasoning``/``cache``) and
  ``cost``;
- ``error`` — terminal session error (``error.name`` + ``error.data``).

Known limitation: file version reconstruction
(``file_history.extract_file_versions()``) is not wired for opencode; file
edits surface only as tool_call steps with key arguments.
"""

from __future__ import annotations

import json

from ai4sci_bench.core.trajectory import Trajectory, TrajectoryStep, TrajectorySummary

_VALUE_KEYS = ("filePath", "file_path", "path", "command", "pattern", "url", "query")


def _timestamp_ms(event: dict) -> int | None:
    raw = event.get("timestamp")
    if raw is None:
        return None
    try:
        return int(raw)
    except (ValueError, TypeError):
        return None


def _key_args(inputs) -> dict:
    if not isinstance(inputs, dict):
        return {}
    return {k: str(inputs[k]) for k in _VALUE_KEYS if k in inputs}


def extract_from_jsonl(jsonl_text: str, instance_id: str = "") -> Trajectory:
    """Parse opencode ``--format json`` stream into a Trajectory object."""
    steps: list[TrajectoryStep] = []
    step_idx = 0

    for line in jsonl_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            steps.append(TrajectoryStep(
                step_index=step_idx,
                timestamp_ms=None,
                step_type="error",
                content=f"Malformed JSON: {stripped[:200]}",
                metadata={"parse_error": True},
            ))
            step_idx += 1
            continue

        etype = event.get("type", "")
        ts_ms = _timestamp_ms(event)
        part = event.get("part")
        part = part if isinstance(part, dict) else {}

        if etype == "text":
            text = str(part.get("text", ""))
            if text:
                steps.append(TrajectoryStep(
                    step_index=step_idx,
                    timestamp_ms=ts_ms,
                    step_type="llm_response",
                    content=text,
                    metadata={"role": "assistant"},
                ))
                step_idx += 1
        elif etype == "reasoning":
            text = str(part.get("text", ""))
            if text:
                steps.append(TrajectoryStep(
                    step_index=step_idx,
                    timestamp_ms=ts_ms,
                    step_type="thinking",
                    content=text,
                    metadata={"thinking_length": len(text)},
                ))
                step_idx += 1
        elif etype == "tool_use":
            tool_name = str(part.get("tool", "unknown"))
            state = part.get("state")
            state = state if isinstance(state, dict) else {}
            status = str(state.get("status", ""))
            metadata = {
                "tool_name": tool_name,
                "key_args": _key_args(state.get("input")),
                "status": status,
                "tool_call_id": str(part.get("callID", "") or ""),
            }
            if status == "error":
                metadata["is_error"] = True
                error = state.get("error")
                content = f"{tool_name} failed: {error}"
            else:
                output = state.get("output")
                content = str(output) if output is not None else tool_name
            steps.append(TrajectoryStep(
                step_index=step_idx,
                timestamp_ms=ts_ms,
                step_type="tool_call",
                content=content[:2000],
                metadata=metadata,
            ))
            step_idx += 1
        elif etype == "error":
            error = event.get("error", {})
            if isinstance(error, dict):
                data = error.get("data")
                message = data.get("message") if isinstance(data, dict) else None
                content = f"{error.get('name', 'UnknownError')}: {message or error}"
                name = str(error.get("name", "UnknownError"))
            else:
                content = str(error)
                name = "UnknownError"
            steps.append(TrajectoryStep(
                step_index=step_idx,
                timestamp_ms=ts_ms,
                step_type="error",
                content=content,
                metadata={"error_name": name},
            ))
            step_idx += 1
        # step_start / step_finish carry boundaries and token accounting but
        # no unique trajectory content; they are skipped here (tokens feed
        # the adapter-level CostInfo extraction).

    return Trajectory(
        instance_id=instance_id,
        adapter_type="opencode",
        total_steps=len(steps),
        total_duration_ms=None,
        steps=steps,
        summary=TrajectorySummary.from_steps(steps),
    )
