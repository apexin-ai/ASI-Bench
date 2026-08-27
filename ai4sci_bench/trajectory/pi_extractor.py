"""Extract structured trajectory from pi ``--mode json`` event streams.

pi (verified v0.84.3) emits a session header followed by events:

- ``agent_start`` / ``agent_end`` — agent lifecycle;
- ``turn_start`` / ``turn_end`` — turn lifecycle (``turn_end`` carries the
  final assistant message + tool results);
- ``message_start`` / ``message_update`` / ``message_end`` — message
  lifecycle; ``message_end`` carries the authoritative final message;
- ``tool_execution_start`` / ``tool_execution_update`` /
  ``tool_execution_end`` — tool calls (``toolName``, ``args``, ``result``,
  ``isError``).

Assistant messages carry per-message ``usage`` dicts
(``input``/``output``/``cacheRead``/``cacheWrite``/``totalTokens``).

Known limitation: pi's write/edit tool arguments are not mapped into
``file_history.extract_file_versions()`` (schemas vary between pi versions);
file version reconstruction is only available for the claude/codex schemas.
"""

from __future__ import annotations

import json

from ai4sci_bench.core.trajectory import Trajectory, TrajectoryStep, TrajectorySummary

_VALUE_KEYS = ("path", "filePath", "file_path", "command", "pattern", "url", "query")


def _timestamp_ms(event: dict) -> int | None:
    raw = event.get("timestamp")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (ValueError, TypeError):
        return None
    # pi timestamps are unix milliseconds already.
    return int(value if value > 1e11 else value * 1000)


def _key_args(args) -> dict:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            return {}
    if not isinstance(args, dict):
        return {}
    return {k: str(args[k]) for k in _VALUE_KEYS if k in args}


def extract_from_jsonl(jsonl_text: str, instance_id: str = "") -> Trajectory:
    """Parse pi JSONL event stream into a Trajectory object."""
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

        if etype == "session":
            steps.append(TrajectoryStep(
                step_index=step_idx,
                timestamp_ms=ts_ms,
                step_type="system",
                content=f"session {event.get('id', '')}",
                metadata={"session_id": event.get("id", "")},
            ))
            step_idx += 1
        elif etype == "agent_start":
            steps.append(TrajectoryStep(
                step_index=step_idx,
                timestamp_ms=ts_ms,
                step_type="system",
                content="agent_start",
                metadata={},
            ))
            step_idx += 1
        elif etype == "message_end":
            message = event.get("message", {})
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            for block in message.get("content", []) or []:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text = str(block.get("text", ""))
                    if not text:
                        continue
                    steps.append(TrajectoryStep(
                        step_index=step_idx,
                        timestamp_ms=ts_ms,
                        step_type="llm_response",
                        content=text,
                        metadata={"role": "assistant"},
                    ))
                    step_idx += 1
                elif btype == "thinking":
                    text = str(block.get("thinking", ""))
                    if not text:
                        continue
                    steps.append(TrajectoryStep(
                        step_index=step_idx,
                        timestamp_ms=ts_ms,
                        step_type="thinking",
                        content=text,
                        metadata={"thinking_length": len(text)},
                    ))
                    step_idx += 1
                # toolCall blocks are skipped: the authoritative call/result
                # pairs come from tool_execution_* events.
            stop = message.get("stopReason")
            if stop == "error":
                steps.append(TrajectoryStep(
                    step_index=step_idx,
                    timestamp_ms=ts_ms,
                    step_type="error",
                    content=str(message.get("errorMessage") or "unknown error"),
                    metadata={"stop_reason": stop},
                ))
                step_idx += 1
            elif stop == "aborted":
                steps.append(TrajectoryStep(
                    step_index=step_idx,
                    timestamp_ms=ts_ms,
                    step_type="error",
                    content="agent aborted",
                    metadata={"stop_reason": stop},
                ))
                step_idx += 1
        elif etype == "tool_execution_start":
            tool_name = str(event.get("toolName", "unknown"))
            steps.append(TrajectoryStep(
                step_index=step_idx,
                timestamp_ms=ts_ms,
                step_type="tool_call",
                content=tool_name,
                metadata={
                    "tool_name": tool_name,
                    "key_args": _key_args(event.get("args")),
                    "tool_call_id": str(event.get("toolCallId", "") or ""),
                },
            ))
            step_idx += 1
        elif etype == "tool_execution_end":
            result = event.get("result")
            if isinstance(result, dict):
                output = str(result.get("output", result.get("text", "")))
            else:
                output = str(result or "")
            steps.append(TrajectoryStep(
                step_index=step_idx,
                timestamp_ms=ts_ms,
                step_type="tool_result",
                content=output[:2000],
                metadata={
                    "parent_tool_name": str(event.get("toolName", "unknown")),
                    "is_error": bool(event.get("isError")),
                    "tool_call_id": str(event.get("toolCallId", "") or ""),
                },
            ))
            step_idx += 1
        elif etype == "error":
            steps.append(TrajectoryStep(
                step_index=step_idx,
                timestamp_ms=ts_ms,
                step_type="error",
                content=str(event.get("error") or stripped[:500]),
                metadata={},
            ))
            step_idx += 1
        # turn_start/turn_end/message_start/message_update/agent_end and
        # extension events carry no unique information for the trajectory.

    return Trajectory(
        instance_id=instance_id,
        adapter_type="pi",
        total_steps=len(steps),
        total_duration_ms=None,
        steps=steps,
        summary=TrajectorySummary.from_steps(steps),
    )
