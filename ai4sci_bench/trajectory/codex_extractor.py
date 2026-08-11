"""Extract structured trajectory from Codex CLI JSONL output."""

from __future__ import annotations

import json
from pathlib import Path

from ai4sci_bench.core.trajectory import Trajectory, TrajectoryStep, TrajectorySummary


def _coerce_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _first_change_path(item: dict[str, object]) -> str:
    changes = item.get("changes", [])
    if not isinstance(changes, list):
        return ""
    for change in changes:
        if isinstance(change, dict) and change.get("path"):
            return _coerce_text(change["path"])
    return ""


def extract_from_jsonl(jsonl_text: str, instance_id: str = "") -> Trajectory:
    """Parse Codex CLI JSONL into a Trajectory object."""
    steps: list[TrajectoryStep] = []
    step_idx = 0
    first_ts: float | None = None
    last_tool_name: str = ""

    text = _coerce_text(jsonl_text)
    for line in text.splitlines():
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

        ts_raw = event.get("timestamp")
        ts_ms = None
        if ts_raw is not None:
            try:
                ts_val = float(ts_raw)
                ts_ms = int(ts_val * 1000)
                if first_ts is None:
                    first_ts = ts_val
            except (ValueError, TypeError):
                pass

        etype = event.get("type", "")

        if etype == "message":
            role = event.get("role", "")
            content = event.get("content", "")

            if isinstance(content, (str, bytes, bytearray)):
                msg_text = _coerce_text(content)
                step_type = "llm_response" if role == "assistant" else "llm_request"
                steps.append(TrajectoryStep(
                    step_index=step_idx,
                    timestamp_ms=ts_ms,
                    step_type=step_type,
                    content=msg_text,
                    metadata={"role": role},
                ))
                step_idx += 1
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")
                    if btype == "text":
                        block_text = _coerce_text(block.get("text", ""))
                        step_type = "llm_response" if role == "assistant" else "llm_request"
                        steps.append(TrajectoryStep(
                            step_index=step_idx,
                            timestamp_ms=ts_ms,
                            step_type=step_type,
                            content=block_text,
                            metadata={"role": role},
                        ))
                        step_idx += 1
                    elif btype == "tool_use":
                        tool_name = block.get("name", "unknown")
                        last_tool_name = tool_name
                        inputs = block.get("input", {}) or {}
                        key_args = {}
                        if isinstance(inputs, dict):
                            for k in ("file_path", "path", "command", "pattern", "url"):
                                if k in inputs:
                                    key_args[k] = str(inputs[k])
                        steps.append(TrajectoryStep(
                            step_index=step_idx,
                            timestamp_ms=ts_ms,
                            step_type="tool_call",
                            content=tool_name,
                            metadata={"tool_name": tool_name, "key_args": key_args},
                        ))
                        step_idx += 1
                    elif btype == "tool_result":
                        inner = block.get("content", "")
                        if isinstance(inner, list):
                            parts = [
                                _coerce_text(item.get("text", ""))
                                for item in inner
                                if isinstance(item, dict) and item.get("type") == "text"
                            ]
                            result_text = "\n".join(parts)
                        else:
                            result_text = _coerce_text(inner)
                        steps.append(TrajectoryStep(
                            step_index=step_idx,
                            timestamp_ms=ts_ms,
                            step_type="tool_result",
                            content=result_text,
                            metadata={
                                "is_error": bool(block.get("is_error")),
                                "parent_tool_name": last_tool_name,
                            },
                        ))
                        step_idx += 1

        elif etype in ("function_call", "tool_use"):
            tool_name = _coerce_text(event.get("name", "unknown"))
            last_tool_name = tool_name
            steps.append(TrajectoryStep(
                step_index=step_idx,
                timestamp_ms=ts_ms,
                step_type="tool_call",
                content=tool_name,
                metadata={"tool_name": tool_name},
            ))
            step_idx += 1

        elif etype in ("function_call_output", "tool_result"):
            output = event.get("output", event.get("content", ""))
            output_text = _coerce_text(output)
            steps.append(TrajectoryStep(
                step_index=step_idx,
                timestamp_ms=ts_ms,
                step_type="tool_result",
                content=output_text,
                metadata={
                    "is_error": bool(event.get("is_error")),
                    "parent_tool_name": last_tool_name,
                },
            ))
            step_idx += 1

        elif etype == "error":
            msg = _coerce_text(event.get("message", str(event)))
            steps.append(TrajectoryStep(
                step_index=step_idx,
                timestamp_ms=ts_ms,
                step_type="error",
                content=msg,
            ))
            step_idx += 1

        elif etype in ("rate_limit_event", "rate_limit"):
            msg = _coerce_text(event.get("message") or str(event))
            steps.append(TrajectoryStep(
                step_index=step_idx,
                timestamp_ms=ts_ms,
                step_type="rate_limit",
                content=msg,
            ))
            step_idx += 1

        elif etype in ("reasoning", "thinking"):
            think_text = _coerce_text(event.get("text") or event.get("content") or "")
            if think_text:
                steps.append(TrajectoryStep(
                    step_index=step_idx,
                    timestamp_ms=ts_ms,
                    step_type="thinking",
                    content=think_text,
                    metadata={"thinking_length": len(think_text)},
                ))
                step_idx += 1

        elif etype in ("item.started", "item.completed"):
            item = event.get("item", {})
            if not isinstance(item, dict):
                continue
            item_type = item.get("type", "")

            if item_type == "agent_message" and etype == "item.completed":
                msg_text = _coerce_text(item.get("text", ""))
                if msg_text:
                    steps.append(TrajectoryStep(
                        step_index=step_idx,
                        timestamp_ms=ts_ms,
                        step_type="llm_response",
                        content=msg_text,
                        metadata={"role": "assistant", "item_id": item.get("id")},
                    ))
                    step_idx += 1

            elif item_type == "command_execution":
                command = _coerce_text(item.get("command", ""))
                if etype == "item.started":
                    last_tool_name = "command_execution"
                    steps.append(TrajectoryStep(
                        step_index=step_idx,
                        timestamp_ms=ts_ms,
                        step_type="tool_call",
                        content=command,
                        metadata={
                            "tool_name": "command_execution",
                            "key_args": {"command": command} if command else {},
                            "item_id": item.get("id"),
                        },
                    ))
                    step_idx += 1
                else:
                    exit_code = item.get("exit_code")
                    output_text = _coerce_text(
                        item.get("aggregated_output")
                        or item.get("output")
                        or item.get("stdout")
                        or ""
                    )
                    steps.append(TrajectoryStep(
                        step_index=step_idx,
                        timestamp_ms=ts_ms,
                        step_type="tool_result",
                        content=output_text,
                        metadata={
                            "is_error": exit_code not in (None, 0),
                            "exit_code": exit_code,
                            "parent_tool_name": "command_execution",
                            "item_id": item.get("id"),
                        },
                    ))
                    step_idx += 1

            elif item_type == "file_change" and etype == "item.completed":
                path = _first_change_path(item)
                key_args = {"file_path": path} if path else {}
                steps.append(TrajectoryStep(
                    step_index=step_idx,
                    timestamp_ms=ts_ms,
                    step_type="tool_call",
                    content="file_change",
                    metadata={
                        "tool_name": "file_change",
                        "key_args": key_args,
                        "changes": item.get("changes", []),
                        "item_id": item.get("id"),
                    },
                ))
                step_idx += 1

    total_duration = None
    if first_ts is not None and steps:
        last_ts = None
        for s in reversed(steps):
            if s.timestamp_ms is not None:
                last_ts = s.timestamp_ms
                break
        if last_ts is not None:
            total_duration = last_ts - int(first_ts * 1000)

    summary = TrajectorySummary.from_steps(steps)
    return Trajectory(
        instance_id=instance_id,
        adapter_type="codex_cli",
        total_steps=len(steps),
        total_duration_ms=total_duration,
        steps=steps,
        summary=summary,
    )


def extract_from_file(path: Path, instance_id: str = "") -> Trajectory:
    """Extract trajectory from a Codex CLI JSONL sidecar file."""
    return extract_from_jsonl(path.read_text(encoding="utf-8"), instance_id)
