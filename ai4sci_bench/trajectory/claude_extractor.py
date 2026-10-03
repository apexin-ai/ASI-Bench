"""Extract structured trajectory from Claude Code JSONL output."""

from __future__ import annotations

import json
from pathlib import Path

from ai4sci_bench.core.trajectory import KEY_ARG_NAMES, Trajectory, TrajectoryStep, TrajectorySummary


def _message_blocks(event: dict) -> tuple[dict, list[dict]]:
    """(message, content blocks) of an assistant/user event. A message that is not an
    object, or content that is not a list, yields no blocks instead of raising: one odd
    event must not cost the whole trajectory."""
    message = event.get("message")
    if not isinstance(message, dict):
        return {}, []
    content = message.get("content")
    if not isinstance(content, list):
        return message, []
    return message, [block for block in content if isinstance(block, dict)]


def extract_from_jsonl(jsonl_text: str, instance_id: str = "") -> Trajectory:
    """Parse Claude Code JSONL into a Trajectory object."""
    steps: list[TrajectoryStep] = []
    step_idx = 0
    first_ts: float | None = None
    tool_id_to_name: dict[str, str] = {}

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

        if etype == "assistant":
            message, blocks = _message_blocks(event)
            assistant_turn_had_response = False
            for block in blocks:
                btype = block.get("type", "")
                if btype == "text":
                    assistant_turn_had_response = True
                    steps.append(TrajectoryStep(
                        step_index=step_idx,
                        timestamp_ms=ts_ms,
                        step_type="llm_response",
                        content=str(block.get("text", "")),
                        metadata={"role": "assistant"},
                    ))
                    step_idx += 1
                elif btype == "tool_use":
                    tool_name = block.get("name", "unknown")
                    tool_call_id = block.get("id", "")
                    if tool_call_id:
                        tool_id_to_name[tool_call_id] = tool_name
                    inputs = block.get("input", {}) or {}
                    key_args = {}
                    if isinstance(inputs, dict):
                        for k in KEY_ARG_NAMES:
                            if k in inputs:
                                key_args[k] = str(inputs[k])
                    steps.append(TrajectoryStep(
                        step_index=step_idx,
                        timestamp_ms=ts_ms,
                        step_type="tool_call",
                        content=f"{tool_name}",
                        metadata={
                            "tool_name": tool_name,
                            "key_args": key_args,
                            "tool_call_id": tool_call_id,
                        },
                    ))
                    step_idx += 1
                elif btype == "thinking":
                    assistant_turn_had_response = True
                    text = str(block.get("thinking", ""))
                    steps.append(TrajectoryStep(
                        step_index=step_idx,
                        timestamp_ms=ts_ms,
                        step_type="thinking",
                        content=text,
                        metadata={"thinking_length": len(text)},
                    ))
                    step_idx += 1
            if not assistant_turn_had_response and message.get("content"):
                steps.append(TrajectoryStep(
                    step_index=step_idx,
                    timestamp_ms=ts_ms,
                    step_type="llm_response",
                    content="",
                    metadata={"role": "assistant", "synthetic_turn_marker": True},
                ))
                step_idx += 1

        elif etype == "user":
            for block in _message_blocks(event)[1]:
                if block.get("type") == "tool_result":
                    inner = block.get("content", "")
                    # Non-text blocks (e.g. MCP images) are not copied into the
                    # trajectory, but their presence and media type are recorded
                    # so that a returned image stays observable.
                    content_types: list[str] = []
                    image_media_types: list[str] = []
                    if isinstance(inner, list):
                        parts = []
                        for item in inner:
                            if not isinstance(item, dict):
                                continue
                            item_type = str(item.get("type", ""))
                            content_types.append(item_type)
                            if item_type == "text":
                                parts.append(str(item.get("text", "")))
                            elif item_type == "image":
                                source = item.get("source") if isinstance(item.get("source"), dict) else {}
                                image_media_types.append(str(
                                    source.get("media_type") or item.get("media_type") or item.get("mimeType") or ""
                                ))
                        text = "\n".join(parts)
                    else:
                        text = str(inner)
                        content_types.append("text")
                    is_error = bool(block.get("is_error"))
                    result_tool_call_id = block.get("tool_use_id", "")
                    parent_tool = tool_id_to_name.get(result_tool_call_id, "")
                    metadata = {
                        "tool_call_id": result_tool_call_id,
                        "is_error": is_error,
                        "output_length": len(text),
                        "parent_tool_name": parent_tool,
                        "content_types": content_types,
                    }
                    if image_media_types:
                        metadata["image_media_types"] = image_media_types
                    steps.append(TrajectoryStep(
                        step_index=step_idx,
                        timestamp_ms=ts_ms,
                        step_type="tool_result",
                        content=text,
                        metadata=metadata,
                    ))
                    step_idx += 1

        elif etype == "system":
            msg = event.get("message") or event.get("text") or ""
            steps.append(TrajectoryStep(
                step_index=step_idx,
                timestamp_ms=ts_ms,
                step_type="system",
                content=str(msg),
                metadata={"subtype": event.get("subtype", "")},
            ))
            step_idx += 1

        elif etype == "error":
            msg = event.get("message") or event.get("error") or str(event)
            steps.append(TrajectoryStep(
                step_index=step_idx,
                timestamp_ms=ts_ms,
                step_type="error",
                content=str(msg),
            ))
            step_idx += 1

        elif etype in ("rate_limit_event", "rate_limit"):
            msg = event.get("message") or str(event)
            steps.append(TrajectoryStep(
                step_index=step_idx,
                timestamp_ms=ts_ms,
                step_type="rate_limit",
                content=str(msg),
            ))
            step_idx += 1

        elif etype == "result":
            meta = {}
            if "num_turns" in event:
                meta["num_turns"] = event["num_turns"]
            if "usage" in event:
                meta["usage"] = event["usage"]
            steps.append(TrajectoryStep(
                step_index=step_idx,
                timestamp_ms=ts_ms,
                step_type="system",
                content=f"result: {event.get('subtype', '')}",
                metadata=meta,
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
        adapter_type="claude_code_cli",
        total_steps=len(steps),
        total_duration_ms=total_duration,
        steps=steps,
        summary=summary,
    )


def extract_from_file(path: Path, instance_id: str = "") -> Trajectory:
    """Extract trajectory from a Claude Code JSONL sidecar file."""
    return extract_from_jsonl(path.read_text(encoding="utf-8"), instance_id)
