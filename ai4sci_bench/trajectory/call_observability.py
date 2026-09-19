"""Versioned, denominator-aware model-call observability records.

Visible trajectory steps are intentionally kept separate from these records:
a completion can be empty, fail before producing a message, or contain only a
tool call.  Parsers only assert facts present in the adapter stream.  Missing
boundaries are represented explicitly so reporting never turns missing
evidence into a zero-percent empty-response rate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


CALL_OBSERVABILITY_SCHEMA_VERSION = 1
_INCOMPLETE_FINISH_REASONS = frozenset({
    "incomplete", "length", "max_output", "max_output_tokens", "max_tokens",
})


@dataclass
class ModelCallRecord:
    """Observable lifecycle of one completion, or one explicit coverage gap."""

    schema_version: int = CALL_OBSERVABILITY_SCHEMA_VERSION
    call_id: str | None = None
    call_index: int = 0
    attempted: bool | None = None
    adapter: str = "unknown"
    provider: str | None = None
    model: str | None = None
    protocol: str | None = None
    endpoint: str | None = None
    transport_observability: str = "unavailable"
    boundary_kind: str = "unknown"
    run_id: str | None = None
    instance_id: str | None = None
    prompt_level: str | None = None
    benchmark_attempt: int | None = None
    benchmark_attempt_kind: str | None = None
    benchmark_retry_index: int | None = None
    benchmark_retry_cause: str | None = None
    benchmark_retry_initiator: str | None = None
    request_id: str | None = None
    response_id: str | None = None
    thread_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    item_id: str | None = None
    parent_call_id: str | None = None
    parent_call_index: int | None = None
    benchmark_parent_call_id: str | None = None
    retry_index: int = 0
    retry_cause: str | None = None
    retry_initiator: str | None = None
    retry_backoff_seconds: float | None = None
    request_started: bool | None = None
    request_sent: bool | None = None
    http_headers_received: bool | None = None
    stream_opened: bool | None = None
    first_byte_received: bool | None = None
    first_token_received: bool | None = None
    stream_ended: bool | None = None
    call_finished: bool | None = None
    response_received: bool | None = None
    partial_content: bool = False
    content_state: str = "unknown"  # omitted | empty | nonempty | unknown
    outcome: str = "unknown"
    finish_reason: str | None = None
    provider_status: str | None = None
    http_status: int | None = None
    connection_close_reason: str | None = None
    timeout_phase: str | None = None
    error_type: str | None = None
    error_class: str | None = None
    error_message: str | None = None
    process_exit_code: int | None = None
    termination_signal: int | None = None
    event_count: int = 0
    chunk_count: int | None = None
    parse_error_count: int = 0
    token_count: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    cost_usd: float | None = None
    started_at: str | None = None
    ended_at: str | None = None
    request_started_at: str | None = None
    request_ended_at: str | None = None
    response_started_at: str | None = None
    response_ended_at: str | None = None
    first_chunk_at: str | None = None
    last_chunk_at: str | None = None
    duration_ms: int | None = None
    first_event_sequence: int | None = None
    last_event_sequence: int | None = None
    boundary_observed: bool = False
    evidence_status: str = "not_observed"
    evidence_reason: str | None = None
    event_sequences: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _event_id(event: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = event.get(key)
        if value is not None:
            return _coerce_str(value)
    return None


def _item(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("item")
    return value if isinstance(value, dict) else {}


def _timestamp(event: dict[str, Any]) -> str | None:
    return _coerce_str(event.get("timestamp") or event.get("created_at"))


def _duration_ms(start: str | None, end: str | None) -> int | None:
    if start is None or end is None:
        return None
    try:
        start_value = float(start)
        end_value = float(end)
        # Unix timestamps from adapters may be seconds or milliseconds.
        scale = 1 if max(abs(start_value), abs(end_value)) > 1e11 else 1000
        return max(0, int((end_value - start_value) * scale))
    except (TypeError, ValueError):
        pass
    try:
        normalized_start = start.replace("Z", "+00:00")
        normalized_end = end.replace("Z", "+00:00")
        delta = datetime.fromisoformat(normalized_end) - datetime.fromisoformat(
            normalized_start
        )
        return max(0, int(delta.total_seconds() * 1000))
    except ValueError:
        return None


def _usage(record: ModelCallRecord, value: Any) -> None:
    if not isinstance(value, dict):
        return
    aliases = {
        "input_tokens": ("input_tokens", "prompt_tokens", "input"),
        "output_tokens": ("output_tokens", "completion_tokens", "output"),
        "cache_read_tokens": ("cache_read_tokens", "cache_read", "cacheRead"),
        "cache_write_tokens": ("cache_write_tokens", "cache_write", "cacheWrite"),
    }
    for attr, keys in aliases.items():
        for key in keys:
            number = _coerce_int(value.get(key))
            if number is not None:
                setattr(record, attr, number)
                break
    if record.token_count is None:
        total = _coerce_int(value.get("total_tokens") or value.get("totalTokens"))
        if total is not None:
            record.token_count = total
        elif record.input_tokens is not None and record.output_tokens is not None:
            record.token_count = record.input_tokens + record.output_tokens


def _observe(record: ModelCallRecord, sequence: int, event: dict[str, Any]) -> None:
    record.event_sequences.append(sequence)
    record.event_count += 1
    if record.first_event_sequence is None:
        record.first_event_sequence = sequence
    record.last_event_sequence = sequence
    timestamp = _timestamp(event)
    if timestamp:
        record.first_chunk_at = record.first_chunk_at or timestamp
        record.last_chunk_at = timestamp


def _start_record(
    *,
    index: int,
    adapter: str,
    instance_id: str | None,
    provider: str | None,
    model: str | None,
    event: dict[str, Any] | None = None,
    boundary_kind: str = "adapter_event",
    request_started: bool | None = None,
    request_sent: bool | None = None,
) -> ModelCallRecord:
    event = event or {}
    started_at = _timestamp(event)
    return ModelCallRecord(
        call_index=index,
        attempted=True,
        adapter=adapter,
        provider=provider,
        model=model,
        instance_id=instance_id,
        request_started=request_started,
        request_sent=request_sent,
        # CLI JSONL is an application event stream, not proof that a provider
        # HTTP/SSE stream was opened. Transport fields stay unknown unless the
        # source explicitly supplies them.
        transport_observability="cli_event_stream_only",
        boundary_kind=boundary_kind,
        boundary_observed=True,
        evidence_status="partial",
        started_at=started_at,
        request_started_at=started_at,
    )


def _mark_response(record: ModelCallRecord, event: dict[str, Any]) -> None:
    timestamp = _timestamp(event)
    # A response event proves the logical request was dispatched, but does not
    # expose HTTP headers, a first network byte, or SSE establishment.
    record.request_sent = True
    record.response_received = True
    record.response_started_at = record.response_started_at or timestamp


def _complete_record(record: ModelCallRecord, *, event: dict[str, Any] | None = None) -> None:
    """Mark an explicitly terminated call without inferring transport events."""
    event = event or {}
    timestamp = _timestamp(event)
    record.call_finished = True
    record.ended_at = timestamp or record.ended_at or record.last_chunk_at
    record.request_ended_at = record.request_ended_at or record.ended_at
    record.response_ended_at = record.response_ended_at or record.ended_at
    record.duration_ms = _duration_ms(record.started_at, record.ended_at)
    record.evidence_status = "complete"
    if (
        record.error_type is None
        and record.finish_reason
        and record.finish_reason.lower() in _INCOMPLETE_FINISH_REASONS
    ):
        record.error_type = "incomplete_response"
    if record.error_type == "timeout":
        record.outcome = "timeout"
    elif record.error_type in {"parse_error", "read_error"}:
        record.outcome = "response_received_not_parseable"
    elif record.error_type == "incomplete_response":
        record.outcome = "incomplete_response"
    elif record.error_type == "provider_error":
        record.outcome = "provider_error"
    elif record.error_type:
        record.outcome = "transport_error"
    elif record.response_received and record.content_state == "empty":
        record.outcome = "complete_empty_response"
    elif record.response_received and record.content_state == "nonempty":
        record.outcome = "complete_nonempty_response"
    elif record.response_received and record.content_state == "omitted":
        record.outcome = "complete_tool_only_response"
    elif record.response_received:
        record.outcome = "response_received_unknown"
    else:
        record.outcome = "request_sent_no_response"


def _incomplete_record(record: ModelCallRecord, *, reason: str) -> None:
    """Close local parsing while preserving that the provider call never ended."""
    record.call_finished = False
    if record.stream_opened is True and record.stream_ended is None:
        record.stream_ended = False
    record.evidence_status = "incomplete"
    record.evidence_reason = reason
    record.ended_at = record.ended_at or record.last_chunk_at
    record.duration_ms = _duration_ms(record.started_at, record.ended_at)
    if record.first_byte_received or record.response_received:
        record.partial_content = record.content_state == "nonempty"
        record.outcome = "partial_response" if record.partial_content else "incomplete_response"
    elif record.stream_opened:
        record.outcome = "transport_established_no_token"
    elif record.request_sent:
        record.outcome = "request_sent_no_response"
    elif record.request_started:
        record.outcome = "request_send_state_unknown"
    else:
        record.outcome = "unknown"


def _logical_boundary_gap(
    record: ModelCallRecord,
    *,
    event: dict[str, Any] | None = None,
    reason: str,
) -> None:
    """Retain a logical agent boundary without treating it as a completion.

    Codex ``turn`` events delimit one agent invocation, which may contain an
    unknown number of provider completions when tools are used.  Their IDs,
    timing, usage, and event sequences are still useful evidence, but they
    cannot establish the attempted-call denominator.
    """
    event = event or {}
    timestamp = _timestamp(event)
    record.attempted = None
    record.request_started = None
    record.request_sent = None
    record.http_headers_received = None
    record.stream_opened = None
    record.first_byte_received = None
    record.first_token_received = None
    record.stream_ended = None
    record.call_finished = None
    record.response_received = None
    record.partial_content = False
    record.content_state = "unknown"
    record.outcome = "no_observable_boundary"
    record.boundary_observed = False
    record.evidence_status = "not_observed"
    record.evidence_reason = reason
    record.ended_at = timestamp or record.ended_at or record.last_chunk_at
    record.request_started_at = None
    record.request_ended_at = None
    record.response_started_at = None
    record.response_ended_at = None
    record.duration_ms = _duration_ms(record.started_at, record.ended_at)


def _unobservable_record(
    *,
    adapter: str,
    instance_id: str | None,
    provider: str | None,
    model: str | None,
    reason: str,
    error_type: str | None = None,
) -> ModelCallRecord:
    return ModelCallRecord(
        call_index=1,
        attempted=None,
        adapter=adapter,
        provider=provider,
        model=model,
        instance_id=instance_id,
        transport_observability="cli_event_stream_only",
        boundary_kind="unobservable",
        outcome="no_observable_boundary",
        content_state="unknown",
        boundary_observed=False,
        evidence_status="not_observed",
        evidence_reason=reason,
        error_type=error_type,
    )


def _parse_gap_record(
    *,
    index: int,
    adapter: str,
    instance_id: str | None,
    provider: str | None,
    model: str | None,
    sequences: list[int],
) -> ModelCallRecord:
    record = _unobservable_record(
        adapter=adapter,
        instance_id=instance_id,
        provider=provider,
        model=model,
        reason="malformed_jsonl_may_hide_call_boundaries",
        error_type="parse_error",
    )
    record.call_index = index
    record.error_class = "JSONDecodeError"
    record.parse_error_count = len(sequences)
    record.event_sequences = sequences
    record.event_count = len(sequences)
    record.first_event_sequence = sequences[0]
    record.last_event_sequence = sequences[-1]
    return record


def _missing_stream_evidence(jsonl_text: str | None) -> tuple[str, str]:
    """Distinguish an observed zero-byte stream from unavailable evidence."""
    if jsonl_text is None:
        return "event_stream_unavailable", "unavailable_stream"
    return "empty_event_stream", "empty_stream"


def apply_process_evidence(
    records: list[dict[str, Any]],
    *,
    process_exit_code: int | None = None,
    termination_signal: int | None = None,
    timeout_phase: str | None = None,
    process_error: str | None = None,
) -> list[dict[str, Any]]:
    """Attach subprocess evidence and refine only otherwise-incomplete outcomes."""
    for record in records:
        record["process_exit_code"] = process_exit_code
        record["termination_signal"] = termination_signal
        if timeout_phase:
            record["timeout_phase"] = timeout_phase
            if record.get("call_finished") is not True:
                record["error_type"] = "timeout"
                record["outcome"] = "timeout"
                record["evidence_reason"] = "adapter_process_timed_out"
        elif (process_exit_code not in (None, 0) or termination_signal is not None):
            if record.get("call_finished") is not True:
                record["error_type"] = record.get("error_type") or "process_failure"
                record["outcome"] = "process_failure"
                record["evidence_reason"] = "adapter_process_exited_before_call_finished"
        elif process_error and record.get("call_finished") is not True:
            record["error_type"] = "process_failure"
            record["outcome"] = "process_failure"
            record["evidence_reason"] = "adapter_process_failed_before_call_finished"
        if process_error and not record.get("error_message"):
            record["error_message"] = process_error
    return records


def parse_codex_call_records(
    jsonl_text: str | None,
    *,
    instance_id: str | None = None,
    adapter: str = "codex_cli",
    provider: str | None = None,
    model: str | None = None,
    process_exit_code: int | None = None,
    termination_signal: int | None = None,
    timeout_phase: str | None = None,
    process_error: str | None = None,
) -> list[dict[str, Any]]:
    """Parse Codex JSONL into records for each emitted completion boundary."""
    text = jsonl_text or ""
    records: list[ModelCallRecord] = []
    active: ModelCallRecord | None = None
    turn_scoped = False
    context: dict[str, str | None] = {}
    malformed_sequences: list[int] = []
    pending_retry: dict[str, Any] | None = None

    def start(
        event: dict[str, Any],
        boundary_kind: str,
        *,
        request_started: bool | None = None,
        request_sent: bool | None = None,
    ) -> ModelCallRecord:
        nonlocal pending_retry
        record = _start_record(
            index=len(records) + 1,
            adapter=adapter,
            instance_id=instance_id,
            provider=provider,
            model=model,
            event=event,
            boundary_kind=boundary_kind,
            request_started=request_started,
            request_sent=request_sent,
        )
        item = _item(event)
        record.thread_id = context.get("thread_id")
        record.session_id = context.get("session_id")
        record.turn_id = _event_id(event, "turn_id") or context.get("turn_id")
        record.item_id = _event_id(item, "id") or _event_id(event, "item_id")
        record.request_id = _event_id(event, "request_id")
        record.response_id = _event_id(event, "response_id")
        if pending_retry is not None:
            record.retry_index = int(pending_retry.get("retry_index") or 1)
            record.retry_cause = pending_retry.get("cause")
            record.retry_initiator = pending_retry.get("initiator") or "adapter"
            record.retry_backoff_seconds = pending_retry.get("backoff")
            record.parent_call_index = len(records) or None
            pending_retry = None
        return record

    for sequence, raw_line in enumerate(text.splitlines(), 1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            malformed_sequences.append(sequence)
            if active is not None:
                active.parse_error_count += 1
            continue
        if not isinstance(event, dict):
            malformed_sequences.append(sequence)
            continue
        etype = str(event.get("type", ""))
        if etype in {"retry", "request.retry", "response.retry", "model.retry"}:
            pending_retry = {
                "retry_index": _coerce_int(event.get("retry_index") or event.get("attempt")),
                "cause": _coerce_str(event.get("reason") or event.get("cause")),
                "initiator": _coerce_str(event.get("initiator")),
                "backoff": _coerce_float(
                    event.get("backoff_seconds") or event.get("retry_after")
                ),
            }
            if active is not None and active.call_finished is not True:
                if active.boundary_kind == "turn":
                    _logical_boundary_gap(
                        active,
                        event=event,
                        reason="codex_turn_does_not_expose_completion_boundaries",
                    )
                else:
                    _incomplete_record(active, reason="adapter_reported_retry")
                records.append(active)
                active = None
                turn_scoped = False
            continue
        if etype == "thread.started":
            context["thread_id"] = _event_id(event, "thread_id", "id")
            continue
        if etype in {"request.created", "request.started", "request.sent"}:
            if active is None:
                active = start(
                    event,
                    "provider_request",
                    request_started=True,
                    request_sent=(True if etype == "request.sent" else None),
                )
            _observe(active, sequence, event)
            active.request_started = True
            if etype == "request.sent":
                active.request_sent = True
            active.request_id = _event_id(event, "request_id", "id") or active.request_id
            continue
        if etype in {"http.headers", "response.headers", "stream.opened"}:
            if active is None:
                active = start(event, "provider_response")
            _observe(active, sequence, event)
            active.request_sent = True
            active.http_headers_received = True
            active.http_status = _coerce_int(
                event.get("http_status") or event.get("status_code")
            )
            if etype == "stream.opened":
                active.stream_opened = True
            continue
        if etype in {
            "response.output_text.delta", "output_text.delta", "content.delta",
        }:
            if active is None:
                active = start(event, "provider_response")
            _observe(active, sequence, event)
            _mark_response(active, event)
            active.stream_opened = True
            active.first_byte_received = True
            active.chunk_count = (active.chunk_count or 0) + 1
            delta = event.get("delta") or event.get("text")
            if delta:
                active.first_token_received = True
                active.content_state = "nonempty"
            continue
        if etype in {"stream.closed", "stream.ended"}:
            if active is None:
                active = start(event, "provider_response")
            _observe(active, sequence, event)
            active.stream_opened = True
            active.stream_ended = True
            active.connection_close_reason = _coerce_str(
                event.get("reason") or event.get("close_reason")
            )
            continue
        if etype == "turn.started":
            if active is not None:
                if active.boundary_kind == "turn":
                    _logical_boundary_gap(
                        active,
                        event=event,
                        reason="codex_turn_does_not_expose_completion_boundaries",
                    )
                else:
                    _incomplete_record(
                        active,
                        reason="new_turn_started_before_previous_terminal_event",
                    )
                records.append(active)
            active = start(
                event,
                "turn",
                request_started=True,
                request_sent=True,
            )
            active.turn_id = _event_id(event, "turn_id", "id") or active.turn_id
            context["turn_id"] = active.turn_id
            turn_scoped = True
            _observe(active, sequence, event)
            continue
        if etype in {"turn.completed", "turn.failed", "turn.cancelled"}:
            if active is None:
                active = start(event, "turn")
            _observe(active, sequence, event)
            active.turn_id = _event_id(event, "turn_id", "id") or active.turn_id
            active.provider_status = _event_id(event, "status") or etype.removeprefix("turn.")
            active.finish_reason = _event_id(event, "finish_reason", "stop_reason")
            _usage(active, event.get("usage"))
            if etype != "turn.completed":
                active.error_type = "harness_error"
                active.error_message = _coerce_str(event.get("message") or event.get("error"))
            _logical_boundary_gap(
                active,
                event=event,
                reason="codex_turn_does_not_expose_completion_boundaries",
            )
            records.append(active)
            active = None
            turn_scoped = False
            continue
        if etype == "message" and event.get("role") == "assistant":
            if active is not None:
                _incomplete_record(active, reason="new_message_boundary_before_terminal_event")
                records.append(active)
            active = start(event, "assistant_message")
            content = event.get("content")
            _observe(active, sequence, event)
            _mark_response(active, event)
            active.first_token_received = bool(content)
            active.content_state = "nonempty" if content else "empty"
            active.finish_reason = _event_id(event, "finish_reason", "stop_reason")
            _complete_record(active, event=event)
            records.append(active)
            active = None
            continue
        if etype in {"response.completed", "response.failed", "response.incomplete"}:
            if active is None:
                active = start(event, "response")
            else:
                active.boundary_kind = "provider_response"
                active.boundary_observed = True
                active.attempted = True
            _observe(active, sequence, event)
            active.response_id = _event_id(event, "response_id", "id") or active.response_id
            active.provider_status = etype.removeprefix("response.")
            active.finish_reason = _event_id(event, "finish_reason", "status")
            _mark_response(active, event)
            if active.stream_opened is True:
                active.stream_ended = True
            if etype != "response.completed":
                active.error_type = "incomplete_response"
            elif active.content_state == "unknown":
                active.content_state = "empty"
            _complete_record(active, event=event)
            records.append(active)
            active = None
            turn_scoped = False
            continue

        item = _item(event)
        item_type = str(item.get("type", ""))
        relevant = etype in {
            "item.started", "item.completed", "item.updated", "function_call",
            "tool_use", "error", "stream.error", "rate_limit_event", "rate_limit",
        }
        if relevant and active is None:
            active = start(event, "item" if item_type else "adapter_event")
        if active is None:
            continue
        _observe(active, sequence, event)
        active.request_id = _event_id(event, "request_id") or active.request_id
        active.response_id = _event_id(event, "response_id") or active.response_id
        active.item_id = _event_id(item, "id") or active.item_id
        _usage(active, event.get("usage"))
        if etype == "item.completed" and item_type == "agent_message":
            content = item.get("text")
            _mark_response(active, event)
            active.first_token_received = bool(content)
            active.content_state = "nonempty" if content else "empty"
            active.finish_reason = _event_id(item, "finish_reason", "stop_reason")
            if not turn_scoped:
                _complete_record(active, event=event)
                records.append(active)
                active = None
        elif etype in {"function_call", "tool_use"} or item_type in {
            "command_execution", "mcp_tool_call", "web_search", "file_change",
        }:
            _mark_response(active, event)
            if active.content_state == "unknown":
                active.content_state = "omitted"
        elif item_type == "reasoning" or etype in {"reasoning", "thinking"}:
            _mark_response(active, event)
            if active.content_state == "unknown":
                active.content_state = "omitted"
        elif etype in {"error", "stream.error"}:
            active.error_type = "provider_error"
            active.error_message = _coerce_str(event.get("message") or event.get("error"))
            active.connection_close_reason = _coerce_str(event.get("reason"))

    if active is not None:
        if active.boundary_kind == "turn":
            _logical_boundary_gap(
                active,
                reason="codex_turn_does_not_expose_completion_boundaries",
            )
        else:
            _incomplete_record(active, reason="event_stream_ended_without_terminal_boundary")
        records.append(active)
    if malformed_sequences:
        records.append(_parse_gap_record(
            index=len(records) + 1,
            adapter=adapter,
            instance_id=instance_id,
            provider=provider,
            model=model,
            sequences=malformed_sequences,
        ))
    if not records:
        reason, error_type = _missing_stream_evidence(jsonl_text)
        records.append(_unobservable_record(
            adapter=adapter,
            instance_id=instance_id,
            provider=provider,
            model=model,
            reason=(reason if not text else "no_completion_boundary_in_event_stream"),
            error_type=(error_type if not text else None),
        ))
    values = [record.to_dict() for record in records]
    return apply_process_evidence(
        values,
        process_exit_code=process_exit_code,
        termination_signal=termination_signal,
        timeout_phase=timeout_phase,
        process_error=process_error,
    )


def parse_pi_call_records(
    jsonl_text: str | None,
    *,
    instance_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    **process_evidence: Any,
) -> list[dict[str, Any]]:
    """Extract pi ``message_start``/``message_end`` completion boundaries."""
    text = jsonl_text or ""
    records: list[ModelCallRecord] = []
    active: ModelCallRecord | None = None
    session_id: str | None = None
    malformed: list[int] = []
    for sequence, line in enumerate(text.splitlines(), 1):
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            malformed.append(sequence)
            continue
        if not isinstance(event, dict):
            malformed.append(sequence)
            continue
        etype = str(event.get("type", ""))
        if etype == "session":
            session_id = _event_id(event, "id")
            continue
        message = event.get("message")
        message = message if isinstance(message, dict) else {}
        if etype == "message_start" and message.get("role", "assistant") == "assistant":
            if active is not None:
                _incomplete_record(active, reason="new_message_started_before_previous_message_ended")
                records.append(active)
            active = _start_record(
                index=len(records) + 1,
                adapter="pi_cli",
                instance_id=instance_id,
                provider=provider,
                model=model,
                event=event,
                boundary_kind="message",
                request_started=True,
                request_sent=True,
            )
            active.session_id = session_id
            active.item_id = _event_id(message, "id")
            _observe(active, sequence, event)
            continue
        if etype == "message_end" and message.get("role") == "assistant":
            if active is None:
                active = _start_record(
                    index=len(records) + 1,
                    adapter="pi_cli",
                    instance_id=instance_id,
                    provider=provider,
                    model=model,
                    event=event,
                    boundary_kind="message",
                )
                active.session_id = session_id
            _observe(active, sequence, event)
            blocks = message.get("content") or []
            texts = [
                str(block.get("text", ""))
                for block in blocks
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            has_non_text = any(
                isinstance(block, dict) and block.get("type") != "text"
                for block in blocks
            )
            _mark_response(active, event)
            active.first_token_received = any(texts)
            active.content_state = "nonempty" if any(texts) else "omitted" if has_non_text else "empty"
            active.finish_reason = _event_id(message, "stopReason", "stop_reason")
            active.provider_status = active.finish_reason
            _usage(active, message.get("usage"))
            if active.finish_reason in {"error", "aborted"}:
                active.error_type = "provider_error"
                active.error_message = _coerce_str(message.get("errorMessage"))
            _complete_record(active, event=event)
            records.append(active)
            active = None
            continue
        if active is not None:
            _observe(active, sequence, event)
            if etype == "message_update":
                _mark_response(active, event)
                delta = event.get("delta") or event.get("assistantMessageEvent")
                if delta:
                    active.first_token_received = True
                    active.content_state = "nonempty"
    if active is not None:
        _incomplete_record(active, reason="event_stream_ended_before_message_end")
        records.append(active)
    if malformed:
        records.append(_parse_gap_record(
            index=len(records) + 1, adapter="pi_cli", instance_id=instance_id,
            provider=provider, model=model, sequences=malformed,
        ))
    if not records:
        reason, error_type = _missing_stream_evidence(jsonl_text)
        records.append(_unobservable_record(
            adapter="pi_cli", instance_id=instance_id, provider=provider, model=model,
            reason=(reason if not text else "no_message_boundary_in_event_stream"),
            error_type=(error_type if not text else None),
        ))
    return apply_process_evidence([record.to_dict() for record in records], **process_evidence)


def parse_opencode_call_records(
    jsonl_text: str | None,
    *,
    instance_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    **process_evidence: Any,
) -> list[dict[str, Any]]:
    """Extract opencode ``step_start``/``step_finish`` model-step boundaries."""
    text = jsonl_text or ""
    records: list[ModelCallRecord] = []
    active: ModelCallRecord | None = None
    malformed: list[int] = []
    for sequence, line in enumerate(text.splitlines(), 1):
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            malformed.append(sequence)
            continue
        if not isinstance(event, dict):
            malformed.append(sequence)
            continue
        etype = str(event.get("type", ""))
        part = event.get("part")
        part = part if isinstance(part, dict) else {}
        if etype == "step_start":
            if active is not None:
                _incomplete_record(active, reason="new_step_started_before_previous_step_finished")
                records.append(active)
            active = _start_record(
                index=len(records) + 1,
                adapter="opencode_cli",
                instance_id=instance_id,
                provider=provider,
                model=model,
                event=event,
                boundary_kind="step",
                request_started=True,
                request_sent=True,
            )
            active.session_id = _event_id(event, "sessionID", "session_id")
            active.item_id = _event_id(part, "id")
        elif active is None and etype in {"text", "reasoning", "tool_use", "step_finish", "error"}:
            active = _start_record(
                index=len(records) + 1,
                adapter="opencode_cli",
                instance_id=instance_id,
                provider=provider,
                model=model,
                event=event,
                boundary_kind="step",
            )
            active.session_id = _event_id(event, "sessionID", "session_id")
        if active is None:
            continue
        _observe(active, sequence, event)
        if etype in {"text", "reasoning", "tool_use"}:
            _mark_response(active, event)
            if etype == "text" and part.get("text"):
                active.first_token_received = True
                active.content_state = "nonempty"
            elif active.content_state == "unknown":
                active.content_state = "omitted"
        elif etype == "step_finish":
            _mark_response(active, event)
            active.finish_reason = _event_id(part, "reason", "finish_reason")
            if active.content_state == "unknown":
                active.content_state = "empty"
            _usage(active, part.get("tokens"))
            active.cost_usd = _coerce_float(part.get("cost"))
            _complete_record(active, event=event)
            records.append(active)
            active = None
        elif etype == "error":
            error = event.get("error")
            active.error_type = "provider_error"
            active.error_class = _event_id(error, "name") if isinstance(error, dict) else None
            active.error_message = _coerce_str(error)
            _complete_record(active, event=event)
            records.append(active)
            active = None
    if active is not None:
        _incomplete_record(active, reason="event_stream_ended_before_step_finish")
        records.append(active)
    if malformed:
        records.append(_parse_gap_record(
            index=len(records) + 1, adapter="opencode_cli", instance_id=instance_id,
            provider=provider, model=model, sequences=malformed,
        ))
    if not records:
        reason, error_type = _missing_stream_evidence(jsonl_text)
        records.append(_unobservable_record(
            adapter="opencode_cli", instance_id=instance_id, provider=provider, model=model,
            reason=(reason if not text else "no_step_boundary_in_event_stream"),
            error_type=(error_type if not text else None),
        ))
    return apply_process_evidence([record.to_dict() for record in records], **process_evidence)


def parse_claude_call_records(
    jsonl_text: str | None,
    *,
    instance_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    **process_evidence: Any,
) -> list[dict[str, Any]]:
    """Treat each authoritative Claude assistant event as one completion."""
    text = jsonl_text or ""
    records: list[ModelCallRecord] = []
    malformed: list[int] = []
    for sequence, line in enumerate(text.splitlines(), 1):
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            malformed.append(sequence)
            continue
        if not isinstance(event, dict):
            malformed.append(sequence)
            continue
        if event.get("type") != "assistant":
            continue
        message = event.get("message")
        message = message if isinstance(message, dict) else {}
        blocks = message.get("content") or []
        texts = [
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        has_non_text = any(
            isinstance(block, dict) and block.get("type") != "text"
            for block in blocks
        )
        record = _start_record(
            index=len(records) + 1,
            adapter="claude_code_cli",
            instance_id=instance_id,
            provider=provider,
            model=model,
            event=event,
            boundary_kind="assistant_message",
        )
        record.request_id = _event_id(event, "request_id")
        record.response_id = _event_id(message, "id") or _event_id(event, "response_id")
        _observe(record, sequence, event)
        _mark_response(record, event)
        record.first_token_received = any(texts)
        record.content_state = "nonempty" if any(texts) else "omitted" if has_non_text else "empty"
        record.finish_reason = _event_id(message, "stop_reason", "stopReason")
        _usage(record, message.get("usage"))
        _complete_record(record, event=event)
        records.append(record)
    if malformed:
        records.append(_parse_gap_record(
            index=len(records) + 1, adapter="claude_code_cli", instance_id=instance_id,
            provider=provider, model=model, sequences=malformed,
        ))
    if not records:
        reason, error_type = _missing_stream_evidence(jsonl_text)
        records.append(_unobservable_record(
            adapter="claude_code_cli", instance_id=instance_id, provider=provider, model=model,
            reason=(reason if not text else "no_assistant_boundary_in_event_stream"),
            error_type=(error_type if not text else None),
        ))
    return apply_process_evidence([record.to_dict() for record in records], **process_evidence)


def summarize_model_calls(records: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Compute counts without inventing an attempted-call denominator."""
    items = records or []
    known_attempts = [item for item in items if item.get("attempted") is True]
    unknown_attempts = [item for item in items if item.get("attempted") is None]
    known_content_states = {"empty", "nonempty", "omitted"}
    summary: dict[str, Any] = {
        "call_records_emitted": len(items),
        "calls_attempted": len(known_attempts) if not unknown_attempts else None,
        "calls_attempted_observed": len(known_attempts),
        "calls_with_response": 0,
        "calls_with_nonempty_content": 0,
        "calls_with_empty_content": 0,
        "calls_with_omitted_content": 0,
        "calls_with_known_content_state": 0,
        "calls_with_unknown_content_state": 0,
        "calls_with_partial_content": 0,
        "calls_with_transport_error": 0,
        "calls_with_timeout": 0,
        "calls_retried": 0,
        "calls_without_observable_boundary": len(unknown_attempts),
    }
    for item in items:
        outcome = item.get("outcome")
        attempted = item.get("attempted") is True
        content_state = item.get("content_state")
        if attempted and item.get("response_received"):
            summary["calls_with_response"] += 1
        if attempted and content_state == "nonempty":
            summary["calls_with_nonempty_content"] += 1
        if attempted and content_state == "empty":
            summary["calls_with_empty_content"] += 1
        if attempted and content_state == "omitted":
            summary["calls_with_omitted_content"] += 1
        if attempted and content_state in known_content_states:
            summary["calls_with_known_content_state"] += 1
        elif attempted:
            summary["calls_with_unknown_content_state"] += 1
        if attempted and (item.get("partial_content") or outcome == "partial_response"):
            summary["calls_with_partial_content"] += 1
        if attempted and outcome in {"transport_error", "process_failure", "request_sent_no_response"}:
            summary["calls_with_transport_error"] += 1
        if attempted and outcome == "timeout":
            summary["calls_with_timeout"] += 1
        if attempted and (
            int(item.get("retry_index") or 0) > 0
            or item.get("benchmark_attempt_kind") == "failure_retry"
        ):
            summary["calls_retried"] += 1

    denominator_known = (
        bool(known_attempts)
        and not unknown_attempts
        and summary["calls_with_unknown_content_state"] == 0
    )
    summary["empty_response_rate"] = (
        summary["calls_with_empty_content"] / len(known_attempts)
        if denominator_known else None
    )
    summary["empty_response_rate_status"] = "observed" if denominator_known else "unknown"
    if not denominator_known:
        if unknown_attempts:
            summary["reason"] = "one_or_more_call_boundaries_were_not_observable"
        elif not known_attempts:
            summary["reason"] = "no_attempted_calls_were_observed"
        else:
            summary["reason"] = "one_or_more_calls_have_unknown_content_state"
    return summary
