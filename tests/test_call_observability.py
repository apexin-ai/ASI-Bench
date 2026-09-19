"""Model-call boundary, transport-evidence, persistence, and privacy tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from ai4sci_bench.adapters.direct_llm import DirectLLMAdapter
from ai4sci_bench.core.agent_interface import AgentAdapter
from ai4sci_bench.core.types import (
    AgentOutput,
    EvalResult,
    PromptLevel,
    RunStatus,
)
from ai4sci_bench.runner.orchestrator import BenchmarkOrchestrator, RunConfig
from ai4sci_bench.reporting.result_loader import parse_eval_result
from ai4sci_bench.reporting.aggregator import aggregate_results
from ai4sci_bench.trajectory.call_observability import (
    ModelCallRecord,
    apply_process_evidence,
    parse_claude_call_records,
    parse_codex_call_records,
    parse_opencode_call_records,
    parse_pi_call_records,
    summarize_model_calls,
)


def _jsonl(*events: dict) -> str:
    return "\n".join(json.dumps(event) for event in events)


class _NoopAgent(AgentAdapter):
    def solve(self, task_instance):  # pragma: no cover - not used directly
        raise AssertionError("not used")


def _orchestrator(tmp_path) -> BenchmarkOrchestrator:
    tasks = tmp_path / "tasks"
    tasks.mkdir(exist_ok=True)
    return BenchmarkOrchestrator(RunConfig(
        agent=_NoopAgent(),
        tasks_dir=str(tasks),
        output_dir=str(tmp_path / "results"),
    ))


def _eval_result(tmp_path, output: AgentOutput, *, attempt: int = 1) -> EvalResult:
    return EvalResult(
        instance_id=output.instance_id,
        task_id="physics.test_task",
        prompt_level=PromptLevel.B2,
        agent_name="CodexCLIAdapter",
        parameters={},
        gate_results=[],
        gates_passed=True,
        score_results=[],
        final_score=0.0,
        status=output.status,
        attempt=attempt,
        agent_output=output,
    )


def test_codex_turn_preserves_evidence_without_inventing_completion_boundary():
    records = parse_codex_call_records(_jsonl(
        {"type": "thread.started", "thread_id": "thread-1"},
        {
            "type": "turn.started",
            "turn_id": "turn-1",
            "request_id": "request-1",
            "timestamp": "2026-01-01T00:00:00Z",
        },
        {
            "type": "item.completed",
            "response_id": "response-1",
            "item": {"id": "item-1", "type": "agent_message", "text": "done"},
        },
        {
            "type": "turn.completed",
            "turn_id": "turn-1",
            "finish_reason": "stop",
            "usage": {"input_tokens": 11, "output_tokens": 3},
            "timestamp": "2026-01-01T00:00:01Z",
        },
    ))

    assert len(records) == 1
    record = records[0]
    assert record["attempted"] is None
    assert record["thread_id"] == "thread-1"
    assert record["turn_id"] == "turn-1"
    assert record["item_id"] == "item-1"
    assert record["request_id"] == "request-1"
    assert record["response_id"] == "response-1"
    assert record["finish_reason"] == "stop"
    assert record["request_sent"] is None
    assert record["stream_opened"] is None
    assert record["stream_ended"] is None
    assert record["call_finished"] is None
    assert record["content_state"] == "unknown"
    assert record["outcome"] == "no_observable_boundary"
    assert record["boundary_observed"] is False
    assert record["evidence_reason"] == (
        "codex_turn_does_not_expose_completion_boundaries"
    )
    assert record["token_count"] == 14
    assert record["duration_ms"] == 1000


def test_codex_turn_without_visible_content_does_not_prove_empty_completion():
    records = parse_codex_call_records(_jsonl(
        {"type": "turn.started", "turn_id": "turn-empty"},
        {"type": "turn.completed", "turn_id": "turn-empty", "finish_reason": "stop"},
    ))
    summary = summarize_model_calls(records)

    assert records[0]["content_state"] == "unknown"
    assert records[0]["outcome"] == "no_observable_boundary"
    assert summary["calls_attempted"] is None
    assert summary["calls_with_empty_content"] == 0
    assert summary["empty_response_rate"] is None


def test_codex_explicit_provider_completion_can_prove_empty_response():
    records = parse_codex_call_records(_jsonl(
        {"type": "request.sent", "request_id": "request-empty"},
        {
            "type": "response.completed",
            "id": "response-empty",
            "finish_reason": "stop",
        },
    ))
    summary = summarize_model_calls(records)

    assert records[0]["boundary_kind"] == "provider_response"
    assert records[0]["attempted"] is True
    assert records[0]["content_state"] == "empty"
    assert records[0]["outcome"] == "complete_empty_response"
    assert summary["calls_attempted"] == 1
    assert summary["calls_with_empty_content"] == 1
    assert summary["empty_response_rate"] == 1.0


def test_empty_event_stream_emits_coverage_record_without_inventing_attempt():
    records = parse_codex_call_records("")
    summary = summarize_model_calls(records)

    assert records[0]["attempted"] is None
    assert records[0]["evidence_reason"] == "empty_event_stream"
    assert records[0]["outcome"] == "no_observable_boundary"
    assert summary["call_records_emitted"] == 1
    assert summary["calls_attempted"] is None
    assert summary["calls_attempted_observed"] == 0
    assert summary["empty_response_rate"] is None
    assert summary["empty_response_rate_status"] == "unknown"


def test_unavailable_event_stream_is_distinct_from_observed_empty_stream():
    record = parse_codex_call_records(None)[0]

    assert record["attempted"] is None
    assert record["evidence_reason"] == "event_stream_unavailable"
    assert record["error_type"] == "unavailable_stream"


def test_malformed_json_makes_denominator_unknown_even_with_valid_call():
    stream = _jsonl({
        "type": "message", "role": "assistant", "content": "ok",
        "response_id": "response-1",
    }) + "\n{broken"
    records = parse_codex_call_records(stream)
    summary = summarize_model_calls(records)

    assert len(records) == 2
    assert records[1]["error_type"] == "parse_error"
    assert records[1]["parse_error_count"] == 1
    assert summary["calls_attempted_observed"] == 1
    assert summary["calls_attempted"] is None
    assert summary["empty_response_rate"] is None


def test_pi_partial_token_then_disconnect_remains_unfinished():
    records = parse_pi_call_records(_jsonl(
        {"type": "session", "id": "session-1"},
        {
            "type": "message_start",
            "timestamp": 1000,
            "message": {"role": "assistant", "id": "message-1"},
        },
        {"type": "message_update", "timestamp": 1001, "delta": {"text": "partial"}},
    ))
    record = records[0]

    assert record["session_id"] == "session-1"
    assert record["first_token_received"] is True
    assert record["partial_content"] is True
    assert record["stream_ended"] is None
    assert record["call_finished"] is False
    assert record["outcome"] == "partial_response"


def test_call_started_with_zero_tokens_then_eof_is_not_complete_empty():
    records = parse_opencode_call_records(_jsonl(
        {"type": "step_start", "sessionID": "session-1", "timestamp": 1000},
    ))
    record = records[0]

    assert record["request_sent"] is True
    assert record["stream_opened"] is None
    assert record["first_token_received"] is None
    assert record["call_finished"] is False
    assert record["content_state"] == "unknown"
    assert record["outcome"] == "request_sent_no_response"
    assert summarize_model_calls(records)["empty_response_rate"] is None


def test_explicit_provider_stream_open_with_zero_tokens_preserves_transport_evidence():
    records = parse_codex_call_records(_jsonl(
        {"type": "request.sent", "request_id": "request-1"},
        {"type": "stream.opened", "status_code": 200},
        {"type": "stream.closed", "reason": "unexpected_eof"},
    ))
    record = records[0]

    assert record["request_sent"] is True
    assert record["http_headers_received"] is True
    assert record["stream_opened"] is True
    assert record["stream_ended"] is True
    assert record["first_token_received"] is None
    assert record["call_finished"] is False
    assert record["connection_close_reason"] == "unexpected_eof"
    assert record["outcome"] == "transport_established_no_token"


def test_tool_only_response_is_omitted_content_not_empty():
    records = parse_opencode_call_records(_jsonl(
        {"type": "step_start", "sessionID": "session-1"},
        {"type": "tool_use", "part": {"tool": "bash"}},
        {"type": "step_finish", "part": {"reason": "tool-calls"}},
    ))
    summary = summarize_model_calls(records)

    assert records[0]["content_state"] == "omitted"
    assert records[0]["outcome"] == "complete_tool_only_response"
    assert summary["calls_with_omitted_content"] == 1
    assert summary["calls_with_empty_content"] == 0
    assert summary["empty_response_rate"] == 0.0


def test_known_non_attempt_does_not_create_an_unknown_call_boundary():
    record = ModelCallRecord(
        attempted=False,
        boundary_observed=True,
        request_started=False,
        call_finished=True,
        outcome="no_request_created",
    ).to_dict()
    summary = summarize_model_calls([record])

    assert summary["calls_attempted"] == 0
    assert summary["calls_without_observable_boundary"] == 0
    assert summary["empty_response_rate"] is None
    assert summary["reason"] == "no_attempted_calls_were_observed"


def test_run_report_keeps_rate_unknown_when_any_result_has_coverage_gap(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    known_record = ModelCallRecord(
        attempted=True,
        boundary_observed=True,
        response_received=True,
        call_finished=True,
        content_state="nonempty",
        outcome="complete_nonempty_response",
    ).to_dict()
    unknown_record = ModelCallRecord(
        attempted=None,
        boundary_observed=False,
        content_state="unknown",
        outcome="no_observable_boundary",
    ).to_dict()

    def result(
        instance_id: str,
        records: list[dict],
        *,
        include_manifest: bool = True,
    ) -> EvalResult:
        output = AgentOutput(
            instance_id=instance_id,
            output_dir=workspace,
            code_files=[],
            data_files=[],
            log="",
            execution_time_seconds=0.1,
            status=RunStatus.COMPLETED,
            model_call_records=records,
            model_call_observability=(
                {"summary": summarize_model_calls(records)}
                if include_manifest
                else None
            ),
        )
        return _eval_result(tmp_path, output)

    report = aggregate_results([
        result("known", [known_record]),
        result("unknown", [unknown_record], include_manifest=False),
    ])

    assert report.model_call_summary is not None
    assert report.model_call_summary["calls_attempted"] is None
    assert report.model_call_summary["calls_attempted_observed"] == 1
    assert report.model_call_summary["empty_response_rate"] is None
    assert report.model_call_summary["results_with_unknown_denominator"] == 1
    assert report.model_call_summary["results_without_call_observability"] == 1


def test_run_report_empty_rate_uses_attempted_call_denominator(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def result(instance_id: str, content_state: str) -> EvalResult:
        record = ModelCallRecord(
            attempted=True,
            boundary_observed=True,
            response_received=True,
            call_finished=True,
            content_state=content_state,
            outcome=(
                "complete_empty_response"
                if content_state == "empty"
                else "complete_nonempty_response"
            ),
        ).to_dict()
        output = AgentOutput(
            instance_id=instance_id,
            output_dir=workspace,
            code_files=[],
            data_files=[],
            log="",
            execution_time_seconds=0.1,
            status=RunStatus.COMPLETED,
            model_call_records=[record],
            model_call_observability={"summary": summarize_model_calls([record])},
        )
        return _eval_result(tmp_path, output)

    report = aggregate_results([
        result("empty", "empty"),
        result("nonempty", "nonempty"),
    ])

    assert report.model_call_summary is not None
    assert report.model_call_summary["calls_attempted"] == 2
    assert report.model_call_summary["calls_with_empty_content"] == 1
    assert report.model_call_summary["empty_response_rate"] == 0.5
    assert report.model_call_summary["empty_response_rate_status"] == "observed"


def test_incomplete_provider_terminal_state_is_not_clean_stream_end():
    records = parse_codex_call_records(_jsonl(
        {"type": "response.incomplete", "id": "response-1", "status": "max_output"},
    ))
    record = records[0]

    assert record["response_id"] == "response-1"
    assert record["call_finished"] is True
    assert record["stream_ended"] is None
    assert record["error_type"] == "incomplete_response"
    assert record["outcome"] == "incomplete_response"


def test_timeout_and_process_signal_are_preserved_on_unfinished_call():
    records = parse_opencode_call_records(
        _jsonl({"type": "step_start"}),
        process_exit_code=None,
        termination_signal=9,
        timeout_phase="process_execution",
        process_error="killed after timeout",
    )
    record = records[0]

    assert record["termination_signal"] == 9
    assert record["timeout_phase"] == "process_execution"
    assert record["error_type"] == "timeout"
    assert record["outcome"] == "timeout"
    assert record["error_message"] == "killed after timeout"


def test_process_failure_does_not_reclassify_completed_response():
    record = ModelCallRecord(
        attempted=True,
        boundary_observed=True,
        call_finished=True,
        content_state="nonempty",
        outcome="complete_nonempty_response",
    ).to_dict()
    updated = apply_process_evidence(
        [record], process_exit_code=2, termination_signal=None, timeout_phase=None,
    )[0]

    assert updated["process_exit_code"] == 2
    assert updated["outcome"] == "complete_nonempty_response"


def test_process_error_without_exit_code_classifies_unfinished_call():
    record = parse_codex_call_records(
        None,
        process_error="adapter reader failed",
    )[0]

    assert record["attempted"] is None
    assert record["process_exit_code"] is None
    assert record["error_type"] == "process_failure"
    assert record["error_message"] == "adapter reader failed"
    assert record["outcome"] == "process_failure"
    assert record["evidence_reason"] == (
        "adapter_process_failed_before_call_finished"
    )


def test_claude_multiple_calls_keep_order_and_provider_ids():
    records = parse_claude_call_records(_jsonl(
        {
            "type": "assistant", "request_id": "request-1",
            "message": {
                "id": "response-1", "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "name": "bash"}],
            },
        },
        {
            "type": "assistant", "request_id": "request-2",
            "message": {
                "id": "response-2", "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "done"}],
            },
        },
    ))

    assert [record["call_index"] for record in records] == [1, 2]
    assert [record["request_id"] for record in records] == ["request-1", "request-2"]
    assert [record["response_id"] for record in records] == ["response-1", "response-2"]
    assert [record["finish_reason"] for record in records] == ["tool_use", "end_turn"]
    assert [record["content_state"] for record in records] == ["omitted", "nonempty"]


def test_finish_reason_marks_truncated_response_incomplete():
    records = parse_claude_call_records(_jsonl({
        "type": "assistant",
        "message": {
            "id": "response-truncated",
            "stop_reason": "max_tokens",
            "content": [{"type": "text", "text": "partial"}],
        },
    }))

    assert records[0]["finish_reason"] == "max_tokens"
    assert records[0]["content_state"] == "nonempty"
    assert records[0]["error_type"] == "incomplete_response"
    assert records[0]["outcome"] == "incomplete_response"


def test_adapter_retry_links_to_previous_call_index():
    records = parse_codex_call_records(_jsonl(
        {"type": "message", "role": "assistant", "content": "first"},
        {
            "type": "request.retry", "retry_index": 1,
            "reason": "rate_limit", "retry_after": 0.5, "initiator": "provider_client",
        },
        {"type": "message", "role": "assistant", "content": "second"},
    ))

    assert records[1]["retry_index"] == 1
    assert records[1]["retry_cause"] == "rate_limit"
    assert records[1]["retry_initiator"] == "provider_client"
    assert records[1]["retry_backoff_seconds"] == 0.5
    assert records[1]["parent_call_index"] == 1


def test_direct_llm_records_response_transport_metadata(sample_task_instance):
    adapter = DirectLLMAdapter(model="openai/test")
    choice = SimpleNamespace(
        message=SimpleNamespace(content="```python\nprint('ok')\n```"),
        finish_reason="stop",
    )
    response = SimpleNamespace(
        id="response-123",
        choices=[choice],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=2),
        _hidden_params={
            "status_code": 200,
            "additional_headers": {
                "x-request-id": "request-123",
                "authorization": "Bearer must-not-persist",
            },
        },
    )
    with (
        patch("litellm.completion", return_value=response),
        patch.object(adapter, "_execute", return_value=(True, "ok", "", "")),
    ):
        output = adapter.solve(sample_task_instance)

    record = output.model_call_records[0]
    assert record["response_id"] == "response-123"
    assert record["request_id"] == "request-123"
    assert record["http_status"] == 200
    assert record["http_headers_received"] is True
    assert record["finish_reason"] == "stop"
    assert record["content_state"] == "nonempty"
    assert record["token_count"] == 9
    assert "must-not-persist" not in json.dumps(record)


def test_direct_llm_records_complete_empty_response(sample_task_instance):
    adapter = DirectLLMAdapter(model="openai/test")
    response = SimpleNamespace(
        id="response-empty",
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=None), finish_reason="stop",
        )],
        usage=None,
        _hidden_params={},
    )
    with patch("litellm.completion", return_value=response):
        output = adapter.solve(sample_task_instance)

    record = output.model_call_records[0]
    assert record["response_id"] == "response-empty"
    assert record["content_state"] == "empty"
    assert record["outcome"] == "complete_empty_response"
    assert output.status == RunStatus.FAILED  # no code was available to execute


def test_orchestrator_assigns_unique_ids_and_distinguishes_repeat_from_retry(tmp_path):
    orchestrator = _orchestrator(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def result(attempt: int, status: RunStatus) -> EvalResult:
        output = AgentOutput(
            instance_id="instance-1", output_dir=workspace, code_files=[], data_files=[],
            log="", execution_time_seconds=0.1, status=status,
            model_call_records=[ModelCallRecord(
                call_index=1, attempted=True, boundary_observed=True,
                call_finished=True, content_state="nonempty",
                outcome="complete_nonempty_response",
            ).to_dict()],
        )
        return _eval_result(tmp_path, output, attempt=attempt)

    first = result(1, RunStatus.COMPLETED)
    orchestrator._annotate_model_call_records(
        first, benchmark_attempt=1, previous_result=None,
    )
    repeated = result(2, RunStatus.COMPLETED)
    orchestrator._annotate_model_call_records(
        repeated, benchmark_attempt=2, previous_result=first,
    )
    failed = result(2, RunStatus.FAILED)
    retry = result(3, RunStatus.COMPLETED)
    orchestrator._annotate_model_call_records(
        failed, benchmark_attempt=2, previous_result=first,
    )
    orchestrator._annotate_model_call_records(
        retry, benchmark_attempt=3, previous_result=failed,
    )

    first_record = first.agent_output.model_call_records[0]
    repeat_record = repeated.agent_output.model_call_records[0]
    retry_record = retry.agent_output.model_call_records[0]
    assert first_record["call_id"] != repeat_record["call_id"]
    assert repeat_record["benchmark_attempt_kind"] == "configured_repeat"
    assert repeat_record["parent_call_id"] == first_record["call_id"]
    assert retry_record["benchmark_attempt_kind"] == "failure_retry"
    assert retry_record["benchmark_retry_cause"] == "previous_attempt_failed"
    assert retry_record["benchmark_retry_initiator"] == "benchmark_runner"
    assert retry_record["benchmark_parent_call_id"] == (
        failed.agent_output.model_call_records[0]["call_id"]
    )


def test_persistence_keeps_empty_raw_stream_and_redacts_sensitive_evidence(tmp_path):
    orchestrator = _orchestrator(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    record = ModelCallRecord(
        call_index=1,
        attempted=None,
        adapter="codex_cli",
        instance_id="instance-1",
        evidence_status="not_observed",
        evidence_reason="empty_event_stream",
        error_type="empty_stream",
        error_message="Authorization: Bearer secret-token",
    ).to_dict()
    record["authorization"] = "Bearer another-secret"
    output = AgentOutput(
        instance_id="instance-1",
        output_dir=workspace,
        code_files=[],
        data_files=[],
        log="",
        execution_time_seconds=0.1,
        status=RunStatus.FAILED,
        raw_stdout="",
        raw_stdout_format="jsonl",
        model_call_records=[record],
    )
    result = _eval_result(tmp_path, output)
    orchestrator._annotate_model_call_records(
        result, benchmark_attempt=1, previous_result=None,
    )
    orchestrator._save_result(result)

    result_dir = tmp_path / "results" / "physics.test_task"
    data = json.loads((result_dir / "instance-1__b2.json").read_text())
    observation = data["agent_output"]["model_call_observability"]
    records_text = (result_dir / observation["records_file"]).read_text()
    assert observation["capture_status"] == "partial"
    assert observation["summary"]["calls_attempted"] is None
    assert observation["summary"]["empty_response_rate"] is None
    assert "secret-token" not in records_text
    assert "another-secret" not in records_text
    assert "<redacted>" in records_text
    assert (result_dir / "instance-1__b2.agent_stdout.jsonl").is_file()
    assert (result_dir / "instance-1__b2.agent_stdout.jsonl").read_text() == ""
    trajectory_path = result_dir / data["agent_output"]["trajectory_file"]
    assert json.loads(trajectory_path.read_text()) == []

    loaded = parse_eval_result(
        data,
        result_path=result_dir / "instance-1__b2.json",
    )
    assert loaded.agent_output is not None
    assert loaded.agent_output.model_call_observability == observation
    assert len(loaded.agent_output.model_call_records) == 1
    assert loaded.agent_output.model_call_records[0]["evidence_reason"] == "empty_event_stream"


def test_raw_json_evidence_redacts_prompts_credentials_but_keeps_endpoint(tmp_path):
    orchestrator = _orchestrator(tmp_path)
    raw = json.dumps({
        "system_prompt": "private instructions",
        "user_prompt": "private task",
        "headers": {
            "Authorization": "Bearer secret-token",
            "Cookie": "session=secret-cookie",
        },
        "messages": [{"role": "user", "content": "private nested prompt"}],
        "endpoint": "https://api.example.test/v1/responses",
    })
    sanitized = orchestrator._sanitize_raw_artifact_text(
        raw, raw_format="json", workspace=None,
    )

    assert "private instructions" not in sanitized
    assert "private task" not in sanitized
    assert "private nested prompt" not in sanitized
    assert "secret-token" not in sanitized
    assert "secret-cookie" not in sanitized
    assert "https://api.example.test/v1/responses" in sanitized
    assert sanitized.count("<redacted>") == 5
