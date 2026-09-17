"""Offline protocol regressions for the Claude-facing API proxy."""

import io
import json
import sys
import types

import pytest

from ai4sci_bench.adapters.api_proxy import _LiteLLMProxyHandler, _anthropic_messages_to_openai_standard


def chunk(delta=None, finish=None, usage=None):
    result = {
        "id": "msg-test", "model": "test-model",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
    }
    if usage is not None:
        result["usage"] = usage
    return result


def handler(body=None):
    h = _LiteLLMProxyHandler.__new__(_LiteLLMProxyHandler)
    h.litellm_model = "openai/test-model"
    h.litellm_api_base = "http://unused.invalid/v1"
    h.litellm_api_key = "test-placeholder"
    h.wfile = io.BytesIO()
    h.statuses = []
    h.sent_headers = []
    h.send_response = h.statuses.append
    h.send_header = lambda name, value: h.sent_headers.append((name, value))
    h.end_headers = lambda: None
    h.close_connection = False
    payload = json.dumps(body or {"messages": [], "stream": True}).encode()
    h.rfile = io.BytesIO(payload)
    h.headers = {"Content-Length": str(len(payload))}
    h.path = "/v1/messages"
    return h


def events(h):
    return [json.loads(line[6:]) for line in h.wfile.getvalue().decode().splitlines()
            if line.startswith("data: ")]


def assert_valid_blocks(items):
    active = {}
    seen = set()
    blocks = []
    for event in items:
        kind = event["type"]
        if kind == "content_block_start":
            idx = event["index"]
            assert idx not in seen, "a content block index was reused"
            assert not active, "content blocks must close before changing type"
            block = dict(event["content_block"])
            active[idx] = block
            blocks.append(block)
            seen.add(idx)
        elif kind == "content_block_delta":
            block = active[event["index"]]
            delta = event["delta"]
            expected = {"thinking_delta": "thinking", "text_delta": "text",
                        "input_json_delta": "tool_use", "signature_delta": "thinking"}
            assert block["type"] == expected[delta["type"]]
            key = {"thinking_delta": "thinking", "text_delta": "text",
                   "input_json_delta": "arguments", "signature_delta": "signature"}[delta["type"]]
            block[key] = block.get(key, "") + delta.get(key, delta.get("partial_json", ""))
        elif kind == "content_block_stop":
            del active[event["index"]]
        elif kind == "message_stop":
            assert not active
    assert not active
    return blocks


def wire(items, newline="\n"):
    return "".join(f"event: {event['type']}{newline}data: {json.dumps(event)}{newline}{newline}"
                   for event in items).encode()


def raw_message():
    return [
        {"type": "message_start", "message": {
            "id": "msg-raw", "type": "message", "role": "assistant", "model": "test-model",
            "content": [], "usage": {"input_tokens": 3, "output_tokens": 0},
            "stop_reason": None, "stop_sequence": None,
        }},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "hello"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
         "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    ]


def install_completion(monkeypatch, responses):
    calls = []
    iterator = iter(responses)

    def completion(**kwargs):
        calls.append(kwargs)
        value = next(iterator)
        if isinstance(value, Exception):
            raise value
        return iter(value)

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=completion))
    return calls


def test_thinking_text_tool_transitions_are_type_correct():
    h = handler()
    outcome = h._handle_streaming([
        chunk({"reasoning_content": "first"}),
        chunk({"content": "answer"}),
        chunk({"reasoning_content": "again"}),
        chunk({"tool_calls": [{"index": 0, "id": "call-0",
                              "function": {"name": "Bash", "arguments": '{"command":"pwd"}'}}]}),
        chunk(finish="tool_calls"),
    ])
    assert outcome == "ok"
    blocks = assert_valid_blocks(events(h))
    assert [b["type"] for b in blocks] == ["thinking", "text", "thinking", "tool_use"]
    assert [b.get("thinking") for b in blocks if b["type"] == "thinking"] == ["first", "again"]
    assert json.loads(blocks[-1]["arguments"]) == {"command": "pwd"}


def test_tool_arguments_before_fragmented_name_and_parallel_tools_are_preserved():
    h = handler()
    h._handle_streaming([
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"command":'}}]}),
        chunk({"tool_calls": [{"index": 0, "id": "call-0", "function": {"name": "Ba"}},
                              {"index": 1, "id": "call-1", "function": {"name": "Read", "arguments": '{"path":'}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"name": "sh", "arguments": '"pwd"}'}},
                              {"index": 1, "function": {"arguments": '"x.txt"}'}}]}),
        chunk(finish="tool_calls"),
    ])
    blocks = assert_valid_blocks(events(h))
    assert [(b["name"], json.loads(b["arguments"])) for b in blocks] == [
        ("Bash", {"command": "pwd"}), ("Read", {"path": "x.txt"}),
    ]


@pytest.mark.parametrize("response", [[], [chunk()], [chunk(usage={"completion_tokens": 1})]])
def test_empty_or_unfinished_response_fails_without_writing_headers(response):
    h = handler()
    assert h._handle_streaming(response) == "integrity_failed_before_write"
    assert h.statuses == []
    assert h.wfile.getvalue() == b""


def test_prewrite_retry_emits_exactly_one_http_response(monkeypatch):
    monkeypatch.setenv("ASIBENCH_STREAM_RETRY_ATTEMPTS", "2")
    calls = install_completion(monkeypatch, [ConnectionResetError("transport"), [chunk({"content": "ok"}, "stop")]])
    h = handler()
    h.do_POST()
    assert len(calls) == 2
    assert h.statuses == [200]
    assert [e["type"] for e in events(h)].count("message_stop") == 1
    assert not any(e["type"] == "error" for e in events(h))


def test_exhausted_prewrite_failures_return_one_structured_502(monkeypatch):
    monkeypatch.setenv("ASIBENCH_STREAM_RETRY_ATTEMPTS", "2")
    install_completion(monkeypatch, [ConnectionResetError("transport"), ConnectionResetError("transport")])
    h = handler()
    h.do_POST()
    assert h.statuses == [502]
    assert json.loads(h.wfile.getvalue())["type"] == "error"


@pytest.mark.parametrize("failure, expected", [
    (RuntimeError("upstream failed"), "nonretryable_before_write"),
    (ConnectionResetError("upstream reset"), "failed_before_write"),
])
def test_only_transport_exception_before_any_data_is_retryable(failure, expected):
    def broken():
        raise failure
        yield

    h = handler()
    assert h._handle_streaming(broken()) == expected
    assert h.statuses == []


def test_exception_after_data_emits_error_and_never_completion():
    def broken():
        yield chunk({"content": "partial"})
        raise RuntimeError("upstream failed")

    h = handler()
    assert h._handle_streaming(broken()) == "failed_after_write"
    assert events(h)[-1]["type"] == "error"
    assert not any(e["type"] == "message_stop" for e in events(h))


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("fragment", [1, 17, 100000])
def test_raw_anthropic_sse_supports_fragmentation_and_coalescing(newline, fragment):
    data = wire(raw_message(), newline)
    h = handler()
    assert h._handle_streaming([data[i:i + fragment] for i in range(0, len(data), fragment)]) == "ok"
    assert events(h) == raw_message()


@pytest.mark.parametrize("tail", [[], [{"type": "error", "error": {"type": "api_error", "message": "bad"}}]])
def test_raw_stop_reason_without_message_stop_is_not_success(tail):
    h = handler()
    outcome = h._handle_streaming([wire(raw_message()[:-1] + tail)])
    assert outcome in {"failed_after_write", "incomplete_after_write"}
    assert events(h)[-1]["type"] == "error"
    assert not any(e["type"] == "message_stop" for e in events(h))


def test_raw_terminal_followed_by_exception_cannot_commit_success():
    def broken():
        yield wire(raw_message())
        raise RuntimeError("truncated upstream")

    h = handler()
    assert h._handle_streaming(broken()) == "failed_after_write"
    assert not any(e["type"] == "message_stop" for e in events(h))


def test_force_streaming_aggregates_json_and_preserves_usage(monkeypatch):
    monkeypatch.setenv("ASIBENCH_FORCE_UPSTREAM_STREAMING", "1")
    calls = install_completion(monkeypatch, [[
        chunk({"reasoning_content": "reason"}), chunk({"content": "answer"}),
        chunk(finish="length"), {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 12}},
    ]])
    h = handler({"messages": [], "stream": False})
    h.do_POST()
    assert calls[0]["stream"] is True
    assert h.statuses == [200]
    result = json.loads(h.wfile.getvalue())
    assert result["content"] == [{"type": "thinking", "thinking": "reason"}, {"type": "text", "text": "answer"}]
    assert result["stop_reason"] == "max_tokens"
    assert result["usage"] == {"input_tokens": 7, "output_tokens": 12}


def test_force_streaming_never_returns_partial_json_as_success(monkeypatch):
    monkeypatch.setenv("ASIBENCH_FORCE_UPSTREAM_STREAMING", "1")
    monkeypatch.setenv("ASIBENCH_STREAM_RETRY_ATTEMPTS", "1")
    calls = install_completion(monkeypatch, [[chunk({"content": "partial"})]])
    h = handler({"messages": [], "stream": False})
    h.do_POST()
    assert calls[0]["stream"] is True
    assert h.statuses == [502]
    assert json.loads(h.wfile.getvalue())["type"] == "error"


def test_legacy_switch_still_uses_nonstream_when_force_is_off(monkeypatch):
    monkeypatch.delenv("ASIBENCH_FORCE_UPSTREAM_STREAMING", raising=False)
    monkeypatch.setenv("ASIBENCH_REAL_STREAMING_PROXY", "0")
    h = handler()
    calls = []

    def complete(kwargs):
        calls.append(kwargs)
        return {"id": "msg-legacy", "model": "test", "content": [{"type": "text", "text": "ok"}],
                "usage": {}, "stop_reason": "end_turn"}

    h._call_litellm_anthropic = complete
    h.do_POST()
    assert calls[0]["stream"] is False
    assert events(h)[-1]["type"] == "message_stop"


def test_force_flag_takes_precedence_over_legacy_compatibility(monkeypatch):
    monkeypatch.setenv("ASIBENCH_FORCE_UPSTREAM_STREAMING", "1")
    monkeypatch.setenv("ASIBENCH_REAL_STREAMING_PROXY", "0")
    calls = install_completion(monkeypatch, [[chunk({"content": "ok"}, "stop")]])
    h = handler()
    h.do_POST()
    assert calls[0]["stream"] is True
    assert events(h)[-1]["type"] == "message_stop"


def test_incomplete_stream_after_visible_data_is_never_replayed(monkeypatch):
    monkeypatch.setenv("ASIBENCH_STREAM_RETRY_ATTEMPTS", "3")
    calls = install_completion(monkeypatch, [[chunk({"content": "partial"})], []])
    h = handler()
    h.do_POST()
    assert len(calls) == 1
    assert h.statuses == [200]
    assert events(h)[-1]["type"] == "error"


def test_force_streaming_accepts_raw_anthropic_and_aggregates_tool_json(monkeypatch):
    monkeypatch.setenv("ASIBENCH_FORCE_UPSTREAM_STREAMING", "1")
    raw = raw_message()
    raw[1]["content_block"] = {"type": "tool_use", "id": "call-0", "name": "Read", "input": {}}
    raw[2]["delta"] = {"type": "input_json_delta", "partial_json": '{"path":"x.txt"}'}
    raw[4]["delta"]["stop_reason"] = "tool_use"
    install_completion(monkeypatch, [[wire(raw)]])
    h = handler({"messages": [], "stream": False})
    h.do_POST()
    assert h.statuses == [200]
    assert json.loads(h.wfile.getvalue())["content"] == [
        {"type": "tool_use", "id": "call-0", "name": "Read", "input": {"path": "x.txt"}},
    ]


def test_raw_unicode_multiline_data_and_ping_survive_byte_fragmentation():
    raw = raw_message()
    raw[2]["delta"]["text"] = "\u6d4b\u8bd5"
    data = b": heartbeat\r\nevent: ping\r\ndata: {\"type\":\"ping\"}\r\n\r\n"
    data += b"".join(
        b"data: " + json.dumps(e, ensure_ascii=False).encode() + b"\r\n\r\n" for e in raw
    )
    h = handler()
    assert h._handle_streaming([bytes([byte]) for byte in data]) == "ok"
    assert events(h) == [{"type": "ping"}] + raw


@pytest.mark.parametrize("ending", [b"data: {", b"data: {\"type\":\"ping\"}", b"\xe6"])
def test_truncated_raw_tail_never_commits_message_stop(ending):
    h = handler()
    assert h._handle_streaming([wire(raw_message()) + ending]) == "failed_after_write"
    assert not any(e["type"] == "message_stop" for e in events(h))


@pytest.mark.parametrize("arguments", ['{"command":', '[]', '"value"'])
def test_invalid_tool_input_never_exposes_executable_complete_block(arguments):
    h = handler()
    assert h._handle_streaming([
        chunk({"tool_calls": [{"index": 0, "id": "call-0",
                              "function": {"name": "Bash", "arguments": arguments}}]}),
        chunk(finish="tool_calls"),
    ]) == "failed_after_write"
    assert not any(e["type"] in {"content_block_stop", "message_stop"} for e in events(h))


def test_raw_delta_type_mismatch_is_rejected():
    raw = raw_message()
    raw[1]["content_block"] = {"type": "thinking", "thinking": ""}
    h = handler()
    assert h._handle_streaming([wire(raw)]) == "failed_after_write"
    assert events(h)[-1]["type"] == "error"


def test_raw_content_after_stop_reason_is_rejected():
    raw = raw_message()
    raw.insert(-1, {"type": "content_block_start", "index": 1,
                    "content_block": {"type": "text", "text": "extra"}})
    raw.insert(-1, {"type": "content_block_stop", "index": 1})
    h = handler()
    assert h._handle_streaming([wire(raw)]) == "failed_after_write"
    assert not any(e["type"] == "message_stop" for e in events(h))


def test_valid_empty_model_response_is_not_answer_conditioned_retry(monkeypatch):
    calls = install_completion(monkeypatch, [[chunk(finish="stop")]])
    h = handler()
    h.do_POST()
    assert len(calls) == 1
    assert events(h)[-1]["type"] == "message_stop"
    assert not any(e["type"] == "content_block_start" for e in events(h))


def test_disconnect_during_downstream_write_is_not_upstream_retry():
    h = handler()

    def disconnected(_event, _data):
        raise ConnectionResetError("downstream reset")

    h._write_sse = disconnected
    assert h._handle_streaming([chunk({"content": "ok"}, "stop")]) == "client_gone"


def test_text_after_tool_delta_preserves_first_appearance_order():
    h = handler()
    assert h._handle_streaming([
        chunk({"tool_calls": [{"index": 0, "id": "call-0",
                              "function": {"name": "Read", "arguments": '{"path":"x"}'}}]}),
        chunk({"content": "after"}), chunk(finish="tool_calls"),
    ]) == "ok"
    assert [b["type"] for b in assert_valid_blocks(events(h))] == ["tool_use", "text"]


@pytest.mark.parametrize("stream", [True, False])
def test_http_roundtrip_prewrite_retry_has_one_response(monkeypatch, stream):
    import urllib.request

    from ai4sci_bench.adapters import api_proxy

    monkeypatch.setattr(api_proxy, "_configure_proxy_logging", lambda: None)
    monkeypatch.setenv("ASIBENCH_FORCE_UPSTREAM_STREAMING", "1")
    monkeypatch.setenv("ASIBENCH_STREAM_RETRY_ATTEMPTS", "2")
    calls = install_completion(monkeypatch, [ConnectionResetError("transport"), [chunk({"content": "ok"}, "stop")]])
    proxy = api_proxy.LiteLLMProxy(model="openai/test-model")
    try:
        url = proxy.start()
        request = urllib.request.Request(url + "/v1/messages", method="POST",
                                         data=json.dumps({"messages": [], "stream": stream}).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 200
            data = response.read()
            assert b"HTTP/1." not in data
            assert b'"type": "error"' not in data
            if stream:
                assert data.count(b"event: message_stop") == 1
            else:
                assert json.loads(data)["content"] == [{"type": "text", "text": "ok"}]
        assert len(calls) == 2
        assert all(call["stream"] for call in calls)
    finally:
        proxy.stop()


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("bad_response", [[], [chunk()], [chunk({"content": "partial"})],
                                             [{"error": {"message": "bad"}}]])
def test_integrity_failure_is_not_regenerated_even_with_retry_budget(monkeypatch, stream, bad_response):
    monkeypatch.setenv("ASIBENCH_FORCE_UPSTREAM_STREAMING", "1")
    monkeypatch.setenv("ASIBENCH_STREAM_RETRY_ATTEMPTS", "3")
    calls = install_completion(monkeypatch, [bad_response, [chunk({"content": "replacement"}, "stop")]])
    h = handler({"messages": [], "stream": stream})
    h.do_POST()
    assert len(calls) == 1
    if h.statuses == [200]:
        assert events(h)[-1]["type"] == "error"
    else:
        assert h.statuses == [502]


def test_aggregation_transport_failure_after_content_is_not_regenerated(monkeypatch):
    monkeypatch.setenv("ASIBENCH_FORCE_UPSTREAM_STREAMING", "1")
    monkeypatch.setenv("ASIBENCH_STREAM_RETRY_ATTEMPTS", "3")

    def broken():
        yield chunk({"content": "partial"})
        raise ConnectionResetError("reset after generation")

    calls = install_completion(monkeypatch, [broken(), [chunk({"content": "replacement"}, "stop")]])
    h = handler({"messages": [], "stream": False})
    h.do_POST()
    assert len(calls) == 1
    assert h.statuses == [502]


@pytest.mark.parametrize("first", [chunk(), {"choices": [], "usage": {"completion_tokens": 1}}, b"data: "])
def test_transport_failure_after_unemitted_upstream_item_is_not_regenerated(monkeypatch, first):
    monkeypatch.setenv("ASIBENCH_FORCE_UPSTREAM_STREAMING", "1")
    monkeypatch.setenv("ASIBENCH_STREAM_RETRY_ATTEMPTS", "3")

    def broken():
        yield first
        raise ConnectionResetError("upstream already began")

    calls = install_completion(monkeypatch, [broken(), [chunk({"content": "replacement"}, "stop")]])
    h = handler({"messages": [], "stream": False})
    h.do_POST()
    assert len(calls) == 1
    assert h.statuses == [502]


@pytest.mark.parametrize("finish", ["length", "tool_calls"])
@pytest.mark.parametrize("arguments", [None, ""])
def test_missing_tool_arguments_never_become_empty_executable_input(finish, arguments):
    function = {"name": "Bash"}
    if arguments is not None:
        function["arguments"] = arguments
    h = handler()
    outcome = h._handle_streaming([
        chunk({"tool_calls": [{"index": 0, "id": "call", "function": function}]}),
        chunk(finish=finish),
    ])
    assert outcome != "ok"
    assert not any(e.get("content_block", {}).get("type") == "tool_use" for e in events(h))
    assert not any(e["type"] == "message_stop" for e in events(h))


def test_explicit_empty_object_tool_input_remains_valid():
    h = handler()
    assert h._handle_streaming([
        chunk({"tool_calls": [{"index": 0, "id": "call", "function": {"name": "Ping", "arguments": "{}"}}]}),
        chunk(finish="tool_calls"),
    ]) == "ok"
    assert json.loads(assert_valid_blocks(events(h))[0]["arguments"]) == {}


@pytest.mark.parametrize("environment", ["environment", [{"type": "text", "text": "environment"}]])
def test_cc_system_context_is_preserved_at_start_of_openai_messages(environment):
    body = {"system": [{"type": "text", "text": "policy"}], "messages": [
        {"role": "user", "content": "attribution"},
        {"role": "system", "content": environment},
        {"role": "user", "content": "task"},
    ]}
    messages = _anthropic_messages_to_openai_standard(body)
    assert [m["role"] for m in messages] == ["system", "user", "user"]
    assert messages[0]["content"] == "policy\n\nenvironment"
    assert messages[1:]==[{"role":"user","content":"attribution"},{"role":"user","content":"task"}]
    assert body["messages"][1]["role"] == "system"


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 503])
def test_upstream_http_status_is_preserved_instead_of_made_retryable(monkeypatch, status):
    error = RuntimeError("fixture upstream HTTP failure")
    error.status_code = status
    monkeypatch.setenv("ASIBENCH_STREAM_RETRY_ATTEMPTS", "1")
    calls = install_completion(monkeypatch, [error])
    h = handler()
    h.do_POST()
    assert len(calls) == 1
    assert h.statuses == [status]
    if status == 429:
        assert json.loads(h.wfile.getvalue())["error"]["type"] == "rate_limit_error"
