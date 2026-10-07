"""Issue #8: lossless Anthropic Messages to Responses reasoning replay."""

from __future__ import annotations

import io
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ai4sci_bench.adapters.api_proxy import _LiteLLMProxyHandler


def handler(*, key: bytes = b"k" * 32):
    h = object.__new__(_LiteLLMProxyHandler)
    h.litellm_model = "openai/test-model"
    h.litellm_api_base = "https://example.invalid/v1"
    h.litellm_api_key = "test-only"
    h.anthropic_via_responses = True
    h.reasoning_replay_key = key
    h.supports_image_input = True
    h.wfile = io.BytesIO()
    h.send_response = Mock()
    h.send_header = Mock()
    h.end_headers = Mock()
    return h


def responses_result(output=None, status="completed"):
    return {
        "id": "resp_original",
        "model": "test-model",
        "status": status,
        "output": output or [],
        "error": None,
        "incomplete_details": None,
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }


def response_body(h):
    return json.loads(h.wfile.getvalue())


def sse_events(h):
    return [json.loads(line[6:]) for line in h.wfile.getvalue().decode().splitlines()
            if line.startswith("data: ")]


def test_two_turn_reasoning_replay_preserves_id_and_encrypted_content(monkeypatch):
    reasoning = {
        "type": "reasoning",
        "id": "rs_original",
        "summary": [{"type": "summary_text", "text": "Checked the workspace."}],
        "encrypted_content": "opaque-ciphertext",
    }
    tool_call = {
        "type": "function_call", "id": "fc_original", "call_id": "call_original",
        "name": "shell", "arguments": '{"command":"pwd"}', "status": "completed",
    }
    upstream = Mock(side_effect=[
        responses_result([reasoning, tool_call]),
        responses_result([{
            "type": "message", "id": "msg_final", "role": "assistant",
            "status": "completed", "content": [{"type": "output_text", "text": "Done"}],
        }]),
    ])
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(responses=upstream))

    first = handler()
    first._handle_anthropic_via_responses({
        "model": "ignored", "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Inspect the workspace."}],
        "tools": [{"name": "shell", "description": "Run a command",
                   "input_schema": {"type": "object"}}],
    })
    first_body = response_body(first)
    thinking = next(block for block in first_body["content"] if block["type"] == "thinking")
    returned_tool = next(block for block in first_body["content"] if block["type"] == "tool_use")
    assert thinking["thinking"] == "Checked the workspace."
    assert thinking["signature"].startswith("asibench-reasoning-v1.")
    assert returned_tool["id"].startswith("asibench-tool-v1.")

    second = handler()
    second._handle_anthropic_via_responses({
        "model": "ignored", "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Inspect the workspace."},
            {"role": "assistant", "content": [thinking, returned_tool]},
            {"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": returned_tool["id"],
                "content": "/workspace",
            }]},
        ],
    })
    replay_input = upstream.call_args_list[1].kwargs["input"]
    replayed = next(item for item in replay_input if item.get("type") == "reasoning")
    assert replayed == reasoning
    replayed_tool = next(item for item in replay_input if item.get("type") == "function_call")
    assert replayed_tool == tool_call
    assert upstream.call_args_list[1].kwargs["include"] == ["reasoning.encrypted_content"]
    assert upstream.call_args_list[1].kwargs["store"] is False


def test_redacted_reasoning_round_trip(monkeypatch):
    reasoning = {
        "type": "reasoning", "id": "rs_redacted", "summary": [],
        "encrypted_content": "opaque-redacted",
    }
    upstream = Mock(side_effect=[responses_result([reasoning]), responses_result()])
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(responses=upstream))

    first = handler()
    first._handle_anthropic_via_responses({
        "max_tokens": 10, "messages": [{"role": "user", "content": "Think"}],
    })
    redacted = response_body(first)["content"][0]
    assert redacted["type"] == "redacted_thinking"
    assert redacted["data"].startswith("asibench-reasoning-v1.")

    second = handler()
    second._handle_anthropic_via_responses({
        "max_tokens": 10,
        "messages": [
            {"role": "user", "content": "Think"},
            {"role": "assistant", "content": [redacted]},
        ],
    })
    assert upstream.call_args_list[1].kwargs["input"] == [
        {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "Think"}]},
        reasoning,
    ]


def test_redacted_reasoning_stream_carries_replay_data(monkeypatch):
    reasoning = {
        "type": "reasoning", "id": "rs_stream", "summary": [],
        "encrypted_content": "opaque-stream",
    }
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(
        responses=Mock(return_value=responses_result([reasoning])),
    ))
    h = handler()
    h._handle_anthropic_via_responses({
        "max_tokens": 10, "stream": True,
        "messages": [{"role": "user", "content": "Think"}],
    })
    start = next(event for event in sse_events(h)
                 if event["type"] == "content_block_start")
    assert start["content_block"]["type"] == "redacted_thinking"
    assert start["content_block"]["data"].startswith("asibench-reasoning-v1.")


@pytest.mark.parametrize("mutation", ["tamper", "other_proxy"])
def test_invalid_or_cross_proxy_replay_fails_before_upstream(monkeypatch, mutation):
    reasoning = {
        "type": "reasoning", "id": "rs_private", "summary": [],
        "encrypted_content": "opaque-private",
    }
    upstream = Mock(return_value=responses_result([reasoning]))
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(responses=upstream))
    first = handler()
    first._handle_anthropic_via_responses({
        "max_tokens": 10, "messages": [{"role": "user", "content": "Think"}],
    })
    token = response_body(first)["content"][0]["data"]
    upstream.reset_mock()
    if mutation == "tamper":
        token = token[:-1] + ("A" if token[-1] != "A" else "B")
        second = handler()
    else:
        second = handler(key=b"z" * 32)
    second._handle_anthropic_via_responses({
        "max_tokens": 10,
        "messages": [
            {"role": "user", "content": "Think"},
            {"role": "assistant", "content": [
                {"type": "redacted_thinking", "data": token},
            ]},
        ],
    })
    second.send_response.assert_called_once_with(400)
    upstream.assert_not_called()


def test_replay_is_bound_to_the_original_conversation(monkeypatch):
    reasoning = {
        "type": "reasoning", "id": "rs_bound", "summary": [],
        "encrypted_content": "opaque-bound",
    }
    upstream = Mock(return_value=responses_result([reasoning]))
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(responses=upstream))
    h = handler()
    h._handle_anthropic_via_responses({
        "max_tokens": 10, "messages": [{"role": "user", "content": "session A"}],
    })
    token = response_body(h)["content"][0]["data"]
    upstream.reset_mock()
    other_session = handler()
    other_session._handle_anthropic_via_responses({
        "max_tokens": 10,
        "messages": [
            {"role": "user", "content": "session B"},
            {"role": "assistant", "content": [
                {"type": "redacted_thinking", "data": token},
            ]},
        ],
    })
    other_session.send_response.assert_called_once_with(400)
    upstream.assert_not_called()


def test_concurrent_reasoning_replay_has_no_shared_session_state(monkeypatch):
    def upstream(**kwargs):
        prompt = kwargs["input"][0]["content"][0]["text"]
        return responses_result([{
            "type": "reasoning", "id": f"rs_{prompt}", "summary": [],
            "encrypted_content": f"opaque_{prompt}",
        }])

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(responses=upstream))
    shared_key = b"s" * 32

    def run(value):
        h = handler(key=shared_key)
        h._handle_anthropic_via_responses({
            "max_tokens": 10, "messages": [{"role": "user", "content": value}],
        })
        token = response_body(h)["content"][0]["data"]
        replay = h._decode_reasoning_replay(token)
        assert replay["id"] == f"rs_{value}"
        assert replay["encrypted_content"] == f"opaque_{value}"

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, [str(i) for i in range(40)]))


@pytest.mark.parametrize("status", ["failed", "incomplete"])
def test_non_completed_responses_state_is_not_reported_as_success(monkeypatch, status):
    result = responses_result(status=status)
    if status == "failed":
        result["error"] = {"code": "server_error", "message": "upstream failed"}
    else:
        result["incomplete_details"] = {"reason": "max_output_tokens"}
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(
        responses=Mock(return_value=result),
    ))
    h = handler()
    h._handle_anthropic_via_responses({
        "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}],
    })
    if status == "failed":
        h.send_response.assert_called_once_with(502)
    else:
        h.send_response.assert_called_once_with(200)
        assert response_body(h)["stop_reason"] == "max_tokens"


def test_native_anthropic_reasoning_without_replay_envelope_is_rejected(monkeypatch):
    upstream = Mock(return_value=responses_result())
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(responses=upstream))
    h = handler()
    h._handle_anthropic_via_responses({
        "max_tokens": 10,
        "messages": [{"role": "assistant", "content": [{
            "type": "thinking", "thinking": "private", "signature": "provider-signature",
        }]}],
    })
    h.send_response.assert_called_once_with(400)
    upstream.assert_not_called()


def test_reasoning_output_without_encrypted_content_fails_closed(monkeypatch):
    upstream = Mock(return_value=responses_result([{
        "type": "reasoning", "id": "rs_missing", "summary": [],
    }]))
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(responses=upstream))
    h = handler()
    h._handle_anthropic_via_responses({
        "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}],
    })
    h.send_response.assert_called_once_with(502)


def test_locked_litellm_preserves_reasoning_wire_fields(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    captured = []

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            captured.append((self.path, json.loads(self.rfile.read(length))))
            payload = json.dumps({
                "id": "resp_wire", "object": "response", "created_at": 1,
                "status": "completed", "model": "test-model",
                "output": [{
                    "type": "reasoning", "id": "rs_wire", "summary": [],
                    "encrypted_content": "wire-ciphertext",
                }],
                "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
                "error": None, "incomplete_details": None, "instructions": None,
                "metadata": {}, "temperature": None, "top_p": None,
                "max_output_tokens": 10, "previous_response_id": None,
                "reasoning": None, "store": False,
                "text": {"format": {"type": "text"}}, "truncation": "disabled",
                "usage": {
                    "input_tokens": 1, "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1, "output_tokens_details": {"reasoning_tokens": 1},
                    "total_tokens": 2,
                },
                "user": None,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        h = handler()
        h.litellm_api_base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        h._handle_anthropic_via_responses({
            "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}],
        })
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert captured[0][0] == "/v1/responses"
    wire = captured[0][1]
    assert wire["include"] == ["reasoning.encrypted_content"]
    assert wire["store"] is False
    redacted = response_body(h)["content"][0]
    assert redacted["type"] == "redacted_thinking"
    replay = h._decode_reasoning_replay(redacted["data"])
    assert replay["id"] == "rs_wire"
    assert replay["encrypted_content"] == "wire-ciphertext"
