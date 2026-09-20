"""Issue #8: Responses bridge contracts, without external model calls."""

from __future__ import annotations

import copy
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ai4sci_bench.adapters.api_proxy import _LiteLLMOpenAIProxyHandler
from ai4sci_bench.adapters import api_proxy


@pytest.fixture
def custom_support(monkeypatch):
    monkeypatch.setattr(api_proxy, "_litellm_supports_custom_tools", lambda: True)


def handler():
    h = object.__new__(_LiteLLMOpenAIProxyHandler)
    h.litellm_model = "hosted_vllm/test-model"
    h.litellm_api_base = "https://example.invalid/v1"
    h.litellm_api_key = "test-only"
    h.wfile = io.BytesIO()
    h.send_response = Mock()
    h.send_header = Mock()
    h.end_headers = Mock()
    return h


def events(h):
    return [json.loads(line[6:]) for line in h.wfile.getvalue().decode().splitlines()
            if line.startswith("data: ")]


def response(output=None, status="completed"):
    return {"id": "resp_test", "status": status, "output": output or [],
            "error": None, "incomplete_details": None}


@pytest.mark.parametrize("global_drop", [False, True])
def test_translation_preserves_request_fields_without_global_mutation(monkeypatch, global_drop, custom_support):
    upstream = Mock(return_value=response())
    llm = SimpleNamespace(responses=upstream, drop_params=global_drop)
    monkeypatch.setitem(sys.modules, "litellm", llm)
    body = {
        "input": [{"id": "rs_original", "type": "reasoning", "summary": [],
                   "encrypted_content": "opaque-ciphertext"}],
        "include": ["reasoning.encrypted_content"],
        "additional_tools": [{"type": "custom", "name": "exec"}],
        "max_tool_calls": 3, "store": False, "tool_choice": "auto",
    }
    handler()._handle_responses_translated(json.dumps(body).encode())
    kwargs = upstream.call_args.kwargs
    for key, value in body.items():
        if key != "additional_tools":
            assert kwargs[key] == value
    assert kwargs["tools"] == body["additional_tools"]
    assert "additional_tools" not in kwargs
    assert kwargs["drop_params"] is False
    assert llm.drop_params is global_drop


def test_unknown_request_parameter_is_not_silently_dropped(monkeypatch):
    upstream = Mock(return_value=response())
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(responses=upstream))
    h = handler()
    h._handle_responses_translated(b'{"input":"hi","future_critical_option":true}')
    upstream.assert_not_called()
    h.send_response.assert_called_once_with(400)
    assert "future_critical_option" in h.wfile.getvalue().decode()


def test_additional_tools_preserves_existing_declarations(monkeypatch, custom_support):
    upstream = Mock(return_value=response())
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(responses=upstream))
    ordinary = {"type": "function", "name": "read", "parameters": {"type": "object"}}
    custom = {"type": "custom", "name": "exec", "format": {"type": "text"}}
    body = {"input": "hi", "tools": [ordinary], "additional_tools": [custom]}
    handler()._handle_responses_translated(json.dumps(body).encode())
    assert upstream.call_args.kwargs["tools"] == [ordinary, custom]


@pytest.mark.parametrize("extra", [None, {}, "exec"])
def test_invalid_additional_tools_fails_before_upstream(monkeypatch, extra):
    upstream = Mock(return_value=response())
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(responses=upstream))
    h = handler()
    h._handle_responses_translated(json.dumps({"input": "hi", "additional_tools": extra}).encode())
    upstream.assert_not_called()
    h.send_response.assert_called_once_with(400)


@pytest.mark.parametrize("stream", [False, True])
def test_lossy_custom_tool_response_fails_explicitly(monkeypatch, custom_support, stream):
    wrong = {"id": "fc_wrong", "type": "function_call", "name": "exec",
             "call_id": "call_original", "arguments": '{"content":"print(1)"}'}
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(
        responses=Mock(return_value=response([wrong])),
    ))
    h = handler()
    h._handle_responses_translated(json.dumps({
        "input": "hi", "stream": stream, "tools": [{"type": "custom", "name": "exec"}],
    }).encode())
    h.send_response.assert_called_once_with(502)
    assert "lost the custom tool type" in h.wfile.getvalue().decode()


def test_concurrent_translation_does_not_share_reasoning_or_global_policy(monkeypatch):
    def upstream(**kwargs):
        assert kwargs["drop_params"] is False
        return response(kwargs["input"])

    llm = SimpleNamespace(responses=upstream, drop_params=True)
    monkeypatch.setitem(sys.modules, "litellm", llm)

    def run_session(n):
        item = {"type": "reasoning", "id": f"rs_{n}", "summary": [],
                "encrypted_content": f"opaque_{n}"}
        h = handler()
        h._handle_responses_translated(json.dumps({
            "input": [item], "include": ["reasoning.encrypted_content"], "stream": True,
        }).encode())
        assert events(h)[-1]["response"]["output"] == [item]

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run_session, range(12)))
    assert llm.drop_params is True


@pytest.mark.parametrize("item", [
    {"type": "reasoning", "encrypted_content": "opaque", "summary": []},
    {"type": "future_tool_call", "id": "future_original"},
])
def test_unsupported_or_unreplayable_item_fails_before_stream_headers(item):
    h = handler()
    h._handle_synthetic_responses_streaming(response([item]))
    h.send_response.assert_called_once_with(502)
    assert not events(h)


@pytest.mark.parametrize("item_type,field,event_prefix", [
    ("custom_tool_call", "input", "response.custom_tool_call_input"),
    ("function_call", "arguments", "response.function_call_arguments"),
])
@pytest.mark.parametrize("payload", ["", 'print("hello 世界")\n'])
def test_tool_stream_uses_correct_events_and_fields(item_type, field, event_prefix, payload):
    item = {"id": "tool_original", "call_id": "call_original", "type": item_type,
            "name": "exec", field: payload, "status": "completed"}
    original = response([item])
    snapshot = copy.deepcopy(original)
    h = handler()
    h._handle_synthetic_responses_streaming(original)
    stream = events(h)
    delta = next(e for e in stream if e["type"] == event_prefix + ".delta")
    done = next(e for e in stream if e["type"] == event_prefix + ".done")
    assert delta["delta"] == done[field] == payload
    assert done["item_id"] == "tool_original"
    added = next(e for e in stream if e["type"] == "response.output_item.added")
    assert added["item"][field] == ""
    assert added["item"]["call_id"] == "call_original"
    assert stream[-1]["response"] == original
    assert original == snapshot
    assert [e["sequence_number"] for e in stream] == list(range(len(stream)))


def test_reasoning_items_and_original_ids_survive_streaming():
    output = [
        {"id": "rs_original", "type": "reasoning", "summary": [],
         "encrypted_content": "opaque-ciphertext"},
        {"id": "provider-message-id", "type": "message", "role": "assistant",
         "status": "completed", "content": [
             {"type": "output_text", "text": "Done", "annotations": []}]},
    ]
    original = response(output)
    original["id"] = "provider-response-id"
    h = handler()
    h._handle_synthetic_responses_streaming(original)
    stream = events(h)
    done = [e["item"] for e in stream if e["type"] == "response.output_item.done"]
    assert done == output
    assert stream[-1]["response"] == original
    assert stream[0]["response"]["output"] == []
    assert stream[0]["response"]["status"] == "in_progress"


@pytest.mark.parametrize("status", ["completed", "failed", "incomplete"])
def test_terminal_status_is_not_rewritten_as_success(status):
    original = response(status=status)
    if status == "failed":
        original["error"] = {"code": "server_error", "message": "Upstream failed"}
    if status == "incomplete":
        original["incomplete_details"] = {"reason": "max_output_tokens"}
    h = handler()
    h._handle_synthetic_responses_streaming(original)
    stream = events(h)
    assert stream[-1]["type"] == "response." + status
    assert stream[-1]["response"] == original


@pytest.mark.parametrize("status", ["queued", "in_progress", "cancelled"])
def test_unrepresentable_status_fails_before_stream_headers(status):
    h = handler()
    h._handle_synthetic_responses_streaming(response(status=status))
    h.send_response.assert_called_once_with(502)
    assert not events(h)


def test_real_litellm_custom_tool_round_trip(monkeypatch):
    """Exercise installed LiteLLM conversion; replace only the model completion."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import litellm
    import httpx
    from litellm.utils import ProviderConfigManager

    # Simulate a Chat-only provider. Fail immediately on any accidental HTTP.
    monkeypatch.setattr(ProviderConfigManager, "get_provider_responses_api_config",
                        Mock(return_value=None))
    monkeypatch.setattr(httpx.Client, "send", Mock(side_effect=AssertionError("No network in this test")))

    script = 'print("hello 世界")\n'
    completion = Mock(return_value=litellm.ModelResponse(
        model="test-model",
        choices=[{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "tool_calls": [{
                "id": "call_exec", "type": "function", "function": {
                    "name": "exec", "arguments": json.dumps({"content": script}),
                },
            }],
        }}],
    ))
    monkeypatch.setattr(litellm, "completion", completion)
    body = {
        "input": "Write a file using exec.", "stream": True,
        "additional_tools": [{"type": "custom", "name": "exec",
                              "description": "Execute Python code."}],
    }
    h = handler()
    h._handle_responses_translated(json.dumps(body).encode())
    if not api_proxy._litellm_supports_custom_tools():
        h.send_response.assert_called_once_with(400)
        completion.assert_not_called()
        assert "cannot preserve custom tools" in h.wfile.getvalue().decode()
        return
    stream = events(h)
    assert stream, h.wfile.getvalue().decode()
    item = next(e["item"] for e in stream if e["type"] == "response.output_item.done"
                and e["item"]["type"] == "custom_tool_call")
    assert item["type"] == "custom_tool_call"
    assert item["input"] == script
    assert item["call_id"] == "call_exec"
    assert completion.call_args.kwargs["tools"][0]["function"]["name"] == "exec"

    # Replay the original call followed by its result on the next turn.
    body["input"] = [
        {"role": "user", "content": "Write a file using exec."}, item,
        {"type": "custom_tool_call_output", "call_id": "call_exec", "output": "ok"},
    ]
    h = handler()
    h._handle_responses_translated(json.dumps(body).encode())
    assert events(h), h.wfile.getvalue().decode()
    messages = completion.call_args.kwargs["messages"]
    tool_message = next(m for m in messages if m["role"] == "tool")
    assert tool_message["tool_call_id"] == "call_exec"
    assistant = next(m for m in messages if m["role"] == "assistant")
    assert script in assistant["tool_calls"][0]["function"]["arguments"] or (
        json.loads(assistant["tool_calls"][0]["function"]["arguments"])["content"] == script
    )
