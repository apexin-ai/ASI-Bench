"""Strict OpenAI sessions preserve empty answers without hidden CC resampling."""
import json
import threading
import uuid
from types import SimpleNamespace

import pytest

from ai4sci_bench.adapters.api_proxy import _LiteLLMProxyHandler, LiteLLMProxy
from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
from ai4sci_bench.adapters.native_anthropic_guard import EMPTY_RECOVERY
from tests.test_claude_proxy_integrity import chunk, events, handler, install_completion
from tests.test_claude_strict_attempts import body


@pytest.fixture(autouse=True)
def strict_sessions(monkeypatch):
    monkeypatch.setenv("ASIBENCH_STRICT_STREAM_ATTEMPTS", "1")
    monkeypatch.setenv("ASIBENCH_FORCE_UPSTREAM_STREAMING", "1")
    monkeypatch.setenv("ASIBENCH_STREAM_RETRY_ATTEMPTS", "1")
    monkeypatch.setattr(_LiteLLMProxyHandler, "_attempts", {})


def recovery(session, blocks=False, trailing_system=False):
    request = body(session)
    content = "task\n" + EMPTY_RECOVERY
    if blocks:
        content = [{"type": "text", "text": "task\n"},
                   {"type": "text", "text": EMPTY_RECOVERY}]
    request["messages"][-1]["content"] = content
    if trailing_system:
        request["messages"].append({"role": "system", "content": "Today's date is 2026-09-18."})
    return request


def proxy_for_state():
    proxy = LiteLLMProxy("fixture")
    proxy._handler_cls = SimpleNamespace(_attempts_lock=_LiteLLMProxyHandler._attempts_lock,
                                         _attempts=_LiteLLMProxyHandler._attempts)
    return proxy


@pytest.mark.parametrize("response", [
    [chunk(finish="stop")],
    [chunk({"content": ""}, "stop")],
    [chunk({"content": " \n\t"}, "stop")],
    [chunk({"reasoning_content": "private reasoning"}, "stop")],
])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("blocks,trailing_system", [(False, False), (True, True)])
def test_complete_empty_answer_is_preserved_but_recovery_is_sticky(
    monkeypatch, response, stream, blocks, trailing_system,
):
    session = str(uuid.uuid4())
    calls = install_completion(monkeypatch, [response])
    first = handler(body(session, stream))
    first.do_POST()
    assert first.statuses == [200]
    if stream:
        assert events(first)[-1]["type"] == "message_stop"
    else:
        assert json.loads(first.wfile.getvalue())["stop_reason"] == "end_turn"
    proxy = proxy_for_state()
    assert proxy.attempt_failure(session) is None

    hidden = handler(recovery(session, blocks, trailing_system))
    hidden.do_POST()
    assert hidden.statuses == [400]
    assert b"Hidden client recovery" in hidden.wfile.getvalue()
    assert len(calls) == 1
    assert proxy.attempt_failure(session) == "client_recovery_after_empty_response"
    again = handler(body(session))
    again.do_POST()
    assert again.statuses == [400]
    assert len(calls) == 1

    adapter = ClaudeCodeCLIAdapter(model="fixture")
    adapter._proxy = proxy
    apparent_success = json.dumps({"type": "result", "session_id": session,
                                   "is_error": False, "result": "replacement"})
    assert "client_recovery_after_empty_response" in adapter._proxy_attempt_error(apparent_success)


def test_strict_off_retains_existing_empty_recovery_behavior(monkeypatch):
    monkeypatch.setenv("ASIBENCH_STRICT_STREAM_ATTEMPTS", "0")
    session = str(uuid.uuid4())
    calls = install_completion(monkeypatch, [[chunk(finish="stop")], [chunk({"content": "recovered"}, "stop")]])
    for request in [body(session), recovery(session, True, True)]:
        h = handler(request)
        h.do_POST()
        assert h.statuses == [200]
    assert len(calls) == 2


def test_ordinary_followup_after_empty_is_allowed_and_resets_empty_state(monkeypatch):
    session = str(uuid.uuid4())
    calls = install_completion(monkeypatch, [
        [chunk(finish="stop")], [chunk({"content": "visible"}, "stop")],
        [chunk({"content": "still visible"}, "stop")],
    ])
    ordinary = body(session)
    ordinary["messages"].append({"role": "user", "content": "A different, explicitly requested question."})
    for request in [body(session), ordinary, recovery(session)]:
        h = handler(request)
        h.do_POST()
        assert h.statuses == [200]
    assert len(calls) == 3
    assert proxy_for_state().attempt_failure(session) is None


def test_tool_only_turn_and_tool_result_marker_are_not_empty_recovery(monkeypatch):
    session = str(uuid.uuid4())
    tool = {"index": 0, "id": "tool_1", "function": {"name": "Read", "arguments": '{"file_path":"x"}'}}
    calls = install_completion(monkeypatch, [
        [chunk({"reasoning_content": "use a tool"}), chunk({"tool_calls": [tool]}, "tool_calls")],
        [chunk({"content": "done"}, "stop")],
    ])
    first = handler(body(session))
    first.do_POST()
    assert first.statuses == [200]
    followup = body(session)
    followup["messages"].append({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "tool_1", "content": EMPTY_RECOVERY},
    ]})
    second = handler(followup)
    second.do_POST()
    assert second.statuses == [200]
    assert len(calls) == 2


@pytest.mark.parametrize("last", [
    {"role": "assistant", "content": EMPTY_RECOVERY},
    {"role": "user", "content": [{"type": "tool_result", "content": EMPTY_RECOVERY, "tool_use_id": "t"}]},
    {"role": "user", "content": "ordinary question"},
])
def test_older_or_non_user_text_marker_does_not_block_followup(monkeypatch, last):
    session = str(uuid.uuid4())
    calls = install_completion(monkeypatch, [[chunk(finish="stop")], [chunk({"content": "ok"}, "stop")]])
    first = handler(body(session))
    first.do_POST()
    followup = recovery(session)
    followup["messages"].append(last)
    h = handler(followup)
    h.do_POST()
    assert h.statuses == [200]
    assert len(calls) == 2


def test_new_session_can_run_after_empty_recovery_failure(monkeypatch):
    session = str(uuid.uuid4())
    calls = install_completion(monkeypatch, [[chunk(finish="stop")], [chunk({"content": "fresh"}, "stop")]])
    first = handler(body(session))
    first.do_POST()
    hidden = handler(recovery(session))
    hidden.do_POST()
    assert hidden.statuses == [400]
    fresh = handler(recovery(str(uuid.uuid4())))
    fresh.do_POST()
    assert fresh.statuses == [200]
    assert len(calls) == 2


def test_queued_recovery_observes_completed_empty_turn(monkeypatch):
    session = str(uuid.uuid4())
    entered, release = threading.Event(), threading.Event()
    def slow_empty():
        entered.set()
        assert release.wait(2)
        yield chunk(finish="stop")
    calls = install_completion(monkeypatch, [slow_empty()])
    first, second = handler(body(session)), handler(recovery(session, True, True))
    a, b = threading.Thread(target=first.do_POST), threading.Thread(target=second.do_POST)
    a.start()
    assert entered.wait(2)
    b.start()
    release.set()
    a.join(2)
    b.join(2)
    assert not a.is_alive() and not b.is_alive()
    assert first.statuses == [200] and second.statuses == [400]
    assert len(calls) == 1
