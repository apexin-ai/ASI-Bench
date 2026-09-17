"""Strict session mode prevents CC retries from silently replacing an answer."""
import json
import threading
import uuid
from types import SimpleNamespace

import pytest

from tests.test_claude_proxy_integrity import handler, install_completion, chunk
from ai4sci_bench.adapters.api_proxy import _LiteLLMProxyHandler, LiteLLMProxy
from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter


def body(session, stream=True):
    return {"messages": [{"role": "user", "content": "task"}], "stream": stream,
            "metadata": {"user_id": json.dumps({"session_id": session})}}


@pytest.fixture(autouse=True)
def isolated_strict_state(monkeypatch):
    monkeypatch.setenv("ASIBENCH_STRICT_STREAM_ATTEMPTS", "1")
    monkeypatch.setenv("ASIBENCH_FORCE_UPSTREAM_STREAMING", "1")
    monkeypatch.setenv("ASIBENCH_STREAM_RETRY_ATTEMPTS", "1")
    monkeypatch.setattr(_LiteLLMProxyHandler, "_attempts", {})


@pytest.mark.parametrize("broken", [[], [chunk({"content": "partial"})], RuntimeError("upstream failed")])
@pytest.mark.parametrize("retry_stream", [True, False])
def test_new_cc_http_request_cannot_replace_failed_session(monkeypatch, broken, retry_stream):
    session = str(uuid.uuid4())
    calls = install_completion(monkeypatch, [broken, [chunk({"content": "replacement"}, "stop")]])
    first = handler(body(session))
    first.do_POST()
    for _ in range(4):
        retried = body(session, retry_stream)
        retried["messages"].append({"role": "user", "content": "CC internal recovery"})
        second = handler(retried)
        second.do_POST()
        assert second.statuses == [400]
    assert len(calls) == 1
    # A separately declared CLI attempt remains independent.
    fresh = handler(body(str(uuid.uuid4())))
    fresh.do_POST()
    assert fresh.statuses == [200]
    assert len(calls) == 2


def test_successful_tool_followup_is_allowed(monkeypatch):
    session = str(uuid.uuid4())
    calls = install_completion(monkeypatch, [
        [chunk({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "Ping", "arguments": "{}"}}]}, "tool_calls")],
        [chunk({"content": "done"}, "stop")],
    ])
    for _ in range(2):
        h = handler(body(session))
        h.do_POST()
        assert h.statuses == [200]
    assert len(calls) == 2


def test_missing_session_fails_before_model_call(monkeypatch):
    calls = install_completion(monkeypatch, [])
    h = handler({"messages": [], "stream": True})
    h.do_POST()
    assert h.statuses == [400]
    assert not calls


def test_proxy_failure_overrides_apparent_cli_success_without_waiting():
    session = str(uuid.uuid4())
    proxy = LiteLLMProxy("fixture")
    lock = threading.Lock()
    lock.acquire()
    state = {"lock": lock, "failure": "failed_after_write", "active": True}
    proxy._handler_cls = SimpleNamespace(_attempts_lock=threading.Lock(), _attempts={session: state})
    adapter = ClaudeCodeCLIAdapter(model="fixture")
    adapter._proxy = proxy
    raw = json.dumps({"type": "result", "session_id": session, "is_error": False, "result": "partial"})
    assert "failed_after_write" in adapter._proxy_attempt_error(raw)
    lock.release()


def test_queued_request_after_failure_never_calls_upstream(monkeypatch):
    session = str(uuid.uuid4())
    entered, release = threading.Event(), threading.Event()
    def slow_failure():
        entered.set()
        assert release.wait(2)
        yield chunk({"content": "partial"})
    calls = install_completion(monkeypatch, [slow_failure()])
    first, second = handler(body(session)), handler(body(session))
    a = threading.Thread(target=first.do_POST)
    b = threading.Thread(target=second.do_POST)
    a.start()
    assert entered.wait(2)
    b.start()
    release.set()
    a.join(2)
    b.join(2)
    assert not a.is_alive() and not b.is_alive()
    assert second.statuses == [400]
    assert len(calls) == 1
