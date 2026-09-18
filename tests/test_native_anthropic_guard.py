"""Native Messages protection uses real HTTP, never an OpenAI conversion."""
import contextlib
import gzip
import http.client
import http.server
import json
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid

import pytest

from ai4sci_bench.adapters.native_anthropic_guard import NativeAnthropicGuard


def frame(kind, **values):
    return ("event: " + kind + "\ndata: " + json.dumps({"type": kind, **values}) + "\n\n").encode()


def stream(text="done", thinking=False, tool=False):
    out = [frame("message_start", message={"id": "msg_fixture", "type": "message", "role": "assistant", "content": [], "stop_reason": None, "usage": {"input_tokens": 3}})]
    index = 0
    if thinking:
        out += [frame("content_block_start", index=index, content_block={"type": "thinking", "thinking": ""}),
                frame("content_block_delta", index=index, delta={"type": "thinking_delta", "thinking": "reasoning"}),
                frame("content_block_delta", index=index, delta={"type": "signature_delta", "signature": "NATIVE_SIGNATURE"}),
                frame("content_block_stop", index=index)]
        index += 1
    if tool:
        out += [frame("content_block_start", index=index, content_block={"type": "tool_use", "id": "tool_native", "name": "Write", "input": {}}),
                frame("content_block_delta", index=index, delta={"type": "input_json_delta", "partial_json": '{"path":'}),
                frame("content_block_delta", index=index, delta={"type": "input_json_delta", "partial_json": '"result.txt"}'}),
                frame("content_block_stop", index=index)]
    else:
        out += [frame("content_block_start", index=index, content_block={"type": "text", "text": ""}),
                frame("content_block_delta", index=index, delta={"type": "text_delta", "text": text}),
                frame("content_block_stop", index=index)]
    out += [frame("message_delta", delta={"stop_reason": "tool_use" if tool else "end_turn", "stop_sequence": None}, usage={"output_tokens": 3}), frame("message_stop")]
    return out


def body(session=None, stream_value=True):
    return {"model": "native-fixture", "max_tokens": 300, "stream": stream_value,
            "metadata": {"user_id": json.dumps({"session_id": session or str(uuid.uuid4())})},
            "messages": [{"role": "user", "content": "task"}]}


@contextlib.contextmanager
def running(responses):
    calls = []
    class Upstream(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *_):
            pass
        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            calls.append({"raw": raw, "headers": dict(self.headers), "path": self.path})
            response = responses[min(len(calls)-1, len(responses)-1)]
            status, chunks = response if isinstance(response, tuple) else (200, response)
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream" if status == 200 else "application/json")
            self.send_header("Connection", "close")
            self.send_header("request-id", "fixture-request")
            if status == 429:
                self.send_header("retry-after", "1")
            self.end_headers()
            self.close_connection = True
            try:
                for chunk in chunks:
                    if isinstance(chunk, threading.Event):
                        chunk.wait(3)
                    else:
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream.daemon_threads = True
    worker = threading.Thread(target=upstream.serve_forever, daemon=True)
    worker.start()
    guard = NativeAnthropicGuard(api_base=f"http://127.0.0.1:{upstream.server_port}/v1", api_key="fixture-real-key", timeout_seconds=2)
    guard.start()
    try:
        yield guard, calls
    finally:
        guard.stop()
        upstream.shutdown()
        upstream.server_close()
        worker.join(3)


def request(guard, payload, headers=None, raw=None):
    req = urllib.request.Request(guard.local_url + "/v1/messages?beta=true", data=raw or json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=4) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


@pytest.mark.parametrize("thinking,tool", [(False, False), (True, False), (True, True)])
def test_valid_native_bytes_and_auth_are_preserved(thinking, tool):
    wire = b"".join(stream(thinking=thinking, tool=tool))
    payload = body()
    payload["thinking"] = {"type": "enabled", "budget_tokens": 200}
    payload["messages"].insert(0, {"role": "assistant", "content": [{"type": "thinking", "thinking": "prior", "signature": "HISTORICAL_SIGNATURE"}]})
    raw = json.dumps(payload, indent=3).encode()
    with running([[wire]]) as (guard, calls):
        status, returned, _ = request(guard, payload, headers={"anthropic-version": "2023-06-01", "anthropic-beta": "fixture-beta", "X-Stainless-Retry-Count": "0", "Authorization": "Bearer fake-client"}, raw=raw)
        assert status == 200 and returned == wire
        assert calls[0]["raw"] == raw
        headers = {k.lower(): v for k, v in calls[0]["headers"].items()}
        assert headers["authorization"] == "Bearer fixture-real-key"
        assert headers["x-api-key"] == "fixture-real-key"
        assert headers["anthropic-beta"] == "fixture-beta"
        assert headers["x-stainless-retry-count"] == "0"
        assert calls[0]["path"] == "/v1/messages?beta=true"


def test_gzip_body_is_forwarded_without_reencoding():
    payload = body()
    raw = gzip.compress(json.dumps(payload).encode())
    with running([stream()]) as (guard, calls):
        assert request(guard, payload, {"Content-Encoding": "gzip"}, raw)[0] == 200
        assert calls[0]["raw"] == raw


@pytest.mark.parametrize("status", [400, 404, 429, 500])
@pytest.mark.parametrize("retry_stream", [True, False])
def test_http_failure_closes_session_before_any_retry(status, retry_stream):
    session = str(uuid.uuid4())
    with running([(status, [b'{"error":"fixture"}']), stream()]) as (guard, calls):
        assert request(guard, body(session))[0] == status
        assert request(guard, body(session, retry_stream))[0] == 400
        assert len(calls) == 1
        assert guard.attempt_failure(session) == f"upstream_http_{status}"
        assert request(guard, body())[0] == 200


def test_http_error_preserves_provider_reason_status_and_safe_headers():
    error = json.dumps({"error": {"message": "Only admitted clients: fixture-real-key", "type": "access_error"}}).encode()
    with running([(429, [error])]) as (guard, _):
        status, raw, headers = request(guard, body())
        assert status == 429
        assert "retry-after" in {k.lower(): v for k, v in headers.items()}
        assert "fixture-request" in headers.values()
        assert b"Only admitted clients" in raw
        assert b"fixture-real-key" not in raw
        assert json.loads(raw)["upstream_error"]["body"]["error"]["type"] == "access_error"


def test_http_error_body_is_bounded(monkeypatch):
    import ai4sci_bench.adapters.native_anthropic_guard as module
    monkeypatch.setattr(module, "MAX_ERROR_BODY", 128)
    with running([(503, [b"x" * 4096])]) as (guard, _):
        status, raw, _ = request(guard, body())
        assert status == 503
        assert json.loads(raw)["upstream_error"] == {"body": "x" * 128, "truncated": True}


@pytest.mark.parametrize("mutate", [
    lambda chunks: chunks[:-1],
    lambda chunks: chunks[:1] + [frame("message_stop")],
    lambda chunks: chunks + [frame("message_stop")],
    lambda chunks: chunks[:1] + [frame("content_block_start", index=0, content_block={"type": "unknown"})],
    lambda chunks: chunks[:2] + [frame("content_block_delta", index=0, delta={"type": "thinking_delta", "thinking": "bad"})],
    lambda chunks: chunks[:1] + [b"event: content_block_start\ndata: {bad json}\n\n"],
    lambda chunks: chunks[:1] + [frame("error", error={"type": "overloaded_error", "message": "fixture"})],
])
def test_invalid_stream_never_gets_success_terminal_and_is_sticky(mutate):
    session = str(uuid.uuid4())
    with running([mutate(stream())]) as (guard, calls):
        status, raw, _ = request(guard, body(session))
        assert b'"type": "message_stop"' not in raw
        assert status == 502 or b'"type": "error"' in raw
        assert guard.attempt_failure(session)
        assert request(guard, body(session))[0] == 400
        assert len(calls) == 1


def test_crlf_multiline_data_ping_preserved():
    chunks = stream(thinking=True)
    wire = b": keepalive\r\n\r\n" + frame("ping") + b"".join(chunks)
    wire = wire.replace(b'"type": "message_start", "message"', b'"type": "message_start",\ndata: "message"')
    wire = wire.replace(b"\n", b"\r\n").replace(b"\r\r\n", b"\r\n")
    with running([[wire[:37], wire[37:]]]) as (guard, _):
        assert request(guard, body())[1] == wire


def test_missing_session_and_nonstream_do_not_reach_model():
    with running([stream()]) as (guard, calls):
        assert request(guard, {"stream": True, "messages": []})[0] == 400
        session = str(uuid.uuid4())
        assert request(guard, body(session, False))[0] == 400
        assert request(guard, body(session))[0] == 400
        assert not calls


def test_tool_followup_allowed_but_empty_recovery_is_blocked():
    session = str(uuid.uuid4())
    with running([stream(tool=True), stream(""), stream()]) as (guard, calls):
        assert request(guard, body(session))[0] == 200
        assert request(guard, body(session))[0] == 200
        assert guard.attempt_failure(session) is None
        recovery = body(session)
        recovery["messages"].append({"role": "user", "content": "[Your previous response had no visible output. Please continue and produce a user-visible response.]"})
        assert request(guard, recovery)[0] == 400
        assert len(calls) == 2
        assert guard.attempt_failure(session) == "client_recovery_after_empty_response"


def test_first_event_is_not_buffered_until_64k_or_eof():
    release = threading.Event()
    chunks = stream()
    with running([[chunks[0], release, *chunks[1:]]]) as (guard, _):
        req = urllib.request.Request(guard.local_url + "/v1/messages", data=json.dumps(body()).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1) as response:
            assert b"message_start" in response.read1(65536)
            release.set()
            assert b"message_stop" in response.read()


def test_disconnect_is_sticky_and_stop_closes_waiting_upstream():
    release = threading.Event()
    session = str(uuid.uuid4())
    with running([[stream()[0], release]]) as (guard, calls):
        from urllib.parse import urlsplit
        target = urlsplit(guard.local_url)
        conn = http.client.HTTPConnection(target.hostname, target.port, timeout=1)
        conn.request("POST", "/v1/messages", json.dumps(body(session)), {"Content-Type": "application/json"})
        response = conn.getresponse()
        assert response.read1(65536)
        response.close()
        conn.close()
        until = time.monotonic() + 2
        while not guard.attempt_failure(session) and time.monotonic() < until:
            time.sleep(.02)
        assert guard.attempt_failure(session)
        assert request(guard, body(session))[0] == 400
        assert len(calls) == 1
        release.set()


def test_stop_interrupts_active_response_without_waiting_idle_timeout():
    release = threading.Event()
    session = str(uuid.uuid4())
    with running([[stream()[0], release]]) as (guard, _):
        result = []
        client = threading.Thread(target=lambda: result.append(request(guard, body(session))))
        client.start()
        until = time.monotonic() + 1
        while not guard.attempt_failure(session) and time.monotonic() < until:
            time.sleep(.01)
        started = time.monotonic()
        guard.stop()
        client.join(1)
        assert not client.is_alive()
        assert time.monotonic() - started < 1.5
        assert guard.attempt_failure(session) == "proxy_stopped"
        release.set()


def test_active_session_serializes_followup_after_failure():
    release = threading.Event()
    session = str(uuid.uuid4())
    with running([[stream()[0], release]]) as (guard, calls):
        results = []
        first = threading.Thread(target=lambda: results.append(request(guard, body(session))))
        second = threading.Thread(target=lambda: results.append(request(guard, body(session))))
        first.start()
        until = time.monotonic() + 1
        while not calls and time.monotonic() < until:
            time.sleep(.01)
        second.start()
        release.set()
        first.join(3)
        second.join(3)
        assert len(calls) == 1
        assert sorted(r[0] for r in results) == [200, 400]


def test_adapter_opt_in_routing_and_failure_lookup(monkeypatch):
    from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
    monkeypatch.setenv("ASIBENCH_NATIVE_STREAM_GUARD", "1")
    adapter = ClaudeCodeCLIAdapter(model="fixture", api_base="https://api.apexin.ai/v1", api_protocol="anthropic", api_key="fixture-real-key")
    try:
        env = adapter._build_api_env()
        assert env["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1:")
        assert env["ANTHROPIC_API_KEY"] == "sk-proxy-placeholder"
        assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-proxy-placeholder"
        assert isinstance(adapter._proxy, NativeAnthropicGuard)
        assert adapter._prepare_prompt("task") == "task"
    finally:
        adapter.teardown()
    openai = ClaudeCodeCLIAdapter(model="fixture", api_base="https://api.apexin.ai/v1", api_protocol="openai")
    tokenrouter = ClaudeCodeCLIAdapter(model="claude-fixture", api_base="https://api.tokenrouter.com/v1", api_protocol="anthropic")
    assert not openai._uses_native_stream_guard
    assert not tokenrouter._uses_native_stream_guard
