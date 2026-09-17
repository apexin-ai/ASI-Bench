"""Exercise real locked LiteLLM over HTTP, including synthesized EOF finishes."""

import http.server
import json
import threading
import urllib.error
import urllib.request

import pytest

from ai4sci_bench.adapters.api_proxy import LiteLLMProxy
from ai4sci_bench.adapters import api_proxy


@pytest.mark.parametrize("upstream_end", [
    "valid", "valid_usage", "valid_finish_eof", "valid_marker_text", "eof",
    "done_without_finish", "empty", "error", "malformed", "content_after_finish",
])
@pytest.mark.parametrize("client_stream", [True, False])
def test_real_sdk_cannot_turn_unfinished_wire_into_success(monkeypatch, upstream_end, client_stream):
    pytest.importorskip("litellm")
    monkeypatch.setattr(api_proxy, "_configure_proxy_logging", lambda: None)
    monkeypatch.setenv("ASIBENCH_FORCE_UPSTREAM_STREAMING", "1")
    monkeypatch.setenv("ASIBENCH_STREAM_RETRY_ATTEMPTS", "3")
    requests = []

    class Upstream(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            valid = upstream_end.startswith("valid")
            text = "literal [DONE] text" if upstream_end == "valid_marker_text" else "answer"
            parts = [({"role": "assistant", "reasoning_content": "think"}, None),
                     ({"content": text}, None)]
            if upstream_end == "empty":
                parts = []
            if valid or upstream_end == "content_after_finish":
                parts.append(({}, "stop"))
            if upstream_end == "content_after_finish":
                parts.append(({"content": "unexpected"}, None))
            for delta, finish in parts:
                obj = {"id": "fixture", "object": "chat.completion.chunk", "created": 1,
                       "model": "fixture", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode())
                self.wfile.flush()
            if upstream_end == "valid_usage":
                self.wfile.write(b'data: {"id":"fixture","object":"chat.completion.chunk","created":1,"model":"fixture","choices":[],"usage":{"prompt_tokens":13,"completion_tokens":8,"total_tokens":21}}\n\n')
            if upstream_end == "error":
                self.wfile.write(b'data: {"error":{"message":"fixture error","type":"api_error"}}\n\n')
            if upstream_end == "malformed":
                self.wfile.write(b'data: {bad json}\n\n')
            if upstream_end not in {"eof", "empty", "valid_finish_eof"}:
                self.wfile.write(b"data: [DONE]\n\n")

    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    proxy = LiteLLMProxy("openai/fixture", api_base=f"http://127.0.0.1:{upstream.server_port}/v1",
                         api_key="fixture")
    url = proxy.start()
    try:
        req = urllib.request.Request(url + "/v1/messages", data=json.dumps({
            "model": "fixture", "stream": client_stream, "max_tokens": 20,
            "messages": [{"role": "user", "content": "fixture"}],
        }).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                status, raw = response.status, response.read().decode()
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read().decode()
        assert len(requests) == 1
        assert requests[0]["stream"] is True
        if upstream_end.startswith("valid"):
            assert status == 200
            if client_stream:
                assert '"type": "message_stop"' in raw
            else:
                message = json.loads(raw)
                assert [b["type"] for b in message["content"]] == ["thinking", "text"]
                expected_text = "literal [DONE] text" if upstream_end == "valid_marker_text" else "answer"
                assert message["content"][-1]["text"] == expected_text
                if upstream_end == "valid_usage":
                    assert message["usage"] == {"input_tokens": 13, "output_tokens": 8}
        elif client_stream and upstream_end != "empty":
            assert '"type": "error"' in raw
            assert '"type": "message_stop"' not in raw
        else:
            assert status == 502
            assert json.loads(raw)["type"] == "error"
    finally:
        proxy_server = proxy._server
        proxy.stop()
        proxy_server.server_close()
        upstream.shutdown()
        upstream.server_close()
        thread.join()


def test_locked_sdk_inner_generator_releases_http_connection_on_early_exit():
    import httpx
    from types import SimpleNamespace
    from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
    from ai4sci_bench.adapters.api_proxy import _close_upstream_stream

    class Wire(httpx.SyncByteStream):
        closed = False
        def __iter__(self):
            try:
                yield b'data: {"choices":[]}\n\n'
                yield b'data: [DONE]\n\n'
            finally:
                self.closed = True
        def close(self):
            self.closed = True

    wire = Wire()
    response = httpx.Response(200, stream=wire)
    iterator = BaseModelResponseIterator(response.iter_lines(), sync_stream=True)
    next(iterator)
    assert not wire.closed
    _close_upstream_stream(SimpleNamespace(completion_stream=iterator))
    assert wire.closed, "an incomplete SDK iterator must not rely on later garbage collection"
