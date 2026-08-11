"""Regression tests for filtering image attachments for text-only endpoints."""

from __future__ import annotations

import http.server
import json
import threading
import urllib.request

from ai4sci_bench.adapters.api_proxy import (
    LiteLLMProxy,
    LiteLLMOpenAIProxy,
    _LiteLLMProxyHandler,
    replace_unsupported_image_inputs,
)
from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter


def test_replace_unsupported_image_inputs_handles_responses_and_chat_shapes():
    payload = {
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Inspect the generated chart."},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAE",
                    },
                ],
            },
            {
                "type": "function_call_output",
                "call_id": "call_read",
                "output": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBORw0KGgoAAAANSUhEUgAAAAE",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.test/chart.png"},
                    },
                ],
            },
        ],
    }

    rewritten, replaced = replace_unsupported_image_inputs(payload)

    assert replaced == 3
    serialized = json.dumps(rewritten)
    assert "iVBOR" not in serialized
    assert "data:image/" not in serialized
    assert "https://example.test/chart.png" not in serialized
    assert serialized.count("text-only") == 3
    assert rewritten["input"][0]["content"][0] == {
        "type": "input_text",
        "text": "Inspect the generated chart.",
    }


def test_replace_unsupported_image_inputs_preserves_tool_schema_fields():
    payload = {
        "tools": [
            {
                "type": "function",
                "name": "download",
                "parameters": {
                    "type": "object",
                    "properties": {"image_url": {"type": "string"}},
                },
            },
        ],
    }

    rewritten, replaced = replace_unsupported_image_inputs(payload)

    assert replaced == 0
    assert rewritten == payload


def test_proxy_adapters_default_to_text_only_without_model_matching():
    adapter = CodexCLIAdapter(
        model="vendor/arbitrary-model",
        api_key="test",
        api_base="https://gateway.example.test/v1",
    )

    adapter._ensure_proxy()
    try:
        assert adapter._proxy.supports_image_input is False
    finally:
        adapter.teardown()


def test_visual_endpoints_can_opt_in_without_model_matching():
    openai_proxy = LiteLLMOpenAIProxy(
        model="vendor/arbitrary-model",
        supports_image_input=True,
    )
    anthropic_proxy = LiteLLMProxy(
        model="vendor/another-model",
        supports_image_input=True,
    )
    try:
        assert openai_proxy.supports_image_input is True
        assert anthropic_proxy.supports_image_input is True
    finally:
        openai_proxy.stop()
        anthropic_proxy.stop()


def test_text_only_proxy_filters_image_before_responses_passthrough():
    received: list[dict] = []

    class CaptureHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            received.append(json.loads(self.rfile.read(length)))
            response = json.dumps({"id": "resp_test", "output": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format, *args):
            pass

    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    upstream_url = f"http://127.0.0.1:{upstream.server_address[1]}/v1"
    proxy = LiteLLMOpenAIProxy(
        model="openai/text-model",
        api_base=upstream_url,
        api_key="test",
    )
    proxy_url = proxy.start()

    try:
        request = urllib.request.Request(
            f"{proxy_url}/v1/responses",
            data=json.dumps({
                "model": "ignored",
                "input": [{
                    "role": "user",
                    "content": [{
                        "type": "input_image",
                        "image_url": "data:image/png;base64,SECRET_IMAGE_BYTES",
                    }],
                }],
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 200

        assert len(received) == 1
        forwarded = json.dumps(received[0])
        assert "SECRET_IMAGE_BYTES" not in forwarded
        assert "text-only" in forwarded
        assert received[0]["model"] == "text-model"
    finally:
        proxy.stop()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)


def test_text_only_proxy_filters_anthropic_messages(monkeypatch):
    captured: list[dict] = []

    def fake_call(kwargs):
        captured.append(kwargs)
        return {
            "id": "msg_test",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {},
        }

    monkeypatch.setattr(
        _LiteLLMProxyHandler,
        "_call_litellm_anthropic",
        staticmethod(fake_call),
    )
    proxy = LiteLLMProxy(model="vendor/arbitrary-model")
    proxy_url = proxy.start()

    try:
        request = urllib.request.Request(
            f"{proxy_url}/v1/messages",
            data=json.dumps({
                "model": "ignored",
                "max_tokens": 100,
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "SECRET_ANTHROPIC_IMAGE",
                        },
                    }],
                }],
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 200

        forwarded = json.dumps(captured[0]["messages"])
        assert "SECRET_ANTHROPIC_IMAGE" not in forwarded
        assert "text-only" in forwarded
    finally:
        proxy.stop()
