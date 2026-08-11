"""Tests for third-party API integration via api_key + api_base.

Focus: OpenRouter (OpenAI-compatible protocol) integration with
claude_code_cli, openhands, hermes, and codewhale adapters.

Tests are organized in three tiers:
  1. Unit tests — no network, pure logic (init, validation, proxy routing)
  2. Proxy tests — start local proxy, verify HTTP contract
  3. E2E integration — actual OpenRouter API calls via litellm
"""

import json
import os
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ── Constants ──────────────────────────────────────────────────

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "sk-or-test-key")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
# Use a cheap, fast model for testing
OPENROUTER_MODEL = "anthropic/claude-sonnet-4-6"
OPENROUTER_MODEL_CHEAP = "openai/gpt-4.1-mini"


# ════════════════════════════════════════════════════════════════
# §1  UNIT TESTS — Initialization, Validation, Routing Logic
# ════════════════════════════════════════════════════════════════


class TestResolveModel:
    """Test resolve_litellm_model helper."""

    def test_no_protocol_passthrough(self):
        from ai4sci_bench.adapters.api_proxy import resolve_litellm_model
        assert resolve_litellm_model("claude-opus-4-6", None) == "claude-opus-4-6"

    def test_openai_protocol_prefix(self):
        from ai4sci_bench.adapters.api_proxy import resolve_litellm_model
        assert resolve_litellm_model("gpt-4", "openai") == "openai/gpt-4"

    def test_anthropic_protocol_prefix(self):
        from ai4sci_bench.adapters.api_proxy import resolve_litellm_model
        assert resolve_litellm_model("claude-opus-4-6", "anthropic") == "anthropic/claude-opus-4-6"

    def test_invalid_protocol_raises(self):
        from ai4sci_bench.adapters.api_proxy import resolve_litellm_model
        with pytest.raises(ValueError, match="Invalid api_protocol"):
            resolve_litellm_model("model", "grpc")

    def test_openrouter_model_with_openai_protocol(self):
        """OpenRouter model names already have provider prefix; adding openai/ is correct
        because litellm uses it as a routing hint to the OpenAI-compatible endpoint."""
        from ai4sci_bench.adapters.api_proxy import resolve_litellm_model
        result = resolve_litellm_model("anthropic/claude-sonnet-4-6", "openai")
        assert result == "openai/anthropic/claude-sonnet-4-6"


class TestChatCompletionsRemap:
    """Anthropic-in proxy must force chat/completions, not the Responses API.

    litellm routes the ``openai/`` provider to ``POST /v1/responses``; OpenAI-
    compatible gateways (DeepSeek, TokenRouter, OpenRouter, vLLM) implement only
    ``/v1/chat/completions`` and 500 on Responses → we remap to ``hosted_vllm/``.
    """

    def test_openai_prefix_remapped(self):
        from ai4sci_bench.adapters.api_proxy import to_chat_completions_model
        assert to_chat_completions_model("openai/deepseek/deepseek-v4-pro") == \
            "hosted_vllm/deepseek/deepseek-v4-pro"

    def test_openai_prefix_simple(self):
        from ai4sci_bench.adapters.api_proxy import to_chat_completions_model
        assert to_chat_completions_model("openai/gpt-4.1-mini") == "hosted_vllm/gpt-4.1-mini"

    def test_non_openai_passthrough(self):
        from ai4sci_bench.adapters.api_proxy import to_chat_completions_model
        assert to_chat_completions_model("anthropic/claude-opus-4-6") == "anthropic/claude-opus-4-6"
        assert to_chat_completions_model("hosted_vllm/x") == "hosted_vllm/x"
        assert to_chat_completions_model("deepseek-v4-pro") == "deepseek-v4-pro"

    def test_proxy_build_kwargs_uses_chat_completions(self):
        """The Anthropic-in proxy handler must build kwargs with the remapped model."""
        from ai4sci_bench.adapters.api_proxy import _LiteLLMProxyHandler
        handler = _LiteLLMProxyHandler.__new__(_LiteLLMProxyHandler)
        handler.litellm_model = "openai/deepseek/deepseek-v4-pro"
        handler.litellm_api_base = "https://api.tokenrouter.com/v1"
        handler.litellm_api_key = "sk-test"
        kwargs = handler._build_litellm_kwargs({"messages": [], "max_tokens": 8})
        assert kwargs["model"] == "hosted_vllm/deepseek/deepseek-v4-pro"


class TestAnthropicTokenRouterRewrites:
    """TokenRouter native Anthropic routes need small model-specific rewrites."""

    def test_claude_thinking_enabled_becomes_adaptive_with_effort(self):
        from ai4sci_bench.adapters.api_proxy import rewrite_anthropic_payload_for_tokenrouter
        body = {
            "model": "anthropic/claude-opus-4.8",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 4096},
        }
        rewritten = rewrite_anthropic_payload_for_tokenrouter(
            body,
            model="anthropic/claude-opus-4.8",
            effort="xhigh",
        )
        assert rewritten["thinking"] == {"type": "adaptive"}
        assert rewritten["output_config"]["effort"] == "high"

    def test_gemini_tool_schema_markers_are_removed_recursively(self):
        from ai4sci_bench.adapters.api_proxy import rewrite_anthropic_payload_for_tokenrouter
        body = {
            "model": "google/gemini-3.1-pro-preview",
            "tools": [{
                "name": "Write",
                "input_schema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "path": {
                            "$schema": "nested",
                            "type": "string",
                        },
                        "count": {
                            "type": "integer",
                            "exclusiveMinimum": 0,
                        },
                        "pattern": {
                            "type": "string",
                            "pattern": "^[a-z]+$",
                        },
                    },
                    "required": ["path", "pattern"],
                },
            }],
        }
        rewritten = rewrite_anthropic_payload_for_tokenrouter(
            body,
            model="google/gemini-3.1-pro-preview",
            effort="medium",
        )
        schema = rewritten["tools"][0]["input_schema"]
        assert "$schema" not in schema
        assert "$schema" not in schema["properties"]["path"]
        assert "exclusiveMinimum" not in schema["properties"]["count"]
        assert "pattern" in schema["properties"]
        assert "pattern" not in schema["properties"]["pattern"]
        assert "pattern" in schema["required"]
        assert schema["properties"]["path"]["type"] == "string"

    def test_gemini_history_tool_use_is_textualized(self):
        from ai4sci_bench.adapters.api_proxy import rewrite_anthropic_payload_for_tokenrouter
        body = {
            "model": "google/gemini-3.1-pro-preview",
            "messages": [
                {"role": "user", "content": "list files"},
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    }],
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "ok",
                    }],
                },
            ],
        }
        rewritten = rewrite_anthropic_payload_for_tokenrouter(
            body,
            model="google/gemini-3.1-pro-preview",
            effort="medium",
        )
        assistant_text = rewritten["messages"][1]["content"][0]
        user_content = rewritten["messages"][2]["content"]
        assert assistant_text["type"] == "text"
        assert "prior assistant tool request" in assistant_text["text"]
        assert "command" not in assistant_text["text"]
        assert user_content[0] == {
            "type": "text",
            "text": "Runtime observation for the prior tool execution:\nok",
        }
        assert "use the provided tool-calling interface" in user_content[1]["text"]

    def test_gemini_history_tool_use_textualization_ignores_signature_cache(self):
        from ai4sci_bench.adapters.api_proxy import rewrite_anthropic_payload_for_tokenrouter
        body = {
            "model": "google/gemini-3.1-pro-preview",
            "messages": [{
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "Bash",
                    "input": {"command": "pwd"},
                }],
            }],
        }
        rewritten = rewrite_anthropic_payload_for_tokenrouter(
            body,
            model="google/gemini-3.1-pro-preview",
            effort="medium",
            gemini_thought_signature_cache={"toolu_abc": "real-sig"},
        )
        block = rewritten["messages"][0]["content"][0]
        assert block["type"] == "text"
        assert "real-sig" not in block["text"]
        assert "pwd" not in block["text"]

    def test_gemini_terminal_tool_result_is_textualized_without_continue_text(self):
        from ai4sci_bench.adapters.api_proxy import rewrite_anthropic_payload_for_tokenrouter
        body = {
            "model": "google/gemini-3.1-pro-preview",
            "messages": [
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Read",
                        "input": {"file_path": "x"},
                    }],
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "ok",
                    }],
                },
            ],
        }
        rewritten = rewrite_anthropic_payload_for_tokenrouter(
            body,
            model="google/gemini-3.1-pro-preview",
            effort="medium",
        )
        assert rewritten["messages"][-1]["content"] == [{
            "type": "text",
            "text": "Runtime observation for the prior tool execution:\nok",
        }, {
            "type": "text",
            "text": (
                "Continue from this observation. If you need another command, file read, "
                "or file write, use the provided tool-calling interface; do not describe "
                "a tool call in plain text."
            ),
        }]

    def test_gemini_terminal_tool_result_preserves_existing_text_blocks(self):
        from ai4sci_bench.adapters.api_proxy import rewrite_anthropic_payload_for_tokenrouter
        body = {
            "model": "google/gemini-3.1-pro-preview",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
                    {"type": "text", "text": "Please continue."},
                ],
            }],
        }
        rewritten = rewrite_anthropic_payload_for_tokenrouter(
            body,
            model="google/gemini-3.1-pro-preview",
            effort="medium",
        )
        assert rewritten["messages"][-1]["content"] == [
            {"type": "text", "text": "Runtime observation for the prior tool execution:\nok"},
            {"type": "text", "text": "Please continue."},
            {
                "type": "text",
                "text": (
                    "Continue from this observation. If you need another command, file read, "
                    "or file write, use the provided tool-calling interface; do not describe "
                    "a tool call in plain text."
                ),
            },
        ]

    def test_extract_gemini_tool_signature_from_response_json(self):
        from ai4sci_bench.adapters.api_proxy import _extract_gemini_tool_thought_signatures
        data = json.dumps({
            "content": [{
                "type": "tool_use",
                "id": "toolu_json",
                "name": "Bash",
                "thought_signature": "sig-json",
            }],
        }).encode()
        assert _extract_gemini_tool_thought_signatures(data) == {"toolu_json": "sig-json"}

    def test_openai_reasoning_effort_maps_claude_effort(self):
        from ai4sci_bench.adapters.api_proxy import _openai_reasoning_effort_from_claude

        assert _openai_reasoning_effort_from_claude("low") == "low"
        assert _openai_reasoning_effort_from_claude("medium") == "medium"
        assert _openai_reasoning_effort_from_claude("high") == "high"
        assert _openai_reasoning_effort_from_claude("xhigh") == "high"
        assert _openai_reasoning_effort_from_claude("max") == "high"

    def test_extract_gemini_tool_signature_from_sse(self):
        from ai4sci_bench.adapters.api_proxy import _extract_gemini_tool_thought_signatures
        data = (
            'event: content_block_start\n'
            'data: {"type":"content_block_start","index":0,'
            '"content_block":{"type":"tool_use","id":"toolu_sse","name":"Bash"}}\n\n'
            'event: content_block_delta\n'
            'data: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"signature_delta","signature":"sig-sse"}}\n\n'
        ).encode()
        assert _extract_gemini_tool_thought_signatures(data) == {"toolu_sse": "sig-sse"}

    def test_rewrite_gemini_response_tool_use_ids_json(self):
        from ai4sci_bench.adapters.api_proxy import _rewrite_gemini_response_tool_use_ids

        ids = iter(["toolu_gemini_000001", "toolu_gemini_000002"])
        data = json.dumps({
            "content": [
                {"type": "tool_use", "id": "call_gemini_0", "name": "Bash"},
                {"type": "tool_use", "id": "call_gemini_0", "name": "Read"},
            ],
        }).encode()

        rewritten = json.loads(_rewrite_gemini_response_tool_use_ids(data, lambda: next(ids)))
        assert rewritten["content"][0]["id"] == "toolu_gemini_000001"
        assert rewritten["content"][1]["id"] == "toolu_gemini_000002"

    def test_rewrite_gemini_response_tool_use_ids_sse(self):
        from ai4sci_bench.adapters.api_proxy import _rewrite_gemini_response_tool_use_ids

        data = (
            'event: content_block_start\n'
            'data: {"type":"content_block_start","index":0,'
            '"content_block":{"type":"tool_use","id":"call_gemini_0","name":"Bash"}}\n\n'
        ).encode()

        rewritten = _rewrite_gemini_response_tool_use_ids(data, lambda: "toolu_gemini_000001").decode()
        assert '"id": "toolu_gemini_000001"' in rewritten
        assert "call_gemini_0" not in rewritten


class TestProxyThinkingPassthrough:
    """Reasoning models return a ``thinking`` block; the synthetic SSE stream
    must replay it as thinking_delta, not silently drop it (regression: the
    block fell into the text branch and ``block.get('text')`` was empty)."""

    @staticmethod
    def _drive(resp):
        from ai4sci_bench.adapters.api_proxy import _LiteLLMProxyHandler
        h = _LiteLLMProxyHandler.__new__(_LiteLLMProxyHandler)
        h.litellm_model = "hosted_vllm/x"
        events = []
        h._write_sse = lambda ev, data: events.append((ev, data))
        h.send_response = lambda *a, **k: None
        h.send_header = lambda *a, **k: None
        h.end_headers = lambda *a, **k: None
        h._handle_synthetic_streaming(resp)
        return events

    def test_thinking_block_streamed_as_thinking_delta(self):
        events = self._drive({
            "id": "m", "model": "x", "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 2},
            "content": [
                {"type": "thinking", "thinking": "because 17*23=391"},
                {"type": "text", "text": "391"},
            ],
        })
        starts = [d["content_block"]["type"] for ev, d in events if ev == "content_block_start"]
        thinking = "".join(d["delta"]["thinking"] for ev, d in events
                           if ev == "content_block_delta" and d["delta"]["type"] == "thinking_delta")
        text = "".join(d["delta"]["text"] for ev, d in events
                       if ev == "content_block_delta" and d["delta"]["type"] == "text_delta")
        assert "thinking" in starts, "no thinking content_block emitted"
        assert thinking == "because 17*23=391"
        assert text == "391"

    def test_signature_forwarded_when_present(self):
        events = self._drive({
            "id": "m", "model": "x", "stop_reason": "end_turn", "usage": {},
            "content": [{"type": "thinking", "thinking": "x", "signature": "sig123"}],
        })
        sigs = [d["delta"]["signature"] for ev, d in events
                if ev == "content_block_delta" and d["delta"].get("type") == "signature_delta"]
        assert sigs == ["sig123"]


class TestClaudeCodeCLIInit:
    """Unit tests for ClaudeCodeCLIAdapter initialization with third-party API."""

    def test_api_base_requires_protocol(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        with pytest.raises(ValueError, match="api_protocol is required"):
            ClaudeCodeCLIAdapter(
                api_key="sk-test",
                api_base="https://openrouter.ai/api/v1",
            )

    def test_valid_openai_protocol(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        adapter = ClaudeCodeCLIAdapter(
            api_key=OPENROUTER_API_KEY,
            api_base=OPENROUTER_API_BASE,
            api_protocol="openai",
        )
        assert adapter._uses_proxy is True
        assert adapter.api_protocol == "openai"

    def test_valid_anthropic_protocol_no_proxy(self):
        """Anthropic-native endpoints are hit directly — no litellm proxy."""
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        adapter = ClaudeCodeCLIAdapter(
            api_key="sk-ant-test",
            api_base="https://api.anthropic.com",
            api_protocol="anthropic",
        )
        assert adapter._uses_proxy is False

    def test_invalid_protocol_raises(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        with pytest.raises(ValueError, match="Invalid api_protocol"):
            ClaudeCodeCLIAdapter(
                api_key="sk-test",
                api_base="https://example.com",
                api_protocol="grpc",
            )

    def test_build_api_env_direct_anthropic_strips_v1(self):
        """Anthropic protocol: ANTHROPIC_BASE_URL is set directly (trailing
        /v1 stripped) and the real api_key is forwarded — no proxy."""
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        adapter = ClaudeCodeCLIAdapter(
            model="z-ai/glm-5.2",
            api_key="sk-real-key",
            api_base="https://api.tokenrouter.com/v1",
            api_protocol="anthropic",
        )
        env = adapter._build_api_env()
        assert env["ANTHROPIC_BASE_URL"] == "https://api.tokenrouter.com"
        assert env["ANTHROPIC_API_KEY"] == "sk-real-key"
        assert adapter._proxy is None
        assert adapter._uses_anthropic_rewrite_proxy is False

    def test_tokenrouter_claude_uses_anthropic_rewrite_proxy(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        adapter = ClaudeCodeCLIAdapter(
            model="anthropic/claude-opus-4.8",
            api_key="sk-real-key",
            api_base="https://api.tokenrouter.com/v1",
            api_protocol="anthropic",
            effort="medium",
        )
        assert adapter._uses_proxy is False
        assert adapter._uses_anthropic_rewrite_proxy is True
        with patch.object(
            adapter,
            "_ensure_anthropic_rewrite_proxy",
            return_value="http://127.0.0.1:9999",
        ):
            env = adapter._build_api_env()
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9999"
        assert env["ANTHROPIC_API_KEY"] == "sk-proxy-placeholder"

    def test_tokenrouter_claude_46_uses_anthropic_rewrite_proxy(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        adapter = ClaudeCodeCLIAdapter(
            model="anthropic/claude-opus-4.6",
            api_key="sk-real-key",
            api_base="https://api.tokenrouter.com/v1",
            api_protocol="anthropic",
            effort="medium",
        )
        assert adapter._uses_proxy is False
        assert adapter._uses_anthropic_rewrite_proxy is True

    def test_tokenrouter_any_claude_uses_anthropic_rewrite_proxy(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        for model in (
            "claude-3-5-sonnet",
            "anthropic/claude-opus-4.9",
            "anthropic/claude-sonnet-5.0",
            "claude-opus-5-1",
        ):
            adapter = ClaudeCodeCLIAdapter(
                model=model,
                api_key="sk-real-key",
                api_base="https://api.tokenrouter.com/v1",
                api_protocol="anthropic",
                effort="medium",
            )
            assert adapter._uses_proxy is False
            assert adapter._uses_anthropic_rewrite_proxy is True

    def test_tokenrouter_gemini_uses_openai_chat_proxy(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        adapter = ClaudeCodeCLIAdapter(
            model="google/gemini-3.1-pro-preview",
            api_key="sk-real-key",
            api_base="https://api.tokenrouter.com/v1",
            api_protocol="anthropic",
        )
        assert adapter._uses_proxy is False
        assert adapter._uses_anthropic_rewrite_proxy is False
        assert adapter._uses_tokenrouter_openai_chat_proxy is True

    def test_normalize_anthropic_base(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        norm = ClaudeCodeCLIAdapter._normalize_anthropic_base
        assert norm("https://api.tokenrouter.com/v1") == "https://api.tokenrouter.com"
        assert norm("https://api.tokenrouter.com/v1/") == "https://api.tokenrouter.com"
        assert norm("https://api.tokenrouter.com") == "https://api.tokenrouter.com"
        assert norm("https://host/api/v1") == "https://host/api"

    def test_api_key_only_no_proxy(self):
        """api_key without api_base should NOT use proxy."""
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        adapter = ClaudeCodeCLIAdapter(api_key="sk-ant-test")
        assert adapter._uses_proxy is False

    def test_no_credentials_no_proxy(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        adapter = ClaudeCodeCLIAdapter()
        assert adapter._uses_proxy is False

    def test_effort_validation(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        with pytest.raises(ValueError, match="Invalid effort"):
            ClaudeCodeCLIAdapter(effort="invalid")

    def test_build_api_env_proxy_mode(self):
        """When using proxy, env should have ANTHROPIC_BASE_URL and placeholder key."""
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        adapter = ClaudeCodeCLIAdapter(
            api_key=OPENROUTER_API_KEY,
            api_base=OPENROUTER_API_BASE,
            api_protocol="openai",
        )
        # Mock the proxy to avoid actually starting it
        mock_proxy = MagicMock()
        mock_proxy.local_url = "http://127.0.0.1:9999"
        adapter._proxy = mock_proxy

        env = adapter._build_api_env()
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9999"
        assert env["ANTHROPIC_API_KEY"] == "sk-proxy-placeholder"

    def test_build_api_env_direct_key(self):
        """api_key only should set ANTHROPIC_API_KEY directly."""
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        adapter = ClaudeCodeCLIAdapter(api_key="sk-ant-real-key")
        env = adapter._build_api_env()
        assert env["ANTHROPIC_API_KEY"] == "sk-ant-real-key"
        assert "ANTHROPIC_BASE_URL" not in env

    def test_api_key_and_base_can_come_from_env_names(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        with patch.dict(os.environ, {
            "TOKENROUTER_TEST_KEY": "sk-env-key",
            "TOKENROUTER_TEST_BASE": "https://api.tokenrouter.com/v1",
        }):
            adapter = ClaudeCodeCLIAdapter(
                model="google/gemini-3.1-pro-preview",
                api_key_env="TOKENROUTER_TEST_KEY",
                api_base_env="TOKENROUTER_TEST_BASE",
                api_protocol="anthropic",
            )
        assert adapter.api_key == "sk-env-key"
        assert adapter.api_base == "https://api.tokenrouter.com/v1"
        assert adapter.api_key_env == "TOKENROUTER_TEST_KEY"
        assert adapter.api_base_env == "TOKENROUTER_TEST_BASE"
        assert adapter._uses_tokenrouter_openai_chat_proxy is True
        assert adapter._uses_anthropic_rewrite_proxy is False

    def test_missing_api_key_env_raises(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="TOKENROUTER_MISSING_KEY"):
                ClaudeCodeCLIAdapter(api_key_env="TOKENROUTER_MISSING_KEY")

    def test_build_api_env_no_credentials(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        adapter = ClaudeCodeCLIAdapter()
        env = adapter._build_api_env()
        assert env == {}


class TestOpenHandsInit:
    """Unit tests for OpenHandsAdapter initialization with third-party API."""

    def test_api_base_requires_protocol(self):
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
        with pytest.raises(ValueError, match="api_protocol is required"):
            OpenHandsAdapter(
                model="gpt-4",
                api_key="sk-test",
                api_base="https://openrouter.ai/api/v1",
            )

    def test_openai_protocol_resolves_model(self):
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
        adapter = OpenHandsAdapter(
            model="anthropic/claude-sonnet-4-6",
            api_key=OPENROUTER_API_KEY,
            api_base=OPENROUTER_API_BASE,
            api_protocol="openai",
        )
        # model should be prefixed with openai/
        assert adapter.model == "openai/anthropic/claude-sonnet-4-6"
        assert adapter.api_base == OPENROUTER_API_BASE

    def test_no_protocol_passthrough(self):
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
        adapter = OpenHandsAdapter(model="openai/gpt-4o")
        assert adapter.model == "openai/gpt-4o"

    def test_api_key_fallback_env(self):
        """When api_key is not provided, falls back to OPENAI_API_KEY env."""
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            adapter = OpenHandsAdapter(model="openai/gpt-4o")
            assert adapter.api_key == "env-key"


class TestOpenHandsToolIsolation:
    """Unit tests for OpenHands tool isolation (restricted / search modes)."""

    def test_default_tool_mode_is_restricted(self):
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
        from ai4sci_bench.core.types import ToolMode
        adapter = OpenHandsAdapter()
        assert adapter.tool_mode == ToolMode.RESTRICTED

    def test_allow_external_tools_sets_search(self):
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
        from ai4sci_bench.core.types import ToolMode
        adapter = OpenHandsAdapter(allow_external_tools=True)
        assert adapter.tool_mode == ToolMode.SEARCH

    def test_explicit_tool_mode_overrides_allow_external_tools(self):
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
        from ai4sci_bench.core.types import ToolMode
        adapter = OpenHandsAdapter(allow_external_tools=True, tool_mode="restricted")
        assert adapter.tool_mode == ToolMode.RESTRICTED

    def test_restricted_mode_helper_config_disables_browser(self):
        """In restricted mode, enable_browser must be False in the helper config."""
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
        adapter = OpenHandsAdapter(tool_mode="restricted")
        # Simulate what solve() builds for the helper config
        from ai4sci_bench.core.types import ToolMode
        enable_browser = adapter.tool_mode == ToolMode.SEARCH
        assert enable_browser is False

    def test_search_mode_helper_config_enables_browser(self):
        """In search mode, enable_browser must be True in the helper config."""
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
        adapter = OpenHandsAdapter(tool_mode="search")
        from ai4sci_bench.core.types import ToolMode
        enable_browser = adapter.tool_mode == ToolMode.SEARCH
        assert enable_browser is True

    def test_helper_script_contains_enable_browser_guard(self):
        """The helper script must conditionally register browser_use based on enable_browser."""
        from ai4sci_bench.adapters.openhands_agent import _HELPER_SCRIPT
        assert "enable_browser" in _HELPER_SCRIPT
        assert "browser_use" in _HELPER_SCRIPT

    def test_helper_script_restricted_tools_are_exactly_three(self):
        """Without enable_browser, the helper script registers exactly 3 tools."""
        from ai4sci_bench.adapters.openhands_agent import _HELPER_SCRIPT
        # The base tools list always has these three
        assert 'Tool(name="terminal")' in _HELPER_SCRIPT
        assert 'Tool(name="file_editor")' in _HELPER_SCRIPT
        assert 'Tool(name="task_tracker")' in _HELPER_SCRIPT

    def test_build_agent_passes_tool_mode_to_openhands(self):
        """cli._build_agent must pass tool_mode to OpenHandsAdapter (not pop it)."""
        from ai4sci_bench.cli import _build_agent
        from ai4sci_bench.core.types import ToolMode
        adapter = _build_agent(None, "openhands", {"api_key": "test"}, tool_mode="restricted")
        assert adapter.tool_mode == ToolMode.RESTRICTED
        adapter = _build_agent(None, "openhands", {"api_key": "test"}, tool_mode="search")
        assert adapter.tool_mode == ToolMode.SEARCH


class TestHermesInit:
    """Unit tests for HermesAgentAdapter initialization with third-party API."""

    def test_api_base_requires_protocol(self):
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        with pytest.raises(ValueError, match="api_protocol is required"):
            HermesAgentAdapter(
                api_key="sk-test",
                api_base="https://openrouter.ai/api/v1",
            )

    def test_openai_protocol_no_proxy(self):
        """OpenAI-compatible endpoint should NOT use proxy."""
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        adapter = HermesAgentAdapter(
            model=OPENROUTER_MODEL,
            api_key=OPENROUTER_API_KEY,
            api_base=OPENROUTER_API_BASE,
            api_protocol="openai",
        )
        assert adapter._uses_proxy is False

    def test_anthropic_protocol_uses_proxy(self):
        """Non-OpenAI endpoint should use proxy."""
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        adapter = HermesAgentAdapter(
            model="claude-opus-4-6",
            api_key="sk-ant-test",
            api_base="https://api.anthropic.com",
            api_protocol="anthropic",
        )
        assert adapter._uses_proxy is True

    def test_resolve_hermes_params_openai_direct(self):
        """With OpenAI protocol, params go directly to Hermes (no proxy)."""
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        adapter = HermesAgentAdapter(
            model=OPENROUTER_MODEL,
            api_key=OPENROUTER_API_KEY,
            api_base=OPENROUTER_API_BASE,
            api_protocol="openai",
        )
        params = adapter._resolve_hermes_params()
        assert params["base_url"] == OPENROUTER_API_BASE
        assert params["api_key"] == OPENROUTER_API_KEY
        assert params["model"] == OPENROUTER_MODEL

    def test_resolve_hermes_params_anthropic_proxy(self):
        """With Anthropic protocol, params should point to local proxy."""
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        adapter = HermesAgentAdapter(
            model="claude-opus-4-6",
            api_key="sk-ant-test",
            api_base="https://api.anthropic.com",
            api_protocol="anthropic",
        )
        mock_proxy = MagicMock()
        mock_proxy.local_url = "http://127.0.0.1:8888"
        adapter._proxy = mock_proxy

        params = adapter._resolve_hermes_params()
        assert params["base_url"] == "http://127.0.0.1:8888"
        assert params["api_key"] == "sk-proxy-placeholder"


class TestHermesToolIsolation:
    """Unit tests for Hermes tool isolation (restricted / search modes)."""

    def test_default_tool_mode_is_restricted(self):
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        from ai4sci_bench.core.types import ToolMode
        adapter = HermesAgentAdapter()
        assert adapter.tool_mode == ToolMode.RESTRICTED

    def test_allow_external_tools_sets_search(self):
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        from ai4sci_bench.core.types import ToolMode
        adapter = HermesAgentAdapter(allow_external_tools=True)
        assert adapter.tool_mode == ToolMode.SEARCH

    def test_explicit_tool_mode_overrides_allow_external_tools(self):
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        from ai4sci_bench.core.types import ToolMode
        adapter = HermesAgentAdapter(allow_external_tools=True, tool_mode="restricted")
        assert adapter.tool_mode == ToolMode.RESTRICTED

    def test_restricted_disables_web(self):
        """Restricted mode must disable the 'web' toolset."""
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        adapter = HermesAgentAdapter(tool_mode="restricted")
        params = adapter._resolve_hermes_params()
        assert "web" in params["disabled_toolsets"]

    def test_restricted_disables_browser(self):
        """Restricted mode must disable the 'browser' toolset."""
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        adapter = HermesAgentAdapter(tool_mode="restricted")
        params = adapter._resolve_hermes_params()
        assert "browser" in params["disabled_toolsets"]

    def test_restricted_disables_image_gen(self):
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        adapter = HermesAgentAdapter(tool_mode="restricted")
        params = adapter._resolve_hermes_params()
        assert "image_gen" in params["disabled_toolsets"]

    def test_restricted_disables_delegation(self):
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        adapter = HermesAgentAdapter(tool_mode="restricted")
        params = adapter._resolve_hermes_params()
        assert "delegation" in params["disabled_toolsets"]

    def test_restricted_disables_cronjob(self):
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        adapter = HermesAgentAdapter(tool_mode="restricted")
        params = adapter._resolve_hermes_params()
        assert "cronjob" in params["disabled_toolsets"]

    def test_restricted_disables_memory(self):
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        adapter = HermesAgentAdapter(tool_mode="restricted")
        params = adapter._resolve_hermes_params()
        assert "memory" in params["disabled_toolsets"]

    def test_restricted_preserves_core_toolsets(self):
        """Restricted mode must NOT disable terminal, file, code_execution, todo."""
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        adapter = HermesAgentAdapter(tool_mode="restricted")
        params = adapter._resolve_hermes_params()
        disabled = params["disabled_toolsets"]
        for core in ("terminal", "file", "code_execution", "todo", "clarify", "skills", "vision"):
            assert core not in disabled, f"core toolset '{core}' should not be disabled"

    def test_search_allows_web(self):
        """Search mode must NOT disable the 'web' toolset."""
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        adapter = HermesAgentAdapter(tool_mode="search")
        params = adapter._resolve_hermes_params()
        assert "web" not in params["disabled_toolsets"]

    def test_search_still_disables_browser(self):
        """Search mode allows web search but still disables browser automation."""
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        adapter = HermesAgentAdapter(tool_mode="search")
        params = adapter._resolve_hermes_params()
        assert "browser" in params["disabled_toolsets"]

    def test_search_still_disables_image_gen(self):
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        adapter = HermesAgentAdapter(tool_mode="search")
        params = adapter._resolve_hermes_params()
        assert "image_gen" in params["disabled_toolsets"]

    def test_unrestricted_no_extra_disabled(self):
        """Unrestricted mode should not add any disabled toolsets beyond user-specified ones."""
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        adapter = HermesAgentAdapter(tool_mode="unrestricted")
        params = adapter._resolve_hermes_params()
        assert "disabled_toolsets" not in params

    def test_user_disabled_toolsets_preserved(self):
        """User-specified disabled_toolsets should be preserved and merged with mode defaults."""
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        adapter = HermesAgentAdapter(
            tool_mode="restricted",
            disabled_toolsets=["skills"],
        )
        params = adapter._resolve_hermes_params()
        disabled = params["disabled_toolsets"]
        assert "skills" in disabled  # user-specified
        assert "web" in disabled     # mode-enforced

    def test_no_duplicate_disabled_toolsets(self):
        """If user already disabled a toolset that the mode also disables, no duplicates."""
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        adapter = HermesAgentAdapter(
            tool_mode="restricted",
            disabled_toolsets=["web", "browser"],
        )
        params = adapter._resolve_hermes_params()
        disabled = params["disabled_toolsets"]
        assert disabled.count("web") == 1
        assert disabled.count("browser") == 1

    def test_build_agent_passes_tool_mode_to_hermes(self):
        from ai4sci_bench.cli import _build_agent
        from ai4sci_bench.core.types import ToolMode
        adapter = _build_agent(None, "hermes", {}, tool_mode="restricted")
        assert adapter.tool_mode == ToolMode.RESTRICTED
        adapter = _build_agent(None, "hermes", {}, tool_mode="search")
        assert adapter.tool_mode == ToolMode.SEARCH


class TestCodeWhaleInit:
    """Unit tests for CodeWhaleAdapter initialization with third-party API."""

    def test_api_base_requires_protocol(self):
        from ai4sci_bench.adapters.codewhale_agent import CodeWhaleAdapter
        with pytest.raises(ValueError, match="api_protocol is required"):
            CodeWhaleAdapter(
                api_key="sk-test",
                api_base="https://openrouter.ai/api/v1",
            )

    def test_openai_protocol_no_proxy(self):
        from ai4sci_bench.adapters.codewhale_agent import CodeWhaleAdapter
        adapter = CodeWhaleAdapter(
            model="gpt-4",
            api_key="sk-test",
            api_base="https://openrouter.ai/api/v1",
            api_protocol="openai",
        )
        assert adapter._uses_proxy is False

    def test_anthropic_protocol_uses_proxy(self):
        from ai4sci_bench.adapters.codewhale_agent import CodeWhaleAdapter
        adapter = CodeWhaleAdapter(
            model="claude-opus-4-6",
            api_key="sk-ant-test",
            api_base="https://api.anthropic.com",
            api_protocol="anthropic",
        )
        assert adapter._uses_proxy is True

    def test_provider_inference_openrouter(self):
        from ai4sci_bench.adapters.codewhale_agent import CodeWhaleAdapter
        adapter = CodeWhaleAdapter(
            model="openrouter/deepseek/deepseek-chat",
            api_key="sk-or-test",
            api_base="https://openrouter.ai/api/v1",
            api_protocol="openai",
        )
        assert adapter.provider == "openrouter"

    def test_provider_inference_deepseek(self):
        from ai4sci_bench.adapters.codewhale_agent import CodeWhaleAdapter
        adapter = CodeWhaleAdapter(
            model="deepseek-chat",
            api_key="sk-test",
            api_base="https://api.deepseek.com/beta",
            api_protocol="openai",
        )
        assert adapter.provider == "deepseek"

    def test_generate_config_openai_direct(self, tmp_path):
        """Config for OpenAI-compatible endpoint should use provider directly."""
        from ai4sci_bench.adapters.codewhale_agent import CodeWhaleAdapter
        adapter = CodeWhaleAdapter(
            model="gpt-4",
            api_key="sk-test-key",
            api_base="https://openrouter.ai/api/v1",
            api_protocol="openai",
            provider="openrouter",
        )
        config_path = adapter._generate_config(tmp_path)
        config_text = config_path.read_text()
        assert 'provider = "openrouter"' in config_text
        assert 'api_key = "sk-test-key"' in config_text
        assert 'base_url = "https://openrouter.ai/api/v1"' in config_text
        # Non-deepseek model should be mapped to "auto"
        assert 'default_text_model = "auto"' in config_text

    def test_generate_config_proxy_mode(self, tmp_path):
        """Config with proxy should point to local proxy URL."""
        from ai4sci_bench.adapters.codewhale_agent import CodeWhaleAdapter
        adapter = CodeWhaleAdapter(
            model="claude-opus-4-6",
            api_key="sk-ant-test",
            api_base="https://api.anthropic.com",
            api_protocol="anthropic",
        )
        mock_proxy = MagicMock()
        mock_proxy.local_url = "http://127.0.0.1:7777"
        adapter._proxy = mock_proxy

        config_path = adapter._generate_config(tmp_path)
        config_text = config_path.read_text()
        assert 'base_url = "http://127.0.0.1:7777"' in config_text
        assert 'api_key = "sk-proxy-placeholder"' in config_text
        assert 'provider = "openai"' in config_text
        assert 'default_text_model = "auto"' in config_text


class TestCodeWhaleToolIsolation:
    """Unit tests for CodeWhale tool isolation (restricted / search modes)."""

    def test_default_tool_mode_is_restricted(self):
        from ai4sci_bench.adapters.codewhale_agent import CodeWhaleAdapter
        from ai4sci_bench.core.types import ToolMode
        adapter = CodeWhaleAdapter()
        assert adapter.tool_mode == ToolMode.RESTRICTED

    def test_allow_external_tools_sets_search(self):
        from ai4sci_bench.adapters.codewhale_agent import CodeWhaleAdapter
        from ai4sci_bench.core.types import ToolMode
        adapter = CodeWhaleAdapter(allow_external_tools=True)
        assert adapter.tool_mode == ToolMode.SEARCH

    def test_explicit_tool_mode_overrides_allow_external_tools(self):
        from ai4sci_bench.adapters.codewhale_agent import CodeWhaleAdapter
        from ai4sci_bench.core.types import ToolMode
        adapter = CodeWhaleAdapter(allow_external_tools=True, tool_mode="restricted")
        assert adapter.tool_mode == ToolMode.RESTRICTED

    def test_restricted_features_disable_web_search(self):
        """Restricted config disables web_search feature."""
        from ai4sci_bench.adapters.codewhale_agent import CodeWhaleAdapter
        adapter = CodeWhaleAdapter(tool_mode="restricted")
        section = adapter._generate_features_section()
        assert "web_search = false" in section

    def test_restricted_features_disable_mcp(self):
        """Restricted config disables mcp feature."""
        from ai4sci_bench.adapters.codewhale_agent import CodeWhaleAdapter
        adapter = CodeWhaleAdapter(tool_mode="restricted")
        section = adapter._generate_features_section()
        assert "mcp = false" in section

    def test_search_features_allow_web_search(self):
        """Search mode must NOT disable web_search in config."""
        from ai4sci_bench.adapters.codewhale_agent import CodeWhaleAdapter
        adapter = CodeWhaleAdapter(tool_mode="search")
        section = adapter._generate_features_section()
        assert "web_search = false" not in section

    def test_search_features_still_disable_mcp(self):
        """Search mode allows web search but still disables MCP in config."""
        from ai4sci_bench.adapters.codewhale_agent import CodeWhaleAdapter
        adapter = CodeWhaleAdapter(tool_mode="search")
        section = adapter._generate_features_section()
        assert "mcp = false" in section

    def test_unrestricted_no_features_section(self):
        """Unrestricted mode should not add any feature restrictions."""
        from ai4sci_bench.adapters.codewhale_agent import CodeWhaleAdapter
        adapter = CodeWhaleAdapter(tool_mode="unrestricted")
        section = adapter._generate_features_section()
        assert section == ""

    def test_generate_config_includes_features(self):
        """_generate_config should include the [features] section."""
        from ai4sci_bench.adapters.codewhale_agent import CodeWhaleAdapter
        import tempfile
        adapter = CodeWhaleAdapter(tool_mode="restricted")
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            config_path = adapter._generate_config(workspace)
            content = config_path.read_text()
            assert "[features]" in content
            assert "web_search = false" in content
            assert "mcp = false" in content

    def test_build_agent_passes_tool_mode_to_codewhale(self):
        from ai4sci_bench.cli import _build_agent
        from ai4sci_bench.core.types import ToolMode
        adapter = _build_agent(None, "codewhale", {}, tool_mode="restricted")
        assert adapter.tool_mode == ToolMode.RESTRICTED
        adapter = _build_agent(None, "codewhale", {}, tool_mode="search")
        assert adapter.tool_mode == ToolMode.SEARCH


# ════════════════════════════════════════════════════════════════
# §2  PROXY TESTS — LiteLLMProxy and LiteLLMOpenAIProxy
# ════════════════════════════════════════════════════════════════


class TestLiteLLMProxyLifecycle:
    """Test proxy start/stop and HTTP contract (no upstream calls)."""

    def test_proxy_start_stop(self):
        from ai4sci_bench.adapters.api_proxy import LiteLLMProxy
        proxy = LiteLLMProxy(model="openai/gpt-4", api_base="http://fake", api_key="fake")
        url = proxy.start()
        assert url.startswith("http://127.0.0.1:")
        assert proxy.local_url == url

        # GET should return status
        resp = urllib.request.urlopen(url)
        data = json.loads(resp.read())
        assert data["status"] == "ok"

        proxy.stop()

    def test_proxy_not_started_raises(self):
        from ai4sci_bench.adapters.api_proxy import LiteLLMProxy
        proxy = LiteLLMProxy(model="test", api_key="fake")
        with pytest.raises(RuntimeError, match="Proxy not started"):
            _ = proxy.local_url

    def test_openai_proxy_start_stop(self):
        from ai4sci_bench.adapters.api_proxy import LiteLLMOpenAIProxy
        proxy = LiteLLMOpenAIProxy(model="anthropic/claude-opus-4-6", api_base="http://fake", api_key="fake")
        url = proxy.start()
        assert url.startswith("http://127.0.0.1:")

        resp = urllib.request.urlopen(url)
        data = json.loads(resp.read())
        assert data["proxy"] == "litellm-openai"

        proxy.stop()

    def test_proxy_404_wrong_path(self):
        from ai4sci_bench.adapters.api_proxy import LiteLLMProxy
        proxy = LiteLLMProxy(model="test", api_key="fake")
        url = proxy.start()
        try:
            req = urllib.request.Request(
                f"{url}/v1/wrong",
                data=b'{}',
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(req)
            assert exc_info.value.code == 404
        finally:
            proxy.stop()

    def test_openai_proxy_404_wrong_path(self):
        from ai4sci_bench.adapters.api_proxy import LiteLLMOpenAIProxy
        proxy = LiteLLMOpenAIProxy(model="test", api_key="fake")
        url = proxy.start()
        try:
            req = urllib.request.Request(
                f"{url}/v1/wrong",
                data=b'{}',
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(req)
            assert exc_info.value.code == 404
        finally:
            proxy.stop()


class TestClaudeCodeProxyCreation:
    """Test that ClaudeCodeCLIAdapter creates the right type of proxy."""

    def test_ensure_proxy_creates_litellm_proxy(self):
        """Claude Code should use LiteLLMProxy (Anthropic format)."""
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        adapter = ClaudeCodeCLIAdapter(
            api_key=OPENROUTER_API_KEY,
            api_base=OPENROUTER_API_BASE,
            api_protocol="openai",
        )
        url = adapter._ensure_proxy()
        try:
            assert url.startswith("http://127.0.0.1:")
            # The proxy should accept GET
            resp = urllib.request.urlopen(url)
            data = json.loads(resp.read())
            assert data["proxy"] == "litellm"  # Not "litellm-openai"
        finally:
            adapter.teardown()

    def test_ensure_proxy_idempotent(self):
        """Multiple calls should return the same proxy URL."""
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        adapter = ClaudeCodeCLIAdapter(
            api_key="fake",
            api_base="http://fake",
            api_protocol="openai",
        )
        url1 = adapter._ensure_proxy()
        url2 = adapter._ensure_proxy()
        assert url1 == url2
        adapter.teardown()


class TestHermesProxyCreation:
    """Test Hermes proxy creation for non-OpenAI protocols."""

    def test_anthropic_protocol_creates_openai_proxy(self):
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        adapter = HermesAgentAdapter(
            model="claude-opus-4-6",
            api_key="sk-ant-test",
            api_base="https://api.anthropic.com",
            api_protocol="anthropic",
        )
        url = adapter._ensure_proxy()
        try:
            assert url.startswith("http://127.0.0.1:")
            resp = urllib.request.urlopen(url)
            data = json.loads(resp.read())
            assert data["proxy"] == "litellm-openai"  # OpenAI format for Hermes
        finally:
            adapter.teardown()


class TestCodeWhaleProxyCreation:
    """Test CodeWhale proxy creation for non-OpenAI protocols."""

    def test_anthropic_protocol_creates_openai_proxy(self):
        from ai4sci_bench.adapters.codewhale_agent import CodeWhaleAdapter
        adapter = CodeWhaleAdapter(
            model="claude-opus-4-6",
            api_key="sk-ant-test",
            api_base="https://api.anthropic.com",
            api_protocol="anthropic",
        )
        url = adapter._ensure_proxy()
        try:
            assert url.startswith("http://127.0.0.1:")
            resp = urllib.request.urlopen(url)
            data = json.loads(resp.read())
            assert data["proxy"] == "litellm-openai"
        finally:
            adapter.teardown()


# ════════════════════════════════════════════════════════════════
# §3  E2E INTEGRATION — Actual OpenRouter API calls
# ════════════════════════════════════════════════════════════════


@pytest.fixture
def openrouter_available():
    """Check if OpenRouter is reachable. Skip if not."""
    if "OPENROUTER_API_KEY" not in os.environ:
        pytest.skip("OPENROUTER_API_KEY is not set")
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        )
        resp = urllib.request.urlopen(req, timeout=10)
        assert resp.status == 200
    except Exception as e:
        pytest.skip(f"OpenRouter not reachable: {e}")


class TestLiteLLMProxyE2E:
    """Test the Anthropic-format proxy (used by Claude Code) with real OpenRouter calls."""

    @pytest.mark.integration
    def test_anthropic_proxy_non_streaming(self, openrouter_available):
        """Send Anthropic Messages API request through proxy to OpenRouter."""
        from ai4sci_bench.adapters.api_proxy import LiteLLMProxy

        proxy = LiteLLMProxy(
            model=f"openai/{OPENROUTER_MODEL_CHEAP}",
            api_base=OPENROUTER_API_BASE,
            api_key=OPENROUTER_API_KEY,
        )
        url = proxy.start()
        try:
            payload = json.dumps({
                "model": "ignored-by-proxy",
                "max_tokens": 50,
                "messages": [
                    {"role": "user", "content": "Reply with exactly: HELLO_TEST_OK"}
                ],
            }).encode()

            req = urllib.request.Request(
                f"{url}/v1/messages",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read())

            assert resp.status == 200
            # Should have content in Anthropic Messages format
            assert "content" in data or "error" not in data
            print(f"[Anthropic proxy non-stream] Response: {json.dumps(data, indent=2)[:500]}")
        finally:
            proxy.stop()

    @pytest.mark.integration
    def test_anthropic_proxy_streaming(self, openrouter_available):
        """Test streaming SSE through the Anthropic proxy."""
        from ai4sci_bench.adapters.api_proxy import LiteLLMProxy

        proxy = LiteLLMProxy(
            model=f"openai/{OPENROUTER_MODEL_CHEAP}",
            api_base=OPENROUTER_API_BASE,
            api_key=OPENROUTER_API_KEY,
        )
        url = proxy.start()
        try:
            payload = json.dumps({
                "model": "ignored",
                "max_tokens": 50,
                "stream": True,
                "messages": [
                    {"role": "user", "content": "Reply with: STREAM_OK"}
                ],
            }).encode()

            req = urllib.request.Request(
                f"{url}/v1/messages",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=30)
            body = resp.read().decode("utf-8")

            assert resp.status == 200
            # Should have SSE events
            assert "event:" in body
            assert "message_start" in body
            assert "message_stop" in body
            print(f"[Anthropic proxy stream] SSE events ({len(body)} bytes):")
            for line in body.splitlines()[:10]:
                if line.startswith("event:"):
                    print(f"  {line}")
        finally:
            proxy.stop()


class TestLiteLLMOpenAIProxyE2E:
    """Test the OpenAI-format proxy (used by Codex/Hermes/CodeWhale) with real OpenRouter calls."""

    @pytest.mark.integration
    def test_openai_proxy_non_streaming(self, openrouter_available):
        """Send OpenAI Chat Completions request through proxy to OpenRouter."""
        from ai4sci_bench.adapters.api_proxy import LiteLLMOpenAIProxy

        proxy = LiteLLMOpenAIProxy(
            model=f"openai/{OPENROUTER_MODEL_CHEAP}",
            api_base=OPENROUTER_API_BASE,
            api_key=OPENROUTER_API_KEY,
        )
        url = proxy.start()
        try:
            payload = json.dumps({
                "model": "ignored-by-proxy",
                "max_tokens": 50,
                "messages": [
                    {"role": "user", "content": "Reply with exactly: OPENAI_TEST_OK"}
                ],
            }).encode()

            req = urllib.request.Request(
                f"{url}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read())

            assert resp.status == 200
            # Should have OpenAI Chat Completions format
            assert "choices" in data
            assert len(data["choices"]) > 0
            content = data["choices"][0]["message"]["content"]
            print(f"[OpenAI proxy non-stream] Content: {content[:200]}")
        finally:
            proxy.stop()

    @pytest.mark.integration
    def test_openai_proxy_streaming(self, openrouter_available):
        """Test streaming through the OpenAI proxy."""
        from ai4sci_bench.adapters.api_proxy import LiteLLMOpenAIProxy

        proxy = LiteLLMOpenAIProxy(
            model=f"openai/{OPENROUTER_MODEL_CHEAP}",
            api_base=OPENROUTER_API_BASE,
            api_key=OPENROUTER_API_KEY,
        )
        url = proxy.start()
        try:
            payload = json.dumps({
                "model": "ignored",
                "max_tokens": 50,
                "stream": True,
                "messages": [
                    {"role": "user", "content": "Reply with: STREAM_OK"}
                ],
            }).encode()

            req = urllib.request.Request(
                f"{url}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=60)
            # Read SSE line-by-line (keep-alive means read() blocks forever)
            lines = []
            while True:
                line = resp.readline().decode("utf-8")
                lines.append(line)
                if "[DONE]" in line or not line:
                    break

            body = "".join(lines)
            assert resp.status == 200
            assert "data:" in body
            assert "[DONE]" in body
            print(f"[OpenAI proxy stream] Response ({len(body)} bytes, {body.count('data:')} chunks)")
        finally:
            proxy.stop()


class TestClaudeCodeCLIE2E:
    """E2E test for ClaudeCodeCLIAdapter with OpenRouter.

    These tests verify the proxy starts correctly and env vars are set,
    but don't actually run `claude` CLI (which may not be installed).
    """

    @pytest.mark.integration
    def test_proxy_starts_and_accepts_anthropic_format(self, openrouter_available):
        """Verify the adapter's proxy accepts Anthropic Messages API format via OpenRouter."""
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter

        adapter = ClaudeCodeCLIAdapter(
            model=OPENROUTER_MODEL_CHEAP,
            api_key=OPENROUTER_API_KEY,
            api_base=OPENROUTER_API_BASE,
            api_protocol="openai",
        )
        proxy_url = adapter._ensure_proxy()
        try:
            # Send a real Anthropic Messages API request
            payload = json.dumps({
                "model": "anything",
                "max_tokens": 30,
                "messages": [
                    {"role": "user", "content": "Say OK"}
                ],
            }).encode()

            req = urllib.request.Request(
                f"{proxy_url}/v1/messages",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read())
            assert resp.status == 200
            assert "content" in data
            print(f"[ClaudeCode E2E] Anthropic proxy response: {json.dumps(data, indent=2)[:500]}")
        finally:
            adapter.teardown()

    @pytest.mark.integration
    def test_streaming_through_proxy(self, openrouter_available):
        """Verify streaming works (Claude Code uses stream-json)."""
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter

        adapter = ClaudeCodeCLIAdapter(
            model=OPENROUTER_MODEL_CHEAP,
            api_key=OPENROUTER_API_KEY,
            api_base=OPENROUTER_API_BASE,
            api_protocol="openai",
        )
        proxy_url = adapter._ensure_proxy()
        try:
            payload = json.dumps({
                "model": "anything",
                "max_tokens": 30,
                "stream": True,
                "messages": [
                    {"role": "user", "content": "Say hello"}
                ],
            }).encode()

            req = urllib.request.Request(
                f"{proxy_url}/v1/messages",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=30)
            body = resp.read().decode("utf-8")

            assert "message_start" in body
            assert "content_block_delta" in body
            assert "message_stop" in body
            print(f"[ClaudeCode E2E stream] Got {body.count('event:')} SSE events")
        finally:
            adapter.teardown()

    @pytest.mark.integration
    def test_env_vars_correct(self, openrouter_available):
        """Check that _build_api_env returns correct env for Claude CLI."""
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter

        adapter = ClaudeCodeCLIAdapter(
            model=OPENROUTER_MODEL_CHEAP,
            api_key=OPENROUTER_API_KEY,
            api_base=OPENROUTER_API_BASE,
            api_protocol="openai",
        )
        proxy_url = adapter._ensure_proxy()
        try:
            env = adapter._build_api_env()
            assert "ANTHROPIC_BASE_URL" in env
            assert env["ANTHROPIC_BASE_URL"] == proxy_url
            assert env["ANTHROPIC_API_KEY"] == "sk-proxy-placeholder"
        finally:
            adapter.teardown()


class TestOpenHandsE2E:
    """E2E test for OpenHandsAdapter config generation with OpenRouter.

    These verify the adapter correctly prepares config for OpenHands SDK,
    since the actual SDK may not be installed.
    """

    @pytest.mark.integration
    def test_config_passes_correct_api_params(self):
        """Verify OpenHands receives correct model/api_key/api_base."""
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter

        adapter = OpenHandsAdapter(
            model=OPENROUTER_MODEL_CHEAP,
            api_key=OPENROUTER_API_KEY,
            api_base=OPENROUTER_API_BASE,
            api_protocol="openai",
        )
        # model should be prefixed
        assert adapter.model == f"openai/{OPENROUTER_MODEL_CHEAP}"
        assert adapter.api_key == OPENROUTER_API_KEY
        assert adapter.api_base == OPENROUTER_API_BASE

    @pytest.mark.integration
    def test_helper_script_config_json(self, tmp_path):
        """Verify the config JSON written for the helper script is correct."""
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter

        adapter = OpenHandsAdapter(
            model=OPENROUTER_MODEL_CHEAP,
            api_key=OPENROUTER_API_KEY,
            api_base=OPENROUTER_API_BASE,
            api_protocol="openai",
            max_iterations=10,
            temperature=0.0,
        )

        # The adapter writes config to a temp file; simulate that
        config = {
            "workspace": str(tmp_path),
            "prompt": "test prompt",
            "result_path": str(tmp_path / "result.json"),
            "model": adapter.model,
            "api_key": adapter.api_key,
            "api_base": adapter.api_base,
            "max_iterations": adapter.max_iterations,
            "temperature": adapter.temperature,
            "max_output_tokens": adapter.max_output_tokens,
        }
        assert config["model"] == f"openai/{OPENROUTER_MODEL_CHEAP}"
        assert config["api_key"] == OPENROUTER_API_KEY
        assert config["api_base"] == OPENROUTER_API_BASE
        assert config["max_iterations"] == 10

    @pytest.mark.integration
    def test_litellm_direct_call_openrouter(self, openrouter_available):
        """Verify litellm can call OpenRouter directly (what OpenHands SDK does internally)."""
        import litellm
        response = litellm.completion(
            model=f"openai/{OPENROUTER_MODEL_CHEAP}",
            api_base=OPENROUTER_API_BASE,
            api_key=OPENROUTER_API_KEY,
            messages=[{"role": "user", "content": "Reply with exactly: LITELLM_OK"}],
            max_tokens=30,
        )
        content = response.choices[0].message.content
        assert content is not None
        assert len(content) > 0
        print(f"[OpenHands litellm direct] Response: {content[:200]}")


class TestHermesE2E:
    """E2E test for HermesAgentAdapter with OpenRouter.

    Since Hermes uses OpenAI-compatible protocol and OpenRouter is
    OpenAI-compatible, no proxy is needed — direct pass-through.
    """

    @pytest.mark.integration
    def test_params_for_openrouter_direct(self):
        """Verify Hermes params point directly to OpenRouter."""
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter

        adapter = HermesAgentAdapter(
            model=OPENROUTER_MODEL,
            api_key=OPENROUTER_API_KEY,
            api_base=OPENROUTER_API_BASE,
            api_protocol="openai",
        )
        params = adapter._resolve_hermes_params()
        assert params["base_url"] == OPENROUTER_API_BASE
        assert params["api_key"] == OPENROUTER_API_KEY
        assert params["model"] == OPENROUTER_MODEL
        assert adapter._uses_proxy is False

    @pytest.mark.integration
    def test_no_proxy_started(self):
        """With OpenAI protocol, no proxy should be started."""
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter

        adapter = HermesAgentAdapter(
            model=OPENROUTER_MODEL,
            api_key=OPENROUTER_API_KEY,
            api_base=OPENROUTER_API_BASE,
            api_protocol="openai",
        )
        assert adapter._proxy is None
        params = adapter._resolve_hermes_params()
        # Still no proxy
        assert adapter._proxy is None


class TestCodeWhaleE2E:
    """E2E test for CodeWhaleAdapter config with OpenRouter."""

    @pytest.mark.integration
    def test_config_toml_openrouter(self, tmp_path):
        """Verify generated config.toml is correct for OpenRouter."""
        from ai4sci_bench.adapters.codewhale_agent import CodeWhaleAdapter

        adapter = CodeWhaleAdapter(
            model=OPENROUTER_MODEL,
            api_key=OPENROUTER_API_KEY,
            api_base=OPENROUTER_API_BASE,
            api_protocol="openai",
            provider="openrouter",
        )
        config_path = adapter._generate_config(tmp_path)
        config_text = config_path.read_text()

        assert 'provider = "openrouter"' in config_text
        assert f'api_key = "{OPENROUTER_API_KEY}"' in config_text
        assert f'base_url = "{OPENROUTER_API_BASE}"' in config_text
        # Non-DeepSeek model should map to "auto"
        assert 'default_text_model = "auto"' in config_text
        assert 'approval_policy = "never"' in config_text
        print(f"[CodeWhale E2E] Config:\n{config_text}")

    @pytest.mark.integration
    def test_config_toml_deepseek_model_preserved(self, tmp_path):
        """DeepSeek model names should be preserved (not mapped to 'auto')."""
        from ai4sci_bench.adapters.codewhale_agent import CodeWhaleAdapter

        adapter = CodeWhaleAdapter(
            model="deepseek-v4-pro",
            api_key="sk-test",
            api_base="https://api.deepseek.com/beta",
            api_protocol="openai",
        )
        config_path = adapter._generate_config(tmp_path)
        config_text = config_path.read_text()
        assert 'default_text_model = "deepseek-v4-pro"' in config_text


# ════════════════════════════════════════════════════════════════
# §4  CROSS-ADAPTER TESTS — Protocol consistency
# ════════════════════════════════════════════════════════════════


class TestProtocolConsistency:
    """Verify all adapters handle the OpenRouter pattern consistently."""

    ADAPTERS_WITH_PROXY = [
        ("claude_code_cli", "ai4sci_bench.adapters.claude_code_cli", "ClaudeCodeCLIAdapter"),
        ("hermes", "ai4sci_bench.adapters.hermes_agent", "HermesAgentAdapter"),
        ("codewhale", "ai4sci_bench.adapters.codewhale_agent", "CodeWhaleAdapter"),
    ]

    @pytest.mark.parametrize("name,module,cls_name", ADAPTERS_WITH_PROXY)
    def test_all_require_protocol_with_base(self, name, module, cls_name):
        """All adapters should require api_protocol when api_base is set."""
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        with pytest.raises(ValueError, match="api_protocol is required"):
            cls(api_key="sk-test", api_base="https://example.com")

    def test_openhands_requires_protocol_with_base(self):
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
        with pytest.raises(ValueError, match="api_protocol is required"):
            OpenHandsAdapter(model="gpt-4", api_key="sk-test", api_base="https://example.com")

    @pytest.mark.integration
    def test_all_proxies_respond_to_get(self, openrouter_available):
        """All proxy types should respond to GET with status."""
        from ai4sci_bench.adapters.api_proxy import LiteLLMProxy, LiteLLMOpenAIProxy

        proxies = [
            ("LiteLLMProxy", LiteLLMProxy(
                model=f"openai/{OPENROUTER_MODEL_CHEAP}",
                api_base=OPENROUTER_API_BASE,
                api_key=OPENROUTER_API_KEY,
            )),
            ("LiteLLMOpenAIProxy", LiteLLMOpenAIProxy(
                model=f"openai/{OPENROUTER_MODEL_CHEAP}",
                api_base=OPENROUTER_API_BASE,
                api_key=OPENROUTER_API_KEY,
            )),
        ]
        for name, proxy in proxies:
            url = proxy.start()
            try:
                resp = urllib.request.urlopen(url, timeout=5)
                data = json.loads(resp.read())
                assert data["status"] == "ok", f"{name} GET failed"
            finally:
                proxy.stop()


class TestMultipleProxyConcurrency:
    """Test that multiple adapters can run proxies concurrently."""

    def test_two_proxies_different_ports(self):
        from ai4sci_bench.adapters.api_proxy import LiteLLMProxy, LiteLLMOpenAIProxy

        p1 = LiteLLMProxy(model="openai/gpt-4", api_key="fake")
        p2 = LiteLLMOpenAIProxy(model="anthropic/claude", api_key="fake")

        url1 = p1.start()
        url2 = p2.start()
        try:
            assert url1 != url2
            # Both should respond
            r1 = json.loads(urllib.request.urlopen(url1).read())
            r2 = json.loads(urllib.request.urlopen(url2).read())
            assert r1["status"] == "ok"
            assert r2["status"] == "ok"
        finally:
            p1.stop()
            p2.stop()


# ════════════════════════════════════════════════════════════════
# §5  EDGE CASES
# ════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and error handling for third-party API integration."""

    def test_empty_request_body(self):
        """Proxy should return 400 for empty POST body."""
        from ai4sci_bench.adapters.api_proxy import LiteLLMProxy
        proxy = LiteLLMProxy(model="test", api_key="fake")
        url = proxy.start()
        try:
            req = urllib.request.Request(
                f"{url}/v1/messages",
                data=b'',
                headers={"Content-Type": "application/json", "Content-Length": "0"},
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(req)
            assert exc_info.value.code == 400
        finally:
            proxy.stop()

    def test_invalid_json_body(self):
        """Proxy should return 400 for invalid JSON."""
        from ai4sci_bench.adapters.api_proxy import LiteLLMProxy
        proxy = LiteLLMProxy(model="test", api_key="fake")
        url = proxy.start()
        try:
            req = urllib.request.Request(
                f"{url}/v1/messages",
                data=b'not json at all{{{',
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(req)
            assert exc_info.value.code == 400
        finally:
            proxy.stop()

    def test_claude_code_teardown_stops_proxy(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        adapter = ClaudeCodeCLIAdapter(
            api_key="fake",
            api_base="http://fake",
            api_protocol="openai",
        )
        url = adapter._ensure_proxy()
        # Proxy should work
        resp = urllib.request.urlopen(url)
        assert json.loads(resp.read())["status"] == "ok"

        adapter.teardown()
        # After teardown, proxy should be gone
        with pytest.raises(Exception):
            urllib.request.urlopen(url, timeout=1)

    def test_hermes_teardown_stops_proxy(self):
        from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
        adapter = HermesAgentAdapter(
            model="claude-opus-4-6",
            api_key="sk-ant-test",
            api_base="https://api.anthropic.com",
            api_protocol="anthropic",
        )
        url = adapter._ensure_proxy()
        resp = urllib.request.urlopen(url)
        assert json.loads(resp.read())["status"] == "ok"

        adapter.teardown()
        with pytest.raises(Exception):
            urllib.request.urlopen(url, timeout=1)

    def test_codewhale_teardown_stops_proxy(self):
        from ai4sci_bench.adapters.codewhale_agent import CodeWhaleAdapter
        adapter = CodeWhaleAdapter(
            model="claude-opus-4-6",
            api_key="sk-ant-test",
            api_base="https://api.anthropic.com",
            api_protocol="anthropic",
        )
        url = adapter._ensure_proxy()
        resp = urllib.request.urlopen(url)
        assert json.loads(resp.read())["status"] == "ok"

        adapter.teardown()
        with pytest.raises(Exception):
            urllib.request.urlopen(url, timeout=1)

    @pytest.mark.integration
    def test_proxy_upstream_error_returns_structured_error(self, openrouter_available):
        """Bad API key should return structured error, not crash."""
        from ai4sci_bench.adapters.api_proxy import LiteLLMProxy
        proxy = LiteLLMProxy(
            model=f"openai/{OPENROUTER_MODEL_CHEAP}",
            api_base=OPENROUTER_API_BASE,
            api_key="sk-or-INVALID-KEY",
        )
        url = proxy.start()
        try:
            payload = json.dumps({
                "model": "anything",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "test"}],
            }).encode()

            req = urllib.request.Request(
                f"{url}/v1/messages",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(req, timeout=15)
            # Should return 502 (upstream error), not crash
            assert exc_info.value.code == 502
            error_body = json.loads(exc_info.value.read())
            assert "error" in error_body
            print(f"[Error test] Structured error: {error_body}")
        finally:
            proxy.stop()

    @pytest.mark.integration
    def test_openai_proxy_bad_key_structured_error(self, openrouter_available):
        """OpenAI proxy should also return structured error for bad keys."""
        from ai4sci_bench.adapters.api_proxy import LiteLLMOpenAIProxy
        proxy = LiteLLMOpenAIProxy(
            model=f"openai/{OPENROUTER_MODEL_CHEAP}",
            api_base=OPENROUTER_API_BASE,
            api_key="sk-or-INVALID-KEY",
        )
        url = proxy.start()
        try:
            payload = json.dumps({
                "model": "anything",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "test"}],
            }).encode()

            req = urllib.request.Request(
                f"{url}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(req, timeout=15)
            assert exc_info.value.code == 502
            error_body = json.loads(exc_info.value.read())
            assert "error" in error_body
            print(f"[OpenAI proxy error test] Structured error: {error_body}")
        finally:
            proxy.stop()
