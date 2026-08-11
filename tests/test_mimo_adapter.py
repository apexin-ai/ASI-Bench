"""Tests for MiMoCodeCLIAdapter — endpoint resolution, proxy, log parsing, usage."""

from __future__ import annotations

import json
import textwrap

import pytest

from ai4sci_bench.adapters.mimo_code_cli import (
    MIMO_PROXY_DUMMY_MODEL,
    MiMoCodeCLIAdapter,
)


# ═══════════════════════════════════════════════════════════════════
# _resolve_endpoint: auto-detect protocol from URL
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("api_base,expected_base,expected_protocol", [
    # Token Plan (Anthropic wire protocol)
    ("https://token-plan-cn.xiaomimimo.com/anthropic",
     "https://token-plan-cn.xiaomimimo.com/anthropic", "anthropic"),
    # ... with trailing slash
    ("https://token-plan-cn.xiaomimimo.com/anthropic/",
     "https://token-plan-cn.xiaomimimo.com/anthropic", "anthropic"),
    # Official OpenAI endpoint
    ("https://api.xiaomimimo.com/v1",
     "https://api.xiaomimimo.com/v1", "openai"),
    ("https://api.xiaomimimo.com/v1/",
     "https://api.xiaomimimo.com/v1", "openai"),
])
def test_official_endpoint_auto_resolves(api_base, expected_base, expected_protocol):
    base, proto = MiMoCodeCLIAdapter._resolve_endpoint(api_base)
    assert base == expected_base
    assert proto == expected_protocol


def test_unsupported_endpoint_raises():
    with pytest.raises(ValueError, match="not a supported MiMo endpoint"):
        MiMoCodeCLIAdapter._resolve_endpoint("https://api.example.com/v1")


@pytest.mark.parametrize("url", [
    "https://api.deepseek.com/v1",
    "https://openrouter.ai/api/v1",
    "http://localhost:8080",
    "https://api.xiaomimimo.com/v2",
])
def test_various_unsupported_urls_rejected(url):
    with pytest.raises(ValueError, match="not a supported MiMo endpoint"):
        MiMoCodeCLIAdapter._resolve_endpoint(url)


# ═══════════════════════════════════════════════════════════════════
# __init__ — construction & validation
# ═══════════════════════════════════════════════════════════════════

def test_local_login_mode():
    adapter = MiMoCodeCLIAdapter()
    assert not adapter._uses_proxy
    assert adapter._resolved_base is None
    assert adapter._resolved_protocol is None
    assert adapter.model == "xiaomi/mimo-v2.5-pro"


def test_local_login_with_custom_model():
    adapter = MiMoCodeCLIAdapter(model="xiaomi/mimo-v3-pro")
    assert adapter.model == "xiaomi/mimo-v3-pro"
    assert not adapter._uses_proxy


def test_anthropic_endpoint_mode():
    adapter = MiMoCodeCLIAdapter(
        model="mimo-v2.5-pro",
        api_key="tp-xxx",
        api_base="https://token-plan-cn.xiaomimimo.com/anthropic",
    )
    assert adapter._uses_proxy
    assert adapter._resolved_protocol == "anthropic"


def test_openai_endpoint_mode():
    adapter = MiMoCodeCLIAdapter(
        model="mimo-v2.5-pro",
        api_key="sk-xxx",
        api_base="https://api.xiaomimimo.com/v1",
    )
    assert adapter._uses_proxy
    assert adapter._resolved_protocol == "openai"


def test_api_base_without_api_key_raises():
    with pytest.raises(ValueError, match="api_base was given but api_key"):
        MiMoCodeCLIAdapter(api_base="https://api.xiaomimimo.com/v1")


def test_unsupported_endpoint_in_init_raises():
    with pytest.raises(ValueError, match="not a supported MiMo endpoint"):
        MiMoCodeCLIAdapter(
            api_key="sk-xxx",
            api_base="https://api.random-provider.com/v1",
        )


def test_extra_kwargs_silently_ignored():
    """Backward compat: old api_protocol kwarg doesn't crash."""
    adapter = MiMoCodeCLIAdapter(
        api_key="tp-xxx",
        api_base="https://token-plan-cn.xiaomimimo.com/anthropic",
        api_protocol="anthropic",
    )
    assert adapter._uses_proxy


def test_default_tool_mode_is_restricted():
    adapter = MiMoCodeCLIAdapter()
    from ai4sci_bench.core.types import ToolMode
    assert adapter.tool_mode == ToolMode.RESTRICTED


def test_allow_external_tools_sets_search():
    adapter = MiMoCodeCLIAdapter(allow_external_tools=True)
    from ai4sci_bench.core.types import ToolMode
    assert adapter.tool_mode == ToolMode.SEARCH


def test_explicit_tool_mode_overrides():
    adapter = MiMoCodeCLIAdapter(tool_mode="unrestricted")
    from ai4sci_bench.core.types import ToolMode
    assert adapter.tool_mode == ToolMode.UNRESTRICTED


# ═══════════════════════════════════════════════════════════════════
# _build_command — proxy vs local model name
# ═══════════════════════════════════════════════════════════════════

class _StubTaskInstance:
    def __init__(self, workspace_dir):
        self.workspace_dir = workspace_dir


def test_build_command_uses_dummy_model_under_proxy(tmp_path):
    (tmp_path / "prompt.md").write_text("test prompt")
    adapter = MiMoCodeCLIAdapter(
        model="mimo-v2.5-pro",
        api_key="tp-xxx",
        api_base="https://token-plan-cn.xiaomimimo.com/anthropic",
    )
    cmd = adapter._build_command(_StubTaskInstance(tmp_path), None)
    assert MIMO_PROXY_DUMMY_MODEL in cmd
    assert "mimo-v2.5-pro" not in cmd


def test_build_command_uses_real_model_in_local_mode(tmp_path):
    (tmp_path / "prompt.md").write_text("test prompt")
    adapter = MiMoCodeCLIAdapter(model="xiaomi/mimo-v2.5-pro")
    cmd = adapter._build_command(_StubTaskInstance(tmp_path), None)
    assert "xiaomi/mimo-v2.5-pro" in cmd
    assert MIMO_PROXY_DUMMY_MODEL not in cmd


def test_build_command_includes_format_json(tmp_path):
    (tmp_path / "prompt.md").write_text("test")
    adapter = MiMoCodeCLIAdapter()
    cmd = adapter._build_command(_StubTaskInstance(tmp_path), None)
    assert "--format" in cmd
    assert "json" in cmd


def test_build_command_includes_skip_permissions(tmp_path):
    (tmp_path / "prompt.md").write_text("test")
    adapter = MiMoCodeCLIAdapter()
    cmd = adapter._build_command(_StubTaskInstance(tmp_path), None)
    assert "--dangerously-skip-permissions" in cmd


def test_build_command_reads_prompt_from_workspace(tmp_path):
    (tmp_path / "prompt.md").write_text("solve this task")
    adapter = MiMoCodeCLIAdapter()
    cmd = adapter._build_command(_StubTaskInstance(tmp_path), None)
    assert cmd[-1] == "solve this task"


# ═══════════════════════════════════════════════════════════════════
# _build_api_env — env var construction
# ═══════════════════════════════════════════════════════════════════

def test_build_api_env_empty_for_local_login():
    adapter = MiMoCodeCLIAdapter()
    assert adapter._build_api_env() == {}


def test_build_api_env_sets_placeholder_key_for_proxy():
    adapter = MiMoCodeCLIAdapter(
        api_key="tp-real-key",
        api_base="https://token-plan-cn.xiaomimimo.com/anthropic",
    )
    # Don't actually start the proxy — mock _ensure_proxy
    adapter._ensure_proxy = lambda: "http://127.0.0.1:9999"
    env = adapter._build_api_env()
    assert env["OPENAI_API_KEY"] == "sk-proxy-placeholder"
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9999"


# ═══════════════════════════════════════════════════════════════════
# Log parsing — _parse_mimo_log
# ═══════════════════════════════════════════════════════════════════

def _make_event(type_: str, **kw) -> str:
    return json.dumps({"type": type_, **kw})


def test_parse_assistant_text():
    adapter = MiMoCodeCLIAdapter()
    stdout = _make_event("assistant", message={"content": [
        {"type": "text", "text": "Hello world"},
    ]})
    result = adapter._parse_mimo_log(stdout)
    assert "=== Turn 1 ===" in result
    assert "[MiMo] Hello world" in result


def test_parse_assistant_thinking():
    adapter = MiMoCodeCLIAdapter()
    stdout = _make_event("assistant", message={"content": [
        {"type": "thinking", "thinking": "Let me think about this..."},
    ]})
    result = adapter._parse_mimo_log(stdout)
    assert "[thinking] Let me think about this..." in result


def test_parse_tool_use():
    adapter = MiMoCodeCLIAdapter()
    stdout = _make_event("tool_use", name="bash", input={"command": "ls -la"})
    result = adapter._parse_mimo_log(stdout)
    assert "[Tool] bash(command=ls -la)" in result


def test_parse_tool_use_with_long_path():
    adapter = MiMoCodeCLIAdapter()
    long_path = "/very/long/" + "a" * 300
    stdout = _make_event("tool_use", name="read", input={"file_path": long_path})
    result = adapter._parse_mimo_log(stdout)
    assert "[Tool] read(file_path=" in result
    assert "..." in result


def test_parse_tool_result():
    adapter = MiMoCodeCLIAdapter()
    stdout = _make_event("function_call_output", output="file created")
    result = adapter._parse_mimo_log(stdout)
    assert "[ToolResult] file created" in result


def test_parse_error_event():
    adapter = MiMoCodeCLIAdapter()
    stdout = _make_event("error", message="API timeout")
    result = adapter._parse_mimo_log(stdout)
    assert "[error] API timeout" in result


def test_parse_rate_limit_event():
    adapter = MiMoCodeCLIAdapter()
    stdout = _make_event("rate_limit", message="429 too many requests")
    result = adapter._parse_mimo_log(stdout)
    assert "[rate_limit] 429 too many requests" in result


def test_parse_reasoning_event():
    adapter = MiMoCodeCLIAdapter()
    stdout = _make_event("reasoning", text="Considering approach A vs B")
    result = adapter._parse_mimo_log(stdout)
    assert "[thinking] Considering approach A vs B" in result


def test_parse_result_event():
    adapter = MiMoCodeCLIAdapter()
    stdout = _make_event("result", subtype="success", is_error=False, num_turns=3)
    result = adapter._parse_mimo_log(stdout)
    assert "[result] success is_error=False turns=3" in result


def test_parse_system_event():
    adapter = MiMoCodeCLIAdapter()
    stdout = _make_event("system", message="Session started")
    result = adapter._parse_mimo_log(stdout)
    assert "[system] Session started" in result


def test_parse_user_tool_result():
    adapter = MiMoCodeCLIAdapter()
    stdout = _make_event("user", message={"content": [
        {"type": "tool_result", "content": "0\n1\n1\n2\n3", "is_error": False},
    ]})
    result = adapter._parse_mimo_log(stdout)
    assert "[ToolResult] 0\n1\n1\n2\n3" in result


def test_parse_user_tool_result_error():
    adapter = MiMoCodeCLIAdapter()
    stdout = _make_event("user", message={"content": [
        {"type": "tool_result", "content": "Permission denied", "is_error": True},
    ]})
    result = adapter._parse_mimo_log(stdout)
    assert "[ToolResult:err] Permission denied" in result


def test_parse_message_assistant():
    adapter = MiMoCodeCLIAdapter()
    stdout = _make_event("message", role="assistant", content="Done!")
    result = adapter._parse_mimo_log(stdout)
    assert "[assistant] Done!" in result


def test_parse_message_with_content_blocks():
    adapter = MiMoCodeCLIAdapter()
    stdout = _make_event("message", role="assistant", content=[
        {"type": "text", "text": "Here is the result"},
        {"type": "tool_use", "name": "bash", "input": {"command": "python fib.py"}},
    ])
    result = adapter._parse_mimo_log(stdout)
    assert "[assistant] Here is the result" in result
    assert "[Tool] bash(command=python fib.py)" in result


def test_parse_invalid_json_lines_preserved():
    adapter = MiMoCodeCLIAdapter()
    stdout = "not json at all\n" + _make_event("system", message="ok")
    result = adapter._parse_mimo_log(stdout)
    assert "not json at all" in result
    assert "[system] ok" in result


def test_parse_empty_stdout():
    adapter = MiMoCodeCLIAdapter()
    assert adapter._parse_mimo_log("") == ""


def test_parse_multiple_turns():
    adapter = MiMoCodeCLIAdapter()
    lines = [
        _make_event("assistant", message={"content": [{"type": "text", "text": "Step 1"}]}),
        _make_event("assistant", message={"content": [{"type": "text", "text": "Step 2"}]}),
    ]
    result = adapter._parse_mimo_log("\n".join(lines))
    assert "=== Turn 1 ===" in result
    assert "=== Turn 2 ===" in result
    assert "[MiMo] Step 1" in result
    assert "[MiMo] Step 2" in result


def test_parse_tool_use_string_arguments():
    adapter = MiMoCodeCLIAdapter()
    stdout = _make_event("function_call", name="bash", arguments='{"command": "echo hi"}')
    result = adapter._parse_mimo_log(stdout)
    assert "[Tool] bash(command=echo hi)" in result


def test_parse_tool_use_non_dict_input():
    adapter = MiMoCodeCLIAdapter()
    stdout = _make_event("tool_use", name="unknown_tool", input="raw string arg")
    result = adapter._parse_mimo_log(stdout)
    assert "[Tool] unknown_tool" in result


# ═══════════════════════════════════════════════════════════════════
# Usage extraction — _extract_usage_from_jsonl
# ═══════════════════════════════════════════════════════════════════

def test_extract_usage_from_result_event():
    stdout = _make_event("result", usage={
        "input_tokens": 1000,
        "output_tokens": 200,
    })
    cost = MiMoCodeCLIAdapter._extract_usage_from_jsonl(stdout)
    assert cost is not None
    assert cost.input_tokens == 1000
    assert cost.output_tokens == 200
    assert cost.total_tokens == 1200


def test_extract_usage_from_non_result_event():
    stdout = _make_event("step_finish", usage={
        "prompt_tokens": 500,
        "completion_tokens": 100,
    })
    cost = MiMoCodeCLIAdapter._extract_usage_from_jsonl(stdout)
    assert cost is not None
    assert cost.input_tokens == 500
    assert cost.output_tokens == 100


def test_extract_usage_returns_none_when_no_usage():
    stdout = _make_event("assistant", message={"content": []})
    cost = MiMoCodeCLIAdapter._extract_usage_from_jsonl(stdout)
    assert cost is None


def test_extract_usage_empty_stdout():
    assert MiMoCodeCLIAdapter._extract_usage_from_jsonl("") is None


def test_extract_usage_invalid_json_skipped():
    stdout = "not json\n" + _make_event("result", usage={
        "input_tokens": 42, "output_tokens": 7,
    })
    cost = MiMoCodeCLIAdapter._extract_usage_from_jsonl(stdout)
    assert cost is not None
    assert cost.total_tokens == 49


def test_extract_usage_prefers_result_event():
    stdout = "\n".join([
        _make_event("step_finish", usage={"prompt_tokens": 100, "completion_tokens": 50}),
        _make_event("result", usage={"input_tokens": 999, "output_tokens": 111}),
    ])
    cost = MiMoCodeCLIAdapter._extract_usage_from_jsonl(stdout)
    # First match wins; step_finish comes first
    assert cost.input_tokens == 100


# ═══════════════════════════════════════════════════════════════════
# teardown
# ═══════════════════════════════════════════════════════════════════

def test_teardown_without_proxy_is_safe():
    adapter = MiMoCodeCLIAdapter()
    adapter.teardown()


def test_teardown_clears_proxy():
    adapter = MiMoCodeCLIAdapter(
        api_key="tp-xxx",
        api_base="https://token-plan-cn.xiaomimimo.com/anthropic",
    )

    class FakeProxy:
        stopped = False
        def stop(self):
            self.stopped = True

    fake = FakeProxy()
    adapter._proxy = fake
    adapter.teardown()
    assert fake.stopped
    assert adapter._proxy is None
