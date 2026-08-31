"""Tests for PiCLIAdapter: command building, auth modes, log parsing, usage."""

from __future__ import annotations

import json

import pytest

from ai4sci_bench.adapters.pi_cli import PI_CORE_TOOLS, PiCLIAdapter
from ai4sci_bench.cli import _AGENT_CLI_BINARY, _build_agent, _build_agent_metadata
from ai4sci_bench.core.types import RunStatus, ToolMode


# ── Command building ──────────────────────────────────────────

class _StubTaskInstance:
    def __init__(self, workspace_dir):
        self.workspace_dir = workspace_dir


def test_base_command_flags():
    adapter = PiCLIAdapter()
    cmd = adapter._base_command()
    assert cmd[0] == "pi"
    assert "-p" in cmd
    assert "--mode" in cmd and "json" in cmd
    assert "--no-approve" in cmd
    assert "--no-session" in cmd
    assert "--thinking" in cmd and "medium" in cmd
    assert "--model" in cmd and "anthropic/claude-opus-4-6" in cmd


def test_prompt_not_in_argv(monkeypatch, tmp_path):
    prompt = "LONG-PROMPT " + "x" * 20000
    (tmp_path / "prompt.md").write_text(prompt, encoding="utf-8")
    adapter = PiCLIAdapter()
    cmd = adapter._build_command(_StubTaskInstance(tmp_path), None)
    # Prompt must travel via stdin, never argv (#27).
    assert prompt not in " ".join(cmd)
    stdin = adapter._get_stdin_input(_StubTaskInstance(tmp_path))
    assert stdin == prompt


def test_tool_mode_restricted_adds_tools_allowlist():
    adapter = PiCLIAdapter(tool_mode="restricted")
    cmd = adapter._base_command()
    assert "--tools" in cmd
    assert PI_CORE_TOOLS in cmd


def test_tool_mode_search_same_allowlist_no_builtin_web_tools():
    # pi has no built-in web tools; SEARCH shares the coding allowlist.
    adapter = PiCLIAdapter(tool_mode="search")
    cmd = adapter._base_command()
    assert PI_CORE_TOOLS in cmd


def test_tool_mode_unrestricted_no_tools_flag():
    adapter = PiCLIAdapter(tool_mode="unrestricted")
    cmd = adapter._base_command()
    assert "--tools" not in cmd


def test_native_provider_mode_rewrites_model():
    adapter = PiCLIAdapter(
        model="glm-5.3",
        api_key="sk-test",
        api_base="https://api.example.com/v1",
        api_protocol="openai",
    )
    cmd = adapter._base_command()
    assert "bench/glm-5.3" in cmd
    assert cmd[cmd.index("--model") + 1] == "bench/glm-5.3"


def test_effort_validation():
    with pytest.raises(ValueError, match="Invalid effort"):
        PiCLIAdapter(effort="ultramax")


# ── Auth mode resolution ──────────────────────────────────────

def test_local_login_mode():
    adapter = PiCLIAdapter()
    assert not adapter._uses_native_provider
    assert adapter._resolved_provider is None
    assert adapter._build_api_env() == {}


def test_key_only_mode_anthropic_prefix():
    adapter = PiCLIAdapter(model="anthropic/claude-opus-4-6", api_key="sk-test")
    env = adapter._build_api_env()
    assert env == {"ANTHROPIC_API_KEY": "sk-test"}


def test_key_only_mode_explicit_provider():
    adapter = PiCLIAdapter(model="some-model", api_key="sk-test", provider="deepseek")
    env = adapter._build_api_env()
    assert env == {"DEEPSEEK_API_KEY": "sk-test"}


def test_key_only_mode_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        PiCLIAdapter(
            model="mystery-model",
            api_key="sk-test",
            provider="not-a-provider",
        )


def test_api_base_without_key_raises():
    with pytest.raises(ValueError, match="api_base was given but api_key"):
        PiCLIAdapter(api_base="https://api.example.com/v1")


def test_api_protocol_without_base_raises():
    with pytest.raises(ValueError, match="api_protocol was given but api_base"):
        PiCLIAdapter(api_key="sk-test", api_protocol="openai")


def test_invalid_api_protocol_raises():
    with pytest.raises(ValueError, match="Invalid api_protocol"):
        PiCLIAdapter(
            api_key="sk-test",
            api_base="https://api.example.com/v1",
            api_protocol="totally-made-up",
        )


def test_native_mode_generates_temp_config():
    adapter = PiCLIAdapter(
        model="deepseek-chat",
        api_key="sk-ds",
        api_base="https://api.deepseek.com/v1",
        api_protocol="openai",
    )
    try:
        env = adapter._build_api_env()
        config_dir = env["PI_CODING_AGENT_DIR"]
        models = json.loads(open(f"{config_dir}/models.json", encoding="utf-8").read())
        provider = models["providers"]["bench"]
        assert provider["baseUrl"] == "https://api.deepseek.com/v1"
        assert provider["api"] == "openai-completions"
        assert provider["apiKey"] == "sk-ds"
        assert provider["models"][0]["id"] == "deepseek-chat"
    finally:
        adapter.teardown()


def test_native_mode_anthropic_strips_v1():
    adapter = PiCLIAdapter(
        model="glm-5.3",
        api_key="sk-test",
        api_base="https://api.example.com/v1",
        api_protocol="anthropic",
    )
    models = json.loads(adapter._generate_models_json())
    # pi's anthropic-messages client appends /v1/messages itself.
    assert models["providers"]["bench"]["baseUrl"] == "https://api.example.com"
    assert models["providers"]["bench"]["api"] == "anthropic-messages"


def test_env_secret_reading(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_PI_KEY", "sk-from-env")
    adapter = PiCLIAdapter(api_key_env="MY_PI_KEY")
    assert adapter.api_key == "sk-from-env"

    with pytest.raises(ValueError, match="not set"):
        PiCLIAdapter(api_key_env="MY_PI_KEY_UNSET_XYZ")


def test_env_endpoint_and_key_use_native_provider(monkeypatch):
    monkeypatch.setenv("MY_PI_KEY", "sk-from-env")
    monkeypatch.setenv("MY_PI_BASE", "https://api.example.com/v1")
    adapter = PiCLIAdapter(model="glm-5.3", api_key_env="MY_PI_KEY", api_base_env="MY_PI_BASE", api_protocol="openai")
    assert adapter._uses_native_provider is True
    assert adapter._model_arg() == "bench/glm-5.3"


# ── Log parsing ───────────────────────────────────────────────

PI_FIXTURE = "\n".join([
    json.dumps({"type": "session", "version": 3, "id": "ses-1",
                "timestamp": "2026-08-27T15:28:40.139Z", "cwd": "/w"}),
    json.dumps({"type": "agent_start"}),
    json.dumps({"type": "turn_start"}),
    json.dumps({"type": "message_end", "message": {"role": "user", "content": [
        {"type": "text", "text": "solve it"}]}}),
    json.dumps({"type": "tool_execution_start", "toolCallId": "tc-1",
                "toolName": "read", "args": {"path": "/workspace/data.csv"}}),
    json.dumps({"type": "tool_execution_end", "toolCallId": "tc-1",
                "toolName": "read", "result": {"output": "a,b\n1,2"},
                "isError": False}),
    json.dumps({"type": "message_end", "message": {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "let me look"},
            {"type": "text", "text": "All done"},
        ],
        "usage": {"input": 120, "output": 50, "cacheRead": 0, "cacheWrite": 0,
                  "totalTokens": 170},
        "stopReason": "stop",
    }}),
    json.dumps({"type": "turn_end", "message": {"role": "assistant",
                                                "content": [], "stopReason": "stop"},
                "toolResults": []}),
    json.dumps({"type": "agent_end", "messages": []}),
])

PI_ERROR_FIXTURE = "\n".join([
    json.dumps({"type": "session", "id": "ses-2"}),
    json.dumps({"type": "message_end", "message": {
        "role": "assistant", "content": [],
        "stopReason": "error",
        "errorMessage": '401 {"type":"error","error":{"type":"authentication_error"}}',
    }}),
    json.dumps({"type": "agent_end", "messages": [], "willRetry": False}),
])


def test_parse_log_human_summary():
    adapter = PiCLIAdapter()
    log = adapter._parse_log(PI_FIXTURE)
    assert "[session] id=ses-1" in log
    assert "[agent_start]" in log
    assert "=== Turn 1 ===" in log
    assert "[Tool] read(path=/workspace/data.csv)" in log
    assert "[ToolResult] a,b" in log
    assert "[thinking] let me look" in log
    assert "[pi] All done" in log


def test_parse_log_tolerates_malformed_lines():
    adapter = PiCLIAdapter()
    log = adapter._parse_log("not-json\n" + PI_FIXTURE)
    assert "not-json" in log


def test_usage_extraction_sums_assistant_messages():
    adapter = PiCLIAdapter()
    cost = adapter._extract_usage_from_jsonl(PI_FIXTURE)
    assert cost is not None
    assert cost.input_tokens == 120
    assert cost.output_tokens == 50
    assert cost.total_tokens == 170


def test_usage_extraction_none_when_no_usage():
    adapter = PiCLIAdapter()
    assert adapter._extract_usage_from_jsonl('{"type":"agent_start"}') is None


def test_terminal_error_detection_api_error():
    adapter = PiCLIAdapter()
    err = adapter._extract_terminal_error_from_jsonl(PI_ERROR_FIXTURE)
    assert err is not None
    assert "401" in err


def test_terminal_error_detection_clean_run():
    adapter = PiCLIAdapter()
    assert adapter._extract_terminal_error_from_jsonl(PI_FIXTURE) is None


def test_terminal_error_detection_aborted():
    adapter = PiCLIAdapter()
    fixture = json.dumps({"type": "message_end", "message": {
        "role": "assistant", "content": [], "stopReason": "aborted"}})
    err = adapter._extract_terminal_error_from_jsonl(fixture)
    assert err is not None and "aborted" in err


def test_raw_stdout_format_is_jsonl():
    assert PiCLIAdapter()._raw_stdout_format() == "jsonl"


# ── Registration ──────────────────────────────────────────────

def test_build_agent_returns_pi_adapter():
    from ai4sci_bench.adapters.pi_cli import PiCLIAdapter as cls
    assert isinstance(_build_agent(None, "pi_cli", {}), cls)


def test_build_agent_metadata_pi():
    meta = _build_agent_metadata(None, "pi_cli", {})
    assert meta["adapter_class"] == "PiCLIAdapter"


def test_cli_binary_registered():
    assert _AGENT_CLI_BINARY.get("pi_cli") == "pi"


def test_tool_mode_enum_accepted():
    adapter = PiCLIAdapter(tool_mode=ToolMode.RESTRICTED)
    assert adapter.tool_mode == ToolMode.RESTRICTED


# ── OS sandbox integration (mocked Docker) ────────────────────

class TestPiOsSandbox:
    def _adapter(self, tmp_path, **kw):
        from ai4sci_bench.adapters.pi_cli import PiCLIAdapter
        adapter = PiCLIAdapter(**kw)
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        return adapter

    def _mock_result(self, success=True, log="ok", stdout=None):
        return (success, log, stdout, None, "sha256:abc123")

    def _make_task_instance(self, tmp_path):
        from ai4sci_bench.core.types import PromptLevel, TaskInstance
        workspace = tmp_path / "workspace"
        workspace.mkdir(exist_ok=True)
        (workspace / "prompt.md").write_text("Solve this task.")
        return TaskInstance(
            task_id="test.task", instance_id="test.task__seed42",
            task_dir=tmp_path, workspace_dir=workspace,
            reference_dir=tmp_path, prompt_level=PromptLevel.B2,
            parameters={}, metadata={"id": "test.task"},
        )

    def test_supported_modes_include_os(self):
        assert "os" in PiCLIAdapter()._supported_sandbox_modes

    def test_solve_os_delegates_with_agent_type_pi(self, tmp_path):
        from unittest.mock import MagicMock
        adapter = self._adapter(tmp_path)
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = self._mock_result(
            stdout='{"type":"session","id":"s"}')
        adapter._os_sandbox = mock_sb

        result = adapter.solve(self._make_task_instance(tmp_path))
        assert result.status == RunStatus.COMPLETED
        kwargs = mock_sb.run_agent.call_args.kwargs
        assert kwargs["agent_type"] == "pi"

    def test_solve_os_passes_prompt_via_stdin(self, tmp_path):
        from unittest.mock import MagicMock
        adapter = self._adapter(tmp_path)
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = self._mock_result()
        adapter._os_sandbox = mock_sb

        adapter.solve(self._make_task_instance(tmp_path))
        kwargs = mock_sb.run_agent.call_args.kwargs
        assert "Solve this task." not in kwargs["agent_cmd"]
        assert kwargs["stdin_input"] == "Solve this task."

    def test_solve_os_native_mode_mounts_config_dir(self, tmp_path):
        from unittest.mock import MagicMock
        adapter = self._adapter(
            tmp_path, api_key="sk-test",
            api_base="https://api.example.com/v1", api_protocol="openai",
        )
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = self._mock_result()
        adapter._os_sandbox = mock_sb
        try:
            adapter.solve(self._make_task_instance(tmp_path))
            kwargs = mock_sb.run_agent.call_args.kwargs
            env = kwargs["extra_env"]
            assert env["PI_CODING_AGENT_DIR"] == adapter._CONTAINER_PI_CONFIG_DIR
            mounts = kwargs["extra_mounts"]
            assert any(adapter._CONTAINER_PI_CONFIG_DIR in m for m in mounts)
        finally:
            adapter.teardown()

    def test_solve_os_terminal_error_maps_failed(self, tmp_path):
        from unittest.mock import MagicMock
        adapter = self._adapter(tmp_path)
        mock_sb = MagicMock()
        fixture = json.dumps({"type": "message_end", "message": {
            "role": "assistant", "content": [], "stopReason": "error",
            "errorMessage": "401 invalid key"}})
        mock_sb.run_agent.return_value = self._mock_result(stdout=fixture)
        adapter._os_sandbox = mock_sb

        result = adapter.solve(self._make_task_instance(tmp_path))
        assert result.status == RunStatus.FAILED
        assert "401" in result.error_message

    def test_prepare_auth_mounts_pi(self, tmp_path, monkeypatch):
        from ai4sci_bench.runner import os_sandbox as mod
        auth = tmp_path / "auth.json"
        auth.write_text("{}")
        monkeypatch.setattr(mod, "PI_AUTH_PATHS", [auth])
        sb = mod.OSSandbox(tmp_path)
        mounts = sb._prepare_auth_mounts("pi")
        assert len(mounts) == 2  # ["-v", "host:container:ro"]
        assert mounts[1].endswith(":/home/agent/.pi/agent/auth.json:ro")

    def test_prepare_auth_mounts_missing_file_skipped(self, tmp_path, monkeypatch):
        from ai4sci_bench.runner import os_sandbox as mod
        monkeypatch.setattr(mod, "PI_AUTH_PATHS", [tmp_path / "nope.json"])
        sb = mod.OSSandbox(tmp_path)
        assert sb._prepare_auth_mounts("pi") == []


def test_install_command_registered():
    from ai4sci_bench.runner.task_image import AGENT_INSTALL_COMMANDS
    assert AGENT_INSTALL_COMMANDS["pi"] == [
        "npm install -g @earendil-works/pi-coding-agent@0.84.3",
    ]


def test_verified_cli_version_fact():
    assert PiCLIAdapter.VERIFIED_CLI_VERSION == "0.84.3"
    assert PiCLIAdapter.CLI_BINARY == "pi"
