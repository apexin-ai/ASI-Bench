"""Tests for OpenCodeCLIAdapter: command building, auth modes, log parsing."""

from __future__ import annotations

import json

import pytest

from ai4sci_bench.adapters.opencode_cli import OpenCodeCLIAdapter
from ai4sci_bench.cli import _AGENT_CLI_BINARY, _build_agent, _build_agent_metadata
from ai4sci_bench.core.types import RunStatus


# ── Command building ──────────────────────────────────────────

class _StubTaskInstance:
    def __init__(self, workspace_dir):
        self.workspace_dir = workspace_dir


def test_base_command_flags():
    adapter = OpenCodeCLIAdapter()
    cmd = adapter._base_command()
    assert cmd[:3] == ["opencode", "run", "--format"]
    assert "json" in cmd
    assert "--auto" in cmd
    assert "-m" in cmd and "anthropic/claude-opus-4-6" in cmd
    # effort defaults to None → no --variant flag
    assert "--variant" not in cmd


def test_prompt_not_in_argv(monkeypatch, tmp_path):
    prompt = "LONG-PROMPT " + "x" * 20000
    (tmp_path / "prompt.md").write_text(prompt, encoding="utf-8")
    adapter = OpenCodeCLIAdapter()
    cmd = adapter._build_command(_StubTaskInstance(tmp_path), None)
    assert prompt not in " ".join(cmd)
    stdin = adapter._get_stdin_input(_StubTaskInstance(tmp_path))
    assert stdin == prompt


def test_effort_passes_variant_flag():
    adapter = OpenCodeCLIAdapter(effort="high")
    cmd = adapter._base_command()
    assert "--variant" in cmd and "high" in cmd


def test_effort_validation():
    with pytest.raises(ValueError, match="Invalid effort"):
        OpenCodeCLIAdapter(effort="ultramax")


def test_tool_mode_restricted_disables_web_tools():
    adapter = OpenCodeCLIAdapter(tool_mode="restricted")
    env = adapter._build_api_env()
    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    tools = config["agent"]["build"]["tools"]
    assert tools == {"webfetch": False, "websearch": False}


def test_tool_mode_search_keeps_web_tools():
    adapter = OpenCodeCLIAdapter(tool_mode="search")
    env = adapter._build_api_env()
    assert "OPENCODE_CONFIG_CONTENT" not in env


def test_tool_mode_unrestricted_no_config():
    adapter = OpenCodeCLIAdapter(tool_mode="unrestricted")
    assert adapter._build_api_env() == {}


def test_native_provider_mode_rewrites_model():
    adapter = OpenCodeCLIAdapter(
        model="glm-5.3",
        api_key="sk-test",
        api_base="https://api.example.com/v1",
        api_protocol="openai",
    )
    cmd = adapter._base_command()
    assert cmd[cmd.index("-m") + 1] == "bench/glm-5.3"


# ── Auth mode resolution ──────────────────────────────────────

def test_local_login_mode():
    adapter = OpenCodeCLIAdapter()
    assert not adapter._uses_native_provider
    assert adapter._resolved_provider is None
    # Local login still applies RESTRICTED tool isolation via inline config.
    env = adapter._build_api_env()
    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    assert config == {"agent": {"build": {"tools": {
        "webfetch": False, "websearch": False}}}}
    assert "provider" not in config


def test_key_only_mode_injects_provider_env():
    adapter = OpenCodeCLIAdapter(model="deepseek/deepseek-chat", api_key="sk-test")
    env = adapter._build_api_env()
    assert env == {"DEEPSEEK_API_KEY": "sk-test"}


def test_key_only_mode_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        OpenCodeCLIAdapter(model="x/y", api_key="sk-test", provider="nope")


def test_api_base_without_key_raises():
    with pytest.raises(ValueError, match="api_base was given but api_key"):
        OpenCodeCLIAdapter(api_base="https://api.example.com/v1")


def test_api_protocol_without_base_raises():
    with pytest.raises(ValueError, match="api_protocol was given but api_base"):
        OpenCodeCLIAdapter(api_key="sk-test", api_protocol="openai")


def test_invalid_api_protocol_raises():
    with pytest.raises(ValueError, match="Invalid api_protocol"):
        OpenCodeCLIAdapter(
            api_key="sk-test",
            api_base="https://api.example.com/v1",
            api_protocol="made-up",
        )


def test_native_mode_config_openai_protocol():
    adapter = OpenCodeCLIAdapter(
        model="deepseek-chat",
        api_key="sk-ds",
        api_base="https://api.deepseek.com/v1",
        api_protocol="openai",
    )
    env = adapter._build_api_env()
    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    provider = config["provider"]["bench"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "https://api.deepseek.com/v1"
    assert provider["options"]["apiKey"] == "sk-ds"
    assert "deepseek-chat" in provider["models"]


def test_native_mode_config_anthropic_protocol():
    adapter = OpenCodeCLIAdapter(
        model="glm-5.3",
        api_key="sk-test",
        api_base="https://api.example.com/v1",
        api_protocol="anthropic",
    )
    config = json.loads(adapter._build_config_content())
    provider = config["provider"]["bench"]
    assert provider["npm"] == "@ai-sdk/anthropic"
    assert provider["options"]["baseURL"] == "https://api.example.com/v1"


def test_native_mode_config_openai_responses_protocol():
    adapter = OpenCodeCLIAdapter(
        model="gpt-5.5",
        api_key="sk-test",
        api_base="https://api.example.com/v1",
        api_protocol="openai_responses",
    )
    config = json.loads(adapter._build_config_content())
    assert config["provider"]["bench"]["npm"] == "@ai-sdk/openai"


def test_env_secret_reading(monkeypatch):
    monkeypatch.setenv("MY_OC_KEY", "sk-from-env")
    adapter = OpenCodeCLIAdapter(api_key_env="MY_OC_KEY")
    assert adapter.api_key == "sk-from-env"

    with pytest.raises(ValueError, match="not set"):
        OpenCodeCLIAdapter(api_key_env="MY_OC_KEY_UNSET_XYZ")


# ── Log parsing / usage / errors ──────────────────────────────

OPENCODE_FIXTURE = "\n".join([
    json.dumps({"type": "step_start", "timestamp": 1787844581500, "sessionID": "ses-1",
                "part": {"id": "p1", "type": "step-start"}}),
    json.dumps({"type": "reasoning", "timestamp": 1787844581550, "sessionID": "ses-1",
                "part": {"id": "p2", "type": "reasoning", "text": "pondering"}}),
    json.dumps({"type": "tool_use", "timestamp": 1787844581600, "sessionID": "ses-1",
                "part": {"id": "p3", "type": "tool", "callID": "c1", "tool": "bash",
                         "state": {"status": "completed",
                                   "input": {"command": "ls -la"},
                                   "output": "file.txt", "title": "ls -la"}}}),
    json.dumps({"type": "text", "timestamp": 1787844581700, "sessionID": "ses-1",
                "part": {"id": "p4", "type": "text", "text": "PONG"}}),
    json.dumps({"type": "step_finish", "timestamp": 1787844581800, "sessionID": "ses-1",
                "part": {"id": "p5", "type": "step-finish",
                         "tokens": {"total": 200, "input": 120, "output": 60,
                                    "reasoning": 20, "cache": {"read": 0, "write": 0}},
                         "cost": 0.001}}),
])

OPENCODE_ERROR_FIXTURE = json.dumps({
    "type": "error", "timestamp": 1787844567027, "sessionID": "ses-2",
    "error": {"name": "APIError",
              "data": {"message": "API key is invalid.", "statusCode": 401,
                       "isRetryable": False}},
})


def test_parse_log_human_summary():
    adapter = OpenCodeCLIAdapter()
    log = adapter._parse_log(OPENCODE_FIXTURE)
    assert "=== Turn 1 ===" in log
    assert "[thinking] pondering" in log
    assert "[Tool] bash(command=ls -la)" in log
    assert "[opencode] PONG" in log
    assert "[step_finish]" in log


def test_parse_log_error_event():
    adapter = OpenCodeCLIAdapter()
    log = adapter._parse_log(OPENCODE_ERROR_FIXTURE)
    assert "[error] APIError: API key is invalid." in log


def test_usage_extraction_sums_steps():
    adapter = OpenCodeCLIAdapter()
    cost = adapter._extract_usage_from_jsonl(OPENCODE_FIXTURE)
    assert cost is not None
    assert cost.input_tokens == 120
    # output + reasoning tokens
    assert cost.output_tokens == 80
    assert cost.total_tokens == 200


def test_usage_extraction_none_when_empty():
    adapter = OpenCodeCLIAdapter()
    assert adapter._extract_usage_from_jsonl('{"type":"step_start"}') is None


def test_terminal_error_detection():
    adapter = OpenCodeCLIAdapter()
    err = adapter._extract_terminal_error_from_jsonl(OPENCODE_ERROR_FIXTURE)
    assert err is not None
    assert "APIError" in err and "API key is invalid." in err


def test_terminal_error_clean_run():
    adapter = OpenCodeCLIAdapter()
    assert adapter._extract_terminal_error_from_jsonl(OPENCODE_FIXTURE) is None


def test_raw_stdout_format_is_jsonl():
    assert OpenCodeCLIAdapter()._raw_stdout_format() == "jsonl"


# ── Registration ──────────────────────────────────────────────

def test_build_agent_returns_opencode_adapter():
    from ai4sci_bench.adapters.opencode_cli import OpenCodeCLIAdapter as cls
    assert isinstance(_build_agent(None, "opencode_cli", {}), cls)


def test_build_agent_metadata_opencode():
    meta = _build_agent_metadata(None, "opencode_cli", {})
    assert meta["adapter_class"] == "OpenCodeCLIAdapter"


def test_cli_binary_registered():
    assert _AGENT_CLI_BINARY.get("opencode_cli") == "opencode"


# ── OS sandbox integration (mocked Docker) ────────────────────

class TestOpenCodeOsSandbox:
    def _adapter(self, tmp_path, **kw):
        adapter = OpenCodeCLIAdapter(**kw)
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        return adapter

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
        assert "os" in OpenCodeCLIAdapter()._supported_sandbox_modes

    def test_solve_os_delegates_with_agent_type_opencode(self, tmp_path):
        from unittest.mock import MagicMock
        adapter = self._adapter(tmp_path)
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = (True, "ok", '{"type":"step_start"}', None, "sha:1")
        adapter._os_sandbox = mock_sb

        result = adapter.solve(self._make_task_instance(tmp_path))
        assert result.status == RunStatus.COMPLETED
        kwargs = mock_sb.run_agent.call_args.kwargs
        assert kwargs["agent_type"] == "opencode"

    def test_solve_os_cmd_contains_prompt_argv(self, tmp_path):
        from unittest.mock import MagicMock
        adapter = self._adapter(tmp_path)
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = (True, "ok", None, None, None)
        adapter._os_sandbox = mock_sb

        adapter.solve(self._make_task_instance(tmp_path))
        cmd = mock_sb.run_agent.call_args.kwargs["agent_cmd"]
        assert "Solve this task." in cmd

    def test_solve_os_passes_config_content_env(self, tmp_path):
        from unittest.mock import MagicMock
        adapter = self._adapter(
            tmp_path, api_key="sk-test",
            api_base="https://api.example.com/v1", api_protocol="anthropic",
        )
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = (True, "ok", None, None, None)
        adapter._os_sandbox = mock_sb

        adapter.solve(self._make_task_instance(tmp_path))
        env = mock_sb.run_agent.call_args.kwargs["extra_env"]
        assert "OPENCODE_CONFIG_CONTENT" in env
        config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        assert "bench" in config["provider"]

    def test_prepare_auth_mounts_opencode(self, tmp_path, monkeypatch):
        from ai4sci_bench.runner import os_sandbox as mod
        auth = tmp_path / "auth.json"
        auth.write_text("{}")
        monkeypatch.setattr(mod, "OPENCODE_AUTH_PATHS", [auth])
        sb = mod.OSSandbox(tmp_path)
        mounts = sb._prepare_auth_mounts("opencode")
        assert len(mounts) == 2  # ["-v", "host:container:ro"]
        assert mounts[1].endswith(":/home/agent/.local/share/opencode/auth.json:ro")

    def test_needs_api_network_whitelisted(self, tmp_path):
        """pi/opencode containers get network for API calls like claude/codex."""
        from ai4sci_bench.runner import os_sandbox as mod
        sb = mod.OSSandbox(tmp_path)
        cmd = sb._build_docker_cmd(
            container_name="c1", image="img:latest",
            workspace=tmp_path, exec_args=["true"],
            requires_network=False, requires_gpu=False,
            agent_type="pi",
        )
        assert "--network" not in cmd or cmd[cmd.index("--network") + 1] != "none" \
            if "--network" in cmd else True
        cmd2 = sb._build_docker_cmd(
            container_name="c2", image="img:latest",
            workspace=tmp_path, exec_args=["true"],
            requires_network=False, requires_gpu=False,
            agent_type="opencode",
        )
        # no `--network none` appended for whitelisted agent types
        assert "none" not in [cmd2[i + 1] for i in range(len(cmd2)) if cmd2[i] == "--network"]


def test_install_command_registered():
    from ai4sci_bench.runner.task_image import AGENT_INSTALL_COMMANDS
    assert AGENT_INSTALL_COMMANDS["opencode"] == ["npm install -g opencode-ai"]
