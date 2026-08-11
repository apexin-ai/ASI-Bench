"""Tests for SubprocessAgentAdapter base class and shared helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai4sci_bench.adapters.subprocess_base import (
    SubprocessAgentAdapter,
    collect_output_files,
    get_task_environment,
)
from ai4sci_bench.core.types import AgentOutput, PromptLevel, RunStatus, TaskInstance


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def workspace(tmp_path):
    """Create a minimal workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "prompt.md").write_text("Solve the task.")
    return ws


@pytest.fixture
def task_instance(workspace, tmp_path):
    """Create a minimal TaskInstance for testing."""
    ref_dir = tmp_path / "reference"
    ref_dir.mkdir()
    task_dir = tmp_path / "tasks" / "physics" / "test_task"
    task_dir.mkdir(parents=True)
    return TaskInstance(
        task_id="physics.test_task",
        instance_id="physics.test_task__size50_seed0",
        task_dir=task_dir,
        workspace_dir=workspace,
        reference_dir=ref_dir,
        prompt_level=PromptLevel.B1,
        parameters={"size": 50},
        metadata={
            "output": {
                "files": [
                    {"name": "simulation.py", "type": "code"},
                    {"name": "output.npy", "type": "data"},
                ]
            }
        },
    )


# ── Concrete test adapter ────────────────────────────────────────────────────


class EchoAdapter(SubprocessAgentAdapter):
    """Minimal concrete adapter for testing the base class."""

    def _build_command(self, task_instance, task_env):
        return ["echo", "hello"]


class ShellAdapter(SubprocessAgentAdapter):
    """Adapter that uses shell=True."""

    def __init__(self, cmd: str = "echo hello", **kwargs):
        super().__init__(**kwargs)
        self.cmd = cmd

    def _build_command(self, task_instance, task_env):
        return self.cmd

    def _use_shell(self):
        return True


class CustomLogAdapter(SubprocessAgentAdapter):
    """Adapter with custom log parsing."""

    def _build_command(self, task_instance, task_env):
        return ["echo", "raw output"]

    def _parse_log(self, stdout):
        return f"PARSED: {stdout.strip()}"

    def _raw_stdout_format(self):
        return "jsonl"


class CustomEnvAdapter(SubprocessAgentAdapter):
    """Adapter with custom environment."""

    def _build_command(self, task_instance, task_env):
        return ["echo", "hello"]

    def _build_run_env(self, task_instance, task_env):
        env = super()._build_run_env(task_instance, task_env) or {}
        env["CUSTOM_VAR"] = "custom_value"
        env.pop("SECRET_KEY", None)
        return env


class CustomCwdAdapter(SubprocessAgentAdapter):
    """Adapter with custom cwd."""

    def __init__(self, cwd_path, **kwargs):
        super().__init__(**kwargs)
        self.cwd_path = cwd_path

    def _build_command(self, task_instance, task_env):
        return ["echo", "hello"]

    def _get_cwd(self, task_instance):
        return self.cwd_path


# ── Tests: SubprocessAgentAdapter ABC ─────────────────────────────────────────


class TestSubprocessAgentAdapterABC:
    def test_cannot_instantiate_directly(self):
        """SubprocessAgentAdapter is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            SubprocessAgentAdapter()

    def test_concrete_subclass_can_instantiate(self):
        adapter = EchoAdapter()
        assert adapter.timeout_seconds == 10800
        assert adapter.sandbox == "none"

    def test_custom_timeout(self):
        adapter = EchoAdapter(timeout_seconds=120)
        assert adapter.timeout_seconds == 120

    def test_custom_sandbox(self):
        adapter = EchoAdapter(sandbox="task")
        assert adapter.sandbox == "task"

    def test_inherits_agent_adapter(self):
        from ai4sci_bench.core.agent_interface import AgentAdapter
        assert issubclass(SubprocessAgentAdapter, AgentAdapter)

    def test_default_supported_sandbox_modes(self):
        adapter = EchoAdapter()
        assert adapter._supported_sandbox_modes == ("none", "task")


# ── Tests: setup() ───────────────────────────────────────────────────────────


class TestSetup:
    def test_setup_updates_timeout(self):
        adapter = EchoAdapter()
        adapter.setup({"timeout": 999})
        assert adapter.timeout_seconds == 999

    def test_setup_updates_sandbox(self):
        adapter = EchoAdapter()
        adapter.setup({"sandbox": "task"})
        assert adapter.sandbox == "task"
        assert adapter.task_env_manager is not None

    def test_setup_sandbox_none_no_task_env_manager(self):
        adapter = EchoAdapter()
        adapter.setup({"sandbox": "none"})
        assert adapter.task_env_manager is None

    def test_setup_rejects_unsupported_sandbox(self):
        adapter = EchoAdapter(supported_sandbox_modes=("none",))
        with pytest.raises(ValueError, match="does not support"):
            adapter.setup({"sandbox": "task"})

    def test_setup_rejects_unknown_sandbox(self):
        adapter = EchoAdapter()
        with pytest.raises(ValueError, match="Unknown sandbox"):
            adapter.setup({"sandbox": "magic"})

    def test_setup_updates_repo_root(self):
        adapter = EchoAdapter()
        adapter.setup({"repo_root": "/custom/path"})
        assert adapter.repo_root == Path("/custom/path")

    def test_setup_preserves_defaults(self):
        adapter = EchoAdapter(timeout_seconds=120)
        adapter.setup({})
        assert adapter.timeout_seconds == 120


# ── Tests: solve() — success path ────────────────────────────────────────────


class TestSolveSuccess:
    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_successful_run(self, mock_run, task_instance):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="output text", stderr=""
        )
        adapter = EchoAdapter()
        result = adapter.solve(task_instance)

        assert result.status == RunStatus.COMPLETED
        assert result.instance_id == task_instance.instance_id
        assert result.output_dir == task_instance.workspace_dir
        assert result.error_message is None
        assert result.raw_stdout == "output text"
        assert result.raw_stderr == ""
        assert result.execution_time_seconds > 0

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_raw_stdout_format(self, mock_run, task_instance):
        mock_run.return_value = MagicMock(returncode=0, stdout="out", stderr="")
        adapter = EchoAdapter()
        result = adapter.solve(task_instance)
        assert result.raw_stdout_format == "log"  # default

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_custom_raw_stdout_format(self, mock_run, task_instance):
        mock_run.return_value = MagicMock(returncode=0, stdout="out", stderr="")
        adapter = CustomLogAdapter()
        result = adapter.solve(task_instance)
        assert result.raw_stdout_format == "jsonl"

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_collects_existing_output_files(self, mock_run, task_instance):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        # Create one output file
        (task_instance.workspace_dir / "simulation.py").write_text("print('hi')")
        adapter = EchoAdapter()
        result = adapter.solve(task_instance)
        assert "simulation.py" in result.code_files
        assert "output.npy" not in result.data_files

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_subprocess_called_with_correct_args(self, mock_run, task_instance):
        from dataclasses import replace
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter = EchoAdapter(timeout_seconds=120)
        ti = replace(task_instance, effective_timeout_seconds=120)
        adapter.solve(ti)

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["timeout"] == 120
        assert call_kwargs.kwargs["shell"] is False
        # capture_output/text/encoding/errors are now applied internally by
        # run_subprocess_with_graceful_timeout, not passed by solve().
        assert call_kwargs.args[0] == ["echo", "hello"]
        assert call_kwargs.kwargs["cwd"] == str(ti.workspace_dir)


# ── Tests: solve() — failure paths ───────────────────────────────────────────


class TestSolveFailure:
    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_nonzero_exit_code(self, mock_run, task_instance):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="some output", stderr="error msg"
        )
        adapter = EchoAdapter()
        result = adapter.solve(task_instance)

        assert result.status == RunStatus.FAILED
        assert "EchoAdapter exited with code 1" in result.error_message
        assert "error msg" in result.log
        assert "[exit_code] 1" in result.log

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_timeout(self, mock_run, task_instance):
        from dataclasses import replace
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["echo"], timeout=10, output="partial out", stderr="partial err"
        )
        adapter = EchoAdapter(timeout_seconds=10)
        ti = replace(task_instance, effective_timeout_seconds=10)
        result = adapter.solve(ti)

        assert result.status == RunStatus.TIMEOUT
        assert "timed out after 10s" in result.error_message
        assert "timed out" in result.log
        assert result.raw_stdout == "partial out"
        assert result.raw_stderr == "partial err"

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_timeout_with_none_output(self, mock_run, task_instance):
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["echo"], timeout=10
        )
        adapter = EchoAdapter(timeout_seconds=10)
        result = adapter.solve(task_instance)

        assert result.status == RunStatus.TIMEOUT
        assert result.raw_stdout == ""
        assert result.raw_stderr == ""

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_generic_exception(self, mock_run, task_instance):
        mock_run.side_effect = OSError("command not found")
        adapter = EchoAdapter()
        result = adapter.solve(task_instance)

        assert result.status == RunStatus.FAILED
        assert "command not found" in result.error_message
        assert "command not found" in result.log
        assert result.raw_stdout is None
        assert result.raw_stderr is None
        assert result.raw_stdout_format is None
        assert result.code_files == []
        assert result.data_files == []

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_timeout_still_collects_files(self, mock_run, task_instance):
        """Even on timeout, output files produced before timeout should be collected."""
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["echo"], timeout=10, output="", stderr=""
        )
        (task_instance.workspace_dir / "output.npy").write_bytes(b"data")
        adapter = EchoAdapter(timeout_seconds=10)
        result = adapter.solve(task_instance)

        assert result.status == RunStatus.TIMEOUT
        assert "output.npy" in result.data_files


# ── Tests: _build_full_log() ─────────────────────────────────────────────────


class TestBuildFullLog:
    def test_empty_inputs(self):
        adapter = EchoAdapter()
        assert adapter._build_full_log("", "", 0) == ""

    def test_stderr_only(self):
        adapter = EchoAdapter()
        log = adapter._build_full_log("", "some error\n", 0)
        assert "[stderr]" in log
        assert "some error" in log

    def test_stdout_and_stderr_and_exit_code(self):
        adapter = EchoAdapter()
        log = adapter._build_full_log("out text", "err text", 2)
        assert "out text" in log
        assert "[stderr]" in log
        assert "err text" in log
        assert "[exit_code] 2" in log

    def test_none_returncode_no_exit_code(self):
        adapter = EchoAdapter()
        log = adapter._build_full_log("out", "err", None)
        assert "[exit_code]" not in log

    def test_zero_returncode_no_exit_code(self):
        adapter = EchoAdapter()
        log = adapter._build_full_log("out", "", 0)
        assert "[exit_code]" not in log

    def test_custom_parse_log(self):
        adapter = CustomLogAdapter()
        log = adapter._build_full_log("raw output", "", 0)
        assert "PARSED: raw output" in log


# ── Tests: shell mode ────────────────────────────────────────────────────────


class TestShellMode:
    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_shell_adapter(self, mock_run, task_instance):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        adapter = ShellAdapter(cmd="echo hello")
        adapter.solve(task_instance)

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["shell"] is True
        assert call_kwargs.args[0] == "echo hello"

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_non_shell_adapter(self, mock_run, task_instance):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        adapter = EchoAdapter()
        adapter.solve(task_instance)

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["shell"] is False
        assert call_kwargs.args[0] == ["echo", "hello"]


# ── Tests: custom cwd ────────────────────────────────────────────────────────


class TestCustomCwd:
    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_default_cwd_is_workspace(self, mock_run, task_instance):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter = EchoAdapter()
        adapter.solve(task_instance)

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["cwd"] == str(task_instance.workspace_dir)

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_custom_cwd(self, mock_run, task_instance, tmp_path):
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter = CustomCwdAdapter(cwd_path=custom_dir)
        adapter.solve(task_instance)

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["cwd"] == str(custom_dir)


# ── Tests: custom environment ────────────────────────────────────────────────


class TestCustomEnv:
    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_default_env_no_task_env(self, mock_run, task_instance):
        """Without task_env, default env is None (inherit parent)."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter = EchoAdapter()
        adapter.solve(task_instance)

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["env"] is None

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_custom_env(self, mock_run, task_instance):
        """CustomEnvAdapter adds CUSTOM_VAR."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter = CustomEnvAdapter()
        adapter.solve(task_instance)

        call_kwargs = mock_run.call_args
        env = call_kwargs.kwargs["env"]
        assert env["CUSTOM_VAR"] == "custom_value"

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_task_env_used_when_sandbox_task(self, mock_run, task_instance):
        """When task_env_manager returns an env, it's used."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        fake_env = {"PATH": "/sandbox/bin:/usr/bin", "VIRTUAL_ENV": "/sandbox"}
        fake_task_env = MagicMock()
        fake_task_env.build_subprocess_env.return_value = fake_env

        adapter = EchoAdapter()
        adapter.task_env_manager = MagicMock()
        adapter.task_env_manager.ensure_env.return_value = fake_task_env
        adapter.solve(task_instance)

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["env"] == fake_env


# ── Tests: collect_output_files() ────────────────────────────────────────────


class TestCollectOutputFiles:
    def test_no_files_exist(self, task_instance):
        files = collect_output_files(task_instance.workspace_dir, task_instance)
        assert files == []

    def test_some_files_exist(self, task_instance):
        (task_instance.workspace_dir / "simulation.py").write_text("code")
        files = collect_output_files(task_instance.workspace_dir, task_instance)
        assert files == ["simulation.py"]

    def test_all_files_exist(self, task_instance):
        (task_instance.workspace_dir / "simulation.py").write_text("code")
        (task_instance.workspace_dir / "output.npy").write_bytes(b"data")
        files = collect_output_files(task_instance.workspace_dir, task_instance)
        assert set(files) == {"simulation.py", "output.npy"}

    def test_empty_metadata(self, workspace, tmp_path):
        inst = TaskInstance(
            task_id="test",
            instance_id="test__seed0",
            task_dir=tmp_path,
            workspace_dir=workspace,
            reference_dir=tmp_path,
            prompt_level=PromptLevel.B1,
            parameters={},
            metadata={},
        )
        files = collect_output_files(workspace, inst)
        assert files == []

    def test_no_output_key(self, workspace, tmp_path):
        inst = TaskInstance(
            task_id="test",
            instance_id="test__seed0",
            task_dir=tmp_path,
            workspace_dir=workspace,
            reference_dir=tmp_path,
            prompt_level=PromptLevel.B1,
            parameters={},
            metadata={"other": "data"},
        )
        files = collect_output_files(workspace, inst)
        assert files == []


# ── Tests: get_task_environment() ────────────────────────────────────────────


class TestGetTaskEnvironment:
    def test_returns_none_when_no_manager(self, task_instance):
        result = get_task_environment(None, task_instance)
        assert result is None

    def test_calls_ensure_env_when_manager_set(self, task_instance):
        manager = MagicMock()
        manager.ensure_env.return_value = "fake_env"
        result = get_task_environment(manager, task_instance)
        assert result == "fake_env"
        manager.ensure_env.assert_called_once_with(task_instance.metadata)


# ── Tests: inheritance chain ─────────────────────────────────────────────────


class TestInheritanceChain:
    """Verify that all refactored adapters inherit from SubprocessAgentAdapter."""

    def test_cli_agent_inherits(self):
        from ai4sci_bench.adapters.cli_agent import CLIAgentAdapter
        assert issubclass(CLIAgentAdapter, SubprocessAgentAdapter)

    def test_claude_code_inherits(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        assert issubclass(ClaudeCodeCLIAdapter, SubprocessAgentAdapter)

    def test_codex_inherits(self):
        from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter
        assert issubclass(CodexCLIAdapter, SubprocessAgentAdapter)

    def test_direct_llm_does_not_inherit(self):
        """DirectLLMAdapter should NOT inherit SubprocessAgentAdapter."""
        from ai4sci_bench.adapters.direct_llm import DirectLLMAdapter
        assert not issubclass(DirectLLMAdapter, SubprocessAgentAdapter)

    def test_all_are_agent_adapter(self):
        """All adapters should inherit from AgentAdapter."""
        from ai4sci_bench.core.agent_interface import AgentAdapter
        from ai4sci_bench.adapters.cli_agent import CLIAgentAdapter
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter
        from ai4sci_bench.adapters.direct_llm import DirectLLMAdapter
        for cls in [CLIAgentAdapter, ClaudeCodeCLIAdapter, CodexCLIAdapter, DirectLLMAdapter]:
            assert issubclass(cls, AgentAdapter)


# ── Tests: package exports ───────────────────────────────────────────────────


class TestPackageExports:
    def test_subprocess_base_exported(self):
        from ai4sci_bench.adapters import SubprocessAgentAdapter as Cls
        assert Cls is SubprocessAgentAdapter

    def test_shared_helpers_importable(self):
        from ai4sci_bench.adapters.subprocess_base import (
            collect_output_files,
            get_task_environment,
        )
        assert callable(collect_output_files)
        assert callable(get_task_environment)


# ── Tests: CLIAgentAdapter refactored behavior ───────────────────────────────


class TestCLIAgentAdapterRefactored:
    """Verify CLIAgentAdapter preserves behavior after refactor."""

    def test_uses_shell_true(self):
        from ai4sci_bench.adapters.cli_agent import CLIAgentAdapter
        adapter = CLIAgentAdapter(cmd_template="echo {workspace}")
        assert adapter._use_shell() is True

    def test_only_supports_sandbox_none(self):
        from ai4sci_bench.adapters.cli_agent import CLIAgentAdapter
        adapter = CLIAgentAdapter(cmd_template="echo")
        with pytest.raises(ValueError, match="does not support"):
            adapter.setup({"sandbox": "task"})

    def test_backward_compat_timeout_attr(self):
        """CLIAgentAdapter(timeout=999) maps to timeout_seconds=999."""
        from ai4sci_bench.adapters.cli_agent import CLIAgentAdapter
        adapter = CLIAgentAdapter(cmd_template="echo", timeout=999)
        assert adapter.timeout_seconds == 999


# ── Tests: ClaudeCodeCLIAdapter refactored behavior ──────────────────────────


class TestClaudeCodeCLIAdapterRefactored:
    """Verify ClaudeCodeCLIAdapter preserves behavior after refactor."""

    def test_uses_shell_false(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        adapter = ClaudeCodeCLIAdapter()
        assert adapter._use_shell() is False

    def test_raw_stdout_format_jsonl(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        adapter = ClaudeCodeCLIAdapter()
        assert adapter._raw_stdout_format() == "jsonl"

    def test_setup_preserves_permission_mode(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        adapter = ClaudeCodeCLIAdapter(permission_mode="bypassPermissions")
        adapter.setup({})
        assert adapter.permission_mode == "bypassPermissions"

    def test_setup_overrides_permission_mode(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        adapter = ClaudeCodeCLIAdapter(permission_mode="bypassPermissions")
        adapter.setup({"permission_mode": "plan"})
        assert adapter.permission_mode == "plan"

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_disallowed_tools_in_command(self, mock_run, task_instance):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter = ClaudeCodeCLIAdapter(allow_external_tools=False)
        adapter.solve(task_instance)

        cmd = mock_run.call_args.args[0]
        assert "--tools" in cmd
        tools = cmd[cmd.index("--tools") + 1].split(",")
        assert "WebSearch" not in tools
        assert "WebFetch" not in tools
        assert "--strict-mcp-config" in cmd
        assert "--disable-slash-commands" in cmd

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_external_tools_allowed(self, mock_run, task_instance):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter = ClaudeCodeCLIAdapter(allow_external_tools=True)
        adapter.solve(task_instance)

        cmd = mock_run.call_args.args[0]
        assert "--disallowed-tools" not in cmd


# ── Tests: CodexCLIAdapter refactored behavior ──────────────────────────────


class TestCodexCLIAdapterRefactored:
    """Verify CodexCLIAdapter preserves behavior after refactor."""

    def test_raw_stdout_format_jsonl(self):
        from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter
        adapter = CodexCLIAdapter()
        assert adapter._raw_stdout_format() == "jsonl"

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_openai_api_key_stripped(self, mock_run, task_instance):
        from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        import os
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            adapter = CodexCLIAdapter()
            adapter.solve(task_instance)

        env = mock_run.call_args.kwargs["env"]
        assert "OPENAI_API_KEY" not in env

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_web_search_disabled_by_default(self, mock_run, task_instance):
        from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter = CodexCLIAdapter(allow_external_tools=False)
        adapter.solve(task_instance)

        cmd = mock_run.call_args.args[0]
        assert "--config" in cmd
        config_idx = cmd.index("--config")
        assert 'web_search="disabled"' in cmd[config_idx + 1]

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_web_search_not_disabled_when_allowed(self, mock_run, task_instance):
        from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter = CodexCLIAdapter(allow_external_tools=True)
        adapter.solve(task_instance)

        cmd = mock_run.call_args.args[0]
        assert "--config" not in cmd

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_legacy_full_auto_uses_host_bypass(self, mock_run, task_instance):
        from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter = CodexCLIAdapter(full_auto=True)
        adapter.solve(task_instance)
        cmd = mock_run.call_args.args[0]
        assert "--full-auto" not in cmd
        assert "--dangerously-bypass-approvals-and-sandbox" in cmd

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_no_full_auto_flag(self, mock_run, task_instance):
        from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter = CodexCLIAdapter(full_auto=False)
        adapter.solve(task_instance)
        assert "--full-auto" not in mock_run.call_args.args[0]


# ── Tests: DirectLLMAdapter uses shared helpers ──────────────────────────────


class TestDirectLLMSharedHelpers:
    def test_uses_shared_get_task_environment(self):
        from ai4sci_bench.adapters.direct_llm import DirectLLMAdapter
        adapter = DirectLLMAdapter()
        # Should return None when no manager set
        result = adapter._get_task_environment(MagicMock())
        assert result is None


# ── Tests: HTTPAgentAdapter security fix ─────────────────────────────────────


class TestHTTPAgentSecurityFix:
    def test_instance_id_not_in_payload(self, task_instance):
        """instance_id should NOT be sent in the HTTP payload (security fix)."""
        from ai4sci_bench.adapters.http_agent import HTTPAgentAdapter

        adapter = HTTPAgentAdapter(base_url="http://fake:8080")
        # Mock _post to capture the payload
        captured_payload = {}

        def fake_post(endpoint, payload, **kwargs):
            captured_payload.update(payload)
            return {
                "status": "completed",
                "code_files": {},
                "data_files": {},
                "log": "ok",
            }

        adapter._post = fake_post
        adapter.solve(task_instance)

        assert "instance_id" not in captured_payload
        assert "task_id" in captured_payload
        assert "prompt_level" in captured_payload
        assert "prompt" in captured_payload


# ── TODO-13: Signal exit code translation tests ──────────────────────────


class TestSignalExitCodeTranslation:
    """Tests for _format_exit_error signal translation."""

    def test_sigkill_exit_code_translated(self, workspace, task_instance):
        from ai4sci_bench.adapters.subprocess_base import _format_exit_error
        msg = _format_exit_error("TestAdapter", -9)
        assert "SIGKILL" in msg
        assert "OOM" in msg or "killed" in msg.lower()

    def test_sigterm_exit_code_translated(self, workspace, task_instance):
        from ai4sci_bench.adapters.subprocess_base import _format_exit_error
        msg = _format_exit_error("TestAdapter", -15)
        assert "SIGTERM" in msg

    def test_sigsegv_exit_code_translated(self, workspace, task_instance):
        from ai4sci_bench.adapters.subprocess_base import _format_exit_error
        msg = _format_exit_error("TestAdapter", -11)
        assert "SIGSEGV" in msg

    def test_positive_exit_code_unchanged(self, workspace, task_instance):
        from ai4sci_bench.adapters.subprocess_base import _format_exit_error
        msg = _format_exit_error("TestAdapter", 1)
        assert msg == "TestAdapter exited with code 1"
        assert "SIG" not in msg

    def test_unknown_signal_number_graceful(self, workspace, task_instance):
        from ai4sci_bench.adapters.subprocess_base import _format_exit_error
        msg = _format_exit_error("TestAdapter", -99)
        assert "exited with code -99" in msg
        assert "signal 99" in msg
