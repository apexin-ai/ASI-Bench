"""Tests for agent adapters."""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ai4sci_bench.core.agent_interface import AgentAdapter
from ai4sci_bench.core.types import AgentOutput, PromptLevel, RunStatus, TaskInstance
from ai4sci_bench.adapters.cli_agent import CLIAgentAdapter
from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
from ai4sci_bench.adapters.direct_llm import DirectLLMAdapter
from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter
from ai4sci_bench.runner.task_env import TaskEnvironment


class TestAdapterPackageExports:
    """Verify all adapters are importable from the package."""

    def test_codex_cli_importable_from_package(self):
        from ai4sci_bench.adapters import CodexCLIAdapter
        assert CodexCLIAdapter is not None

    def test_all_exports(self):
        import ai4sci_bench.adapters as adapters
        assert hasattr(adapters, "DirectLLMAdapter")
        assert hasattr(adapters, "ClaudeCodeCLIAdapter")
        assert hasattr(adapters, "CLIAgentAdapter")
        assert hasattr(adapters, "CodexCLIAdapter")


class TestAgentAdapterInterface:
    def test_abstract(self):
        """AgentAdapter is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            AgentAdapter()

    def test_subclass(self):
        class DummyAgent(AgentAdapter):
            def solve(self, task_instance):
                return AgentOutput(
                    instance_id=task_instance.instance_id,
                    output_dir=task_instance.workspace_dir,
                    code_files=[],
                    data_files=[],
                    log="done",
                    execution_time_seconds=0.1,
                    status=RunStatus.COMPLETED,
                )

        agent = DummyAgent()
        assert agent is not None
        agent.setup({})
        agent.teardown()


class TestCLIAgentAdapter:
    def test_init(self):
        adapter = CLIAgentAdapter(cmd_template="echo hello --workspace {workspace}")
        assert adapter.cmd_template == "echo hello --workspace {workspace}"
        assert adapter.timeout_seconds == 10800

    def test_solve_echo(self, sample_task_instance):
        """Test with a simple echo command."""
        adapter = CLIAgentAdapter(cmd_template="echo 'test output'", timeout=30)
        result = adapter.solve(sample_task_instance)
        assert result.status == RunStatus.COMPLETED
        assert "test output" in result.log
        assert result.raw_stdout is not None
        assert "test output" in result.raw_stdout
        assert result.raw_stdout_format == "log"

    def test_solve_failing_command(self, sample_task_instance):
        adapter = CLIAgentAdapter(cmd_template="exit 1", timeout=10)
        result = adapter.solve(sample_task_instance)
        assert result.status == RunStatus.FAILED

    def test_template_substitution(self, sample_task_instance):
        adapter = CLIAgentAdapter(
            cmd_template="echo task_id={task_id} level={prompt_level}",
            timeout=10,
        )
        result = adapter.solve(sample_task_instance)
        assert "task_id=physics.test_task" in result.log
        assert "level=b2" in result.log

    def test_timeout(self, sample_task_instance):
        from dataclasses import replace
        adapter = CLIAgentAdapter(cmd_template="sleep 10", timeout=1)
        inst = replace(sample_task_instance, effective_timeout_seconds=1)
        result = adapter.solve(inst)
        assert result.status == RunStatus.TIMEOUT

    def test_setup_rejects_task_sandbox(self):
        adapter = CLIAgentAdapter(cmd_template="echo hello")
        with pytest.raises(ValueError, match="does not support sandbox 'task'"):
            adapter.setup({"sandbox": "task"})

    @patch("ai4sci_bench.adapters.subprocess_base.validate_sandbox_mode")
    @patch("ai4sci_bench.adapters.subprocess_base.LinuxNSSandbox")
    def test_setup_accepts_linux_ns(self, mock_linux_ns_cls, mock_validate):
        adapter = CLIAgentAdapter(cmd_template="echo hello")
        adapter.setup({"sandbox": "linux_ns"})
        assert adapter.sandbox == "linux_ns"
        assert adapter._linux_ns_sandbox is mock_linux_ns_cls.return_value

    @patch("ai4sci_bench.adapters.subprocess_base.validate_sandbox_mode")
    @patch("ai4sci_bench.adapters.subprocess_base.get_task_environment", return_value=None)
    @patch("ai4sci_bench.adapters.subprocess_base.LinuxNSSandbox")
    def test_solve_linux_ns_uses_namespace_runner(
        self,
        mock_linux_ns_cls,
        mock_get_task_env,
        mock_validate,
        sample_task_instance,
    ):
        adapter = CLIAgentAdapter(cmd_template="echo 'test output'", timeout=30)
        adapter.setup({"sandbox": "linux_ns"})
        mock_linux_ns = mock_linux_ns_cls.return_value
        mock_linux_ns.run_agent.return_value = (True, "test output\n", "test output\n", "")

        with patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout") as mock_run:
            result = adapter.solve(sample_task_instance)

        assert result.status == RunStatus.COMPLETED
        assert "test output" in result.log
        mock_run.assert_not_called()
        mock_linux_ns.run_agent.assert_called_once()
        assert mock_linux_ns.run_agent.call_args.kwargs["shell"] is True

    def test_template_values_are_shell_quoted(self, tmp_dir, sample_task_dir):
        """Bug P3: substituted values must be shlex-quoted so that
        paths or ids containing shell metachars cannot break out of
        their argument slot.
        """
        import yaml
        # Use a workspace path that contains a space (legal on macOS/Linux)
        ws = tmp_dir / "space dir"
        ws.mkdir()
        (ws / "prompt.md").write_text("# prompt")

        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["_task_dir"] = sample_task_dir
        ti = TaskInstance(
            task_id="physics.test_task",
            instance_id="physics.test_task__space_id",
            task_dir=sample_task_dir,
            workspace_dir=ws,
            reference_dir=tmp_dir / "ref",
            prompt_level=PromptLevel.B2,
            parameters={},
            metadata=metadata,
        )

        # Capture the command actually passed to subprocess
        captured = {}
        real_run = subprocess.run

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            # Minimal stand-in so solve() completes without actually shelling out.
            class R:
                returncode = 0
                stdout = "ok"
                stderr = ""
            return R()

        adapter = CLIAgentAdapter(cmd_template="printf %s {workspace}", timeout=10)
        with patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout", side_effect=fake_run):
            adapter.solve(ti)

        # shlex.quote wraps the path with a space in single quotes.
        assert "'" in captured["cmd"], captured["cmd"]
        assert "space dir" in captured["cmd"]
        # And the final shell command remains a single string (shell=True path).
        assert isinstance(captured["cmd"], str)

    def test_template_values_reject_injection_via_instance_id(self, tmp_dir, sample_task_dir):
        """An instance_id containing shell metacharacters must be neutralized."""
        import yaml
        ws = tmp_dir / "ws"
        ws.mkdir()
        (ws / "prompt.md").write_text("# prompt")

        metadata = yaml.safe_load((sample_task_dir / "task.yaml").read_text())
        metadata["_task_dir"] = sample_task_dir
        # A malicious/typo instance_id that would normally inject a sub-command.
        evil = "inst; touch /tmp/ai4sci_pwned_$$"
        ti = TaskInstance(
            task_id="physics.test_task",
            instance_id=evil,
            task_dir=sample_task_dir,
            workspace_dir=ws,
            reference_dir=tmp_dir / "ref",
            prompt_level=PromptLevel.B2,
            parameters={},
            metadata=metadata,
        )

        adapter = CLIAgentAdapter(
            cmd_template="echo pre {instance_id} post",
            timeout=10,
        )
        result = adapter.solve(ti)
        # The echo output should contain the evil string literally, proving
        # the shell interpreted it as a single argument rather than splitting
        # it on `;`.
        assert result.status == RunStatus.COMPLETED
        assert evil in result.log


class TestClaudeCodeCLIAdapter:
    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_uses_claude_cmd_on_windows(self, mock_run, sample_task_instance):
        """Issue #5: On Windows, subprocess should invoke 'claude.cmd' not 'claude'."""
        adapter = ClaudeCodeCLIAdapter()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with patch("ai4sci_bench.adapters.claude_code_cli.os") as mock_os:
            mock_os.name = "nt"
            adapter.solve(sample_task_instance)

        cmd = mock_run.call_args.args[0]
        assert cmd[0] == "claude.cmd"

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_uses_claude_on_posix(self, mock_run, sample_task_instance):
        """On POSIX, subprocess should invoke 'claude' (no .cmd)."""
        adapter = ClaudeCodeCLIAdapter()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with patch("ai4sci_bench.adapters.claude_code_cli.os") as mock_os:
            mock_os.name = "posix"
            adapter.solve(sample_task_instance)

        cmd = mock_run.call_args.args[0]
        assert cmd[0] == "claude"

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_routes_through_graceful_timeout_helper(self, mock_run, sample_task_instance):
        """Issue #6: stdout is decoded as UTF-8 with errors='replace'.

        Encoding is now enforced inside run_subprocess_with_graceful_timeout
        (see tests/test_proc_util.py::test_decodes_invalid_utf8_without_crashing);
        solve() must route through that helper rather than calling subprocess.run
        directly, so the UTF-8 guarantee holds for every CLI adapter."""
        adapter = ClaudeCodeCLIAdapter()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        adapter.solve(sample_task_instance)

        mock_run.assert_called_once()

    def test_init(self):
        adapter = ClaudeCodeCLIAdapter(
            model="claude-sonnet-4-6",
            permission_mode="bypassPermissions",
        )
        assert adapter.model == "claude-sonnet-4-6"
        assert adapter.permission_mode == "bypassPermissions"
        assert adapter.allow_external_tools is False

    def test_init_allow_external_tools(self):
        adapter = ClaudeCodeCLIAdapter(allow_external_tools=True)
        assert adapter.allow_external_tools is True

    def test_setup_accepts_os_sandbox(self):
        adapter = ClaudeCodeCLIAdapter()
        adapter.setup({"sandbox": "os"})
        assert adapter.sandbox == "os"
        assert adapter._os_sandbox is not None

    def test_setup_rejects_unsupported_sandbox(self):
        adapter = ClaudeCodeCLIAdapter()
        with pytest.raises(ValueError, match="Unknown sandbox mode 'bogus'"):
            adapter.setup({"sandbox": "bogus"})

    # -- OS Sandbox solve path tests ----------------------------------

    @patch("ai4sci_bench.adapters.claude_code_cli.OSSandbox")
    def test_solve_os_sandbox_success(self, mock_os_sandbox_cls, sample_task_instance):
        """solve() with sandbox=os delegates to OSSandbox.run_agent()."""
        adapter = ClaudeCodeCLIAdapter()
        adapter.setup({"sandbox": "os"})
        # Replace the OSSandbox instance created during setup
        mock_sandbox = MagicMock()
        mock_sandbox.run_agent.return_value = (
            True, "agent completed", '{"type":"result"}', "", "sha256:abc"
        )
        adapter._os_sandbox = mock_sandbox

        result = adapter.solve(sample_task_instance)

        assert result.status == RunStatus.COMPLETED
        assert result.error_message is None
        assert result.raw_stdout == '{"type":"result"}'
        assert result.raw_stdout_format == "jsonl"
        mock_sandbox.run_agent.assert_called_once()
        call_kwargs = mock_sandbox.run_agent.call_args[1]
        assert call_kwargs["agent_type"] == "claude_code"
        assert call_kwargs["allow_external_tools"] is False

    @patch("ai4sci_bench.adapters.claude_code_cli.OSSandbox")
    def test_solve_os_sandbox_failure(self, mock_os_sandbox_cls, sample_task_instance):
        """solve() with sandbox=os returns FAILED on non-zero exit."""
        adapter = ClaudeCodeCLIAdapter()
        adapter.setup({"sandbox": "os"})
        mock_sandbox = MagicMock()
        mock_sandbox.run_agent.return_value = (
            False, "error: auth failed", "", "auth error", "sha256:abc"
        )
        adapter._os_sandbox = mock_sandbox

        result = adapter.solve(sample_task_instance)

        assert result.status == RunStatus.FAILED
        assert result.error_message is not None
        assert "error: auth failed" in result.error_message

    @patch("ai4sci_bench.adapters.claude_code_cli.OSSandbox")
    def test_solve_os_sandbox_timeout(self, mock_os_sandbox_cls, sample_task_instance):
        """solve() with sandbox=os returns TIMEOUT when OSSandbox reports timeout."""
        adapter = ClaudeCodeCLIAdapter(timeout_seconds=120)
        adapter.setup({"sandbox": "os"})
        mock_sandbox = MagicMock()
        mock_sandbox.run_agent.return_value = (
            False, "OS sandbox agent execution timed out (120s)", "", "", "sha256:abc"
        )
        adapter._os_sandbox = mock_sandbox

        result = adapter.solve(sample_task_instance)

        assert result.status == RunStatus.TIMEOUT

    @patch("ai4sci_bench.adapters.claude_code_cli.OSSandbox")
    def test_solve_os_sandbox_passes_allow_external_tools(
        self, mock_os_sandbox_cls, sample_task_instance
    ):
        """allow_external_tools=True should be forwarded to run_agent."""
        adapter = ClaudeCodeCLIAdapter(allow_external_tools=True)
        adapter.setup({"sandbox": "os"})
        mock_sandbox = MagicMock()
        mock_sandbox.run_agent.return_value = (True, "ok", "", "", "sha256:abc")
        adapter._os_sandbox = mock_sandbox

        adapter.solve(sample_task_instance)

        call_kwargs = mock_sandbox.run_agent.call_args[1]
        assert call_kwargs["allow_external_tools"] is True

    @patch("ai4sci_bench.adapters.claude_code_cli.OSSandbox")
    def test_solve_os_sandbox_stores_image_identity(
        self, mock_os_sandbox_cls, sample_task_instance
    ):
        """solve() should store the image identity for provenance."""
        adapter = ClaudeCodeCLIAdapter()
        adapter.setup({"sandbox": "os"})
        mock_sandbox = MagicMock()
        mock_sandbox.run_agent.return_value = (
            True, "ok", "", "", "sha256:deadbeef123"
        )
        adapter._os_sandbox = mock_sandbox

        adapter.solve(sample_task_instance)

        assert adapter._sandbox_image_identity == "sha256:deadbeef123"

    def test_build_os_agent_cmd_forces_bypass_permissions(self, sample_task_instance):
        """OS sandbox command must force bypassPermissions."""
        adapter = ClaudeCodeCLIAdapter()
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        assert "--permission-mode" in cmd
        idx = cmd.index("--permission-mode")
        assert cmd[idx + 1] == "bypassPermissions"

    def test_build_os_agent_cmd_uses_stream_json(self, sample_task_instance):
        """OS sandbox command should use stream-json output format."""
        adapter = ClaudeCodeCLIAdapter()
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        assert "--output-format" in cmd
        idx = cmd.index("--output-format")
        assert cmd[idx + 1] == "stream-json"

    def test_build_os_agent_cmd_blocks_external_tools_by_default(self, sample_task_instance):
        """Default restricted mode: --tools whitelist with 15 core tools."""
        adapter = ClaudeCodeCLIAdapter()
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        assert "--tools" in cmd
        idx = cmd.index("--tools")
        tools = cmd[idx + 1].split(",")
        assert set(tools) == {
            "Bash", "Read", "Write", "Edit", "Glob", "Grep", "TodoWrite",
            "Agent", "NotebookEdit", "ToolSearch",
            "Monitor", "TaskOutput", "TaskStop",
            "EnterWorktree", "ExitWorktree",
        }
        assert "--disallowed-tools" not in cmd
        assert "--strict-mcp-config" in cmd
        assert "--disable-slash-commands" in cmd

    def test_build_os_agent_cmd_allows_external_tools(self, sample_task_instance):
        """allow_external_tools=True: search mode adds WebSearch,WebFetch."""
        adapter = ClaudeCodeCLIAdapter(allow_external_tools=True)
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        assert "--tools" in cmd
        idx = cmd.index("--tools")
        tools = cmd[idx + 1].split(",")
        assert "WebSearch" in tools
        assert "WebFetch" in tools
        assert "--disallowed-tools" not in cmd
        assert "--strict-mcp-config" in cmd
        assert "--disable-slash-commands" in cmd

    def test_build_os_agent_cmd_prompt_is_last_arg(self, sample_task_instance):
        """Prompt text should be the last argument after --."""
        adapter = ClaudeCodeCLIAdapter()
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        prompt_text = (sample_task_instance.workspace_dir / "prompt.md").read_text()
        assert cmd[-1] == prompt_text
        assert cmd[-2] == "--"

    def test_build_os_agent_cmd_uses_correct_model(self, sample_task_instance):
        """Command should use the configured model."""
        adapter = ClaudeCodeCLIAdapter(model="claude-sonnet-4-6")
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-sonnet-4-6"

    @patch("ai4sci_bench.adapters.claude_code_cli.OSSandbox")
    def test_solve_os_sandbox_collects_output_files(
        self, mock_os_sandbox_cls, sample_task_instance
    ):
        """solve() should collect output files from workspace after run_agent."""
        adapter = ClaudeCodeCLIAdapter()
        adapter.setup({"sandbox": "os"})
        mock_sandbox = MagicMock()
        mock_sandbox.run_agent.return_value = (True, "ok", "", "", "sha256:abc")
        adapter._os_sandbox = mock_sandbox
        # Create an output file in workspace
        (sample_task_instance.workspace_dir / "output.npy").write_bytes(b"data")

        result = adapter.solve(sample_task_instance)

        assert "output.npy" in result.data_files

    @patch("ai4sci_bench.adapters.claude_code_cli.OSSandbox")
    def test_solve_os_sandbox_does_not_call_subprocess_directly(
        self, mock_os_sandbox_cls, sample_task_instance
    ):
        """In OS sandbox mode, solve() should NOT call subprocess.run directly."""
        adapter = ClaudeCodeCLIAdapter()
        adapter.setup({"sandbox": "os"})
        mock_sandbox = MagicMock()
        mock_sandbox.run_agent.return_value = (True, "ok", "", "", "sha256:abc")
        adapter._os_sandbox = mock_sandbox

        with patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout") as mock_run:
            adapter.solve(sample_task_instance)
            mock_run.assert_not_called()

    def test_parse_claude_log_json(self):
        adapter = ClaudeCodeCLIAdapter()
        log_data = json.dumps({
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "I will write the code now."},
                    {"type": "tool_use", "name": "Write", "input": {"path": "sim.py"}},
                ]
            }
        })
        result = adapter._parse_claude_log(log_data)
        assert "[Claude] I will write the code now." in result
        assert "[Tool] Write" in result

    def test_parse_claude_log_plain(self):
        adapter = ClaudeCodeCLIAdapter()
        result = adapter._parse_claude_log("plain text line")
        assert result == "plain text line"

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_blocks_external_tools_by_default(self, mock_run, sample_task_instance):
        """Default restricted mode: --tools whitelist via _build_command()."""
        adapter = ClaudeCodeCLIAdapter()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = adapter.solve(sample_task_instance)

        cmd = mock_run.call_args.args[0]
        assert "--tools" in cmd
        assert "--strict-mcp-config" in cmd
        assert "--disable-slash-commands" in cmd
        assert "--disallowed-tools" not in cmd
        assert "--verbose" in cmd
        assert result.status == RunStatus.COMPLETED

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_allows_external_tools_when_enabled(self, mock_run, sample_task_instance):
        """allow_external_tools=True: search mode with WebSearch,WebFetch."""
        adapter = ClaudeCodeCLIAdapter(allow_external_tools=True)
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        adapter.solve(sample_task_instance)

        cmd = mock_run.call_args.args[0]
        assert "--tools" in cmd
        idx = cmd.index("--tools")
        tools = cmd[idx + 1].split(",")
        assert "WebSearch" in tools
        assert "WebFetch" in tools
        assert "--disallowed-tools" not in cmd

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_uses_permission_mode_when_configured(self, mock_run, sample_task_instance):
        adapter = ClaudeCodeCLIAdapter(permission_mode="bypassPermissions")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        adapter.solve(sample_task_instance)

        cmd = mock_run.call_args.args[0]
        assert "--permission-mode" in cmd
        assert "bypassPermissions" in cmd

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_uses_task_sandbox_env_when_enabled(self, mock_run, sample_task_instance, tmp_dir):
        adapter = ClaudeCodeCLIAdapter()
        adapter.setup({"sandbox": "task", "repo_root": str(tmp_dir), "timeout": 123})
        fake_env = TaskEnvironment(
            env_dir=tmp_dir / "env",
            python_executable=tmp_dir / "env" / "bin" / "python",
            bin_dir=tmp_dir / "env" / "bin",
            cache_key="abc",
            python_requirement=">=3.11",
            packages=["taichi>=1.7.4"],
            cache_hit=True,
        )
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(adapter.task_env_manager, "ensure_env", return_value=fake_env):
            adapter.solve(sample_task_instance)

        env = mock_run.call_args.kwargs["env"]
        assert env["VIRTUAL_ENV"] == str(fake_env.env_dir)
        assert env["PATH"].startswith(str(fake_env.bin_dir))

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_includes_stderr_and_exit_code_on_failure(self, mock_run, sample_task_instance):
        adapter = ClaudeCodeCLIAdapter()
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout='{"type":"assistant","message":{"content":[{"type":"text","text":"thinking"}]}}\n',
            stderr="claude authentication failed",
        )

        result = adapter.solve(sample_task_instance)

        assert result.status == RunStatus.FAILED
        assert "claude authentication failed" in result.log
        assert "[exit_code] 1" in result.log
        assert result.error_message == "ClaudeCodeCLIAdapter exited with code 1"
        assert result.raw_stdout is not None
        assert "thinking" in result.raw_stdout
        assert result.raw_stderr == "claude authentication failed"
        assert result.raw_stdout_format == "jsonl"


    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_timeout_captures_partial_output(self, mock_run, sample_task_instance):
        """Timeout should capture partial stdout/stderr and set TIMEOUT status."""
        from dataclasses import replace
        adapter = ClaudeCodeCLIAdapter(timeout_seconds=5)
        inst = replace(sample_task_instance, effective_timeout_seconds=5)
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd="claude", timeout=5, output="partial output", stderr="partial stderr"
        )

        result = adapter.solve(inst)

        assert result.status == RunStatus.TIMEOUT
        assert "timed out after 5s" in result.log
        assert "partial stderr" in result.log
        assert result.error_message.startswith("ClaudeCodeCLIAdapter timed out after 5s")
        assert result.raw_stdout == "partial output"
        assert result.raw_stderr == "partial stderr"
        assert result.raw_stdout_format == "jsonl"

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_generic_exception(self, mock_run, sample_task_instance):
        """Non-subprocess exceptions should be caught and reported."""
        adapter = ClaudeCodeCLIAdapter()
        mock_run.side_effect = FileNotFoundError("claude: command not found")

        result = adapter.solve(sample_task_instance)

        assert result.status == RunStatus.FAILED
        assert "command not found" in result.log
        assert result.error_message == "claude: command not found"

    def test_build_log_empty_inputs(self):
        """Empty stdout/stderr and success returncode should produce empty log."""
        adapter = ClaudeCodeCLIAdapter()
        log = adapter._build_full_log("", "", 0)
        assert log == ""

    def test_build_log_only_stderr(self):
        adapter = ClaudeCodeCLIAdapter()
        log = adapter._build_full_log("", "some error\n", 0)
        assert "[stderr]" in log
        assert "some error" in log

    def test_build_log_combines_all_parts(self):
        adapter = ClaudeCodeCLIAdapter()
        log = adapter._build_full_log("plain stdout text", "error text", 2)
        assert "plain stdout text" in log
        assert "[stderr]" in log
        assert "error text" in log
        assert "[exit_code] 2" in log

    def test_parse_claude_log_malformed_json_lines(self):
        """Malformed JSON lines should be included as-is."""
        adapter = ClaudeCodeCLIAdapter()
        result = adapter._parse_claude_log("not json\n{bad json\nplain line")
        assert "not json" in result
        assert "{bad json" in result
        assert "plain line" in result

    def test_collect_output_files_missing_files(self, sample_task_instance):
        """Files listed in metadata but not in workspace should be excluded."""
        from ai4sci_bench.adapters.subprocess_base import collect_output_files
        # sample_task_instance has output.npy in metadata but doesn't exist on disk
        files = collect_output_files(
            sample_task_instance.workspace_dir, sample_task_instance
        )
        assert files == []

    def test_collect_output_files_existing(self, sample_task_instance):
        """Files that exist in workspace should be collected."""
        from ai4sci_bench.adapters.subprocess_base import collect_output_files
        (sample_task_instance.workspace_dir / "output.npy").write_bytes(b"data")
        files = collect_output_files(
            sample_task_instance.workspace_dir, sample_task_instance
        )
        assert "output.npy" in files

    # -- Bug #3: summary log must surface tool_use args, tool_result,
    #    system, error and rate_limit events --------------------------

    def test_parse_claude_log_tool_use_highlights_file_path(self):
        """tool_use summary should surface key arg values (not just keys)."""
        adapter = ClaudeCodeCLIAdapter()
        event = json.dumps({
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Write",
                        "input": {
                            "file_path": "/workspace/simulation.py",
                            "content": "print('hi')",
                        },
                    }
                ]
            },
        })
        result = adapter._parse_claude_log(event)
        assert "[Tool] Write" in result
        assert "file_path=/workspace/simulation.py" in result

    def test_parse_claude_log_tool_use_highlights_bash_command(self):
        adapter = ClaudeCodeCLIAdapter()
        event = json.dumps({
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"command": "python simulation.py --size 128"},
                    }
                ]
            },
        })
        result = adapter._parse_claude_log(event)
        assert "command=python simulation.py --size 128" in result

    def test_parse_claude_log_captures_tool_result(self):
        """user-message tool_result events must appear in the summary."""
        adapter = ClaudeCodeCLIAdapter()
        event = json.dumps({
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "content": [
                            {"type": "text", "text": "File written successfully"},
                        ],
                    }
                ]
            },
        })
        result = adapter._parse_claude_log(event)
        assert "[ToolResult] File written successfully" in result

    def test_parse_claude_log_flags_tool_result_errors(self):
        adapter = ClaudeCodeCLIAdapter()
        event = json.dumps({
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "is_error": True,
                        "content": "ModuleNotFoundError: taichi",
                    }
                ]
            },
        })
        result = adapter._parse_claude_log(event)
        assert "[ToolResult:err]" in result
        assert "ModuleNotFoundError" in result

    def test_parse_claude_log_captures_system_events(self):
        adapter = ClaudeCodeCLIAdapter()
        event = json.dumps({"type": "system", "subtype": "init", "message": "session started"})
        result = adapter._parse_claude_log(event)
        assert "[system:init]" in result
        assert "session started" in result

    def test_parse_claude_log_captures_error_and_rate_limit(self):
        adapter = ClaudeCodeCLIAdapter()
        log = "\n".join([
            json.dumps({"type": "error", "message": "auth failed"}),
            json.dumps({"type": "rate_limit_event", "message": "limit hit: 100/100"}),
        ])
        result = adapter._parse_claude_log(log)
        assert "[error] auth failed" in result
        assert "[rate_limit] limit hit: 100/100" in result

    def test_parse_claude_log_no_truncation_by_default(self):
        """Summary log should preserve full text content without truncation."""
        adapter = ClaudeCodeCLIAdapter()
        long_text = "A" * 5000
        event = json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": long_text}]},
        })
        result = adapter._parse_claude_log(event)
        assert result.count("A") == 5000
        assert result.count("[Claude]") == 1

    def test_parse_claude_log_blank_lines_are_skipped(self):
        adapter = ClaudeCodeCLIAdapter()
        result = adapter._parse_claude_log("\n\n\n")
        assert result == ""

    # -- Tool isolation: comprehensive tests ----------------------------

    def test_restricted_mode_whitelist_has_exactly_15_tools(self, sample_task_instance):
        adapter = ClaudeCodeCLIAdapter()
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        idx = cmd.index("--tools")
        tools = cmd[idx + 1].split(",")
        assert len(tools) == 15

    def test_restricted_whitelist_excludes_external_tools(self, sample_task_instance):
        adapter = ClaudeCodeCLIAdapter()
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        idx = cmd.index("--tools")
        tools = cmd[idx + 1].split(",")
        for excluded in ("WebSearch", "WebFetch", "Skill", "AskUserQuestion",
                         "CronCreate", "CronDelete", "CronList",
                         "PushNotification", "RemoteTrigger", "ShareOnboardingGuide",
                         "ScheduleWakeup", "EnterPlanMode", "ExitPlanMode"):
            assert excluded not in tools

    def test_search_mode_whitelist_has_17_tools(self, sample_task_instance):
        adapter = ClaudeCodeCLIAdapter(allow_external_tools=True)
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        idx = cmd.index("--tools")
        tools = cmd[idx + 1].split(",")
        assert len(tools) == 17
        assert "WebSearch" in tools
        assert "WebFetch" in tools

    def test_search_mode_still_has_mcp_and_slash_restrictions(self, sample_task_instance):
        adapter = ClaudeCodeCLIAdapter(allow_external_tools=True)
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        assert "--strict-mcp-config" in cmd
        assert "--disable-slash-commands" in cmd

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_build_command_restricted_matches_os_cmd(self, mock_run, sample_task_instance):
        """_build_command() and _build_os_agent_cmd() tool restriction logic must be consistent."""
        adapter = ClaudeCodeCLIAdapter()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter.solve(sample_task_instance)
        cmd = mock_run.call_args.args[0]
        assert "--tools" in cmd
        assert "--strict-mcp-config" in cmd
        assert "--disable-slash-commands" in cmd
        assert "--disallowed-tools" not in cmd

    def test_tool_mode_from_allow_external_tools_false(self):
        adapter = ClaudeCodeCLIAdapter(allow_external_tools=False)
        from ai4sci_bench.core.types import ToolMode
        assert adapter.tool_mode == ToolMode.RESTRICTED

    def test_tool_mode_from_allow_external_tools_true(self):
        adapter = ClaudeCodeCLIAdapter(allow_external_tools=True)
        from ai4sci_bench.core.types import ToolMode
        assert adapter.tool_mode == ToolMode.SEARCH

    def test_tool_mode_explicit_overrides_allow_external_tools(self):
        adapter = ClaudeCodeCLIAdapter(allow_external_tools=True, tool_mode="restricted")
        from ai4sci_bench.core.types import ToolMode
        assert adapter.tool_mode == ToolMode.RESTRICTED

    def test_tool_mode_unrestricted_no_restrictions(self, sample_task_instance):
        adapter = ClaudeCodeCLIAdapter(tool_mode="unrestricted")
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        assert "--tools" not in cmd
        assert "--strict-mcp-config" not in cmd
        assert "--disable-slash-commands" not in cmd
        assert "--disallowed-tools" not in cmd


class TestDirectLLMAdapter:
    def test_init(self):
        adapter = DirectLLMAdapter(model="openai/gpt-4o")
        assert adapter.model == "openai/gpt-4o"
        assert "expert scientific programmer" in adapter.system_prompt
        assert adapter.api_key is None
        assert adapter.api_base is None

    def test_init_with_api_key_and_base(self):
        adapter = DirectLLMAdapter(
            model="anthropic/claude-3.5-sonnet",
            api_key="sk-or-test-key",
            api_base="https://openrouter.ai/api/v1",
            api_protocol="openai",
        )
        assert adapter.model == "openai/anthropic/claude-3.5-sonnet"
        assert adapter.api_key == "sk-or-test-key"
        assert adapter.api_base == "https://openrouter.ai/api/v1"

    def test_init_openrouter_model_without_explicit_key(self):
        """OpenRouter model without explicit api_key should rely on env var."""
        adapter = DirectLLMAdapter(model="openrouter/anthropic/claude-3.5-sonnet")
        assert adapter.model == "openrouter/anthropic/claude-3.5-sonnet"
        assert adapter.api_key is None

    def test_setup_rejects_unsupported_sandbox(self):
        adapter = DirectLLMAdapter()
        with pytest.raises(ValueError, match="Unknown sandbox mode 'bogus'"):
            adapter.setup({"sandbox": "bogus"})

    @patch("litellm.completion")
    def test_solve_passes_api_key_to_litellm(self, mock_completion, sample_task_instance):
        """When api_key/api_base are set, they should be forwarded to litellm.completion."""
        adapter = DirectLLMAdapter(
            model="anthropic/claude-3.5-sonnet",
            api_key="sk-or-test",
            api_base="https://openrouter.ai/api/v1",
            api_protocol="openai",
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "```python\nprint('hi')\n```"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_completion.return_value = mock_response

        with patch.object(adapter, "_execute", return_value=(True, "ok", "stdout", "stderr")):
            result = adapter.solve(sample_task_instance)

        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["api_key"] == "sk-or-test"
        assert call_kwargs.kwargs["api_base"] == "https://openrouter.ai/api/v1"
        assert result.raw_model_output_format == "json"
        import json as _json
        structured = _json.loads(result.raw_model_output)
        assert isinstance(structured, list)
        assert structured[0]["step"] == "prompt"

    @patch("litellm.completion")
    def test_solve_omits_api_key_when_not_set(self, mock_completion, sample_task_instance):
        """When api_key/api_base are None, they should not appear in litellm.completion kwargs."""
        adapter = DirectLLMAdapter(model="openai/gpt-4o")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "```python\nprint('hi')\n```"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_completion.return_value = mock_response

        with patch.object(adapter, "_execute", return_value=(True, "ok", "stdout", "stderr")):
            result = adapter.solve(sample_task_instance)

        call_kwargs = mock_completion.call_args
        assert "api_key" not in call_kwargs.kwargs
        assert "api_base" not in call_kwargs.kwargs
        assert result.raw_model_output_format == "json"

    def test_extract_code_python_block(self):
        adapter = DirectLLMAdapter()
        content = "Here is the code:\n```python\nprint('hello')\n```\nDone."
        code = adapter._extract_code(content)
        assert code == "print('hello')"

    def test_extract_code_generic_block(self):
        adapter = DirectLLMAdapter()
        content = "```\nfoo = bar\n```"
        code = adapter._extract_code(content)
        assert code == "foo = bar"

    def test_extract_code_no_block(self):
        adapter = DirectLLMAdapter()
        content = "just plain code\nno blocks"
        code = adapter._extract_code(content)
        assert code == "just plain code\nno blocks"

    def test_extract_code_unwraps_analysis_writer_wrapper(self):
        adapter = DirectLLMAdapter()
        content = """```python
from pathlib import Path

code = r'''import numpy as np

np.save("potential.npy", np.zeros((2, 2), dtype=np.float32))
'''
Path("analysis.py").write_text(code)
```"""
        code = adapter._extract_code(
            content,
            code_file="analysis.py",
            expected_output_files=["analysis.py", "potential.npy"],
        )
        assert 'Path("analysis.py").write_text(code)' not in code
        assert 'np.save("potential.npy"' in code

    def test_extract_code_prefers_runnable_candidate_over_wrapper(self):
        adapter = DirectLLMAdapter()
        content = """```python
from pathlib import Path

code = r'''print("wrapper only")'''
Path("analysis.py").write_text(code)
```
```python
import numpy as np

np.save("potential.npy", np.zeros((2, 2), dtype=np.float32))
```
"""
        code = adapter._extract_code(
            content,
            code_file="analysis.py",
            expected_output_files=["analysis.py", "potential.npy"],
        )
        assert 'np.save("potential.npy"' in code
        assert 'Path("analysis.py").write_text(code)' not in code

    def test_execute_runs_script_from_workspace(self, tmp_dir):
        adapter = DirectLLMAdapter()
        workspace = tmp_dir / "workspace"
        workspace.mkdir()
        (workspace / "script.py").write_text("print('ok')", encoding="utf-8")
        task_instance = TaskInstance(
            task_id="test.task",
            instance_id="test.instance",
            task_dir=tmp_dir,
            workspace_dir=workspace,
            reference_dir=tmp_dir / "reference",
            prompt_level=PromptLevel.B2,
            parameters={},
            metadata={"output": {"files": []}},
        )

        success, log, raw_stdout, raw_stderr = adapter._execute(task_instance, "script.py")

        assert success is True
        assert "ok" in log
        assert raw_stdout is not None
        assert "ok" in raw_stdout
        assert raw_stderr == ""

    @patch("ai4sci_bench.adapters.direct_llm.subprocess.run")
    def test_execute_uses_task_sandbox_python(self, mock_run, tmp_dir):
        adapter = DirectLLMAdapter()
        adapter.setup({"sandbox": "task", "repo_root": str(tmp_dir), "timeout": 77})
        workspace = tmp_dir / "workspace"
        workspace.mkdir()
        reference = tmp_dir / "reference"
        reference.mkdir()
        task_instance = TaskInstance(
            task_id="test.task",
            instance_id="test.instance",
            task_dir=tmp_dir,
            workspace_dir=workspace,
            reference_dir=reference,
            prompt_level=PromptLevel.B2,
            parameters={},
            metadata={"output": {"files": []}},
            effective_timeout_seconds=77,
        )
        fake_env = TaskEnvironment(
            env_dir=tmp_dir / "env",
            python_executable=tmp_dir / "env" / "bin" / "python",
            bin_dir=tmp_dir / "env" / "bin",
            cache_key="abc",
            python_requirement=">=3.11",
            packages=["taichi>=1.7.4"],
            cache_hit=True,
        )
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")

        with patch.object(adapter.task_env_manager, "ensure_env", return_value=fake_env):
            success, _, raw_stdout, raw_stderr = adapter._execute(task_instance, "script.py")

        assert success is True
        cmd = mock_run.call_args.args[0]
        assert cmd[0] == str(fake_env.python_executable)
        assert mock_run.call_args.kwargs["timeout"] == 77
        assert mock_run.call_args.kwargs["env"]["VIRTUAL_ENV"] == str(fake_env.env_dir)
        assert raw_stdout == "ok"
        assert raw_stderr == ""

    @patch("ai4sci_bench.adapters.direct_llm.OSSandbox")
    def test_execute_uses_os_sandbox(self, mock_os_sandbox, tmp_dir):
        adapter = DirectLLMAdapter()
        adapter.setup({"sandbox": "os", "repo_root": str(tmp_dir), "timeout": 88})
        workspace = tmp_dir / "workspace"
        workspace.mkdir()
        reference = tmp_dir / "reference"
        reference.mkdir()
        task_instance = TaskInstance(
            task_id="test.task",
            instance_id="test.instance",
            task_dir=tmp_dir,
            workspace_dir=workspace,
            reference_dir=reference,
            prompt_level=PromptLevel.B2,
            parameters={},
            metadata={"output": {"files": []}},
        )
        mock_os_sandbox.return_value.execute_python.return_value = (
            True,
            "ok",
            "stdout",
            "stderr",
            "sha256:test-image",
        )

        success, log, raw_stdout, raw_stderr = adapter._execute(task_instance, "script.py")

        assert success is True
        assert log == "ok"
        assert raw_stdout == "stdout"
        assert raw_stderr == "stderr"
        mock_os_sandbox.assert_called_once_with(tmp_dir)
        mock_os_sandbox.return_value.execute_python.assert_called_once()
        assert adapter.sandbox_image_identity == "sha256:test-image"

    @patch("ai4sci_bench.adapters.direct_llm.validate_sandbox_mode")
    @patch("ai4sci_bench.adapters.direct_llm.LinuxNSSandbox")
    def test_execute_uses_linux_ns_sandbox(self, mock_linux_ns_cls, mock_validate, tmp_dir):
        adapter = DirectLLMAdapter()
        adapter.setup({"sandbox": "linux_ns", "repo_root": str(tmp_dir), "timeout": 66})
        workspace = tmp_dir / "workspace"
        workspace.mkdir()
        reference = tmp_dir / "reference"
        reference.mkdir()
        task_instance = TaskInstance(
            task_id="test.task",
            instance_id="test.instance",
            task_dir=tmp_dir,
            workspace_dir=workspace,
            reference_dir=reference,
            prompt_level=PromptLevel.B2,
            parameters={},
            metadata={"output": {"files": []}},
        )
        fake_env = MagicMock()
        fake_env.python_executable = tmp_dir / "env" / "bin" / "python"
        fake_env.build_subprocess_env.return_value = {"VIRTUAL_ENV": str(tmp_dir / "env")}
        mock_linux_ns_cls.return_value.execute_python.return_value = (
            True,
            "ok",
            "stdout",
            "stderr",
        )

        with patch.object(adapter.task_env_manager, "ensure_env", return_value=fake_env):
            success, log, raw_stdout, raw_stderr = adapter._execute(task_instance, "script.py")

        assert success is True
        assert log == "ok"
        assert raw_stdout == "stdout"
        assert raw_stderr == "stderr"
        mock_linux_ns_cls.return_value.execute_python.assert_called_once()
        call_kwargs = mock_linux_ns_cls.return_value.execute_python.call_args.kwargs
        assert call_kwargs["workspace"] == workspace
        assert call_kwargs["python_executable"] == str(fake_env.python_executable)
        assert call_kwargs["extra_env"]["VIRTUAL_ENV"] == str(tmp_dir / "env")


class TestCodexCLIAdapter:
    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_sends_prompt_via_stdin(self, mock_run, sample_task_instance):
        """Issue #12: prompt should be passed via stdin (not CLI arg) to avoid
        Windows command-line length limits."""
        adapter = CodexCLIAdapter()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        adapter.solve(sample_task_instance)

        kwargs = mock_run.call_args.kwargs
        # Prompt should be sent via stdin (input=)
        assert kwargs.get("input") is not None
        assert len(kwargs["input"]) > 0

    def test_init_defaults(self):
        adapter = CodexCLIAdapter()
        assert adapter.model == "gpt-5.5"
        assert adapter.full_auto is True
        assert adapter.timeout_seconds == 10800
        assert adapter.allow_external_tools is False

    def test_init_custom(self):
        adapter = CodexCLIAdapter(
            model="o4-mini",
            full_auto=False,
            timeout_seconds=600,
        )
        assert adapter.model == "o4-mini"
        assert adapter.full_auto is False
        assert adapter.timeout_seconds == 600

    def test_init_allow_external_tools(self):
        adapter = CodexCLIAdapter(allow_external_tools=True)
        assert adapter.allow_external_tools is True

    def test_setup_accepts_os_sandbox(self):
        adapter = CodexCLIAdapter()
        adapter.setup({"sandbox": "os"})
        assert adapter.sandbox == "os"
        assert adapter._os_sandbox is not None

    @patch("ai4sci_bench.adapters.subprocess_base.validate_sandbox_mode")
    @patch("ai4sci_bench.adapters.subprocess_base.LinuxNSSandbox")
    def test_setup_accepts_linux_ns_sandbox(self, mock_linux_ns_cls, mock_validate):
        adapter = CodexCLIAdapter()
        adapter.setup({"sandbox": "linux_ns"})
        assert adapter.sandbox == "linux_ns"
        assert adapter._linux_ns_sandbox is mock_linux_ns_cls.return_value

    def test_setup_rejects_unsupported_sandbox(self):
        adapter = CodexCLIAdapter()
        with pytest.raises(ValueError, match="Unknown sandbox mode 'bogus'"):
            adapter.setup({"sandbox": "bogus"})

    # -- OS Sandbox solve path tests ----------------------------------

    @patch("ai4sci_bench.adapters.codex_cli.OSSandbox")
    def test_solve_os_sandbox_success(self, mock_os_sandbox_cls, sample_task_instance):
        """solve() with sandbox=os delegates to OSSandbox.run_agent()."""
        adapter = CodexCLIAdapter()
        adapter.setup({"sandbox": "os"})
        mock_sandbox = MagicMock()
        mock_sandbox.run_agent.return_value = (
            True, "agent completed", '{"type":"message"}', "", "sha256:abc"
        )
        adapter._os_sandbox = mock_sandbox

        result = adapter.solve(sample_task_instance)

        assert result.status == RunStatus.COMPLETED
        assert result.error_message is None
        assert result.raw_stdout == '{"type":"message"}'
        assert result.raw_stdout_format == "jsonl"
        mock_sandbox.run_agent.assert_called_once()
        call_kwargs = mock_sandbox.run_agent.call_args[1]
        assert call_kwargs["agent_type"] == "codex"
        assert call_kwargs["allow_external_tools"] is False

    @patch("ai4sci_bench.adapters.subprocess_base.validate_sandbox_mode")
    @patch("ai4sci_bench.adapters.subprocess_base.get_task_environment", return_value=None)
    @patch("ai4sci_bench.adapters.subprocess_base.LinuxNSSandbox")
    def test_solve_linux_ns_uses_namespace_runner(
        self,
        mock_linux_ns_cls,
        mock_get_task_env,
        mock_validate,
        sample_task_instance,
    ):
        adapter = CodexCLIAdapter()
        adapter.setup({"sandbox": "linux_ns"})
        mock_linux_ns = mock_linux_ns_cls.return_value
        stdout = '{"type":"message","role":"assistant","content":"ok"}'
        mock_linux_ns.run_agent.return_value = (True, stdout, stdout, "")

        with patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout") as mock_run:
            result = adapter.solve(sample_task_instance)

        assert result.status == RunStatus.COMPLETED
        assert result.raw_stdout == stdout
        mock_run.assert_not_called()
        mock_linux_ns.run_agent.assert_called_once()
        call_kwargs = mock_linux_ns.run_agent.call_args.kwargs
        assert call_kwargs["shell"] is False
        assert call_kwargs["input_text"] is not None

    @patch("ai4sci_bench.adapters.codex_cli.OSSandbox")
    def test_solve_os_sandbox_failure(self, mock_os_sandbox_cls, sample_task_instance):
        """solve() with sandbox=os returns FAILED on non-zero exit."""
        adapter = CodexCLIAdapter()
        adapter.setup({"sandbox": "os"})
        mock_sandbox = MagicMock()
        mock_sandbox.run_agent.return_value = (
            False, "error: codex auth failed", "", "auth error", "sha256:abc"
        )
        adapter._os_sandbox = mock_sandbox

        result = adapter.solve(sample_task_instance)

        assert result.status == RunStatus.FAILED
        assert result.error_message is not None

    @patch("ai4sci_bench.adapters.codex_cli.OSSandbox")
    def test_solve_os_sandbox_timeout(self, mock_os_sandbox_cls, sample_task_instance):
        """solve() with sandbox=os returns TIMEOUT when OSSandbox reports timeout."""
        adapter = CodexCLIAdapter(timeout_seconds=120)
        adapter.setup({"sandbox": "os"})
        mock_sandbox = MagicMock()
        mock_sandbox.run_agent.return_value = (
            False, "OS sandbox agent execution timed out (120s)", "", "", "sha256:abc"
        )
        adapter._os_sandbox = mock_sandbox

        result = adapter.solve(sample_task_instance)

        assert result.status == RunStatus.TIMEOUT

    @patch("ai4sci_bench.adapters.codex_cli.OSSandbox")
    def test_solve_os_sandbox_passes_allow_external_tools(
        self, mock_os_sandbox_cls, sample_task_instance
    ):
        """allow_external_tools=True should be forwarded to run_agent."""
        adapter = CodexCLIAdapter(allow_external_tools=True)
        adapter.setup({"sandbox": "os"})
        mock_sandbox = MagicMock()
        mock_sandbox.run_agent.return_value = (True, "ok", "", "", "sha256:abc")
        adapter._os_sandbox = mock_sandbox

        adapter.solve(sample_task_instance)

        call_kwargs = mock_sandbox.run_agent.call_args[1]
        assert call_kwargs["allow_external_tools"] is True

    @patch("ai4sci_bench.adapters.codex_cli.OSSandbox")
    def test_solve_os_sandbox_stores_image_identity(
        self, mock_os_sandbox_cls, sample_task_instance
    ):
        """solve() should store the image identity for provenance."""
        adapter = CodexCLIAdapter()
        adapter.setup({"sandbox": "os"})
        mock_sandbox = MagicMock()
        mock_sandbox.run_agent.return_value = (
            True, "ok", "", "", "sha256:deadbeef123"
        )
        adapter._os_sandbox = mock_sandbox

        adapter.solve(sample_task_instance)

        assert adapter._sandbox_image_identity == "sha256:deadbeef123"

    def test_build_os_agent_cmd_disables_bwrap(self, sample_task_instance):
        """OS sandbox command must disable built-in bwrap sandbox."""
        adapter = CodexCLIAdapter()
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        assert "--dangerously-bypass-approvals-and-sandbox" in cmd

    def test_build_os_agent_cmd_uses_workspace_cd(self, sample_task_instance):
        """OS sandbox command should cd to /workspace (container path)."""
        adapter = CodexCLIAdapter()
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        idx = cmd.index("--cd")
        assert cmd[idx + 1] == "/workspace"

    def test_build_os_agent_cmd_uses_json_output(self, sample_task_instance):
        """OS sandbox command should use --json output."""
        adapter = CodexCLIAdapter()
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        assert "--json" in cmd

    def test_build_os_agent_cmd_blocks_external_tools_by_default(self, sample_task_instance):
        """Default restricted mode: web_search disabled + feature flags disabled."""
        adapter = CodexCLIAdapter()
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        assert "--config" in cmd
        idx = cmd.index("--config")
        assert cmd[idx + 1] == 'web_search="disabled"'
        disabled = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--disable"]
        assert "plugins" in disabled
        assert "tool_call_mcp_elicitation" in disabled
        assert "skill_mcp_dependency_install" in disabled
        assert "image_generation" in disabled
        assert "multi_agent" not in disabled

    def test_build_os_agent_cmd_allows_external_tools(self, sample_task_instance):
        """allow_external_tools=True: search mode allows web search but still disables plugins."""
        adapter = CodexCLIAdapter(allow_external_tools=True)
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        assert 'web_search="disabled"' not in " ".join(cmd)
        disabled = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--disable"]
        assert "plugins" in disabled
        assert "image_generation" in disabled
        assert "multi_agent" not in disabled

    def test_build_os_agent_cmd_omits_full_auto_in_os_mode(self, sample_task_instance):
        """OS sandbox uses --dangerously-bypass which conflicts with --full-auto."""
        adapter = CodexCLIAdapter(full_auto=True)
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        # --full-auto is intentionally omitted in OS sandbox mode because
        # --dangerously-bypass-approvals-and-sandbox conflicts with it.
        assert "--full-auto" not in cmd
        assert "--dangerously-bypass-approvals-and-sandbox" in cmd

    @patch("ai4sci_bench.adapters.subprocess_base.validate_sandbox_mode")
    @patch("ai4sci_bench.adapters.subprocess_base.LinuxNSSandbox")
    def test_build_command_linux_ns_disables_internal_sandbox(
        self,
        mock_linux_ns_cls,
        mock_validate,
        sample_task_instance,
    ):
        adapter = CodexCLIAdapter(full_auto=True)
        adapter.setup({"sandbox": "linux_ns"})

        cmd = adapter._build_command(sample_task_instance, task_env=None)

        assert "--dangerously-bypass-approvals-and-sandbox" in cmd
        assert "--full-auto" not in cmd

    @patch("ai4sci_bench.adapters.subprocess_base.validate_sandbox_mode")
    @patch("ai4sci_bench.adapters.subprocess_base.LinuxNSSandbox")
    def test_build_command_linux_ns_keeps_absolute_workspace_cd(
        self,
        mock_linux_ns_cls,
        mock_validate,
        sample_task_instance,
    ):
        adapter = CodexCLIAdapter()
        adapter.setup({"sandbox": "linux_ns"})

        cmd = adapter._build_command(sample_task_instance, task_env=None)

        idx = cmd.index("--cd")
        assert cmd[idx + 1] == str(sample_task_instance.workspace_dir.resolve())

    def test_build_os_agent_cmd_prompt_is_last_arg(self, sample_task_instance):
        """Prompt text should be the last argument after --."""
        adapter = CodexCLIAdapter()
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        prompt_text = (sample_task_instance.workspace_dir / "prompt.md").read_text()
        assert cmd[-1] == prompt_text
        assert cmd[-2] == "--"

    def test_build_os_agent_cmd_uses_correct_model(self, sample_task_instance):
        """Command should use the configured model."""
        adapter = CodexCLIAdapter(model="o3")
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "o3"

    @patch("ai4sci_bench.adapters.codex_cli.OSSandbox")
    def test_solve_os_sandbox_collects_output_files(
        self, mock_os_sandbox_cls, sample_task_instance
    ):
        """solve() should collect output files from workspace after run_agent."""
        adapter = CodexCLIAdapter()
        adapter.setup({"sandbox": "os"})
        mock_sandbox = MagicMock()
        mock_sandbox.run_agent.return_value = (True, "ok", "", "", "sha256:abc")
        adapter._os_sandbox = mock_sandbox
        (sample_task_instance.workspace_dir / "output.npy").write_bytes(b"data")

        result = adapter.solve(sample_task_instance)

        assert "output.npy" in result.data_files

    @patch("ai4sci_bench.adapters.codex_cli.OSSandbox")
    def test_solve_os_sandbox_does_not_call_subprocess_directly(
        self, mock_os_sandbox_cls, sample_task_instance
    ):
        """In OS sandbox mode, solve() should NOT call subprocess.run directly."""
        adapter = CodexCLIAdapter()
        adapter.setup({"sandbox": "os"})
        mock_sandbox = MagicMock()
        mock_sandbox.run_agent.return_value = (True, "ok", "", "", "sha256:abc")
        adapter._os_sandbox = mock_sandbox

        with patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout") as mock_run:
            adapter.solve(sample_task_instance)
            mock_run.assert_not_called()

    def test_setup_overrides(self, tmp_dir):
        adapter = CodexCLIAdapter()
        adapter.setup({"timeout": 999, "sandbox": "none", "repo_root": str(tmp_dir)})
        assert adapter.timeout_seconds == 999
        assert adapter.sandbox == "none"
        assert adapter.task_env_manager is None

    def test_setup_creates_task_env_manager(self, tmp_dir):
        adapter = CodexCLIAdapter()
        adapter.setup({"sandbox": "task", "repo_root": str(tmp_dir)})
        assert adapter.task_env_manager is not None

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_strips_openai_api_key_from_env(self, mock_run, sample_task_instance):
        """Codex CLI should never receive OPENAI_API_KEY in its environment."""
        adapter = CodexCLIAdapter()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-secret", "HOME": "/tmp"}):
            adapter.solve(sample_task_instance)

        env = mock_run.call_args.kwargs["env"]
        assert "OPENAI_API_KEY" not in env

    def test_restricted_mode_uses_isolated_codex_home(self, sample_task_instance, tmp_dir):
        """Restricted Codex runs should not inherit ambient config, MCP, or skills."""
        host_home = tmp_dir / "host_home"
        host_codex = host_home / ".codex"
        host_codex.mkdir(parents=True)
        (host_codex / "auth.json").write_text('{"token": "ok"}')
        (host_codex / "config.toml").write_text("[mcp_servers.local]\n")
        (host_codex / "skills").mkdir()

        adapter = CodexCLIAdapter()
        adapter.setup({"sandbox": "none", "repo_root": str(tmp_dir / "repo")})

        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "sk-secret", "HOME": str(host_home)},
            clear=True,
        ):
            env = adapter._build_run_env(sample_task_instance, task_env=None)

        isolated_home = Path(env["HOME"])
        isolated_codex = isolated_home / ".codex"
        assert isolated_home.is_relative_to(tmp_dir / "repo" / ".ai4sci-bench" / "codex_home")
        assert env["USERPROFILE"] == str(isolated_home)
        assert env["CODEX_HOME"] == str(isolated_codex)
        assert "OPENAI_API_KEY" not in env
        assert (isolated_codex / "auth.json").read_text() == '{"token": "ok"}'
        assert not (isolated_codex / "config.toml").exists()
        assert not (isolated_codex / "skills").exists()

    def test_unrestricted_mode_keeps_ambient_codex_home(self, sample_task_instance, tmp_dir):
        """Unrestricted mode is explicit opt-in to ambient Codex configuration."""
        host_home = tmp_dir / "host_home"
        adapter = CodexCLIAdapter(tool_mode="unrestricted")
        adapter.setup({"sandbox": "none", "repo_root": str(tmp_dir / "repo")})

        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "sk-secret", "HOME": str(host_home)},
            clear=True,
        ):
            env = adapter._build_run_env(sample_task_instance, task_env=None)

        assert env["HOME"] == str(host_home)
        assert "CODEX_HOME" not in env
        assert "OPENAI_API_KEY" not in env

    def test_isolated_codex_home_can_copy_auth_from_codex_home(self, sample_task_instance, tmp_dir):
        host_codex = tmp_dir / "custom_codex_home"
        host_codex.mkdir()
        (host_codex / "auth.json").write_text('{"token": "custom"}')
        (host_codex / "config.toml").write_text("[mcp_servers.local]\n")

        adapter = CodexCLIAdapter()
        adapter.setup({"sandbox": "none", "repo_root": str(tmp_dir / "repo")})

        with patch.dict(
            "os.environ",
            {"CODEX_HOME": str(host_codex), "HOME": str(tmp_dir / "host_home")},
            clear=True,
        ):
            env = adapter._build_run_env(sample_task_instance, task_env=None)

        isolated_codex = Path(env["CODEX_HOME"])
        assert (isolated_codex / "auth.json").read_text() == '{"token": "custom"}'
        assert not (isolated_codex / "config.toml").exists()

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_strips_openai_api_key_with_task_env(self, mock_run, sample_task_instance, tmp_dir):
        """Codex CLI should strip OPENAI_API_KEY even when using task sandbox env."""
        adapter = CodexCLIAdapter()
        adapter.setup({"sandbox": "task", "repo_root": str(tmp_dir)})
        fake_env = TaskEnvironment(
            env_dir=tmp_dir / "env",
            python_executable=tmp_dir / "env" / "bin" / "python",
            bin_dir=tmp_dir / "env" / "bin",
            cache_key="abc",
            python_requirement=">=3.11",
            packages=["numpy"],
            cache_hit=True,
        )
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-secret"}):
            with patch.object(adapter.task_env_manager, "ensure_env", return_value=fake_env):
                adapter.solve(sample_task_instance)

        env = mock_run.call_args.kwargs["env"]
        assert "OPENAI_API_KEY" not in env
        assert env["VIRTUAL_ENV"] == str(fake_env.env_dir)

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_preserves_other_env_vars(self, mock_run, sample_task_instance):
        """Codex CLI should keep other env vars like ANTHROPIC_API_KEY intact."""
        adapter = CodexCLIAdapter()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant", "HOME": "/tmp"}):
            adapter.solve(sample_task_instance)

        env = mock_run.call_args.kwargs["env"]
        assert env.get("ANTHROPIC_API_KEY") == "sk-ant"

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_builds_correct_command(self, mock_run, sample_task_instance):
        adapter = CodexCLIAdapter(model="o3", full_auto=True)
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = adapter.solve(sample_task_instance)

        cmd = mock_run.call_args.args[0]
        assert cmd[0] == "codex"
        assert cmd[1] == "exec"
        assert "--model" in cmd
        assert "o3" in cmd
        assert "--cd" in cmd
        cd_idx = cmd.index("--cd")
        cd_path = cmd[cd_idx + 1]
        assert os.path.isabs(cd_path)
        assert "--skip-git-repo-check" in cmd
        assert "--json" in cmd
        assert "--full-auto" not in cmd
        assert "--dangerously-bypass-approvals-and-sandbox" in cmd
        assert "--sandbox" not in cmd
        assert result.status == RunStatus.COMPLETED

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_blocks_external_tools_by_default(self, mock_run, sample_task_instance):
        """Default restricted mode: web_search disabled + feature flags disabled via _build_command."""
        adapter = CodexCLIAdapter()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        adapter.solve(sample_task_instance)

        cmd = mock_run.call_args.args[0]
        assert "--config" in cmd
        idx = cmd.index("--config")
        assert cmd[idx + 1] == 'web_search="disabled"'
        disabled = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--disable"]
        assert "plugins" in disabled
        assert "image_generation" in disabled
        assert "multi_agent" not in disabled

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_allows_external_tools_when_enabled(self, mock_run, sample_task_instance):
        """allow_external_tools=True: search mode skips web_search disable, keeps plugins disabled."""
        adapter = CodexCLIAdapter(allow_external_tools=True)
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        adapter.solve(sample_task_instance)

        cmd = mock_run.call_args.args[0]
        assert 'web_search="disabled"' not in " ".join(cmd)
        disabled = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--disable"]
        assert "plugins" in disabled

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_without_full_auto(self, mock_run, sample_task_instance):
        adapter = CodexCLIAdapter(full_auto=False)
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        adapter.solve(sample_task_instance)

        cmd = mock_run.call_args.args[0]
        assert "--full-auto" not in cmd
        assert "--dangerously-bypass-approvals-and-sandbox" in cmd
        assert "--sandbox" not in cmd

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_passes_prompt_via_stdin_not_cli_arg(self, mock_run, sample_task_instance):
        """Issue #12: prompt passed via stdin to avoid Windows cmd length limits."""
        adapter = CodexCLIAdapter()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        adapter.solve(sample_task_instance)

        cmd = mock_run.call_args.args[0]
        # Last args should be ["--", "-"] (stdin sentinel)
        assert cmd[-1] == "-"
        assert cmd[-2] == "--"
        # Prompt should be in stdin input
        kwargs = mock_run.call_args.kwargs
        prompt_text = (sample_task_instance.workspace_dir / "prompt.md").read_text()
        assert kwargs["input"] == prompt_text

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_task_sandbox_adds_python_helper_and_prompt_note(
        self, mock_run, sample_task_instance, tmp_dir
    ):
        adapter = CodexCLIAdapter()
        adapter.setup({"sandbox": "task", "repo_root": str(tmp_dir), "timeout": 200})
        bin_dir = tmp_dir / "env" / ("Scripts" if os.name == "nt" else "bin")
        python_executable = bin_dir / ("python.exe" if os.name == "nt" else "python")
        fake_env = TaskEnvironment(
            env_dir=tmp_dir / "env",
            python_executable=python_executable,
            bin_dir=bin_dir,
            cache_key="abc",
            python_requirement=">=3.11",
            packages=["numpy", "taichi"],
            cache_hit=True,
        )
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(adapter.task_env_manager, "ensure_env", return_value=fake_env):
            adapter.solve(sample_task_instance)

        if os.name == "nt":
            helper_path = Path(mock_run.call_args.kwargs["cwd"]) / "_ai4sci_task_python.cmd"
            assert helper_path.exists()
            helper_text = helper_path.read_text(encoding="utf-8")
            assert 'cd /d "%~dp0"' in helper_text
            assert "python %*" in helper_text
        else:
            helper_path = sample_task_instance.workspace_dir / "_ai4sci_task_python"
            assert helper_path.exists()
            helper_text = helper_path.read_text(encoding="utf-8")
            assert 'exec python "$@"' in helper_text
        assert str(fake_env.python_executable) not in helper_text

        env = mock_run.call_args.kwargs["env"]
        assert "AI4SCI_TASK_PYTHON" not in env
        assert env["PATH"].startswith(str(fake_env.bin_dir))

        cmd = mock_run.call_args.args[0]
        assert cmd[-2] == "--"
        assert cmd[-1] == "-"  # stdin sentinel
        # Prompt content is now passed via stdin (input=)
        stdin_input = mock_run.call_args.kwargs["input"]
        if os.name == "nt":
            assert "_ai4sci_task_python.cmd" in stdin_input
            assert "./_ai4sci_task_python" not in stdin_input
        else:
            assert "./_ai4sci_task_python" in stdin_input
        assert str(fake_env.python_executable) not in stdin_input

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_failure_returns_failed_status(self, mock_run, sample_task_instance):
        adapter = CodexCLIAdapter()
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout='{"type":"error","message":"rate limited"}\n',
            stderr="codex auth error",
        )

        result = adapter.solve(sample_task_instance)

        assert result.status == RunStatus.FAILED
        assert "codex auth error" in result.log
        assert "[exit_code] 1" in result.log
        assert result.error_message == "CodexCLIAdapter exited with code 1"
        assert result.raw_stdout is not None
        assert "rate limited" in result.raw_stdout
        assert result.raw_stderr == "codex auth error"
        assert result.raw_stdout_format == "jsonl"

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_timeout(self, mock_run, sample_task_instance):
        from dataclasses import replace
        adapter = CodexCLIAdapter(timeout_seconds=5)
        task_instance = replace(sample_task_instance, effective_timeout_seconds=5)
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd="codex", timeout=5, output="partial output", stderr="partial stderr"
        )

        result = adapter.solve(task_instance)

        assert result.status == RunStatus.TIMEOUT
        assert "timed out after 5s" in result.log
        assert "partial stderr" in result.log
        assert result.error_message.startswith("CodexCLIAdapter timed out after 5s")
        assert result.raw_stdout == "partial output"
        assert result.raw_stderr == "partial stderr"
        assert result.raw_stdout_format == "jsonl"

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_generic_exception(self, mock_run, sample_task_instance):
        adapter = CodexCLIAdapter()
        mock_run.side_effect = FileNotFoundError("codex: command not found")

        result = adapter.solve(sample_task_instance)

        assert result.status == RunStatus.FAILED
        assert "command not found" in result.log
        assert result.error_message == "codex: command not found"

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_uses_task_sandbox_env(self, mock_run, sample_task_instance, tmp_dir):
        adapter = CodexCLIAdapter()
        adapter.setup({"sandbox": "task", "repo_root": str(tmp_dir), "timeout": 200})
        fake_env = TaskEnvironment(
            env_dir=tmp_dir / "env",
            python_executable=tmp_dir / "env" / "bin" / "python",
            bin_dir=tmp_dir / "env" / "bin",
            cache_key="abc",
            python_requirement=">=3.11",
            packages=["numpy"],
            cache_hit=True,
        )
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(adapter.task_env_manager, "ensure_env", return_value=fake_env):
            adapter.solve(sample_task_instance)

        env = mock_run.call_args.kwargs["env"]
        assert env["VIRTUAL_ENV"] == str(fake_env.env_dir)
        assert env["PATH"].startswith(str(fake_env.bin_dir))

    def test_build_log_empty(self):
        adapter = CodexCLIAdapter()
        assert adapter._build_full_log("", "", 0) == ""

    def test_build_log_only_stderr(self):
        adapter = CodexCLIAdapter()
        log = adapter._build_full_log("", "some error\n", 0)
        assert "[stderr]" in log
        assert "some error" in log

    def test_build_log_combines_all(self):
        adapter = CodexCLIAdapter()
        log = adapter._build_full_log("plain text", "error text", 2)
        assert "plain text" in log
        assert "[stderr]" in log
        assert "[exit_code] 2" in log

    def test_parse_codex_log_message_event(self):
        adapter = CodexCLIAdapter()
        line = json.dumps({"type": "message", "role": "assistant", "content": "Hello world"})
        result = adapter._parse_codex_log(line)
        assert "[assistant] Hello world" in result

    def test_parse_codex_log_message_with_content_blocks(self):
        adapter = CodexCLIAdapter()
        line = json.dumps({
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will solve this."},
                {"type": "tool_use", "name": "shell", "input": {"command": "python sim.py"}},
            ]
        })
        result = adapter._parse_codex_log(line)
        assert "[assistant] I will solve this." in result
        assert "[Tool] shell" in result

    def test_parse_codex_log_function_call(self):
        adapter = CodexCLIAdapter()
        line = json.dumps({"type": "function_call", "name": "shell"})
        result = adapter._parse_codex_log(line)
        assert "[Tool] shell" in result

    def test_parse_codex_log_function_call_output(self):
        adapter = CodexCLIAdapter()
        line = json.dumps({"type": "function_call_output", "output": "simulation complete"})
        result = adapter._parse_codex_log(line)
        assert "[ToolResult] simulation complete" in result

    def test_parse_codex_log_error_event(self):
        adapter = CodexCLIAdapter()
        line = json.dumps({"type": "error", "message": "rate limit exceeded"})
        result = adapter._parse_codex_log(line)
        assert "[Error] rate limit exceeded" in result

    def test_parse_codex_log_malformed_json(self):
        adapter = CodexCLIAdapter()
        result = adapter._parse_codex_log("not json\n{bad\nplain line")
        assert "not json" in result
        assert "{bad" in result
        assert "plain line" in result

    def test_parse_codex_log_no_truncation_by_default(self):
        """Summary log should preserve full text content without truncation."""
        adapter = CodexCLIAdapter()
        long_text = "x" * 5000
        line = json.dumps({"type": "message", "role": "assistant", "content": long_text})
        result = adapter._parse_codex_log(line)
        assert result.count("x") == 5000

    def test_parse_codex_log_tool_call_highlights_command(self):
        """Bug #3: function_call should surface key arg values, not only keys."""
        adapter = CodexCLIAdapter()
        event = json.dumps({
            "type": "function_call",
            "name": "shell",
            "arguments": json.dumps({"command": "python simulation.py --size 128"}),
        })
        result = adapter._parse_codex_log(event)
        assert "[Tool] shell" in result
        assert "command=python simulation.py --size 128" in result

    def test_parse_codex_log_tool_call_arguments_as_string(self):
        """Non-JSON argument string should still show a hint in the summary."""
        adapter = CodexCLIAdapter()
        event = json.dumps({
            "type": "function_call",
            "name": "shell",
            "arguments": "not valid json",
        })
        result = adapter._parse_codex_log(event)
        assert "[Tool] shell" in result
        assert "not valid json" in result

    def test_parse_codex_log_captures_rate_limit_event(self):
        adapter = CodexCLIAdapter()
        event = json.dumps({
            "type": "rate_limit_event",
            "message": "7d limit 90%",
        })
        result = adapter._parse_codex_log(event)
        assert "[rate_limit]" in result
        assert "7d limit 90%" in result

    def test_parse_codex_log_tool_result_not_truncated(self):
        """Tool output should be preserved in full without truncation."""
        adapter = CodexCLIAdapter()
        long_output = "y" * 3000
        event = json.dumps({"type": "function_call_output", "output": long_output})
        result = adapter._parse_codex_log(event)
        assert "[ToolResult]" in result
        assert result.count("y") == 3000

    def test_parse_codex_log_blank_lines_are_skipped(self):
        adapter = CodexCLIAdapter()
        assert adapter._parse_codex_log("\n\n  \n") == ""

    def test_collect_output_files_missing(self, sample_task_instance):
        from ai4sci_bench.adapters.subprocess_base import collect_output_files
        files = collect_output_files(
            sample_task_instance.workspace_dir, sample_task_instance
        )
        assert files == []

    def test_collect_output_files_existing(self, sample_task_instance):
        from ai4sci_bench.adapters.subprocess_base import collect_output_files
        (sample_task_instance.workspace_dir / "output.npy").write_bytes(b"data")
        files = collect_output_files(
            sample_task_instance.workspace_dir, sample_task_instance
        )
        assert "output.npy" in files

    # -- Tool isolation: comprehensive tests ----------------------------

    def test_restricted_mode_disables_plugins(self, sample_task_instance):
        adapter = CodexCLIAdapter()
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        disabled = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--disable"]
        assert "plugins" in disabled

    def test_restricted_mode_disables_mcp_elicitation(self, sample_task_instance):
        adapter = CodexCLIAdapter()
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        disabled = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--disable"]
        assert "tool_call_mcp_elicitation" in disabled

    def test_restricted_mode_does_not_disable_multi_agent(self, sample_task_instance):
        adapter = CodexCLIAdapter()
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        disabled = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--disable"]
        assert "multi_agent" not in disabled

    def test_restricted_mode_disables_skill_mcp_dependency_install(self, sample_task_instance):
        adapter = CodexCLIAdapter()
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        disabled = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--disable"]
        assert "skill_mcp_dependency_install" in disabled

    def test_restricted_mode_disables_web_search(self, sample_task_instance):
        adapter = CodexCLIAdapter()
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        assert "--config" in cmd
        idx = cmd.index("--config")
        assert cmd[idx + 1] == 'web_search="disabled"'

    def test_restricted_mode_disables_image_generation(self, sample_task_instance):
        adapter = CodexCLIAdapter()
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        disabled = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--disable"]
        assert "image_generation" in disabled

    def test_restricted_mode_disables_codex_hooks(self, sample_task_instance):
        adapter = CodexCLIAdapter()
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        disabled = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--disable"]
        assert "codex_hooks" in disabled

    def test_restricted_mode_disables_computer_use(self, sample_task_instance):
        adapter = CodexCLIAdapter()
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        disabled = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--disable"]
        assert "computer_use" in disabled

    def test_restricted_mode_ignores_user_config(self, sample_task_instance):
        """--ignore-user-config is the only reliable way to block MCP servers (Appendix F.3)."""
        adapter = CodexCLIAdapter()
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        assert "--ignore-user-config" in cmd

    def test_search_mode_ignores_user_config(self, sample_task_instance):
        adapter = CodexCLIAdapter(allow_external_tools=True)
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        assert "--ignore-user-config" in cmd

    def test_unrestricted_mode_does_not_ignore_user_config(self, sample_task_instance):
        adapter = CodexCLIAdapter(tool_mode="unrestricted")
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        assert "--ignore-user-config" not in cmd

    def test_search_mode_allows_web_search(self, sample_task_instance):
        adapter = CodexCLIAdapter(allow_external_tools=True)
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        assert 'web_search="disabled"' not in " ".join(cmd)

    def test_search_mode_still_disables_plugins(self, sample_task_instance):
        adapter = CodexCLIAdapter(allow_external_tools=True)
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        disabled = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--disable"]
        assert "plugins" in disabled
        assert "tool_call_mcp_elicitation" in disabled
        assert "image_generation" in disabled
        assert "multi_agent" not in disabled

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_build_command_restricted_disables_features(self, mock_run, sample_task_instance):
        adapter = CodexCLIAdapter()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter.solve(sample_task_instance)
        cmd = mock_run.call_args.args[0]
        disabled = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--disable"]
        assert "plugins" in disabled
        assert "multi_agent" not in disabled
        assert "image_generation" in disabled

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_build_command_uses_sandbox_workspace_write(self, mock_run, sample_task_instance):
        adapter = CodexCLIAdapter()
        adapter.setup({"sandbox": "task"})
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter.solve(sample_task_instance)
        cmd = mock_run.call_args.args[0]
        assert "--full-auto" not in cmd
        assert "--sandbox" in cmd
        idx = cmd.index("--sandbox")
        assert cmd[idx + 1] == "workspace-write"

    @patch("ai4sci_bench.adapters.subprocess_base.validate_sandbox_mode")
    @patch("ai4sci_bench.adapters.subprocess_base.LinuxNSSandbox")
    def test_linux_ns_restricted_disables_features(
        self, mock_linux_ns_cls, mock_validate, sample_task_instance
    ):
        adapter = CodexCLIAdapter()
        adapter.setup({"sandbox": "linux_ns"})
        cmd = adapter._build_command(sample_task_instance, task_env=None)
        disabled = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--disable"]
        assert "plugins" in disabled
        assert "image_generation" in disabled

    def test_tool_mode_from_allow_external_tools_false(self):
        adapter = CodexCLIAdapter(allow_external_tools=False)
        from ai4sci_bench.core.types import ToolMode
        assert adapter.tool_mode == ToolMode.RESTRICTED

    def test_tool_mode_from_allow_external_tools_true(self):
        adapter = CodexCLIAdapter(allow_external_tools=True)
        from ai4sci_bench.core.types import ToolMode
        assert adapter.tool_mode == ToolMode.SEARCH

    def test_tool_mode_explicit_overrides_allow_external_tools(self):
        adapter = CodexCLIAdapter(allow_external_tools=True, tool_mode="restricted")
        from ai4sci_bench.core.types import ToolMode
        assert adapter.tool_mode == ToolMode.RESTRICTED

    def test_tool_mode_unrestricted_no_restrictions(self, sample_task_instance):
        adapter = CodexCLIAdapter(tool_mode="unrestricted")
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        disabled = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--disable"]
        assert disabled == []
        assert 'web_search="disabled"' not in " ".join(cmd)

    def test_build_os_agent_cmd_feature_consistency_with_build_command(self, sample_task_instance):
        """_build_os_agent_cmd() and _build_command() must disable the same features."""
        adapter = CodexCLIAdapter()
        os_cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        local_cmd = adapter._build_command(sample_task_instance, task_env=None)
        os_disabled = sorted(os_cmd[i + 1] for i, v in enumerate(os_cmd) if v == "--disable")
        local_disabled = sorted(local_cmd[i + 1] for i, v in enumerate(local_cmd) if v == "--disable")
        assert os_disabled == local_disabled


# -----------------------------------------------------------------------
# Supplementary coverage tests for adapter helper methods
# -----------------------------------------------------------------------


class TestClaudeCodeCLIFormatHelpers:
    """Tests for _format_assistant_blocks, _format_user_blocks, _format_tool_use."""

    def test_format_assistant_blocks_empty_content(self):
        adapter = ClaudeCodeCLIAdapter()
        event = {"message": {"content": []}}
        assert adapter._format_assistant_blocks(event) == []

    def test_format_assistant_blocks_missing_message(self):
        adapter = ClaudeCodeCLIAdapter()
        assert adapter._format_assistant_blocks({}) == []

    def test_format_assistant_blocks_thinking_block(self):
        adapter = ClaudeCodeCLIAdapter()
        event = {"message": {"content": [
            {"type": "thinking", "thinking": "Let me think about this problem..."},
        ]}}
        out = adapter._format_assistant_blocks(event)
        assert len(out) == 1
        assert "[thinking]" in out[0]
        assert "Let me think" in out[0]

    def test_format_assistant_blocks_thinking_not_truncated(self):
        adapter = ClaudeCodeCLIAdapter()
        event = {"message": {"content": [
            {"type": "thinking", "thinking": "x" * 10000},
        ]}}
        out = adapter._format_assistant_blocks(event)
        assert out[0].count("x") == 10000

    def test_format_assistant_blocks_mixed_types(self):
        adapter = ClaudeCodeCLIAdapter()
        event = {"message": {"content": [
            {"type": "text", "text": "Hello"},
            {"type": "tool_use", "name": "Read", "input": {"file_path": "/tmp/a.py"}},
            {"type": "thinking", "thinking": "hmm"},
            {"type": "unknown_block"},  # should be silently skipped
        ]}}
        out = adapter._format_assistant_blocks(event)
        assert len(out) == 3
        assert "[Claude] Hello" in out[0]
        assert "[Tool] Read" in out[1]
        assert "[thinking]" in out[2]

    def test_format_user_blocks_non_list_content(self):
        adapter = ClaudeCodeCLIAdapter()
        event = {"message": {"content": "plain string"}}
        assert adapter._format_user_blocks(event) == []

    def test_format_user_blocks_non_dict_items(self):
        adapter = ClaudeCodeCLIAdapter()
        event = {"message": {"content": ["string item", 42, None]}}
        assert adapter._format_user_blocks(event) == []

    def test_format_user_blocks_tool_result_nested_text(self):
        adapter = ClaudeCodeCLIAdapter()
        event = {"message": {"content": [
            {
                "type": "tool_result",
                "content": [
                    {"type": "text", "text": "line 1"},
                    {"type": "text", "text": "line 2"},
                ],
            }
        ]}}
        out = adapter._format_user_blocks(event)
        assert len(out) == 1
        assert "line 1" in out[0]
        assert "line 2" in out[0]
        assert "[ToolResult]" in out[0]

    def test_format_user_blocks_tool_result_plain_string(self):
        adapter = ClaudeCodeCLIAdapter()
        event = {"message": {"content": [
            {"type": "tool_result", "content": "simple output"},
        ]}}
        out = adapter._format_user_blocks(event)
        assert out[0] == "[ToolResult] simple output"

    def test_format_user_blocks_tool_result_not_truncated(self):
        adapter = ClaudeCodeCLIAdapter()
        event = {"message": {"content": [
            {"type": "tool_result", "content": "z" * 3000},
        ]}}
        out = adapter._format_user_blocks(event)
        assert out[0].count("z") == 3000

    def test_format_tool_use_no_input(self):
        adapter = ClaudeCodeCLIAdapter()
        # Empty dict input → fallback shows empty key list
        assert adapter._format_tool_use({"name": "Foo"}) == "[Tool] Foo([])"

    def test_format_tool_use_non_dict_input(self):
        adapter = ClaudeCodeCLIAdapter()
        assert adapter._format_tool_use({"name": "Foo", "input": "string"}) == "[Tool] Foo"

    def test_format_tool_use_none_input(self):
        adapter = ClaudeCodeCLIAdapter()
        assert adapter._format_tool_use({"name": "Foo", "input": None}) == "[Tool] Foo([])"

    def test_format_tool_use_known_keys(self):
        adapter = ClaudeCodeCLIAdapter()
        block = {
            "name": "Grep",
            "input": {"pattern": "import", "path": "/src"},
        }
        result = adapter._format_tool_use(block)
        assert "pattern=import" in result
        assert "path=/src" in result

    def test_format_tool_use_long_value_truncated(self):
        adapter = ClaudeCodeCLIAdapter()
        block = {
            "name": "Write",
            "input": {"file_path": "x" * 500},
        }
        result = adapter._format_tool_use(block)
        assert "..." in result
        assert result.count("x") == 200

    def test_format_tool_use_fallback_to_keys(self):
        adapter = ClaudeCodeCLIAdapter()
        block = {
            "name": "CustomTool",
            "input": {"custom_arg": "value", "other": 42},
        }
        result = adapter._format_tool_use(block)
        assert "[Tool] CustomTool(['custom_arg', 'other'])" == result

    def test_parse_claude_log_result_event(self):
        """The summary should capture 'result' events (final turn summary)."""
        adapter = ClaudeCodeCLIAdapter()
        event = json.dumps({
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "num_turns": 12,
        })
        result = adapter._parse_claude_log(event)
        assert "[result]" in result
        assert "success" in result
        assert "turns=12" in result

    def test_parse_claude_log_system_no_message(self):
        adapter = ClaudeCodeCLIAdapter()
        event = json.dumps({"type": "system", "subtype": "handshake"})
        result = adapter._parse_claude_log(event)
        assert "[system:handshake]" in result

    def test_extract_terminal_error_from_api_error_result(self):
        event = json.dumps({
            "type": "result",
            "is_error": True,
            "api_error_status": 400,
            "result": "API Error: 400 Function call is missing a thought_signature",
        })
        result = ClaudeCodeCLIAdapter._extract_terminal_error_from_jsonl(event)
        assert result is not None
        assert "API error 400" in result
        assert "thought_signature" in result

    def test_extract_terminal_error_from_tool_use_interrupted(self):
        event = json.dumps({
            "type": "result",
            "is_error": False,
            "api_error_status": None,
            "result": "[Tool use interrupted]",
        })
        result = ClaudeCodeCLIAdapter._extract_terminal_error_from_jsonl(event)
        assert result == "Claude Code ended with interrupted tool use"

    def test_extract_terminal_error_from_plain_text_pseudo_tool_call(self):
        event = json.dumps({
            "type": "result",
            "is_error": False,
            "api_error_status": None,
            "result": (
                "Prior assistant action already executed by the runtime.\n"
                "Tool name: Bash\n"
                "Tool id: toolu_gemini_000007\n"
                "Input JSON: {\"command\":\"cat data/problem_setup.json\"}"
            ),
        })
        result = ClaudeCodeCLIAdapter._extract_terminal_error_from_jsonl(event)
        assert result == "Claude Code ended with a plain-text pseudo tool call"

    def test_extract_terminal_error_clean_result_is_none(self):
        event = json.dumps({
            "type": "result",
            "is_error": False,
            "api_error_status": None,
            "result": "Done",
        })
        assert ClaudeCodeCLIAdapter._extract_terminal_error_from_jsonl(event) is None


class TestCodexCLIFormatHelpers:
    """Tests for _format_codex_message, _coerce_text, _format_codex_tool_call."""

    def test_coerce_text_string(self):
        adapter = CodexCLIAdapter()
        assert adapter._coerce_text("hello") == "hello"

    def test_coerce_text_bytes(self):
        adapter = CodexCLIAdapter()
        assert adapter._coerce_text(b"hello") == "hello"

    def test_coerce_text_bytearray(self):
        adapter = CodexCLIAdapter()
        assert adapter._coerce_text(bytearray(b"\xc3\xa9")) == "é"

    def test_coerce_text_other(self):
        adapter = CodexCLIAdapter()
        assert adapter._coerce_text(42) == "42"
        assert adapter._coerce_text(None) == "None"

    def test_coerce_text_invalid_utf8(self):
        adapter = CodexCLIAdapter()
        result = adapter._coerce_text(b"\xff\xfe")
        assert isinstance(result, str)  # should not raise

    def test_format_codex_message_bytes_content(self):
        adapter = CodexCLIAdapter()
        event = {"role": "assistant", "content": b"binary response"}
        out = adapter._format_codex_message(event)
        assert len(out) == 1
        assert "[assistant] binary response" in out[0]

    def test_format_codex_message_empty_string(self):
        adapter = CodexCLIAdapter()
        event = {"role": "assistant", "content": ""}
        assert adapter._format_codex_message(event) == []

    def test_format_codex_message_list_with_tool_result(self):
        adapter = CodexCLIAdapter()
        event = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "is_error": True,
                    "content": [{"type": "text", "text": "ModuleNotFoundError"}],
                }
            ],
        }
        out = adapter._format_codex_message(event)
        assert "[ToolResult:err]" in out[0]
        assert "ModuleNotFoundError" in out[0]

    def test_format_codex_message_list_non_dict_items(self):
        adapter = CodexCLIAdapter()
        event = {"role": "user", "content": ["string", 42]}
        assert adapter._format_codex_message(event) == []

    def test_format_codex_tool_call_dict_input(self):
        adapter = CodexCLIAdapter()
        event = {"name": "shell", "input": {"command": "ls -la"}}
        result = adapter._format_codex_tool_call(event)
        assert "command=ls -la" in result

    def test_format_codex_tool_call_json_string_arguments(self):
        adapter = CodexCLIAdapter()
        event = {
            "name": "file_write",
            "arguments": json.dumps({"file_path": "/tmp/out.py", "content": "pass"}),
        }
        result = adapter._format_codex_tool_call(event)
        assert "file_path=/tmp/out.py" in result

    def test_format_codex_tool_call_bytes_arguments(self):
        adapter = CodexCLIAdapter()
        event = {
            "name": "shell",
            "arguments": json.dumps({"command": "echo hi"}).encode(),
        }
        result = adapter._format_codex_tool_call(event)
        assert "command=echo hi" in result

    def test_format_codex_tool_call_non_dict_parsed(self):
        """JSON-string arguments that parse to a non-dict (e.g. list) → name only."""
        adapter = CodexCLIAdapter()
        event = {"name": "tool", "arguments": json.dumps([1, 2, 3])}
        result = adapter._format_codex_tool_call(event)
        # Parsed result is a list, not dict → falls through to "[Tool] name"
        assert result == "[Tool] tool([])"

    def test_format_codex_tool_call_no_input(self):
        """No input/arguments keys → name only."""
        adapter = CodexCLIAdapter()
        event = {"name": "tool"}
        result = adapter._format_codex_tool_call(event)
        assert result == "[Tool] tool([])"

    def test_format_codex_tool_call_non_dict_input(self):
        """Non-dict input value → name only."""
        adapter = CodexCLIAdapter()
        event = {"name": "tool", "input": 42}
        result = adapter._format_codex_tool_call(event)
        assert result == "[Tool] tool"

    def test_parse_codex_log_thinking_event(self):
        adapter = CodexCLIAdapter()
        line = json.dumps({"type": "thinking", "text": "reasoning here"})
        result = adapter._parse_codex_log(line)
        assert "[thinking]" in result
        assert "reasoning here" in result

    def test_parse_codex_log_unknown_structured_event(self):
        adapter = CodexCLIAdapter()
        line = json.dumps({"type": "custom_event", "data": "stuff"})
        result = adapter._parse_codex_log(line)
        assert "[custom_event]" in result

    def test_parse_codex_log_tool_result_via_top_level(self):
        adapter = CodexCLIAdapter()
        line = json.dumps({"type": "tool_result", "content": "done"})
        result = adapter._parse_codex_log(line)
        assert "[ToolResult] done" in result

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_build_prompt_without_task_env(self, mock_run, sample_task_instance):
        """_build_prompt without task_env should return plain prompt."""
        adapter = CodexCLIAdapter()
        prompt = adapter._build_prompt(sample_task_instance.workspace_dir, None)
        expected = (sample_task_instance.workspace_dir / "prompt.md").read_text()
        assert prompt == expected

    def test_write_workspace_python_helper(self, sample_task_instance):
        """Helper should create executable script without absolute paths."""
        adapter = CodexCLIAdapter()
        ws = sample_task_instance.workspace_dir
        helper_name = adapter._write_workspace_python_helper(ws, None)
        expected_name = "_ai4sci_task_python.cmd" if os.name == "nt" else "_ai4sci_task_python"
        assert helper_name == expected_name
        helper = ws / helper_name
        assert helper.exists()
        text = helper.read_text()
        if os.name == "nt":
            assert 'cd /d "%~dp0"' in text
            assert "python %*" in text
        else:
            assert "exec python" in text
            # Must be executable
            import stat
            assert helper.stat().st_mode & stat.S_IXUSR

    # ── Codex task sandbox bridge tests ───────────────────────────

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_task_sandbox_windows_uses_bridge_workspace_and_syncs_outputs(
        self, mock_run, sample_task_instance, tmp_dir
    ):
        adapter = CodexCLIAdapter()
        adapter.setup({"sandbox": "task", "repo_root": str(tmp_dir), "timeout": 200})
        fake_env = TaskEnvironment(
            env_dir=tmp_dir / "env",
            python_executable=tmp_dir / "env" / "Scripts" / "python.exe",
            bin_dir=tmp_dir / "env" / "Scripts",
            cache_key="abc",
            python_requirement=">=3.11",
            packages=["numpy"],
            cache_hit=True,
        )

        bridge_cwds = []

        def fake_run(*args, **kwargs):
            bridge_cwds.append(Path(kwargs["cwd"]))
            np.save(Path(kwargs["cwd"]) / "output.npy", np.array([1.0], dtype=np.float32))
            return MagicMock(
                returncode=0,
                stdout='{"type":"message","role":"assistant","content":"ok"}\n',
                stderr="",
            )

        mock_run.side_effect = fake_run

        with (
            patch.object(adapter, "_is_windows_platform", return_value=True),
            patch.object(adapter.task_env_manager, "ensure_env", return_value=fake_env),
        ):
            result = adapter.solve(sample_task_instance)

        bridge_workspace = bridge_cwds[-1]
        assert result.output_dir == sample_task_instance.workspace_dir
        assert "output.npy" in result.data_files
        assert (sample_task_instance.workspace_dir / "output.npy").exists()
        assert result.raw_stdout_format == "jsonl"
        assert result.raw_stdout == '{"type":"message","role":"assistant","content":"ok"}\n'
        assert result.error_message is None
        assert bridge_workspace != sample_task_instance.workspace_dir
        assert str(tmp_dir / ".ai4sci-bench" / "codex_windows_task_bridge") in str(bridge_workspace)
        assert not bridge_workspace.exists()

    @patch("ai4sci_bench.adapters.codex_cli.os.name", "posix")
    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_task_sandbox_posix_keeps_original_workspace(
        self, mock_run, sample_task_instance, tmp_dir
    ):
        adapter = CodexCLIAdapter()
        adapter.setup({"sandbox": "task", "repo_root": str(tmp_dir), "timeout": 200})
        fake_env = TaskEnvironment(
            env_dir=tmp_dir / "env",
            python_executable=tmp_dir / "env" / "bin" / "python",
            bin_dir=tmp_dir / "env" / "bin",
            cache_key="abc",
            python_requirement=">=3.11",
            packages=["numpy"],
            cache_hit=True,
        )
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(adapter.task_env_manager, "ensure_env", return_value=fake_env):
            adapter.solve(sample_task_instance)

        assert str(Path(mock_run.call_args.kwargs["cwd"]).resolve()) == str(
            sample_task_instance.workspace_dir.resolve()
        )


# ── New trajectory logging tests ─────────────────────────────────────────


class TestClaudeCodeCLITrajectoryEnhancements:
    """Tests for TODO-1, TODO-4, TODO-10, TODO-12 on Claude Code CLI adapter."""

    def test_parse_claude_log_thinking_not_truncated(self):
        adapter = ClaudeCodeCLIAdapter()
        thinking = "x" * 10000
        event = json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "thinking", "thinking": thinking}]},
        })
        result = adapter._parse_claude_log(event)
        assert result.count("x") == 10000

    def test_parse_claude_log_tool_result_not_truncated(self):
        adapter = ClaudeCodeCLIAdapter()
        long_content = "z" * 3000
        events = [
            json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "start"}]}}),
            json.dumps({"type": "user", "message": {"content": [
                {"type": "tool_result", "content": long_content},
            ]}}),
        ]
        result = adapter._parse_claude_log("\n".join(events))
        assert result.count("z") == 3000

    def test_parse_claude_log_turn_numbering(self):
        adapter = ClaudeCodeCLIAdapter()
        events = [
            json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "turn1"}]}}),
            json.dumps({"type": "user", "message": {"content": [{"type": "tool_result", "content": "ok"}]}}),
            json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "turn2"}]}}),
            json.dumps({"type": "user", "message": {"content": [{"type": "tool_result", "content": "ok"}]}}),
            json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "turn3"}]}}),
        ]
        result = adapter._parse_claude_log("\n".join(events))
        assert "=== Turn 1" in result
        assert "=== Turn 2" in result
        assert "=== Turn 3" in result

    def test_parse_claude_log_turn_separator_between_not_before_first(self):
        adapter = ClaudeCodeCLIAdapter()
        event = json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}})
        result = adapter._parse_claude_log(event)
        lines = result.splitlines()
        assert lines[0] == "=== Turn 1 ==="

    def test_parse_claude_log_timestamp_from_event(self):
        adapter = ClaudeCodeCLIAdapter()
        events = [
            json.dumps({"type": "assistant", "timestamp": 1000.0, "message": {"content": [{"type": "text", "text": "a"}]}}),
            json.dumps({"type": "assistant", "timestamp": 1002.3, "message": {"content": [{"type": "text", "text": "b"}]}}),
        ]
        result = adapter._parse_claude_log("\n".join(events))
        assert "[+0.0s]" in result
        assert "[+2.3s]" in result

    def test_parse_claude_log_no_timestamp_graceful(self):
        adapter = ClaudeCodeCLIAdapter()
        event = json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}})
        result = adapter._parse_claude_log(event)
        assert "=== Turn 1 ===" in result
        assert "[+" not in result

    def test_parse_claude_log_unknown_event_type_surfaced(self):
        adapter = ClaudeCodeCLIAdapter()
        event = json.dumps({"type": "new_feature", "data": "something"})
        result = adapter._parse_claude_log(event)
        assert "[unknown:new_feature]" in result

    def test_parse_claude_log_unknown_event_content_shown(self):
        adapter = ClaudeCodeCLIAdapter()
        event = json.dumps({"type": "new_feature", "data": "something important"})
        result = adapter._parse_claude_log(event)
        assert "new_feature" in result
        assert "something important" in result

    def test_claude_extract_usage_from_result_event(self):
        adapter = ClaudeCodeCLIAdapter()
        jsonl = json.dumps({
            "type": "result",
            "usage": {"input_tokens": 5000, "output_tokens": 1000},
        })
        cost = adapter._extract_usage_from_jsonl(jsonl)
        assert cost is not None
        assert cost.input_tokens == 5000
        assert cost.output_tokens == 1000
        assert cost.total_tokens == 6000

    def test_claude_no_usage_in_result_event(self):
        adapter = ClaudeCodeCLIAdapter()
        jsonl = json.dumps({"type": "result", "subtype": "success"})
        cost = adapter._extract_usage_from_jsonl(jsonl)
        assert cost is None


class TestCodexCLITrajectoryEnhancements:
    """Tests for TODO-1, TODO-4, TODO-10 on Codex CLI adapter."""

    def test_parse_codex_log_reasoning_not_truncated(self):
        adapter = CodexCLIAdapter()
        long_thinking = "Q" * 5000
        event = json.dumps({"type": "reasoning", "text": long_thinking})
        result = adapter._parse_codex_log(event)
        assert result.count("Q") == 5000

    def test_parse_codex_log_turn_numbering(self):
        adapter = CodexCLIAdapter()
        events = [
            json.dumps({"type": "message", "role": "assistant", "content": "turn1"}),
            json.dumps({"type": "function_call_output", "output": "ok"}),
            json.dumps({"type": "message", "role": "assistant", "content": "turn2"}),
        ]
        result = adapter._parse_codex_log("\n".join(events))
        assert "=== Turn 1" in result
        assert "=== Turn 2" in result

    def test_parse_codex_log_single_turn_no_separator(self):
        adapter = CodexCLIAdapter()
        event = json.dumps({"type": "message", "role": "assistant", "content": "hello"})
        result = adapter._parse_codex_log(event)
        assert "=== Turn 1 ===" in result

    def test_codex_extract_usage(self):
        adapter = CodexCLIAdapter()
        jsonl = json.dumps({
            "type": "result",
            "usage": {"input_tokens": 3000, "output_tokens": 800},
        })
        cost = adapter._extract_usage_from_jsonl(jsonl)
        assert cost is not None
        assert cost.input_tokens == 3000
        assert cost.output_tokens == 800

    def test_codex_no_usage(self):
        adapter = CodexCLIAdapter()
        jsonl = json.dumps({"type": "message", "role": "assistant", "content": "hi"})
        cost = adapter._extract_usage_from_jsonl(jsonl)
        assert cost is None

    def test_parse_codex_log_timestamp_from_event(self):
        adapter = CodexCLIAdapter()
        events = [
            json.dumps({"type": "message", "role": "assistant", "content": "a", "timestamp": 1000.0}),
            json.dumps({"type": "message", "role": "assistant", "content": "b", "timestamp": 1003.5}),
        ]
        result = adapter._parse_codex_log("\n".join(events))
        assert "[+0.0s]" in result
        assert "[+3.5s]" in result


class TestDirectLLMStructuredLog:
    """Tests for TODO-2: Direct LLM adapter structured logging."""

    @patch("litellm.completion")
    def test_direct_llm_structured_log_contains_prompt(self, mock_completion, sample_task_instance):
        adapter = DirectLLMAdapter()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "```python\nprint('hi')\n```"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_completion.return_value = mock_response

        with patch.object(adapter, "_execute", return_value=(True, "ok", "stdout", "stderr")):
            result = adapter.solve(sample_task_instance)

        structured = json.loads(result.raw_model_output)
        prompt_step = [s for s in structured if s["step"] == "prompt"][0]
        assert adapter.system_prompt in prompt_step["system_prompt"]
        assert len(prompt_step["user_prompt"]) > 0

    @patch("litellm.completion")
    def test_direct_llm_structured_log_contains_usage(self, mock_completion, sample_task_instance):
        adapter = DirectLLMAdapter()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "```python\nprint('hi')\n```"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 5000
        mock_response.usage.completion_tokens = 1000
        mock_completion.return_value = mock_response

        with patch.object(adapter, "_execute", return_value=(True, "ok", "", "")):
            result = adapter.solve(sample_task_instance)

        structured = json.loads(result.raw_model_output)
        llm_step = [s for s in structured if s["step"] == "llm_response"][0]
        assert llm_step["content"] == "```python\nprint('hi')\n```"
        assert llm_step["usage"]["input_tokens"] == 5000
        assert llm_step["usage"]["output_tokens"] == 1000

    @patch("litellm.completion")
    def test_direct_llm_structured_log_contains_code_extraction(self, mock_completion, sample_task_instance):
        adapter = DirectLLMAdapter()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "```python\nblock1\n```\n```python\nblock2\n```\n```python\nblock3\n```"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_completion.return_value = mock_response

        with patch.object(adapter, "_execute", return_value=(True, "ok", "", "")):
            result = adapter.solve(sample_task_instance)

        structured = json.loads(result.raw_model_output)
        extract_step = [s for s in structured if s["step"] == "code_extraction"][0]
        assert extract_step["num_candidates"] == 3
        assert len(extract_step["candidate_scores"]) == 3
        assert "selected_index" in extract_step
        assert isinstance(extract_step["selected_code"], str)

    @patch("litellm.completion")
    def test_direct_llm_structured_log_contains_execution_result(self, mock_completion, sample_task_instance):
        adapter = DirectLLMAdapter()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "```python\nprint('hi')\n```"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_completion.return_value = mock_response

        with patch.object(adapter, "_execute", return_value=(True, "ok", "stdout_data", "stderr_data")):
            result = adapter.solve(sample_task_instance)

        structured = json.loads(result.raw_model_output)
        exec_step = [s for s in structured if s["step"] == "code_execution"][0]
        assert "code_file" in exec_step
        assert exec_step["exit_code"] == 0
        assert "duration_ms" in exec_step

    @patch("litellm.completion")
    def test_direct_llm_structured_log_on_api_error(self, mock_completion, sample_task_instance):
        adapter = DirectLLMAdapter()
        mock_completion.side_effect = Exception("AuthenticationError: Invalid API key")

        result = adapter.solve(sample_task_instance)

        assert result.status.value == "failed"
        structured = json.loads(result.raw_model_output)
        error_step = [s for s in structured if s["step"] == "error"][0]
        assert "AuthenticationError" in error_step["error"]


# ── Effort parameter tests ───────────────────────────────────────────────


class TestCodexCLIEffort:
    """Tests for reasoning effort propagation in CodexCLIAdapter."""

    def test_default_effort_is_medium(self):
        adapter = CodexCLIAdapter()
        assert adapter.effort == "medium"

    def test_custom_effort(self):
        adapter = CodexCLIAdapter(effort="xhigh")
        assert adapter.effort == "xhigh"

    def test_all_valid_effort_levels(self):
        for level in ("none", "minimal", "low", "medium", "high", "xhigh"):
            adapter = CodexCLIAdapter(effort=level)
            assert adapter.effort == level

    def test_invalid_effort_rejected(self):
        with pytest.raises(ValueError, match="Invalid effort 'max' for codex_cli"):
            CodexCLIAdapter(effort="max")

    def test_invalid_effort_typo_rejected(self):
        with pytest.raises(ValueError, match="Invalid effort"):
            CodexCLIAdapter(effort="xtra_high")

    def test_default_effort_in_build_command(self, sample_task_instance):
        adapter = CodexCLIAdapter()
        cmd = adapter._build_command(sample_task_instance, task_env=None)
        assert "-c" in cmd
        idx = cmd.index("-c")
        assert cmd[idx + 1] == 'model_reasoning_effort="medium"'

    def test_custom_effort_in_build_command(self, sample_task_instance):
        adapter = CodexCLIAdapter(effort="xhigh")
        cmd = adapter._build_command(sample_task_instance, task_env=None)
        assert "-c" in cmd
        idx = cmd.index("-c")
        assert cmd[idx + 1] == 'model_reasoning_effort="xhigh"'

    def test_default_effort_in_os_command(self, sample_task_instance):
        adapter = CodexCLIAdapter()
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        assert "-c" in cmd
        idx = cmd.index("-c")
        assert cmd[idx + 1] == 'model_reasoning_effort="medium"'

    def test_custom_effort_in_os_command(self, sample_task_instance):
        adapter = CodexCLIAdapter(effort="high")
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        assert "-c" in cmd
        idx = cmd.index("-c")
        assert cmd[idx + 1] == 'model_reasoning_effort="high"'

    def test_effort_via_agent_config(self):
        """effort should be passable via --agent-config JSON."""
        adapter = CodexCLIAdapter(effort="low")
        assert adapter.effort == "low"

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_effort_in_actual_solve(self, mock_run, sample_task_instance):
        """End-to-end: effort flag appears in the command passed to subprocess."""
        adapter = CodexCLIAdapter(effort="xhigh")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter.solve(sample_task_instance)
        cmd = mock_run.call_args.args[0]
        cmd_str = " ".join(cmd)
        assert 'model_reasoning_effort="xhigh"' in cmd_str


class TestClaudeCodeCLIEffort:
    """Tests for reasoning effort propagation in ClaudeCodeCLIAdapter."""

    def test_default_effort_is_medium(self):
        adapter = ClaudeCodeCLIAdapter()
        assert adapter.effort == "medium"

    def test_custom_effort(self):
        adapter = ClaudeCodeCLIAdapter(effort="max")
        assert adapter.effort == "max"

    def test_all_valid_effort_levels(self):
        for level in ("low", "medium", "high", "xhigh", "max"):
            adapter = ClaudeCodeCLIAdapter(effort=level)
            assert adapter.effort == level

    def test_invalid_effort_rejected(self):
        with pytest.raises(ValueError, match="Invalid effort 'none' for claude_code_cli"):
            ClaudeCodeCLIAdapter(effort="none")

    def test_invalid_effort_typo_rejected(self):
        with pytest.raises(ValueError, match="Invalid effort"):
            ClaudeCodeCLIAdapter(effort="maximum")

    def test_default_effort_in_os_command(self, sample_task_instance):
        adapter = ClaudeCodeCLIAdapter()
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        assert "--effort" in cmd
        idx = cmd.index("--effort")
        assert cmd[idx + 1] == "medium"

    def test_custom_effort_in_os_command(self, sample_task_instance):
        adapter = ClaudeCodeCLIAdapter(effort="xhigh")
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        assert "--effort" in cmd
        idx = cmd.index("--effort")
        assert cmd[idx + 1] == "xhigh"

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_default_effort_in_build_command(self, mock_run, sample_task_instance):
        adapter = ClaudeCodeCLIAdapter()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter.solve(sample_task_instance)
        cmd = mock_run.call_args.args[0]
        assert "--effort" in cmd
        idx = cmd.index("--effort")
        assert cmd[idx + 1] == "medium"

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_custom_effort_in_build_command(self, mock_run, sample_task_instance):
        adapter = ClaudeCodeCLIAdapter(effort="high")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter.solve(sample_task_instance)
        cmd = mock_run.call_args.args[0]
        assert "--effort" in cmd
        idx = cmd.index("--effort")
        assert cmd[idx + 1] == "high"

    def test_effort_via_agent_config(self):
        """effort should be passable via --agent-config JSON."""
        adapter = ClaudeCodeCLIAdapter(effort="low")
        assert adapter.effort == "low"


class TestDeriveAgentLabelWithEffort:
    """Tests for derive_agent_label including model + effort for CLI adapters."""

    def test_codex_with_model_and_effort(self):
        from ai4sci_bench.reporting.result_loader import derive_agent_label
        label = derive_agent_label("codex_cli", {"model": "gpt-5.5", "effort": "xhigh"})
        assert label == "codex_cli_gpt-5.5_xhigh"

    def test_codex_with_model_only(self):
        from ai4sci_bench.reporting.result_loader import derive_agent_label
        label = derive_agent_label("codex_cli", {"model": "gpt-5.5"})
        assert label == "codex_cli_gpt-5.5"

    def test_codex_empty_config(self):
        from ai4sci_bench.reporting.result_loader import derive_agent_label
        label = derive_agent_label("codex_cli", {})
        assert label == "codex_cli"

    def test_claude_with_model_and_effort(self):
        from ai4sci_bench.reporting.result_loader import derive_agent_label
        label = derive_agent_label("claude_code_cli", {"model": "claude-opus-4-6", "effort": "high"})
        assert label == "claude_code_cli_claude-opus-4-6_high"

    def test_claude_empty_config(self):
        from ai4sci_bench.reporting.result_loader import derive_agent_label
        label = derive_agent_label("claude_code_cli", {})
        assert label == "claude_code_cli"

    def test_direct_llm_unchanged(self):
        from ai4sci_bench.reporting.result_loader import derive_agent_label
        label = derive_agent_label("direct_llm", {"model": "claude-opus-4-6"})
        assert label == "claude-opus-4-6"

    def test_unknown_agent_unchanged(self):
        from ai4sci_bench.reporting.result_loader import derive_agent_label
        label = derive_agent_label("my_custom_agent", {"effort": "high"})
        assert label == "my_custom_agent"


class TestAgentBanner:
    """Tests for _print_agent_banner and _get_cli_version."""

    def test_banner_prints_effort_with_default_suffix(self, capsys):
        from ai4sci_bench.cli import _print_agent_banner
        adapter = CodexCLIAdapter(effort="medium")
        _print_agent_banner(adapter, "codex_cli", {})
        output = capsys.readouterr().out
        assert "medium (default)" in output

    def test_banner_prints_effort_without_default_when_explicit(self, capsys):
        from ai4sci_bench.cli import _print_agent_banner
        adapter = CodexCLIAdapter(effort="xhigh")
        _print_agent_banner(adapter, "codex_cli", {"effort": "xhigh"})
        output = capsys.readouterr().out
        assert "xhigh" in output
        assert "(default)" not in output

    def test_banner_prints_model(self, capsys):
        from ai4sci_bench.cli import _print_agent_banner
        adapter = ClaudeCodeCLIAdapter(model="claude-sonnet-4-6")
        _print_agent_banner(adapter, "claude_code_cli", {"model": "claude-sonnet-4-6"})
        output = capsys.readouterr().out
        assert "claude-sonnet-4-6" in output

    def test_banner_prints_tool_mode(self, capsys):
        from ai4sci_bench.cli import _print_agent_banner
        adapter = CodexCLIAdapter()
        _print_agent_banner(adapter, "codex_cli", {})
        output = capsys.readouterr().out
        assert "restricted" in output

    def test_get_cli_version_not_found(self):
        from ai4sci_bench.cli import _get_cli_version
        result = _get_cli_version("nonexistent_binary_xyz_123")
        assert result == "not found"

    def test_get_cli_version_codex(self):
        from ai4sci_bench.cli import _get_cli_version
        import shutil
        if shutil.which("codex"):
            result = _get_cli_version("codex")
            assert result != "not found"
            assert result != "unknown"
        else:
            pytest.skip("codex not installed")

    def test_get_cli_version_claude(self):
        from ai4sci_bench.cli import _get_cli_version
        import shutil
        if shutil.which("claude"):
            result = _get_cli_version("claude")
            assert result != "not found"
            assert result != "unknown"
        else:
            pytest.skip("claude not installed")

    def test_banner_direct_llm_no_cli_line(self, capsys):
        """direct_llm has no CLI binary — CLI line should be omitted."""
        from ai4sci_bench.cli import _print_agent_banner
        adapter = DirectLLMAdapter(model="claude-opus-4-6")
        _print_agent_banner(adapter, "direct_llm", {"model": "claude-opus-4-6"})
        output = capsys.readouterr().out
        assert "CLI" not in output
        assert "claude-opus-4-6" in output

    def test_banner_agent_cmd_graceful(self, capsys):
        """--agent-cmd adapter has no model/effort — banner should show N/A."""
        from ai4sci_bench.cli import _print_agent_banner
        from ai4sci_bench.adapters.cli_agent import CLIAgentAdapter
        adapter = CLIAgentAdapter(cmd_template="echo hello")
        _print_agent_banner(adapter, None, {})
        output = capsys.readouterr().out
        assert "CLIAgentAdapter" in output
        assert "N/A" in output

    def test_banner_timeout_value(self, capsys):
        from ai4sci_bench.cli import _print_agent_banner
        adapter = CodexCLIAdapter(timeout_seconds=7200)
        _print_agent_banner(adapter, "codex_cli", {})
        output = capsys.readouterr().out
        assert "7200s" in output

    def test_banner_prefers_cli_timeout_value(self, capsys):
        from ai4sci_bench.cli import _print_agent_banner
        adapter = CodexCLIAdapter(timeout_seconds=7200)
        _print_agent_banner(
            adapter, "codex_cli", {}, cli_timeout_seconds=240
        )
        output = capsys.readouterr().out
        assert "240s" in output
        assert "7200s" not in output


class TestCodexEffortWithToolIsolation:
    """Verify effort `-c` flag coexists correctly with tool isolation flags."""

    def test_effort_and_ignore_user_config_both_present(self, sample_task_instance):
        """Restricted mode adds --ignore-user-config; -c must also be present."""
        adapter = CodexCLIAdapter(effort="xhigh")
        cmd = adapter._build_command(sample_task_instance, task_env=None)
        cmd_str = " ".join(cmd)
        assert "--ignore-user-config" in cmd_str
        assert 'model_reasoning_effort="xhigh"' in cmd_str

    def test_effort_and_ignore_user_config_in_os_cmd(self, sample_task_instance):
        """OS sandbox also has tool isolation; -c must coexist."""
        adapter = CodexCLIAdapter(effort="high")
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        cmd_str = " ".join(cmd)
        assert "--ignore-user-config" in cmd_str
        assert 'model_reasoning_effort="high"' in cmd_str

    def test_effort_with_unrestricted_no_ignore_user_config(self, sample_task_instance):
        """Unrestricted mode: no --ignore-user-config, but -c still present."""
        adapter = CodexCLIAdapter(effort="xhigh", tool_mode="unrestricted")
        cmd = adapter._build_command(sample_task_instance, task_env=None)
        cmd_str = " ".join(cmd)
        assert "--ignore-user-config" not in cmd_str
        assert 'model_reasoning_effort="xhigh"' in cmd_str

    def test_effort_with_search_mode(self, sample_task_instance):
        """Search mode: --ignore-user-config present, web_search NOT disabled, -c present."""
        adapter = CodexCLIAdapter(effort="medium", tool_mode="search")
        cmd = adapter._build_command(sample_task_instance, task_env=None)
        cmd_str = " ".join(cmd)
        assert "--ignore-user-config" in cmd_str
        assert 'web_search="disabled"' not in cmd_str
        assert 'model_reasoning_effort="medium"' in cmd_str


class TestClaudeEffortWithToolIsolation:
    """Verify effort --effort flag coexists correctly with tool isolation."""

    def test_effort_and_tools_whitelist_both_present(self, sample_task_instance):
        adapter = ClaudeCodeCLIAdapter(effort="high")
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        assert "--effort" in cmd
        assert "--tools" in cmd
        idx = cmd.index("--effort")
        assert cmd[idx + 1] == "high"

    def test_effort_with_unrestricted_no_tools_whitelist(self, sample_task_instance):
        adapter = ClaudeCodeCLIAdapter(effort="xhigh", tool_mode="unrestricted")
        cmd = adapter._build_os_agent_cmd(sample_task_instance.workspace_dir)
        assert "--effort" in cmd
        idx = cmd.index("--effort")
        assert cmd[idx + 1] == "xhigh"
        assert "--tools" not in cmd


class TestEffortValidationCrossAdapter:
    """Verify that effort levels unique to one adapter are rejected by the other."""

    def test_max_valid_for_claude_rejected_by_codex(self):
        ClaudeCodeCLIAdapter(effort="max")  # should work
        with pytest.raises(ValueError, match="Invalid effort 'max' for codex_cli"):
            CodexCLIAdapter(effort="max")

    def test_none_valid_for_codex_rejected_by_claude(self):
        CodexCLIAdapter(effort="none")  # should work
        with pytest.raises(ValueError, match="Invalid effort 'none' for claude_code_cli"):
            ClaudeCodeCLIAdapter(effort="none")

    def test_minimal_valid_for_codex_rejected_by_claude(self):
        CodexCLIAdapter(effort="minimal")  # should work
        with pytest.raises(ValueError, match="Invalid effort 'minimal' for claude_code_cli"):
            ClaudeCodeCLIAdapter(effort="minimal")

    def test_build_agent_rejects_invalid_codex_effort(self):
        from ai4sci_bench.cli import _build_agent
        with pytest.raises(ValueError, match="Invalid effort 'max' for codex_cli"):
            _build_agent(None, "codex_cli", {"effort": "max"})

    def test_build_agent_rejects_invalid_claude_effort(self):
        from ai4sci_bench.cli import _build_agent
        with pytest.raises(ValueError, match="Invalid effort 'none' for claude_code_cli"):
            _build_agent(None, "claude_code_cli", {"effort": "none"})


class TestEffortThroughBuildAgent:
    """End-to-end: effort flows from CLI _build_agent through to adapter."""

    def test_codex_effort_via_build_agent(self):
        from ai4sci_bench.cli import _build_agent
        adapter = _build_agent(None, "codex_cli", {"effort": "xhigh"})
        assert adapter.effort == "xhigh"
        assert isinstance(adapter, CodexCLIAdapter)

    def test_claude_effort_via_build_agent(self):
        from ai4sci_bench.cli import _build_agent
        adapter = _build_agent(None, "claude_code_cli", {"effort": "high"})
        assert adapter.effort == "high"
        assert isinstance(adapter, ClaudeCodeCLIAdapter)

    def test_codex_default_effort_via_build_agent(self):
        from ai4sci_bench.cli import _build_agent
        adapter = _build_agent(None, "codex_cli", {})
        assert adapter.effort == "medium"

    def test_claude_default_effort_via_build_agent(self):
        from ai4sci_bench.cli import _build_agent
        adapter = _build_agent(None, "claude_code_cli", {})
        assert adapter.effort == "medium"


class TestLabelDifferentEfforts:
    """Labels must distinguish different effort levels for the same agent+model."""

    def test_different_efforts_produce_different_labels(self):
        from ai4sci_bench.reporting.result_loader import derive_agent_label
        l1 = derive_agent_label("codex_cli", {"model": "gpt-5.5", "effort": "medium"})
        l2 = derive_agent_label("codex_cli", {"model": "gpt-5.5", "effort": "xhigh"})
        assert l1 != l2
        assert "medium" in l1
        assert "xhigh" in l2

    def test_effort_only_difference_still_distinct(self):
        from ai4sci_bench.reporting.result_loader import derive_agent_label
        l1 = derive_agent_label("claude_code_cli", {"effort": "low"})
        l2 = derive_agent_label("claude_code_cli", {"effort": "high"})
        assert l1 != l2
