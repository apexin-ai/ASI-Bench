"""invoke(): running the Claude / Codex CLI without a benchmark task."""
from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter
from ai4sci_bench.adapters.invocation import Invocation
from ai4sci_bench.core.types import RunStatus

RUN = "ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout"

CLAUDE_OK = "\n".join(json.dumps(e) for e in (
    {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}},
    {"type": "result", "subtype": "success", "is_error": False, "result": "done",
     "usage": {"input_tokens": 10, "output_tokens": 5}},
)) + "\n"


def _adapter(cls, tmp_path: Path, **kwargs):
    adapter = cls(**kwargs)
    adapter.setup({"sandbox": "none", "repo_root": str(tmp_path / "root")})
    return adapter


def _call(mock_run):
    args, kwargs = mock_run.call_args
    return args[0], kwargs


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


class TestClaudeInvoke:
    @patch(RUN)
    def test_prompt_goes_on_stdin_and_runs_in_workspace(self, mock_run, tmp_path, workspace):
        mock_run.return_value = MagicMock(returncode=0, stdout=CLAUDE_OK, stderr="")
        adapter = _adapter(ClaudeCodeCLIAdapter, tmp_path, model="m", effort="high")

        result = adapter.invoke(Invocation(prompt="build it", workspace=workspace, run_key="r1"))

        cmd, kwargs = _call(mock_run)
        assert kwargs["input"] == "build it"
        assert kwargs["cwd"] == str(workspace.resolve())
        assert cmd[:5] == ["claude", "--model", "m", "--effort", "high"]
        assert "--system-prompt" not in cmd and "--append-system-prompt" not in cmd
        assert result.status == RunStatus.COMPLETED
        assert result.returncode == 0
        assert result.cost is not None

    @pytest.mark.parametrize("mode, flag", [
        ("append", "--append-system-prompt"),
        ("replace", "--system-prompt"),
    ])
    @patch(RUN)
    def test_system_prompt_mode(self, mock_run, mode, flag, tmp_path, workspace):
        mock_run.return_value = MagicMock(returncode=0, stdout=CLAUDE_OK, stderr="")
        adapter = _adapter(ClaudeCodeCLIAdapter, tmp_path)

        adapter.invoke(Invocation(
            prompt="p", workspace=workspace, run_key="r",
            system_prompt="ROLE", system_prompt_mode=mode,
        ))

        cmd, _ = _call(mock_run)
        assert cmd[cmd.index(flag) + 1] == "ROLE"

    @patch(RUN)
    def test_allowed_tools_and_permission_mode(self, mock_run, tmp_path, workspace):
        mock_run.return_value = MagicMock(returncode=0, stdout=CLAUDE_OK, stderr="")
        adapter = _adapter(
            ClaudeCodeCLIAdapter, tmp_path,
            permission_mode="dontAsk", tool_mode="unrestricted",
        )

        adapter.invoke(Invocation(
            prompt="p", workspace=workspace, run_key="r", allowed_tools="Bash,Read",
        ))

        cmd, _ = _call(mock_run)
        assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
        assert cmd[cmd.index("--allowedTools") + 1] == "Bash,Read"
        assert "--tools" not in cmd  # unrestricted: no isolation flags

    @patch(RUN)
    def test_isolated_home_is_keyed_on_run_key(self, mock_run, tmp_path, workspace):
        mock_run.return_value = MagicMock(returncode=0, stdout=CLAUDE_OK, stderr="")
        adapter = _adapter(ClaudeCodeCLIAdapter, tmp_path, tool_mode="restricted")

        adapter.invoke(Invocation(prompt="p", workspace=workspace, run_key="gen-1"))
        home_1 = _call(mock_run)[1]["env"]["HOME"]
        adapter.invoke(Invocation(prompt="p", workspace=workspace, run_key="gen-2"))
        home_2 = _call(mock_run)[1]["env"]["HOME"]

        assert home_1 != home_2
        assert "gen-1" in home_1 and "gen-2" in home_2

    @patch(RUN)
    def test_base_env_is_layered_under_api_env(self, mock_run, tmp_path, workspace):
        mock_run.return_value = MagicMock(returncode=0, stdout=CLAUDE_OK, stderr="")
        adapter = _adapter(
            ClaudeCodeCLIAdapter, tmp_path, tool_mode="unrestricted",
            api_key="k", api_base="https://gw.example/v1", api_protocol="anthropic",
        )

        adapter.invoke(Invocation(
            prompt="p", workspace=workspace, run_key="r", env={"ONLY_ME": "1"},
        ))

        env = _call(mock_run)[1]["env"]
        assert env["ONLY_ME"] == "1"
        assert env["ANTHROPIC_BASE_URL"] == "https://gw.example"

    @patch(RUN)
    def test_zero_exit_with_error_result_is_failed(self, mock_run, tmp_path, workspace):
        stdout = json.dumps({
            "type": "result", "subtype": "success", "is_error": True,
            "result": "API Error: 529 overloaded",
        }) + "\n"
        mock_run.return_value = MagicMock(returncode=0, stdout=stdout, stderr="")
        adapter = _adapter(ClaudeCodeCLIAdapter, tmp_path)

        result = adapter.invoke(Invocation(prompt="p", workspace=workspace, run_key="r"))

        assert result.status == RunStatus.FAILED
        assert result.error_message

    @patch(RUN)
    def test_timeout(self, mock_run, tmp_path, workspace):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=7)
        adapter = _adapter(ClaudeCodeCLIAdapter, tmp_path)

        result = adapter.invoke(Invocation(
            prompt="p", workspace=workspace, run_key="r", timeout_seconds=7,
        ))

        assert result.status == RunStatus.TIMEOUT
        assert _call(mock_run)[1]["timeout"] == 7

    def test_solve_without_prompt_file_still_raises(self, tmp_path, workspace):
        from ai4sci_bench.core.types import PromptLevel, TaskInstance
        adapter = _adapter(ClaudeCodeCLIAdapter, tmp_path)
        instance = TaskInstance(
            task_id="t", instance_id="t__i", task_dir=tmp_path, workspace_dir=workspace,
            reference_dir=tmp_path, prompt_level=PromptLevel.B2, parameters={},
            metadata={}, effective_timeout_seconds=5,
        )
        with pytest.raises(FileNotFoundError):
            adapter.solve(instance)
        adapter.sandbox = "linux_ns"
        with pytest.raises(FileNotFoundError):
            adapter._build_command(instance, None)

    def test_container_sandboxes_are_rejected(self, tmp_path, workspace):
        adapter = ClaudeCodeCLIAdapter()
        adapter.sandbox = "linux_ns"
        with pytest.raises(ValueError, match="linux_ns"):
            adapter.invoke(Invocation(prompt="p", workspace=workspace, run_key="r"))


class TestCodexInvoke:
    @patch(RUN)
    def test_prompt_goes_on_stdin(self, mock_run, tmp_path, workspace):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter = _adapter(CodexCLIAdapter, tmp_path, model="gpt", effort="high")

        result = adapter.invoke(Invocation(prompt="build it", workspace=workspace, run_key="r"))

        cmd, kwargs = _call(mock_run)
        assert kwargs["input"] == "build it"
        assert cmd[:3] == ["codex", "exec", "--model"]
        assert cmd[cmd.index("--cd") + 1] == str(workspace.resolve())
        assert cmd[-2:] == ["--", "-"]
        assert not any("instructions" in part for part in cmd)
        assert result.status == RunStatus.COMPLETED

    @patch(RUN)
    def test_append_uses_developer_instructions(self, mock_run, tmp_path, workspace):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter = _adapter(CodexCLIAdapter, tmp_path)
        role = 'Line one.\nSay "hi".'

        adapter.invoke(Invocation(
            prompt="p", workspace=workspace, run_key="r", system_prompt=role,
        ))

        cmd, _ = _call(mock_run)
        # codex exec has no --instructions flag; 0.153.4 rejects it outright.
        assert "--instructions" not in cmd
        configs = [cmd[i + 1] for i, part in enumerate(cmd) if part == "--config"]
        (setting,) = [c for c in configs if c.startswith("developer_instructions=")]
        # Codex parses the override as TOML.
        assert tomllib.loads(setting)["developer_instructions"] == role
        assert cmd.index("--config") < cmd.index("--")

    @patch(RUN)
    def test_replace_writes_model_instructions_file(self, mock_run, tmp_path, workspace):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter = _adapter(CodexCLIAdapter, tmp_path)

        adapter.invoke(Invocation(
            prompt="p", workspace=workspace, run_key="gen-7",
            system_prompt="ROLE", system_prompt_mode="replace",
        ))

        cmd, _ = _call(mock_run)
        configs = [cmd[i + 1] for i, part in enumerate(cmd) if part == "--config"]
        (setting,) = [c for c in configs if c.startswith("model_instructions_file=")]
        path = Path(tomllib.loads(setting)["model_instructions_file"])
        assert path.read_text(encoding="utf-8") == "ROLE"
        assert "gen-7" in path.name

    @patch(RUN)
    def test_non_bmp_and_control_characters_stay_valid_toml(self, mock_run, tmp_path, workspace):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter = _adapter(CodexCLIAdapter, tmp_path)
        role = "角色：审稿人 🧪\ttab \\ backslash \u0001 ctrl"

        adapter.invoke(Invocation(
            prompt="p", workspace=workspace, run_key="r", system_prompt=role,
        ))

        cmd, _ = _call(mock_run)
        (setting,) = [
            cmd[i + 1] for i, part in enumerate(cmd)
            if part == "--config" and cmd[i + 1].startswith("developer_instructions=")
        ]
        assert tomllib.loads(setting)["developer_instructions"] == role

    def test_allowed_tools_is_rejected(self, tmp_path, workspace):
        adapter = _adapter(CodexCLIAdapter, tmp_path)
        with pytest.raises(ValueError, match="allow-list"):
            adapter.invoke(Invocation(
                prompt="p", workspace=workspace, run_key="r", allowed_tools="Bash",
            ))

    @patch(RUN)
    def test_codex_home_passes_through_when_unrestricted(self, mock_run, tmp_path, workspace):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        home = tmp_path / "codex-home"
        home.mkdir()
        adapter = _adapter(
            CodexCLIAdapter, tmp_path, codex_home=str(home), tool_mode="unrestricted",
        )

        adapter.invoke(Invocation(prompt="p", workspace=workspace, run_key="r"))

        cmd, kwargs = _call(mock_run)
        assert kwargs["env"]["CODEX_HOME"] == str(home)
        assert "--ignore-user-config" not in cmd

    def test_container_sandboxes_are_rejected(self, workspace):
        adapter = CodexCLIAdapter()
        adapter.sandbox = "os"
        with pytest.raises(ValueError, match="'os'"):
            adapter.invoke(Invocation(prompt="p", workspace=workspace, run_key="r"))


def _task(tmp_path: Path, workspace: Path, prompt: str):
    from ai4sci_bench.core.types import PromptLevel, TaskInstance
    (workspace / "prompt.md").write_text(prompt, encoding="utf-8")
    return TaskInstance(
        task_id="t", instance_id="t__i", task_dir=tmp_path, workspace_dir=workspace,
        reference_dir=tmp_path, prompt_level=PromptLevel.B2, parameters={},
        metadata={}, effective_timeout_seconds=5,
    )


@pytest.mark.parametrize("cls, stdout", [
    (ClaudeCodeCLIAdapter, CLAUDE_OK),
    (CodexCLIAdapter, ""),
])
@patch(RUN)
def test_solve_and_invoke_launch_the_same_process(mock_run, cls, stdout, tmp_path, workspace):
    """solve() keeps its own hooks; this pins it to the same command as invoke()."""
    mock_run.return_value = MagicMock(returncode=0, stdout=stdout, stderr="")
    adapter = _adapter(cls, tmp_path, tool_mode="restricted")
    instance = _task(tmp_path, workspace, "solve the task")

    adapter.solve(instance)
    solve_cmd, solve_kwargs = _call(mock_run)
    adapter.invoke(Invocation(
        prompt="solve the task", workspace=workspace, run_key=instance.run_key,
    ))
    invoke_cmd, invoke_kwargs = _call(mock_run)

    assert solve_cmd == invoke_cmd
    assert solve_kwargs["input"] == invoke_kwargs["input"]
    assert solve_kwargs["env"]["HOME"] == invoke_kwargs["env"]["HOME"]
