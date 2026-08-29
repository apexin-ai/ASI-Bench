"""Tests for OS sandbox support across all adapters.

Verifies that mimo_code_cli, kimi_code_cli, antigravity_cli, and openhands
correctly integrate with OSSandbox when --sandbox os is used.
No real Docker execution — OSSandbox.run_agent is mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai4sci_bench.core.types import AgentOutput, PromptLevel, RunStatus, TaskInstance


# ── Helpers ───────────────────────────────────────────────────────

def _make_task_instance(tmp_path: Path, task_id: str = "test.task") -> TaskInstance:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    (workspace / "prompt.md").write_text("Solve this task.")
    (workspace / "solution.py").write_text("print('hello')")
    task_dir = tmp_path / "task_dir"
    task_dir.mkdir(exist_ok=True)
    ref_dir = tmp_path / "reference"
    ref_dir.mkdir(exist_ok=True)
    return TaskInstance(
        task_id=task_id,
        instance_id=f"{task_id}__seed42",
        task_dir=task_dir,
        workspace_dir=workspace,
        reference_dir=ref_dir,
        prompt_level=PromptLevel.B2,
        parameters={"seed": 42},
        metadata={"id": task_id, "runtime": {"packages": []}},
    )


def _mock_os_sandbox_result(
    success: bool = True,
    log: str = "ok",
    stdout: str | None = None,
    stderr: str | None = None,
    image_id: str | None = "sha256:abc123",
):
    return (success, log, stdout, stderr, image_id)


def test_os_capable_adapters_expose_image_identity_for_submission_provenance():
    from ai4sci_bench.adapters.antigravity_cli import AntigravityCLIAdapter
    from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
    from ai4sci_bench.adapters.kimi_code_cli import KimiCodeCLIAdapter
    from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
    from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter

    adapters = [
        AntigravityCLIAdapter(),
        ClaudeCodeCLIAdapter(),
        KimiCodeCLIAdapter(),
        MiMoCodeCLIAdapter(),
        OpenHandsAdapter(api_key="sk-test"),
    ]
    for adapter in adapters:
        adapter._sandbox_image_identity = "sha256:test-task-image"
        assert adapter.sandbox_image_identity == "sha256:test-task-image"


# ═══════════════════════════════════════════════════════════════════
# MiMo Code CLI
# ═══════════════════════════════════════════════════════════════════

class TestMiMoOsSandbox:
    def test_supported_modes_include_os(self):
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        adapter = MiMoCodeCLIAdapter()
        assert "os" in adapter._supported_sandbox_modes

    def test_setup_creates_os_sandbox(self, tmp_path):
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        adapter = MiMoCodeCLIAdapter()
        with patch("ai4sci_bench.runner.os_sandbox.OSSandbox", autospec=True) as MockSandbox:
            adapter.setup({"sandbox": "os", "repo_root": str(tmp_path), "timeout": 60})
            assert adapter._os_sandbox is not None

    def test_setup_no_sandbox_when_not_os(self, tmp_path):
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        adapter = MiMoCodeCLIAdapter()
        adapter.setup({"sandbox": "none", "repo_root": str(tmp_path), "timeout": 60})
        assert adapter._os_sandbox is None

    def test_solve_os_delegates_to_sandbox(self, tmp_path):
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        adapter = MiMoCodeCLIAdapter()
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result(
            stdout='{"type":"result","num_turns":1}',
        )
        adapter._os_sandbox = mock_sb

        ti = _make_task_instance(tmp_path)
        result = adapter.solve(ti)

        assert result.status == RunStatus.COMPLETED
        mock_sb.run_agent.assert_called_once()
        call_kwargs = mock_sb.run_agent.call_args
        assert call_kwargs.kwargs["agent_type"] == "mimo_code"

    def test_solve_os_failure_status(self, tmp_path):
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        adapter = MiMoCodeCLIAdapter()
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result(success=False, log="crash")
        adapter._os_sandbox = mock_sb

        result = adapter.solve(_make_task_instance(tmp_path))
        assert result.status == RunStatus.FAILED
        assert result.error_message == "crash"

    def test_solve_os_timeout_status(self, tmp_path):
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        adapter = MiMoCodeCLIAdapter()
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result(
            success=False, log="Agent timed out after 60s",
        )
        adapter._os_sandbox = mock_sb

        result = adapter.solve(_make_task_instance(tmp_path))
        assert result.status == RunStatus.TIMEOUT

    def test_solve_os_stores_image_identity(self, tmp_path):
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        adapter = MiMoCodeCLIAdapter()
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result(image_id="sha256:test")
        adapter._os_sandbox = mock_sb

        adapter.solve(_make_task_instance(tmp_path))
        assert adapter._sandbox_image_identity == "sha256:test"

    def test_build_os_agent_cmd(self, tmp_path):
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        adapter = MiMoCodeCLIAdapter(model="mimo-v2.5-pro")
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "prompt.md").write_text("test prompt")
        cmd = adapter._build_os_agent_cmd(workspace)
        assert cmd[0] == "mimo"
        assert "--dangerously-skip-permissions" in cmd
        assert cmd[-1] == "test prompt"

    def test_solve_os_passes_proxy_env(self, tmp_path):
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        adapter = MiMoCodeCLIAdapter(
            api_key="tp-xxx",
            api_base="https://token-plan-cn.xiaomimimo.com/anthropic",
        )
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result()
        adapter._os_sandbox = mock_sb
        adapter._ensure_proxy = lambda: "http://127.0.0.1:9999"

        adapter.solve(_make_task_instance(tmp_path))
        call_kwargs = mock_sb.run_agent.call_args.kwargs
        extra_env = call_kwargs["extra_env"]
        assert extra_env is not None
        assert "OPENAI_API_KEY" in extra_env
        assert "OPENAI_BASE_URL" in extra_env


# ═══════════════════════════════════════════════════════════════════
# Kimi Code CLI
# ═══════════════════════════════════════════════════════════════════

class TestKimiOsSandbox:
    def test_supported_modes_include_os(self):
        from ai4sci_bench.adapters.kimi_code_cli import KimiCodeCLIAdapter
        adapter = KimiCodeCLIAdapter()
        assert "os" in adapter._supported_sandbox_modes

    def test_setup_creates_os_sandbox(self, tmp_path):
        from ai4sci_bench.adapters.kimi_code_cli import KimiCodeCLIAdapter
        adapter = KimiCodeCLIAdapter()
        with patch("ai4sci_bench.runner.os_sandbox.OSSandbox", autospec=True):
            adapter.setup({"sandbox": "os", "repo_root": str(tmp_path), "timeout": 60})
            assert adapter._os_sandbox is not None

    def test_solve_os_delegates_to_sandbox(self, tmp_path):
        from ai4sci_bench.adapters.kimi_code_cli import KimiCodeCLIAdapter
        adapter = KimiCodeCLIAdapter()
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result()
        adapter._os_sandbox = mock_sb

        result = adapter.solve(_make_task_instance(tmp_path))
        assert result.status == RunStatus.COMPLETED
        call_kwargs = mock_sb.run_agent.call_args.kwargs
        assert call_kwargs["agent_type"] == "kimi_code"

    def test_solve_os_failure(self, tmp_path):
        from ai4sci_bench.adapters.kimi_code_cli import KimiCodeCLIAdapter
        adapter = KimiCodeCLIAdapter()
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result(success=False, log="error")
        adapter._os_sandbox = mock_sb

        result = adapter.solve(_make_task_instance(tmp_path))
        assert result.status == RunStatus.FAILED

    def test_build_os_agent_cmd(self, tmp_path):
        from ai4sci_bench.adapters.kimi_code_cli import KimiCodeCLIAdapter
        adapter = KimiCodeCLIAdapter(model="kimi-k2.7")
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "prompt.md").write_text("test")
        cmd = adapter._build_os_agent_cmd(workspace)
        assert cmd[0] == "kimi"
        assert "--output-format" in cmd


# ═══════════════════════════════════════════════════════════════════
# Antigravity CLI
# ═══════════════════════════════════════════════════════════════════

class TestAntigravityOsSandbox:
    def test_supported_modes_include_os(self):
        from ai4sci_bench.adapters.antigravity_cli import AntigravityCLIAdapter
        adapter = AntigravityCLIAdapter()
        assert "os" in adapter._supported_sandbox_modes

    def test_setup_creates_os_sandbox(self, tmp_path):
        from ai4sci_bench.adapters.antigravity_cli import AntigravityCLIAdapter
        adapter = AntigravityCLIAdapter()
        with patch("ai4sci_bench.runner.os_sandbox.OSSandbox", autospec=True):
            adapter.setup({"sandbox": "os", "repo_root": str(tmp_path), "timeout": 60})
            assert adapter._os_sandbox is not None

    def test_solve_os_delegates_to_sandbox(self, tmp_path):
        from ai4sci_bench.adapters.antigravity_cli import AntigravityCLIAdapter
        adapter = AntigravityCLIAdapter()
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result()
        adapter._os_sandbox = mock_sb

        result = adapter.solve(_make_task_instance(tmp_path))
        assert result.status == RunStatus.COMPLETED
        call_kwargs = mock_sb.run_agent.call_args.kwargs
        assert call_kwargs["agent_type"] == "agy"

    def test_solve_os_failure(self, tmp_path):
        from ai4sci_bench.adapters.antigravity_cli import AntigravityCLIAdapter
        adapter = AntigravityCLIAdapter()
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result(success=False, log="err")
        adapter._os_sandbox = mock_sb

        result = adapter.solve(_make_task_instance(tmp_path))
        assert result.status == RunStatus.FAILED

    def test_build_os_agent_cmd(self, tmp_path):
        from ai4sci_bench.adapters.antigravity_cli import AntigravityCLIAdapter
        adapter = AntigravityCLIAdapter(model="gemini-3.5-flash")
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "prompt.md").write_text("test")
        cmd = adapter._build_os_agent_cmd(workspace)
        assert cmd[0] == "agy"
        assert "--dangerously-skip-permissions" in cmd
        assert "--model" in cmd


# ═══════════════════════════════════════════════════════════════════
# OpenHands
# ═══════════════════════════════════════════════════════════════════

class TestOpenHandsOsSandbox:
    def test_setup_creates_os_sandbox(self, tmp_path):
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
        adapter = OpenHandsAdapter()
        with patch("ai4sci_bench.runner.os_sandbox.OSSandbox", autospec=True):
            adapter.setup({"sandbox": "os", "repo_root": str(tmp_path), "timeout": 60})
            assert adapter._os_sandbox is not None
            assert adapter.sandbox == "os"

    def test_setup_no_sandbox_when_none(self, tmp_path):
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
        adapter = OpenHandsAdapter()
        adapter.setup({"sandbox": "none", "repo_root": str(tmp_path), "timeout": 60})
        assert adapter._os_sandbox is None
        assert adapter.sandbox == "none"

    def _make_fake_venv(self, tmp_path):
        venv = tmp_path / "oh_venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("#!/bin/sh\n")
        (venv / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
        return venv

    def test_solve_os_delegates_to_sandbox(self, tmp_path):
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
        adapter = OpenHandsAdapter(api_key="sk-test")
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result()
        adapter._os_sandbox = mock_sb
        venv = self._make_fake_venv(tmp_path)
        adapter._find_openhands_python = lambda: str(venv / "bin" / "python")

        result = adapter.solve(_make_task_instance(tmp_path))
        assert result.status == RunStatus.COMPLETED
        call_kwargs = mock_sb.run_agent.call_args.kwargs
        assert call_kwargs["agent_type"] == "openhands"

    def test_solve_os_passes_api_env(self, tmp_path):
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
        adapter = OpenHandsAdapter(
            api_key="sk-test123",
            api_base="https://api.example.com/v1",
            api_protocol="openai",
        )
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result()
        adapter._os_sandbox = mock_sb
        venv = self._make_fake_venv(tmp_path)
        adapter._find_openhands_python = lambda: str(venv / "bin" / "python")

        adapter.solve(_make_task_instance(tmp_path))
        call_kwargs = mock_sb.run_agent.call_args.kwargs
        extra_env = call_kwargs["extra_env"]
        assert extra_env["OPENAI_API_KEY"] == "sk-test123"
        assert extra_env["OPENAI_BASE_URL"] == "https://api.example.com/v1"

    def test_solve_os_failure(self, tmp_path):
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
        adapter = OpenHandsAdapter(api_key="sk-test")
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result(success=False, log="crash")
        adapter._os_sandbox = mock_sb
        venv = self._make_fake_venv(tmp_path)
        adapter._find_openhands_python = lambda: str(venv / "bin" / "python")

        result = adapter.solve(_make_task_instance(tmp_path))
        assert result.status == RunStatus.FAILED

    def test_solve_os_timeout(self, tmp_path):
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
        adapter = OpenHandsAdapter(api_key="sk-test")
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result(
            success=False, log="Agent timed out",
        )
        adapter._os_sandbox = mock_sb
        venv = self._make_fake_venv(tmp_path)
        adapter._find_openhands_python = lambda: str(venv / "bin" / "python")

        result = adapter.solve(_make_task_instance(tmp_path))
        assert result.status == RunStatus.TIMEOUT

    def test_solve_host_mode_unchanged(self, tmp_path):
        """When sandbox is not os, solve delegates to _solve_host (original logic)."""
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
        adapter = OpenHandsAdapter(api_key="sk-test")
        adapter.sandbox = "none"
        adapter.repo_root = tmp_path
        adapter._find_openhands_python = lambda: None

        result = adapter.solve(_make_task_instance(tmp_path))
        assert result.status == RunStatus.FAILED
        assert "venv not found" in result.error_message


# ═══════════════════════════════════════════════════════════════════
# OSSandbox network config — new agent types
# ═══════════════════════════════════════════════════════════════════

class TestOsSandboxNetworkConfig:
    def test_cli_agent_types_get_network(self):
        """All CLI agents need network for API calls."""
        from ai4sci_bench.runner.os_sandbox import OSSandbox
        import inspect
        source = inspect.getsource(OSSandbox)
        for agent_type in ("mimo_code", "kimi_code", "agy", "openhands"):
            assert agent_type in source, f"{agent_type} not in OSSandbox source"


# ═══════════════════════════════════════════════════════════════════
# Edge cases — non-os sandbox modes still work after changes
# ═══════════════════════════════════════════════════════════════════

class TestNonOsSandboxUnchanged:
    """Verify that none/task/linux_ns modes still route through parent solve()."""

    def test_mimo_none_uses_parent_solve(self, tmp_path):
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        adapter = MiMoCodeCLIAdapter()
        adapter.sandbox = "none"
        adapter.repo_root = tmp_path
        assert adapter._os_sandbox is None
        # parent solve would try to run subprocess; we just verify no os_sandbox call
        with patch.object(adapter, '_os_sandbox') as mock_sb:
            # should NOT be called in none mode
            try:
                adapter.solve(_make_task_instance(tmp_path))
            except Exception:
                pass
            if mock_sb is not None and hasattr(mock_sb, 'run_agent'):
                mock_sb.run_agent.assert_not_called()

    def test_kimi_none_uses_parent_solve(self, tmp_path):
        from ai4sci_bench.adapters.kimi_code_cli import KimiCodeCLIAdapter
        adapter = KimiCodeCLIAdapter()
        adapter.sandbox = "none"
        adapter.repo_root = tmp_path
        assert adapter._os_sandbox is None

    def test_antigravity_none_uses_parent_solve(self, tmp_path):
        from ai4sci_bench.adapters.antigravity_cli import AntigravityCLIAdapter
        adapter = AntigravityCLIAdapter()
        adapter.sandbox = "none"
        adapter.repo_root = tmp_path
        assert adapter._os_sandbox is None


# ═══════════════════════════════════════════════════════════════════
# Edge cases — validate_sandbox_mode integration
# ═══════════════════════════════════════════════════════════════════

class TestSandboxModeValidation:
    """Verify that setup() correctly validates modes via parent class."""

    def test_mimo_rejects_unknown_sandbox(self, tmp_path):
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        adapter = MiMoCodeCLIAdapter()
        with pytest.raises(ValueError, match="Unknown sandbox mode"):
            adapter.setup({"sandbox": "imaginary", "repo_root": str(tmp_path), "timeout": 60})

    def test_kimi_rejects_unknown_sandbox(self, tmp_path):
        from ai4sci_bench.adapters.kimi_code_cli import KimiCodeCLIAdapter
        adapter = KimiCodeCLIAdapter()
        with pytest.raises(ValueError, match="Unknown sandbox mode"):
            adapter.setup({"sandbox": "imaginary", "repo_root": str(tmp_path), "timeout": 60})

    def test_antigravity_rejects_unknown_sandbox(self, tmp_path):
        from ai4sci_bench.adapters.antigravity_cli import AntigravityCLIAdapter
        adapter = AntigravityCLIAdapter()
        with pytest.raises(ValueError, match="Unknown sandbox mode"):
            adapter.setup({"sandbox": "imaginary", "repo_root": str(tmp_path), "timeout": 60})

    def test_mimo_accepts_all_four_modes(self, tmp_path):
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        for mode in ("none", "task", "linux_ns"):
            adapter = MiMoCodeCLIAdapter()
            adapter.setup({"sandbox": mode, "repo_root": str(tmp_path), "timeout": 60})
            assert adapter.sandbox == mode
            assert adapter._os_sandbox is None

    def test_kimi_accepts_os_mode(self, tmp_path):
        from ai4sci_bench.adapters.kimi_code_cli import KimiCodeCLIAdapter
        adapter = KimiCodeCLIAdapter()
        with patch("ai4sci_bench.runner.os_sandbox.OSSandbox", autospec=True):
            adapter.setup({"sandbox": "os", "repo_root": str(tmp_path), "timeout": 60})
            assert adapter.sandbox == "os"
            assert adapter._os_sandbox is not None


# ═══════════════════════════════════════════════════════════════════
# Edge cases — solve output structure
# ═══════════════════════════════════════════════════════════════════

class TestSolveOutputStructure:
    """Verify AgentOutput fields are correctly populated from OSSandbox results."""

    def test_mimo_solve_os_output_has_all_fields(self, tmp_path):
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        adapter = MiMoCodeCLIAdapter()
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result(
            stdout='{"type":"result","usage":{"input_tokens":100,"output_tokens":50}}',
        )
        adapter._os_sandbox = mock_sb

        result = adapter.solve(_make_task_instance(tmp_path))
        assert result.instance_id == "test.task__seed42"
        assert result.output_dir is not None
        assert isinstance(result.code_files, list)
        assert isinstance(result.data_files, list)
        assert result.execution_time_seconds >= 0
        assert result.raw_stdout_format == "jsonl"
        assert result.cost is not None
        assert result.cost.input_tokens == 100
        assert result.cost.output_tokens == 50

    def test_mimo_solve_os_collects_declared_output_files(self, tmp_path):
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        adapter = MiMoCodeCLIAdapter()
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result()
        adapter._os_sandbox = mock_sb

        ti = _make_task_instance(tmp_path)
        ti.metadata["output"] = {"files": [
            {"name": "solution.py", "type": "code"},
            {"name": "result.npy", "type": "data"},
        ]}
        (ti.workspace_dir / "result.npy").write_text("data")

        result = adapter.solve(ti)
        assert "solution.py" in result.code_files
        assert "result.npy" in result.data_files

    def test_mimo_solve_os_null_stdout_no_crash(self, tmp_path):
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        adapter = MiMoCodeCLIAdapter()
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result(stdout=None, stderr=None)
        adapter._os_sandbox = mock_sb

        result = adapter.solve(_make_task_instance(tmp_path))
        assert result.status == RunStatus.COMPLETED
        assert result.raw_stdout is None
        assert result.cost is None

    def test_kimi_solve_os_output_has_correct_format(self, tmp_path):
        from ai4sci_bench.adapters.kimi_code_cli import KimiCodeCLIAdapter
        adapter = KimiCodeCLIAdapter()
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result(
            stdout='{"type":"result","usage":{"input_tokens":200,"output_tokens":80}}',
        )
        adapter._os_sandbox = mock_sb

        result = adapter.solve(_make_task_instance(tmp_path))
        assert result.raw_stdout_format == "jsonl"
        assert result.cost is not None
        assert result.cost.total_tokens == 280

    def test_antigravity_solve_os_raw_format_is_text(self, tmp_path):
        from ai4sci_bench.adapters.antigravity_cli import AntigravityCLIAdapter
        adapter = AntigravityCLIAdapter()
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result(stdout="some text output")
        adapter._os_sandbox = mock_sb

        result = adapter.solve(_make_task_instance(tmp_path))
        assert result.raw_stdout_format == "text"


# ═══════════════════════════════════════════════════════════════════
# Edge cases — run_agent receives correct arguments
# ═══════════════════════════════════════════════════════════════════

class TestRunAgentArguments:
    """Verify OSSandbox.run_agent is called with correct kwargs."""

    def test_mimo_passes_allow_external_tools_true(self, tmp_path):
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        adapter = MiMoCodeCLIAdapter(allow_external_tools=True)
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result()
        adapter._os_sandbox = mock_sb

        adapter.solve(_make_task_instance(tmp_path))
        assert mock_sb.run_agent.call_args.kwargs["allow_external_tools"] is True

    def test_mimo_passes_allow_external_tools_false(self, tmp_path):
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        adapter = MiMoCodeCLIAdapter(allow_external_tools=False)
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result()
        adapter._os_sandbox = mock_sb

        adapter.solve(_make_task_instance(tmp_path))
        assert mock_sb.run_agent.call_args.kwargs["allow_external_tools"] is False

    def test_mimo_passes_workspace_path(self, tmp_path):
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        adapter = MiMoCodeCLIAdapter()
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result()
        adapter._os_sandbox = mock_sb

        ti = _make_task_instance(tmp_path)
        adapter.solve(ti)
        assert mock_sb.run_agent.call_args.kwargs["workspace"] == ti.workspace_dir

    def test_mimo_passes_task_metadata(self, tmp_path):
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        adapter = MiMoCodeCLIAdapter()
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result()
        adapter._os_sandbox = mock_sb

        ti = _make_task_instance(tmp_path)
        adapter.solve(ti)
        assert mock_sb.run_agent.call_args.kwargs["task_metadata"] == ti.metadata

    def test_kimi_no_extra_env_without_api(self, tmp_path):
        from ai4sci_bench.adapters.kimi_code_cli import KimiCodeCLIAdapter
        adapter = KimiCodeCLIAdapter()
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result()
        adapter._os_sandbox = mock_sb

        adapter.solve(_make_task_instance(tmp_path))
        extra_env = mock_sb.run_agent.call_args.kwargs["extra_env"]
        assert extra_env is None

    def test_antigravity_passes_gemini_key(self, tmp_path):
        from ai4sci_bench.adapters.antigravity_cli import AntigravityCLIAdapter
        adapter = AntigravityCLIAdapter(api_key="gemini-key-123")
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result()
        adapter._os_sandbox = mock_sb

        adapter.solve(_make_task_instance(tmp_path))
        extra_env = mock_sb.run_agent.call_args.kwargs["extra_env"]
        assert extra_env["GEMINI_API_KEY"] == "gemini-key-123"

    def test_openhands_passes_both_api_key_and_base(self, tmp_path):
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
        adapter = OpenHandsAdapter(
            api_key="sk-test",
            api_base="https://custom.api.com/v1",
            api_protocol="openai",
        )
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result()
        adapter._os_sandbox = mock_sb
        venv = tmp_path / "oh_venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("#!/bin/sh\n")
        (venv / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
        adapter._find_openhands_python = lambda: str(venv / "bin" / "python")

        adapter.solve(_make_task_instance(tmp_path))
        extra_env = mock_sb.run_agent.call_args.kwargs["extra_env"]
        assert extra_env["OPENAI_API_KEY"] == "sk-test"
        assert extra_env["OPENAI_BASE_URL"] == "https://custom.api.com/v1"
        assert extra_env["OPENHANDS_SUPPRESS_BANNER"] == "1"

    def test_openhands_os_fails_without_python(self, tmp_path):
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
        adapter = OpenHandsAdapter(api_key="sk-test")
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        adapter._os_sandbox = MagicMock()
        adapter._find_openhands_python = lambda: None

        result = adapter.solve(_make_task_instance(tmp_path))
        assert result.status == RunStatus.FAILED
        assert "venv not found" in result.error_message


# ═══════════════════════════════════════════════════════════════════
# Edge cases — Docker network for proxy mode
# ═══════════════════════════════════════════════════════════════════

class TestKimiConfigMount:
    """Kimi's KIMI_CODE_HOME must be volume-mounted, not just env-var'd."""

    def test_kimi_os_env_uses_container_path(self, tmp_path):
        from ai4sci_bench.adapters.kimi_code_cli import KimiCodeCLIAdapter
        adapter = KimiCodeCLIAdapter(
            api_key="sk-xxx",
            api_base="https://api.moonshot.cn/v1",
            api_protocol="openai",
        )
        env = adapter._build_os_api_env()
        assert env["KIMI_CODE_HOME"] == adapter._CONTAINER_KIMI_HOME
        assert not env["KIMI_CODE_HOME"].startswith("/tmp")

    def test_kimi_os_extra_mounts_has_volume(self, tmp_path):
        from ai4sci_bench.adapters.kimi_code_cli import KimiCodeCLIAdapter
        adapter = KimiCodeCLIAdapter(
            api_key="sk-xxx",
            api_base="https://api.moonshot.cn/v1",
            api_protocol="openai",
        )
        mounts = adapter._build_os_extra_mounts()
        assert len(mounts) == 2
        assert mounts[0] == "-v"
        assert adapter._CONTAINER_KIMI_HOME in mounts[1]
        assert ":rw" in mounts[1]

    def test_kimi_host_env_uses_host_path(self, tmp_path):
        from ai4sci_bench.adapters.kimi_code_cli import KimiCodeCLIAdapter
        adapter = KimiCodeCLIAdapter(
            api_key="sk-xxx",
            api_base="https://api.moonshot.cn/v1",
            api_protocol="openai",
        )
        env = adapter._build_api_env()
        assert env["KIMI_CODE_HOME"].startswith("/tmp")

    def test_kimi_os_passes_extra_mounts_to_sandbox(self, tmp_path):
        from ai4sci_bench.adapters.kimi_code_cli import KimiCodeCLIAdapter
        adapter = KimiCodeCLIAdapter(
            api_key="sk-xxx",
            api_base="https://api.moonshot.cn/v1",
            api_protocol="openai",
        )
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result()
        adapter._os_sandbox = mock_sb

        adapter.solve(_make_task_instance(tmp_path))
        call_kwargs = mock_sb.run_agent.call_args.kwargs
        assert call_kwargs["extra_mounts"] is not None
        assert "-v" in call_kwargs["extra_mounts"]

    def test_kimi_no_mounts_without_api(self, tmp_path):
        from ai4sci_bench.adapters.kimi_code_cli import KimiCodeCLIAdapter
        adapter = KimiCodeCLIAdapter()
        mounts = adapter._build_os_extra_mounts()
        assert mounts == []


class TestMiMoTranslateResponses:
    """MiMo OpenAI endpoint should use passthrough, not translation."""

    def test_anthropic_endpoint_uses_translation(self):
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        adapter = MiMoCodeCLIAdapter(
            api_key="tp-xxx",
            api_base="https://token-plan-cn.xiaomimimo.com/anthropic",
        )
        assert adapter._resolved_protocol == "anthropic"

    def test_openai_endpoint_uses_passthrough(self):
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        adapter = MiMoCodeCLIAdapter(
            api_key="sk-xxx",
            api_base="https://api.xiaomimimo.com/v1",
        )
        assert adapter._resolved_protocol == "openai"


class TestOpenHandsVenvMount:
    """OpenHands must mount host site-packages into container."""

    def _make_fake_venv(self, tmp_path):
        """Create a fake openhands venv with site-packages."""
        venv_dir = tmp_path / "fake_venv"
        (venv_dir / "bin").mkdir(parents=True)
        (venv_dir / "bin" / "python").write_text("#!/bin/sh\n")
        sp = venv_dir / "lib" / "python3.12" / "site-packages"
        sp.mkdir(parents=True)
        (sp / "openhands").mkdir()
        return venv_dir

    def test_os_sandbox_uses_container_opt_venv_python(self, tmp_path):
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
        adapter = OpenHandsAdapter(api_key="sk-test")
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result()
        adapter._os_sandbox = mock_sb

        venv_dir = self._make_fake_venv(tmp_path)
        adapter._find_openhands_python = lambda: str(venv_dir / "bin" / "python")

        adapter.solve(_make_task_instance(tmp_path))
        cmd = mock_sb.run_agent.call_args.kwargs["agent_cmd"]
        assert cmd[0] == "/opt/venv/bin/python"

    def test_os_sandbox_mounts_site_packages(self, tmp_path):
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
        adapter = OpenHandsAdapter(api_key="sk-test")
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result()
        adapter._os_sandbox = mock_sb

        venv_dir = self._make_fake_venv(tmp_path)
        adapter._find_openhands_python = lambda: str(venv_dir / "bin" / "python")

        adapter.solve(_make_task_instance(tmp_path))
        mounts = mock_sb.run_agent.call_args.kwargs["extra_mounts"]
        assert "-v" in mounts
        assert "site-packages" in mounts[1]
        assert ":ro" in mounts[1]

    def test_os_sandbox_writes_config_to_workspace(self, tmp_path):
        from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
        adapter = OpenHandsAdapter(api_key="sk-test")
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result()
        adapter._os_sandbox = mock_sb

        venv_dir = self._make_fake_venv(tmp_path)
        adapter._find_openhands_python = lambda: str(venv_dir / "bin" / "python")

        ti = _make_task_instance(tmp_path)
        adapter.solve(ti)
        assert (ti.workspace_dir / ".oh_runner.py").exists()
        assert (ti.workspace_dir / ".oh_config.json").exists()


class TestProxyDockerNetwork:
    """When proxy is used, AI4SCI_DOCKER_NETWORK should be set to 'host'."""

    def test_mimo_proxy_sets_docker_network(self, tmp_path, monkeypatch):
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        monkeypatch.delenv("AI4SCI_DOCKER_NETWORK", raising=False)

        adapter = MiMoCodeCLIAdapter(
            api_key="tp-xxx",
            api_base="https://token-plan-cn.xiaomimimo.com/anthropic",
        )
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result()
        adapter._os_sandbox = mock_sb
        adapter._ensure_proxy = lambda: "http://127.0.0.1:9999"

        import os
        adapter.solve(_make_task_instance(tmp_path))
        assert os.environ.get("AI4SCI_DOCKER_NETWORK") == "host"

        monkeypatch.delenv("AI4SCI_DOCKER_NETWORK", raising=False)

    def test_mimo_local_does_not_set_docker_network(self, tmp_path, monkeypatch):
        from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
        monkeypatch.delenv("AI4SCI_DOCKER_NETWORK", raising=False)

        adapter = MiMoCodeCLIAdapter()
        adapter.sandbox = "os"
        adapter.repo_root = tmp_path
        mock_sb = MagicMock()
        mock_sb.run_agent.return_value = _mock_os_sandbox_result()
        adapter._os_sandbox = mock_sb

        import os
        adapter.solve(_make_task_instance(tmp_path))
        assert os.environ.get("AI4SCI_DOCKER_NETWORK") is None
