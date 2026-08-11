"""Tests for OSSandbox, TaskImageBuilder — auth mounts, run_agent, docker commands, image building."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai4sci_bench.runner.os_sandbox import (
    CLAUDE_AUTH_PATHS,
    CODEX_AUTH_PATHS,
    OSSandbox,
)
from ai4sci_bench.runner.task_image import TaskImageBuilder


def _docker_run_calls(mock_run):
    return [
        call_args[0][0]
        for call_args in mock_run.call_args_list
        if call_args[0] and call_args[0][0][:2] == ["docker", "run"]
    ]


@pytest.fixture
def sandbox(tmp_path: Path) -> OSSandbox:
    builder = MagicMock()
    builder.ensure_image.return_value = "ai4sci-bench-base:latest"
    builder.get_image_identity.return_value = "sha256:abc123"
    return OSSandbox(tmp_path, image_builder=builder)


class TestPrepareAuthMounts:
    """Tests for _prepare_auth_mounts and its sub-methods."""

    def test_no_agent_type_returns_empty(self, sandbox: OSSandbox):
        assert sandbox._prepare_auth_mounts(None) == []

    def test_unknown_agent_type_returns_empty(self, sandbox: OSSandbox):
        assert sandbox._prepare_auth_mounts("unknown_agent") == []

    @patch("ai4sci_bench.runner.os_sandbox.CLAUDE_AUTH_PATHS", [])
    def test_claude_no_paths_returns_empty(self, sandbox: OSSandbox):
        assert sandbox._prepare_auth_mounts("claude_code") == []

    def test_claude_auth_mounts_existing_files(self, sandbox: OSSandbox, tmp_path: Path):
        cred_file = tmp_path / ".credentials.json"
        cred_file.write_text("{}")
        settings_file = tmp_path / "settings.json"
        settings_file.write_text("{}")

        with patch(
            "ai4sci_bench.runner.os_sandbox.CLAUDE_AUTH_PATHS",
            [cred_file, settings_file],
        ):
            mounts = sandbox._prepare_auth_mounts("claude_code")

        assert len(mounts) == 4  # 2 files × 2 args each (-v, path)
        # Verify read-only mount
        assert f"{cred_file}:/home/agent/.claude/.credentials.json:ro" in mounts
        assert f"{settings_file}:/home/agent/.claude/settings.json:ro" in mounts

    def test_claude_auth_skips_missing_files(self, sandbox: OSSandbox, tmp_path: Path):
        existing = tmp_path / ".credentials.json"
        existing.write_text("{}")
        missing = tmp_path / "does_not_exist.json"

        with patch(
            "ai4sci_bench.runner.os_sandbox.CLAUDE_AUTH_PATHS",
            [existing, missing],
        ):
            mounts = sandbox._prepare_auth_mounts("claude_code")

        assert len(mounts) == 2  # Only existing file
        assert f"{existing}:/home/agent/.claude/.credentials.json:ro" in mounts

    def test_codex_auth_mounts_existing_files(self, sandbox: OSSandbox, tmp_path: Path):
        auth_file = tmp_path / "auth.json"
        auth_file.write_text("{}")

        with patch(
            "ai4sci_bench.runner.os_sandbox.CODEX_AUTH_PATHS",
            [auth_file],
        ):
            mounts = sandbox._prepare_auth_mounts("codex")

        assert len(mounts) == 2
        assert f"{auth_file}:/home/agent/.codex/auth.json:ro" in mounts

    def test_codex_auth_skips_missing_files(self, sandbox: OSSandbox):
        with patch(
            "ai4sci_bench.runner.os_sandbox.CODEX_AUTH_PATHS",
            [Path("/nonexistent/auth.json")],
        ):
            mounts = sandbox._prepare_auth_mounts("codex")

        assert mounts == []


class TestBuildDockerCmd:
    """Tests for _build_docker_cmd with auth mounts and extra_env."""

    def test_basic_cmd_structure(self, sandbox: OSSandbox, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        cmd = sandbox._build_docker_cmd(
            container_name="test-container",
            image="test-image",
            workspace=workspace,
            exec_args=["python", "/workspace/solve.py"],
            requires_network=False,
            requires_gpu=False,
        )

        assert cmd[0] == "docker"
        assert cmd[1] == "run"
        assert "--name" in cmd
        assert "test-container" in cmd
        assert "--rm" in cmd
        assert "--network" in cmd
        assert "none" in cmd
        # Exec args at the end
        assert cmd[-3:] == ["test-image", "python", "/workspace/solve.py"]

    def test_env_whitelist_injected(self, sandbox: OSSandbox, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        cmd = sandbox._build_docker_cmd(
            container_name="test",
            image="img",
            workspace=workspace,
            exec_args=["echo", "hi"],
            requires_network=False,
            requires_gpu=False,
        )
        cmd_str = " ".join(cmd)
        assert "AI4SCI_SANDBOX=1" in cmd_str
        assert "HOME=/home/agent" in cmd_str

    def test_extra_env_injected(self, sandbox: OSSandbox, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        cmd = sandbox._build_docker_cmd(
            container_name="test",
            image="img",
            workspace=workspace,
            exec_args=["echo"],
            requires_network=False,
            requires_gpu=False,
            extra_env={"ANTHROPIC_API_KEY": "sk-test123"},
        )
        cmd_str = " ".join(cmd)
        assert "ANTHROPIC_API_KEY=sk-test123" in cmd_str

    def test_auth_mounts_included_for_claude(self, sandbox: OSSandbox, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        cred = tmp_path / "cred.json"
        cred.write_text("{}")

        with patch(
            "ai4sci_bench.runner.os_sandbox.CLAUDE_AUTH_PATHS",
            [cred],
        ):
            cmd = sandbox._build_docker_cmd(
                container_name="test",
                image="img",
                workspace=workspace,
                exec_args=["claude"],
                requires_network=True,
                requires_gpu=False,
                agent_type="claude_code",
            )

        cmd_str = " ".join(cmd)
        assert ":ro" in cmd_str
        assert "/home/agent/.claude/" in cmd_str

    def test_network_requires_dual_opt_in(self, sandbox: OSSandbox, tmp_path: Path):
        """Network only enabled when BOTH requires_network AND allow_external_tools are True."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # requires_network=True but allow_external_tools=False -> still --network none
        cmd = sandbox._build_docker_cmd(
            container_name="test",
            image="img",
            workspace=workspace,
            exec_args=["echo"],
            requires_network=True,
            requires_gpu=False,
            allow_external_tools=False,
        )
        assert "--network" in cmd
        idx = cmd.index("--network")
        assert cmd[idx + 1] == "none"

        # requires_network=False, allow_external_tools=True -> --network none
        cmd2 = sandbox._build_docker_cmd(
            container_name="test2",
            image="img",
            workspace=workspace,
            exec_args=["echo"],
            requires_network=False,
            requires_gpu=False,
            allow_external_tools=True,
        )
        assert "--network" in cmd2

        # Both True -> no --network flag (use default bridge)
        cmd3 = sandbox._build_docker_cmd(
            container_name="test3",
            image="img",
            workspace=workspace,
            exec_args=["echo"],
            requires_network=True,
            requires_gpu=False,
            allow_external_tools=True,
        )
        assert "--network" not in cmd3

    def test_gpu_flags(self, sandbox: OSSandbox, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        cmd = sandbox._build_docker_cmd(
            container_name="test",
            image="img",
            workspace=workspace,
            exec_args=["echo"],
            requires_network=False,
            requires_gpu=True,
        )
        assert "--gpus" in cmd
        assert "all" in cmd
        # shm_size should be GPU variant
        shm_idx = cmd.index("--shm-size")
        assert cmd[shm_idx + 1] == "4g"

    def test_no_host_sensitive_dirs_mounted(self, sandbox: OSSandbox, tmp_path: Path):
        """Verify that instances/, host home, etc. are never mounted."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        cmd = sandbox._build_docker_cmd(
            container_name="test",
            image="img",
            workspace=workspace,
            exec_args=["echo"],
            requires_network=False,
            requires_gpu=False,
        )
        cmd_str = " ".join(cmd)
        # Only one -v mount: the workspace
        v_count = cmd.count("-v")
        assert v_count == 1  # Only workspace mount when no auth

    def test_user_flag_set_on_linux_non_root(self, sandbox: OSSandbox, tmp_path: Path):
        """On Linux non-root hosts, --user must align with host UID/GID so the
        bind-mounted /workspace stays writable inside the container."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        with patch("ai4sci_bench.runner.os_sandbox.platform.system", return_value="Linux"), \
             patch("ai4sci_bench.runner.os_sandbox.os.getuid", return_value=1011), \
             patch("ai4sci_bench.runner.os_sandbox.os.getgid", return_value=1011):
            cmd = sandbox._build_docker_cmd(
                container_name="test",
                image="img",
                workspace=workspace,
                exec_args=["echo"],
                requires_network=False,
                requires_gpu=False,
            )
        assert "--user" in cmd
        idx = cmd.index("--user")
        assert cmd[idx + 1] == "1011:1011"

    def test_user_flag_skipped_on_linux_root(self, sandbox: OSSandbox, tmp_path: Path):
        """If host is root, fall back to image's USER agent so Claude CLI does
        not refuse to run with bypassPermissions."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        with patch("ai4sci_bench.runner.os_sandbox.platform.system", return_value="Linux"), \
             patch("ai4sci_bench.runner.os_sandbox.os.getuid", return_value=0), \
             patch("ai4sci_bench.runner.os_sandbox.os.getgid", return_value=0):
            cmd = sandbox._build_docker_cmd(
                container_name="test",
                image="img",
                workspace=workspace,
                exec_args=["echo"],
                requires_network=False,
                requires_gpu=False,
            )
        assert "--user" not in cmd

    def test_user_flag_skipped_on_macos(self, sandbox: OSSandbox, tmp_path: Path):
        """macOS Docker Desktop already remaps UIDs for bind mounts; do not
        override the image's USER agent there."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        with patch("ai4sci_bench.runner.os_sandbox.platform.system", return_value="Darwin"):
            cmd = sandbox._build_docker_cmd(
                container_name="test",
                image="img",
                workspace=workspace,
                exec_args=["echo"],
                requires_network=False,
                requires_gpu=False,
            )
        assert "--user" not in cmd

    def test_user_flag_different_uid_gid(self, sandbox: OSSandbox, tmp_path: Path):
        """UID and GID may differ; both must appear in --user."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        with patch("ai4sci_bench.runner.os_sandbox.platform.system", return_value="Linux"), \
             patch("ai4sci_bench.runner.os_sandbox.os.getuid", return_value=1011), \
             patch("ai4sci_bench.runner.os_sandbox.os.getgid", return_value=1050):
            cmd = sandbox._build_docker_cmd(
                container_name="test",
                image="img",
                workspace=workspace,
                exec_args=["echo"],
                requires_network=False,
                requires_gpu=False,
            )
        idx = cmd.index("--user")
        assert cmd[idx + 1] == "1011:1050"

    def test_root_on_linux_emits_warning(self, sandbox: OSSandbox, tmp_path: Path, caplog):
        """Running as root on Linux should log a warning about UID mismatch."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        import logging
        with caplog.at_level(logging.WARNING), \
             patch("ai4sci_bench.runner.os_sandbox.platform.system", return_value="Linux"), \
             patch("ai4sci_bench.runner.os_sandbox.os.getuid", return_value=0), \
             patch("ai4sci_bench.runner.os_sandbox.os.getgid", return_value=0):
            sandbox._build_docker_cmd(
                container_name="test",
                image="img",
                workspace=workspace,
                exec_args=["echo"],
                requires_network=False,
                requires_gpu=False,
            )
        assert any("root" in r.message.lower() and "uid=0" in r.message or
                    "uid 1000" in r.message
                    for r in caplog.records), \
            f"Expected root warning, got: {[r.message for r in caplog.records]}"

    def test_user_flag_before_image_arg(self, sandbox: OSSandbox, tmp_path: Path):
        """--user must appear before the image name in the docker command."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        with patch("ai4sci_bench.runner.os_sandbox.platform.system", return_value="Linux"), \
             patch("ai4sci_bench.runner.os_sandbox.os.getuid", return_value=1011), \
             patch("ai4sci_bench.runner.os_sandbox.os.getgid", return_value=1011):
            cmd = sandbox._build_docker_cmd(
                container_name="test",
                image="myimage:latest",
                workspace=workspace,
                exec_args=["echo", "hi"],
                requires_network=False,
                requires_gpu=False,
            )
        user_idx = cmd.index("--user")
        image_idx = cmd.index("myimage:latest")
        assert user_idx < image_idx, "--user must come before image argument"


class TestVerifyWorkspaceWritable:
    """Tests for the _verify_workspace_writable static method."""

    def test_writable_workspace_returns_none(self, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        assert OSSandbox._verify_workspace_writable(workspace) is None

    def test_probe_file_cleaned_up(self, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        OSSandbox._verify_workspace_writable(workspace)
        assert not (workspace / ".ai4sci_write_probe").exists()

    def test_readonly_workspace_returns_error(self, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        with patch.object(Path, "write_text", side_effect=OSError("Permission denied")):
            result = OSSandbox._verify_workspace_writable(workspace)
        assert result is not None
        assert "not writable" in result

    def test_nonexistent_workspace_returns_error(self, tmp_path: Path):
        workspace = tmp_path / "does_not_exist"
        result = OSSandbox._verify_workspace_writable(workspace)
        assert result is not None


class TestWriteProbeIntegration:
    """Verify that run_agent and execute_python fail-closed when workspace is not writable."""

    @patch("ai4sci_bench.runner.os_sandbox.subprocess.run")
    def test_run_agent_fails_on_unwritable_workspace(self, mock_run, sandbox: OSSandbox, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        with patch.object(
            OSSandbox, "_verify_workspace_writable",
            return_value="Workspace /workspace is not writable: Permission denied",
        ):
            success, log, stdout, stderr, img_id = sandbox.run_agent(
                {"difficulty": {}},
                agent_cmd=["claude", "--print", "solve"],
                workspace=workspace,
                timeout=300,
                agent_type="claude_code",
            )
        assert success is False
        assert "Infrastructure failure" in log
        assert "not writable" in log
        assert _docker_run_calls(mock_run) == []

    @patch("ai4sci_bench.runner.os_sandbox.subprocess.run")
    def test_execute_python_fails_on_unwritable_workspace(self, mock_run, sandbox: OSSandbox, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        with patch.object(
            OSSandbox, "_verify_workspace_writable",
            return_value="Workspace /workspace is not writable: Permission denied",
        ):
            success, log, stdout, stderr, img_id = sandbox.execute_python(
                {"difficulty": {}},
                workspace=workspace,
                code_file="solve.py",
                timeout=120,
            )
        assert success is False
        assert "Infrastructure failure" in log
        mock_run.assert_not_called()

    @patch("ai4sci_bench.runner.os_sandbox.subprocess.run")
    def test_run_agent_proceeds_when_writable(self, mock_run, sandbox: OSSandbox, tmp_path: Path):
        """Normal case: writable workspace should not block execution."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")

        success, log, stdout, stderr, img_id = sandbox.run_agent(
            {"difficulty": {}},
            agent_cmd=["echo", "test"],
            workspace=workspace,
            timeout=300,
        )
        assert success is True
        assert mock_run.call_count >= 1
        assert len(_docker_run_calls(mock_run)) == 1


class TestRunAgent:
    """Tests for the run_agent() method."""

    @patch("ai4sci_bench.runner.os_sandbox.subprocess.run")
    def test_run_agent_success(self, mock_run, sandbox: OSSandbox, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="agent output",
            stderr="",
        )

        success, log, stdout, stderr, img_id = sandbox.run_agent(
            {"difficulty": {}},
            agent_cmd=["claude", "--print", "solve this"],
            workspace=workspace,
            timeout=300,
            agent_type="claude_code",
        )

        assert success is True
        assert "agent output" in log
        assert img_id == "sha256:abc123"
        call_args = _docker_run_calls(mock_run)[0]
        assert call_args[0] == "docker"
        assert "claude" in call_args

    @patch("ai4sci_bench.runner.os_sandbox.subprocess.run")
    def test_run_agent_failure(self, mock_run, sandbox: OSSandbox, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="error occurred",
        )

        success, log, stdout, stderr, img_id = sandbox.run_agent(
            {"difficulty": {}},
            agent_cmd=["codex", "solve"],
            workspace=workspace,
            timeout=300,
            agent_type="codex",
        )

        assert success is False
        assert "error occurred" in log

    @patch("ai4sci_bench.runner.os_sandbox.subprocess.run")
    def test_run_agent_timeout(self, mock_run, sandbox: OSSandbox, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["docker", "run"], timeout=330, output="partial", stderr="timeout"
        )

        success, log, stdout, stderr, img_id = sandbox.run_agent(
            {"difficulty": {}},
            agent_cmd=["claude", "--print"],
            workspace=workspace,
            timeout=300,
        )

        assert success is False
        assert "timed out" in log

    @patch("ai4sci_bench.runner.os_sandbox.subprocess.run")
    def test_run_agent_timeout_stops_container_gracefully(
        self, mock_run, sandbox: OSSandbox, tmp_path: Path
    ):
        """On timeout the container is stopped via `docker stop -t` (SIGTERM
        first), not `docker kill` (immediate SIGKILL), so a CLI agent running
        as PID 1 can flush credentials before being killed."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        def side_effect(cmd, *args, **kwargs):
            # The `docker run` (agent) call times out; lifecycle calls succeed.
            if "run" in cmd:
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=330)
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect

        sandbox.run_agent(
            {"difficulty": {}},
            agent_cmd=["claude", "--print"],
            workspace=workspace,
            timeout=300,
        )

        all_cmds = [call.args[0] for call in mock_run.call_args_list]
        stop_cmds = [c for c in all_cmds if "stop" in c]
        kill_cmds = [c for c in all_cmds if "kill" in c]
        assert stop_cmds, f"expected a `docker stop`, got {all_cmds}"
        assert stop_cmds[0][:3] == ["docker", "stop", "-t"]
        assert not kill_cmds, f"should not `docker kill` on graceful path: {kill_cmds}"

    @patch("ai4sci_bench.runner.os_sandbox.subprocess.run")
    def test_run_agent_extra_env(self, mock_run, sandbox: OSSandbox, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")

        sandbox.run_agent(
            {"difficulty": {}},
            agent_cmd=["python", "solve.py"],
            workspace=workspace,
            timeout=300,
            extra_env={"ANTHROPIC_API_KEY": "test-key"},
        )

        call_args = mock_run.call_args_list[0][0][0]
        cmd_str = " ".join(call_args)
        assert "ANTHROPIC_API_KEY=test-key" in cmd_str

    @patch("ai4sci_bench.runner.os_sandbox.subprocess.run")
    def test_run_agent_container_name_prefix(self, mock_run, sandbox: OSSandbox, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        sandbox.run_agent(
            {"difficulty": {}},
            agent_cmd=["echo", "test"],
            workspace=workspace,
            timeout=60,
        )

        call_args = mock_run.call_args_list[0][0][0]
        name_idx = call_args.index("--name")
        container_name = call_args[name_idx + 1]
        assert container_name.startswith("ai4sci-agent-")


class TestExecutePythonRefactored:
    """Verify execute_python still works after refactoring _build_docker_cmd."""

    @patch("ai4sci_bench.runner.os_sandbox.subprocess.run")
    def test_execute_python_uses_exec_args(self, mock_run, sandbox: OSSandbox, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        mock_run.return_value = MagicMock(returncode=0, stdout="result", stderr="")

        success, log, stdout, stderr, img_id = sandbox.execute_python(
            {"difficulty": {}},
            workspace=workspace,
            code_file="solve.py",
            timeout=120,
        )

        assert success is True
        call_args = mock_run.call_args_list[0][0][0]
        # Should end with: image python /workspace/solve.py
        assert call_args[-2] == "python"
        assert call_args[-1] == "/workspace/solve.py"

    @patch("ai4sci_bench.runner.os_sandbox.subprocess.run")
    def test_execute_python_no_auth_mounts(self, mock_run, sandbox: OSSandbox, tmp_path: Path):
        """execute_python should not mount any auth files."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        sandbox.execute_python(
            {"difficulty": {}},
            workspace=workspace,
            code_file="solve.py",
            timeout=120,
        )

        call_args = mock_run.call_args_list[0][0][0]
        v_count = call_args.count("-v")
        assert v_count == 1  # Only workspace mount

    @patch("ai4sci_bench.runner.os_sandbox.subprocess.run")
    def test_execute_python_timeout(self, mock_run, sandbox: OSSandbox, tmp_path: Path):
        """execute_python returns failure on timeout."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["docker", "run"], timeout=150, output="partial", stderr="timeout"
        )

        success, log, stdout, stderr, img_id = sandbox.execute_python(
            {"difficulty": {}},
            workspace=workspace,
            code_file="solve.py",
            timeout=120,
        )

        assert success is False
        assert "timed out" in log

    @patch("ai4sci_bench.runner.os_sandbox.subprocess.run")
    def test_execute_python_exception(self, mock_run, sandbox: OSSandbox, tmp_path: Path):
        """execute_python returns failure on unexpected exception."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        mock_run.side_effect = OSError("Docker daemon crashed")

        success, log, stdout, stderr, img_id = sandbox.execute_python(
            {"difficulty": {}},
            workspace=workspace,
            code_file="solve.py",
            timeout=120,
        )

        assert success is False
        assert "Docker daemon crashed" in log


class TestContainerCleanup:
    """Tests for container lifecycle helpers."""

    @patch("ai4sci_bench.runner.os_sandbox.subprocess.run")
    def test_kill_container_calls_docker_kill(self, mock_run, sandbox: OSSandbox):
        sandbox._kill_container("test-container")
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args == ["docker", "kill", "test-container"]

    @patch("ai4sci_bench.runner.os_sandbox.subprocess.run")
    def test_remove_container_calls_docker_rm(self, mock_run, sandbox: OSSandbox):
        sandbox._remove_container("test-container")
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args == ["docker", "rm", "-f", "test-container"]

    @patch("ai4sci_bench.runner.os_sandbox.subprocess.run")
    def test_kill_container_swallows_exceptions(self, mock_run, sandbox: OSSandbox):
        mock_run.side_effect = Exception("docker not available")
        # Should not raise
        sandbox._kill_container("test-container")

    @patch("ai4sci_bench.runner.os_sandbox.subprocess.run")
    def test_remove_container_swallows_exceptions(self, mock_run, sandbox: OSSandbox):
        mock_run.side_effect = Exception("docker not available")
        # Should not raise
        sandbox._remove_container("test-container")


class TestTaskImageBuilder:
    """Mock tests for TaskImageBuilder methods."""

    @pytest.fixture
    def builder(self, tmp_path: Path) -> TaskImageBuilder:
        return TaskImageBuilder(tmp_path)

    def test_ensure_docker_available_success(self, builder: TaskImageBuilder):
        with patch("ai4sci_bench.runner.task_image.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            builder.ensure_docker_available()  # Should not raise

    def test_ensure_docker_available_not_found(self, builder: TaskImageBuilder):
        with patch("ai4sci_bench.runner.task_image.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("docker not found")
            with pytest.raises(RuntimeError, match="Docker CLI not found"):
                builder.ensure_docker_available()

    def test_ensure_docker_available_timeout(self, builder: TaskImageBuilder):
        with patch("ai4sci_bench.runner.task_image.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["docker"], timeout=10)
            with pytest.raises(RuntimeError, match="Timed out"):
                builder.ensure_docker_available()

    def test_ensure_docker_available_daemon_down(self, builder: TaskImageBuilder):
        with patch("ai4sci_bench.runner.task_image.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stderr="Cannot connect to the Docker daemon",
                stdout="",
            )
            with pytest.raises(RuntimeError, match="Docker is not available"):
                builder.ensure_docker_available()

    def test_base_image_tag_deterministic(self, builder: TaskImageBuilder):
        tag1 = builder._base_image_tag()
        tag2 = builder._base_image_tag()
        assert tag1 == tag2
        assert tag1.startswith("ai4sci-bench-base:")

    def test_task_image_tag_depends_on_packages(self, builder: TaskImageBuilder):
        meta1 = {"_runtime_packages": ["numpy==1.26"]}
        meta2 = {"_runtime_packages": ["scipy==1.12"]}
        tag1 = builder._task_image_tag(meta1)
        tag2 = builder._task_image_tag(meta2)
        assert tag1 != tag2
        assert tag1.startswith("ai4sci-bench-task:")

    def test_ensure_base_image_returns_cached(self, builder: TaskImageBuilder):
        with patch.object(builder, "ensure_docker_available"), \
             patch.object(builder, "_image_exists", return_value=True):
            tag = builder.ensure_base_image()
            assert tag.startswith("ai4sci-bench-base:")

    def test_ensure_base_image_builds_when_missing(self, builder: TaskImageBuilder):
        with patch.object(builder, "ensure_docker_available"), \
             patch.object(builder, "_image_exists", return_value=False), \
             patch.object(builder, "_build_image") as mock_build:
            tag = builder.ensure_base_image()
            assert tag.startswith("ai4sci-bench-base:")
            mock_build.assert_called_once()
            # Dockerfile should contain uv install
            dockerfile_arg = mock_build.call_args[0][1]
            assert "uv" in dockerfile_arg
            assert "git" in dockerfile_arg

    def test_base_image_preserves_task_venv_for_login_shells(self, builder: TaskImageBuilder):
        with patch.object(builder, "ensure_docker_available"), \
             patch.object(builder, "_image_exists", return_value=False), \
             patch.object(builder, "_build_image") as mock_build:
            builder.ensure_base_image()

        dockerfile_arg = mock_build.call_args[0][1]
        assert "/etc/profile.d/ai4sci-venv.sh" in dockerfile_arg
        assert 'export PATH="/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"' in dockerfile_arg
        assert 'export VIRTUAL_ENV="/opt/venv"' in dockerfile_arg

    def test_ensure_image_returns_base_when_no_packages(self, builder: TaskImageBuilder):
        with patch.object(builder, "ensure_base_image", return_value="ai4sci-bench-base:abc"):
            tag = builder.ensure_image({"_runtime_packages": []})
            assert tag == "ai4sci-bench-base:abc"

    def test_ensure_image_returns_base_when_no_runtime_key(self, builder: TaskImageBuilder):
        with patch.object(builder, "ensure_base_image", return_value="ai4sci-bench-base:abc"):
            tag = builder.ensure_image({})
            assert tag == "ai4sci-bench-base:abc"

    def test_ensure_image_builds_task_image_with_packages(self, builder: TaskImageBuilder):
        with patch.object(builder, "ensure_base_image", return_value="ai4sci-bench-base:abc"), \
             patch.object(builder, "_image_exists", return_value=False), \
             patch.object(builder, "_build_image") as mock_build:
            tag = builder.ensure_image({"_runtime_packages": ["numpy==1.26"]})
            assert tag.startswith("ai4sci-bench-task:")
            mock_build.assert_called_once()
            dockerfile_arg = mock_build.call_args[0][1]
            assert "uv pip install numpy==1.26" in dockerfile_arg

    def test_ensure_image_returns_cached_task_image(self, builder: TaskImageBuilder):
        with patch.object(builder, "ensure_base_image", return_value="ai4sci-bench-base:abc"), \
             patch.object(builder, "_image_exists", return_value=True), \
             patch.object(builder, "_build_image") as mock_build:
            tag = builder.ensure_image({"_runtime_packages": ["numpy"]})
            mock_build.assert_not_called()

    def test_get_image_identity_returns_sha(self, builder: TaskImageBuilder):
        with patch.object(builder, "ensure_docker_available"), \
             patch("ai4sci_bench.runner.task_image.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="sha256:abcdef123456\n"
            )
            identity = builder.get_image_identity("test-image")
            assert identity == "sha256:abcdef123456"

    def test_get_image_identity_falls_back_to_tag(self, builder: TaskImageBuilder):
        with patch.object(builder, "ensure_docker_available"), \
             patch("ai4sci_bench.runner.task_image.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            identity = builder.get_image_identity("test-image:latest")
            assert identity == "test-image:latest"

    def test_build_image_raises_on_failure(self, builder: TaskImageBuilder):
        with patch("ai4sci_bench.runner.task_image.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stderr="build error", stdout=""
            )
            with pytest.raises(RuntimeError, match="Failed to build Docker image"):
                builder._build_image("test:tag", "FROM python:3.12-slim\n")

    def test_build_image_writes_standalone_dockerfile_and_cleans_up(
        self, builder: TaskImageBuilder
    ):
        observed_path: Path | None = None
        dockerfile_text = "FROM python:3.12-slim\nRUN echo ok\n"

        def side_effect(cmd, **kwargs):
            nonlocal observed_path
            observed_path = Path(cmd[3])
            assert observed_path.suffix == ".Dockerfile"
            assert observed_path.exists()
            assert observed_path.read_text(encoding="utf-8") == dockerfile_text
            return MagicMock(returncode=0, stderr="", stdout="")

        with patch("ai4sci_bench.runner.task_image.subprocess.run", side_effect=side_effect):
            builder._build_image("test:tag", dockerfile_text)

        assert observed_path is not None
        assert not observed_path.exists()

    def test_build_image_cleans_up_temp_dockerfile_on_failure(
        self, builder: TaskImageBuilder
    ):
        observed_path: Path | None = None

        def side_effect(cmd, **kwargs):
            nonlocal observed_path
            observed_path = Path(cmd[3])
            assert observed_path.exists()
            return MagicMock(returncode=1, stderr="build error", stdout="")

        with patch("ai4sci_bench.runner.task_image.subprocess.run", side_effect=side_effect):
            with pytest.raises(RuntimeError, match="Failed to build Docker image"):
                builder._build_image("test:tag", "FROM python:3.12-slim\n")

        assert observed_path is not None
        assert not observed_path.exists()
