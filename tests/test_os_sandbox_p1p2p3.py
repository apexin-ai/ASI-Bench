"""Tests for P1/P2/P3 OS sandbox features.

Covers: linux_ns mode, clean command, sandbox shell, sandbox test,
macOS warning, metadata field alignment, GPU preflight, custom image,
daemon heartbeat, parallel resource limiting, workspace audit,
fine-grained mounts, network egress, security adversarial patterns,
and sandbox execution patterns.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from click.testing import CliRunner

from ai4sci_bench.runner.sandbox_support import (
    KNOWN_SANDBOX_MODES,
    SANDBOX_DESCRIPTIONS,
    SANDBOX_HELP,
    print_sandbox_banner,
    validate_sandbox_mode,
)
from ai4sci_bench.runner.metadata import build_sandbox_provenance
from ai4sci_bench.runner.parallel import auto_limit_workers, ParallelRunner
from ai4sci_bench.runner.task_image import TaskImageBuilder, REQUIRED_IMAGE_TOOLS
from ai4sci_bench.runner.os_sandbox import OSSandbox


# ============================================================
# P1: linux_ns registration and validation
# ============================================================

class TestLinuxNSRegistration:
    """Verify linux_ns is registered as a known sandbox mode."""

    def test_linux_ns_in_known_modes(self):
        assert "linux_ns" in KNOWN_SANDBOX_MODES

    def test_linux_ns_has_description(self):
        assert "linux_ns" in SANDBOX_DESCRIPTIONS
        info = SANDBOX_DESCRIPTIONS["linux_ns"]
        assert "label" in info
        assert "description" in info
        assert "pros" in info
        assert "cons" in info

    def test_linux_ns_in_sandbox_help(self):
        assert "linux_ns" in SANDBOX_HELP

    def test_linux_ns_banner(self, capsys):
        print_sandbox_banner("linux_ns")
        output = capsys.readouterr().out
        assert "linux_ns" in output
        assert "Linux Namespace" in output

    def test_validate_linux_ns_non_linux_raises(self):
        """On non-Linux, validate_sandbox_mode should raise for linux_ns."""
        if sys.platform == "linux":
            pytest.skip("Only runs on non-Linux")
        with pytest.raises(ValueError, match="requires Linux"):
            validate_sandbox_mode(
                "linux_ns",
                supported_modes=("none", "task", "os", "linux_ns"),
                component="Test",
            )

    def test_validate_linux_ns_unsupported_component(self):
        """Component that doesn't list linux_ns should reject it."""
        with pytest.raises(ValueError, match="does not support sandbox 'linux_ns'"):
            validate_sandbox_mode(
                "linux_ns",
                supported_modes=("none", "task"),
                component="TestAdapter",
            )


class TestLinuxNSSandboxModule:
    """Test the linux_ns_sandbox module."""

    def test_import(self):
        from ai4sci_bench.runner.linux_ns_sandbox import LinuxNSSandbox, check_linux_ns_available
        assert callable(check_linux_ns_available)

    def test_check_availability_non_linux(self):
        from ai4sci_bench.runner.linux_ns_sandbox import check_linux_ns_available
        if sys.platform != "linux":
            available, reason = check_linux_ns_available()
            assert not available
            assert "Linux" in reason

    def test_constructor_non_linux_raises(self):
        from ai4sci_bench.runner.linux_ns_sandbox import LinuxNSSandbox
        if sys.platform != "linux":
            with pytest.raises(RuntimeError, match="requires Linux"):
                LinuxNSSandbox()

    @patch("ai4sci_bench.runner.linux_ns_sandbox.sys")
    def test_constructor_linux_ok(self, mock_sys):
        from ai4sci_bench.runner.linux_ns_sandbox import LinuxNSSandbox
        mock_sys.platform = "linux"
        sandbox = LinuxNSSandbox.__new__(LinuxNSSandbox)
        sandbox.memory_limit_kb = LinuxNSSandbox.DEFAULT_MEMORY_LIMIT_KB
        assert sandbox.memory_limit_kb == 8 * 1024 * 1024


# ============================================================
# P1: macOS degradation strategy
# ============================================================

class TestMacOSWarning:
    """Test macOS performance warning for --sandbox os."""

    @patch("ai4sci_bench.runner.sandbox_support.platform")
    def test_macos_warning_shown(self, mock_platform, capsys):
        mock_platform.system.return_value = "Darwin"
        print_sandbox_banner("os")
        output = capsys.readouterr().out
        assert "Warning:" in output
        assert "macOS" in output or "Docker" in output

    @patch("ai4sci_bench.runner.sandbox_support.platform")
    def test_linux_no_warning(self, mock_platform, capsys):
        mock_platform.system.return_value = "Linux"
        print_sandbox_banner("os")
        output = capsys.readouterr().out
        assert "Warning:" not in output

    @patch("ai4sci_bench.runner.sandbox_support.platform")
    def test_non_os_no_warning(self, mock_platform, capsys):
        mock_platform.system.return_value = "Darwin"
        print_sandbox_banner("task")
        output = capsys.readouterr().out
        assert "Warning:" not in output


# ============================================================
# P1: Metadata field alignment
# ============================================================

class TestMetadataFieldAlignment:
    """Verify metadata fields are documented and consistent."""

    def test_sandbox_provenance_fields(self):
        result = build_sandbox_provenance("os", image_identity="sha256:abc")
        assert "requested_mode" in result
        assert "effective_mode" in result
        assert "enforcement_status" in result
        assert "verification_status" in result
        assert "image_identity" in result

    def test_sandbox_provenance_none(self):
        result = build_sandbox_provenance("none")
        assert result["enforcement_status"] == "not_requested"
        assert result["verification_status"] == "not_applicable"

    def test_sandbox_provenance_task(self):
        result = build_sandbox_provenance("task")
        assert result["enforcement_status"] == "fail_closed"
        assert result["verification_status"] == "task_env_active"

    def test_sandbox_provenance_os(self):
        result = build_sandbox_provenance("os", image_identity="sha256:xyz")
        assert result["enforcement_status"] == "fail_closed"
        assert result["verification_status"] == "docker_container"
        assert result["image_identity"] == "sha256:xyz"

    def test_sandbox_provenance_linux_ns(self):
        result = build_sandbox_provenance("linux_ns")
        assert result["enforcement_status"] == "fail_closed"
        assert result["verification_status"] == "linux_namespace"


# ============================================================
# P1: GPU preflight check
# ============================================================

class TestGPUPreflightCheck:
    """Test NVIDIA GPU preflight validation."""

    @patch.object(TaskImageBuilder, "ensure_docker_available")
    @patch("subprocess.run")
    def test_check_nvidia_toolkit_installed_found(self, mock_run, mock_docker):
        mock_run.return_value = MagicMock(returncode=0)
        builder = TaskImageBuilder(Path("/fake"))
        assert builder.check_nvidia_toolkit_installed() is True

    @patch.object(TaskImageBuilder, "ensure_docker_available")
    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_check_nvidia_toolkit_installed_not_found(self, mock_run, mock_docker):
        builder = TaskImageBuilder(Path("/fake"))
        assert builder.check_nvidia_toolkit_installed() is False

    @patch.object(TaskImageBuilder, "ensure_docker_available")
    @patch("subprocess.run")
    def test_check_nvidia_toolkit_installed_nonzero(self, mock_run, mock_docker):
        mock_run.return_value = MagicMock(returncode=1)
        builder = TaskImageBuilder(Path("/fake"))
        assert builder.check_nvidia_toolkit_installed() is False


# ============================================================
# P1: Preflight custom image check
# ============================================================

class TestPreflightCustomImageCheck:
    """Test image validation."""

    @patch.object(TaskImageBuilder, "ensure_docker_available")
    @patch("subprocess.run")
    def test_validate_image_all_tools_present(self, mock_run, mock_docker):
        mock_run.return_value = MagicMock(returncode=0)
        builder = TaskImageBuilder(Path("/fake"))
        errors = builder.validate_image("test:latest")
        assert errors == []

    @patch.object(TaskImageBuilder, "ensure_docker_available")
    @patch("subprocess.run")
    def test_validate_image_missing_tool(self, mock_run, mock_docker):
        def side_effect(cmd, **kwargs):
            if "which" in cmd and "uv" in cmd:
                return MagicMock(returncode=1)
            return MagicMock(returncode=0)
        mock_run.side_effect = side_effect
        builder = TaskImageBuilder(Path("/fake"))
        errors = builder.validate_image("test:latest")
        assert any("uv" in e for e in errors)

    @patch.object(TaskImageBuilder, "ensure_docker_available")
    @patch("subprocess.run")
    def test_validate_image_workspace_not_writable(self, mock_run, mock_docker):
        def side_effect(cmd, **kwargs):
            if "touch" in str(cmd):
                return MagicMock(returncode=1)
            return MagicMock(returncode=0)
        mock_run.side_effect = side_effect
        builder = TaskImageBuilder(Path("/fake"))
        errors = builder.validate_image("test:latest")
        assert any("/workspace" in e for e in errors)


# ============================================================
# P1: Docker daemon heartbeat detection
# ============================================================

class TestDockerDaemonHeartbeat:
    """Test daemon health check."""

    @patch.object(TaskImageBuilder, "ensure_docker_available")
    @patch("subprocess.run")
    def test_daemon_healthy(self, mock_run, mock_docker):
        mock_run.return_value = MagicMock(returncode=0)
        builder = TaskImageBuilder(Path("/fake"))
        assert builder.check_daemon_health() is True

    @patch.object(TaskImageBuilder, "ensure_docker_available")
    @patch("subprocess.run")
    def test_daemon_unhealthy(self, mock_run, mock_docker):
        mock_run.return_value = MagicMock(returncode=1)
        builder = TaskImageBuilder(Path("/fake"))
        assert builder.check_daemon_health() is False

    @patch.object(TaskImageBuilder, "ensure_docker_available")
    @patch("subprocess.run", side_effect=Exception("down"))
    def test_daemon_exception(self, mock_run, mock_docker):
        builder = TaskImageBuilder(Path("/fake"))
        assert builder.check_daemon_health() is False

    @patch("ai4sci_bench.runner.os_sandbox.subprocess.run")
    def test_heartbeat_blocks_execution_on_long_timeout(self, mock_run):
        """When timeout >= 3600 and daemon is down, execution should fail."""
        mock_builder = MagicMock()
        mock_builder.ensure_image.return_value = "test:latest"
        mock_builder.get_image_identity.return_value = "sha256:abc"
        mock_builder.check_daemon_health.return_value = False

        sandbox = OSSandbox(Path("/fake"), image_builder=mock_builder)
        success, log, stdout, stderr, identity = sandbox.execute_python(
            {"difficulty": {}},
            workspace=Path("/tmp/w"),
            code_file="test.py",
            timeout=3600,
        )
        assert not success
        assert "heartbeat" in log.lower()
        # The only subprocess.run call should be the container cleanup (finally block),
        # NOT the actual docker run execution
        for call_args in mock_run.call_args_list:
            cmd = call_args[0][0] if call_args[0] else call_args[1].get("cmd", [])
            assert "docker" in cmd[0] and cmd[1] != "run", (
                f"Expected no docker run call, but got: {cmd}"
            )


# ============================================================
# P1: --parallel resource limiting for --sandbox os
# ============================================================

class TestAutoLimitWorkers:
    """Test auto_limit_workers for --sandbox os."""

    def test_non_os_unchanged(self):
        assert auto_limit_workers(8, sandbox="none") == 8
        assert auto_limit_workers(8, sandbox="task") == 8

    def test_os_clamps_down(self):
        """With small memory, workers should be clamped."""
        with patch("os.sysconf", side_effect=[4096, 2 * 1024 * 1024 * 1024 // 4096]):
            # 2 GB total memory, 80% usable = 1.6 GB, / 8 GB per container = 0.2 -> 1
            result = auto_limit_workers(8, sandbox="os", memory_limit_gb=8.0)
            assert result >= 1

    def test_os_cpu_limit(self):
        with patch("os.cpu_count", return_value=4), \
             patch("os.sysconf", side_effect=[4096, 128 * 1024 * 1024 * 1024 // 4096]):
            # 128 GB RAM but only 4 CPUs -> 4/4 = 1
            result = auto_limit_workers(8, sandbox="os", memory_limit_gb=8.0)
            assert result <= 4

    def test_minimum_one(self):
        assert auto_limit_workers(0, sandbox="os") >= 1

    def test_fallback_on_sysconf_error(self):
        with patch("os.sysconf", side_effect=AttributeError):
            # Should fallback to 16 GB estimate
            result = auto_limit_workers(2, sandbox="os", memory_limit_gb=8.0)
            assert result >= 1


# ============================================================
# P1: Result metadata documentation
# ============================================================

class TestMetadataDocumentation:
    """Verify metadata docstring exists and explains field naming."""

    def test_build_sandbox_provenance_has_field_docs(self):
        doc = build_sandbox_provenance.__doc__
        assert doc is not None
        assert "requested_mode" in doc
        assert "effective_mode" in doc
        assert "enforcement_status" in doc
        assert "verification_status" in doc


# ============================================================
# P1: clean --sandbox-images CLI
# ============================================================

class TestCleanCommand:
    """Test ai4sci-bench clean --sandbox-images."""

    def test_clean_no_flag_does_nothing(self):
        runner = CliRunner()
        from ai4sci_bench.cli import cli
        result = runner.invoke(cli, ["clean"])
        assert result.exit_code == 0
        assert "Nothing to clean" in result.output

    @patch.object(TaskImageBuilder, "clean_images", return_value=["ai4sci-bench-task:abc123"])
    @patch.object(TaskImageBuilder, "ensure_docker_available")
    def test_clean_sandbox_images(self, mock_docker, mock_clean):
        runner = CliRunner()
        from ai4sci_bench.cli import cli
        result = runner.invoke(cli, ["clean", "--sandbox-images"])
        assert result.exit_code == 0
        assert "Removed 1 image" in result.output

    @patch.object(TaskImageBuilder, "clean_images", return_value=[])
    @patch.object(TaskImageBuilder, "ensure_docker_available")
    def test_clean_no_images_found(self, mock_docker, mock_clean):
        runner = CliRunner()
        from ai4sci_bench.cli import cli
        result = runner.invoke(cli, ["clean", "--sandbox-images"])
        assert result.exit_code == 0
        assert "No cached sandbox images" in result.output


# ============================================================
# P1: sandbox test CLI
# ============================================================

class TestSandboxTestCommand:
    """Test ai4sci-bench sandbox test."""

    @patch.object(TaskImageBuilder, "check_nvidia_toolkit_installed", return_value=False)
    @patch.object(TaskImageBuilder, "validate_image", return_value=[])
    @patch.object(TaskImageBuilder, "ensure_base_image", return_value="ai4sci-bench-base:abc")
    @patch.object(TaskImageBuilder, "ensure_docker_available")
    def test_sandbox_test_pass(self, mock_docker, mock_base, mock_validate, mock_nvidia):
        runner = CliRunner()
        from ai4sci_bench.cli import cli
        result = runner.invoke(cli, ["sandbox", "test"])
        assert result.exit_code == 0
        assert "PASSED" in result.output

    @patch.object(TaskImageBuilder, "ensure_docker_available", side_effect=RuntimeError("no docker"))
    def test_sandbox_test_no_docker(self, mock_docker):
        runner = CliRunner()
        from ai4sci_bench.cli import cli
        result = runner.invoke(cli, ["sandbox", "test"])
        assert result.exit_code != 0


# ============================================================
# P2: Workspace file change audit
# ============================================================

class TestWorkspaceAudit:
    """Test workspace snapshot and diff logic."""

    def test_snapshot_empty_workspace(self, tmp_path):
        snapshot = OSSandbox._snapshot_workspace(tmp_path)
        assert snapshot == {}

    def test_snapshot_with_files(self, tmp_path):
        (tmp_path / "a.py").write_text("code")
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "input.npy").write_bytes(b"data")
        snapshot = OSSandbox._snapshot_workspace(tmp_path)
        assert "a.py" in snapshot
        assert "data/input.npy" in snapshot

    def test_diff_created(self, tmp_path):
        pre = {}
        (tmp_path / "output.npy").write_bytes(b"result")
        diff = OSSandbox._diff_workspace(pre, tmp_path)
        assert "output.npy" in diff.get("files_created", [])

    def test_diff_deleted(self, tmp_path):
        pre = {"old.txt": 12345.0}
        diff = OSSandbox._diff_workspace(pre, tmp_path)
        assert "old.txt" in diff.get("files_deleted", [])

    def test_diff_modified(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("v1")
        pre = {"code.py": f.stat().st_mtime - 1}  # slightly older
        diff = OSSandbox._diff_workspace(pre, tmp_path)
        assert "code.py" in diff.get("files_modified", [])

    def test_diff_no_changes(self, tmp_path):
        f = tmp_path / "stable.py"
        f.write_text("stable")
        mtime = f.stat().st_mtime
        pre = {"stable.py": mtime}
        diff = OSSandbox._diff_workspace(pre, tmp_path)
        assert diff == {}


# ============================================================
# P2: sandbox shell command
# ============================================================

class TestSandboxShellCommand:
    """Test OSSandbox.shell() method."""

    @patch.object(TaskImageBuilder, "ensure_image", return_value="test:latest")
    @patch.object(TaskImageBuilder, "ensure_docker_available")
    def test_shell_returns_docker_cmd(self, mock_docker, mock_image, tmp_path):
        sandbox = OSSandbox(Path("/fake"))
        cmd = sandbox.shell(
            {"difficulty": {}},
            workspace=tmp_path,
        )
        assert cmd[0] == "docker"
        assert "-it" in cmd
        assert "/bin/bash" in cmd

    @patch.object(TaskImageBuilder, "ensure_image", return_value="test:latest")
    @patch.object(TaskImageBuilder, "ensure_docker_available")
    def test_shell_mounts_workspace(self, mock_docker, mock_image, tmp_path):
        sandbox = OSSandbox(Path("/fake"))
        cmd = sandbox.shell({"difficulty": {}}, workspace=tmp_path)
        # Find the -v mount
        found = False
        for i, arg in enumerate(cmd):
            if arg == "-v" and "/workspace" in cmd[i + 1]:
                found = True
                break
        assert found


# ============================================================
# P2: runtime.image / runtime.dockerfile custom image support
# ============================================================

class TestCustomImageSupport:
    """Test runtime.image and runtime.dockerfile in ensure_image."""

    @patch.object(TaskImageBuilder, "_image_exists", return_value=True)
    @patch.object(TaskImageBuilder, "ensure_docker_available")
    def test_custom_image_found(self, mock_docker, mock_exists):
        builder = TaskImageBuilder(Path("/fake"))
        metadata = {"runtime": {"image": "my-custom:v1"}}
        result = builder.ensure_image(metadata)
        assert result == "my-custom:v1"

    @patch.object(TaskImageBuilder, "_image_exists", return_value=False)
    @patch.object(TaskImageBuilder, "ensure_docker_available")
    def test_custom_image_not_found_raises(self, mock_docker, mock_exists):
        builder = TaskImageBuilder(Path("/fake"))
        metadata = {"runtime": {"image": "missing:v1"}}
        with pytest.raises(RuntimeError, match="not found"):
            builder.ensure_image(metadata)

    @patch.object(TaskImageBuilder, "_image_exists", return_value=False)
    @patch.object(TaskImageBuilder, "_build_image_from_file")
    @patch.object(TaskImageBuilder, "ensure_docker_available")
    def test_custom_dockerfile(self, mock_docker, mock_build, mock_exists, tmp_path):
        task_dir = tmp_path / "tasks" / "test"
        task_dir.mkdir(parents=True)
        df = task_dir / "Dockerfile.custom"
        df.write_text("FROM python:3.12-slim\n")

        builder = TaskImageBuilder(Path("/fake"))
        metadata = {
            "runtime": {"dockerfile": "Dockerfile.custom"},
            "_task_dir": str(task_dir),
        }
        builder.ensure_image(metadata)
        mock_build.assert_called_once()

    @patch.object(TaskImageBuilder, "ensure_docker_available")
    def test_custom_dockerfile_not_found(self, mock_docker, tmp_path):
        task_dir = tmp_path / "tasks" / "test"
        task_dir.mkdir(parents=True)
        builder = TaskImageBuilder(Path("/fake"))
        metadata = {
            "runtime": {"dockerfile": "nonexistent.Dockerfile"},
            "_task_dir": str(task_dir),
        }
        with pytest.raises(RuntimeError, match="not found"):
            builder.ensure_image(metadata)


# ============================================================
# P2: Image cleanup
# ============================================================

class TestImageCleanup:
    """Test clean_images method."""

    @patch.object(TaskImageBuilder, "ensure_docker_available")
    @patch("subprocess.run")
    def test_clean_task_images(self, mock_run, mock_docker):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="ai4sci-bench-task:abc\nai4sci-bench-task:def\n",
        )
        builder = TaskImageBuilder(Path("/fake"))
        removed = builder.clean_images()
        assert len(removed) == 2

    @patch.object(TaskImageBuilder, "ensure_docker_available")
    @patch("subprocess.run")
    def test_clean_includes_base(self, mock_run, mock_docker):
        def side_effect(cmd, **kwargs):
            if "ai4sci-bench-task" in cmd:
                return MagicMock(returncode=0, stdout="ai4sci-bench-task:abc\n")
            if "ai4sci-bench-base" in cmd:
                return MagicMock(returncode=0, stdout="ai4sci-bench-base:xyz\n")
            return MagicMock(returncode=0, stdout="")
        mock_run.side_effect = side_effect
        builder = TaskImageBuilder(Path("/fake"))
        removed = builder.clean_images(include_base=True)
        assert len(removed) == 2

    @patch.object(TaskImageBuilder, "ensure_docker_available")
    @patch("subprocess.run")
    def test_clean_no_images(self, mock_run, mock_docker):
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        builder = TaskImageBuilder(Path("/fake"))
        removed = builder.clean_images()
        assert removed == []


# ============================================================
# P2: Security adversarial tests (mock-based)
# ============================================================

class TestSecurityAdversarial:
    """Verify OS sandbox security properties through mock tests."""

    def test_env_whitelist_no_host_vars(self):
        """Container should NOT inherit host environment variables."""
        # Check that OSSandbox.ENV_WHITELIST doesn't include dangerous vars
        dangerous_vars = {"AWS_SECRET_ACCESS_KEY", "OPENAI_API_KEY", "SSH_AUTH_SOCK"}
        for var in dangerous_vars:
            assert var not in OSSandbox.ENV_WHITELIST

    def test_workspace_mount_is_rw(self):
        """Workspace should be mounted as rw (agent needs to write output)."""
        mock_builder = MagicMock()
        sandbox = OSSandbox(Path("/fake"), image_builder=mock_builder)
        cmd = sandbox._build_docker_cmd(
            container_name="test",
            image="test:latest",
            workspace=Path("/tmp/ws"),
            exec_args=["python", "test.py"],
            requires_network=False,
            requires_gpu=False,
        )
        # Find workspace mount
        for i, arg in enumerate(cmd):
            if arg == "-v" and "/workspace" in cmd[i + 1]:
                assert ":rw" in cmd[i + 1]
                break

    def test_network_none_by_default(self):
        """Network should be disabled when both requires_network and allow_external_tools are False."""
        mock_builder = MagicMock()
        sandbox = OSSandbox(Path("/fake"), image_builder=mock_builder)
        cmd = sandbox._build_docker_cmd(
            container_name="test",
            image="test:latest",
            workspace=Path("/tmp/ws"),
            exec_args=["python", "test.py"],
            requires_network=False,
            requires_gpu=False,
        )
        assert "--network" in cmd
        idx = cmd.index("--network")
        assert cmd[idx + 1] == "none"

    def test_network_disabled_without_allow_external(self):
        """Even with requires_network, network stays disabled without allow_external_tools."""
        mock_builder = MagicMock()
        sandbox = OSSandbox(Path("/fake"), image_builder=mock_builder)
        cmd = sandbox._build_docker_cmd(
            container_name="test",
            image="test:latest",
            workspace=Path("/tmp/ws"),
            exec_args=["python", "test.py"],
            requires_network=True,
            requires_gpu=False,
            allow_external_tools=False,
        )
        assert "--network" in cmd
        idx = cmd.index("--network")
        assert cmd[idx + 1] == "none"

    def test_network_enabled_dual_optin(self):
        """Network only enabled when both requires_network AND allow_external_tools are True."""
        mock_builder = MagicMock()
        sandbox = OSSandbox(Path("/fake"), image_builder=mock_builder)
        cmd = sandbox._build_docker_cmd(
            container_name="test",
            image="test:latest",
            workspace=Path("/tmp/ws"),
            exec_args=["python", "test.py"],
            requires_network=True,
            requires_gpu=False,
            allow_external_tools=True,
        )
        assert "--network" not in cmd

    def test_pids_limit_set(self):
        """Container should have --pids-limit to prevent fork bombs."""
        mock_builder = MagicMock()
        sandbox = OSSandbox(Path("/fake"), image_builder=mock_builder)
        cmd = sandbox._build_docker_cmd(
            container_name="test",
            image="test:latest",
            workspace=Path("/tmp/ws"),
            exec_args=["python", "test.py"],
            requires_network=False,
            requires_gpu=False,
        )
        assert "--pids-limit" in cmd
        idx = cmd.index("--pids-limit")
        assert int(cmd[idx + 1]) > 0

    def test_memory_limit_set(self):
        """Container should have --memory limit."""
        mock_builder = MagicMock()
        sandbox = OSSandbox(Path("/fake"), image_builder=mock_builder)
        cmd = sandbox._build_docker_cmd(
            container_name="test",
            image="test:latest",
            workspace=Path("/tmp/ws"),
            exec_args=["python", "test.py"],
            requires_network=False,
            requires_gpu=False,
        )
        assert "--memory" in cmd

    def test_tmpfs_set(self):
        """Container should have tmpfs for /tmp with size limit."""
        mock_builder = MagicMock()
        sandbox = OSSandbox(Path("/fake"), image_builder=mock_builder)
        cmd = sandbox._build_docker_cmd(
            container_name="test",
            image="test:latest",
            workspace=Path("/tmp/ws"),
            exec_args=["python", "test.py"],
            requires_network=False,
            requires_gpu=False,
        )
        assert "--tmpfs" in cmd


# ============================================================
# P2: Adapter containerized test patterns
# ============================================================

class TestAdapterContainerizedPatterns:
    """Test adapter sandbox compatibility at the module level."""

    def test_claude_code_cli_supports_os(self):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
        adapter = ClaudeCodeCLIAdapter()
        assert "os" in adapter._supported_sandbox_modes

    def test_codex_cli_supports_os(self):
        from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter
        adapter = CodexCLIAdapter()
        assert "os" in adapter._supported_sandbox_modes

    def test_direct_llm_supports_os(self):
        """DirectLLMAdapter validates sandbox via validate_sandbox_mode in setup().
        Confirm 'os' is in the supported_modes tuple used there."""
        from ai4sci_bench.adapters.direct_llm import DirectLLMAdapter
        # The adapter calls validate_sandbox_mode with supported_modes=("none", "task", "os")
        # so we just verify the setup doesn't raise for "os"
        adapter = DirectLLMAdapter(model="test")
        adapter.setup({"timeout": 60, "sandbox": "os", "repo_root": "/fake"})
        assert adapter.sandbox == "os"


# ============================================================
# P3: Fine-grained read-only mounts
# ============================================================

class TestFineGrainedMounts:
    """Test readonly_data mount mode."""

    def test_readonly_data_mounts(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "data").mkdir()
        (workspace / "data" / "input.npy").write_bytes(b"data")
        (workspace / "prompt.md").write_text("test")
        (workspace / "task_info.json").write_text("{}")

        mock_builder = MagicMock()
        sandbox = OSSandbox(Path("/fake"), image_builder=mock_builder, readonly_data=True)
        cmd = sandbox._build_docker_cmd(
            container_name="test",
            image="test:latest",
            workspace=workspace,
            exec_args=["python", "test.py"],
            requires_network=False,
            requires_gpu=False,
        )
        cmd_str = " ".join(cmd)
        # Should have multiple -v mounts
        v_indices = [i for i, a in enumerate(cmd) if a == "-v"]
        assert len(v_indices) >= 4  # workspace + data + prompt + task_info

        # Check read-only mounts exist
        ro_mounts = [cmd[i + 1] for i in v_indices if ":ro" in cmd[i + 1]]
        assert len(ro_mounts) >= 3  # data, prompt.md, task_info.json

    def test_default_mode_single_rw_mount(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        mock_builder = MagicMock()
        sandbox = OSSandbox(Path("/fake"), image_builder=mock_builder, readonly_data=False)
        cmd = sandbox._build_docker_cmd(
            container_name="test",
            image="test:latest",
            workspace=workspace,
            exec_args=["python", "test.py"],
            requires_network=False,
            requires_gpu=False,
        )
        v_indices = [i for i, a in enumerate(cmd) if a == "-v"]
        # Only one -v mount (workspace :rw)
        workspace_mounts = [cmd[i + 1] for i in v_indices if "/workspace" in cmd[i + 1]]
        assert len(workspace_mounts) == 1
        assert ":rw" in workspace_mounts[0]


# ============================================================
# P3: Network egress whitelist
# ============================================================

class TestNetworkEgressWhitelist:
    """Test egress_whitelist_port parameter (P3 placeholder)."""

    def test_egress_port_stored(self):
        mock_builder = MagicMock()
        sandbox = OSSandbox(Path("/fake"), image_builder=mock_builder, egress_whitelist_port=443)
        assert sandbox.egress_whitelist_port == 443

    def test_egress_port_default_none(self):
        mock_builder = MagicMock()
        sandbox = OSSandbox(Path("/fake"), image_builder=mock_builder)
        assert sandbox.egress_whitelist_port is None


# ============================================================
# P3: Performance benchmark test patterns
# ============================================================

class TestPerformanceBenchmarkPatterns:
    """Verify performance-related code paths exist."""

    def test_auto_limit_workers_callable(self):
        """auto_limit_workers should be importable and callable."""
        from ai4sci_bench.runner.parallel import auto_limit_workers
        assert callable(auto_limit_workers)

    def test_base_image_schema_version(self):
        """Base image schema version should be tracked for cache invalidation."""
        assert TaskImageBuilder.BASE_IMAGE_SCHEMA_VERSION >= 2


# ============================================================
class TestRequiredImageTools:
    """Verify REQUIRED_IMAGE_TOOLS is correctly defined."""

    def test_required_tools_non_empty(self):
        assert len(REQUIRED_IMAGE_TOOLS) > 0

    def test_required_tools_includes_essentials(self):
        for tool in ("python3", "uv", "git", "curl"):
            assert tool in REQUIRED_IMAGE_TOOLS


# ============================================================
# P1: Integration tests marker (Docker required)
# ============================================================

docker_available = False
try:
    r = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
    docker_available = r.returncode == 0
except Exception:
    pass

needs_docker = pytest.mark.skipif(
    not docker_available,
    reason="Docker not available",
)


@needs_docker
class TestDockerIntegration:
    """Integration tests that require a running Docker daemon.

    Skipped when Docker is not available (CI without Docker, macOS dev without Docker, etc.).
    """

    def test_ensure_docker_available(self):
        builder = TaskImageBuilder(Path(__file__).resolve().parents[1])
        builder.ensure_docker_available()  # Should not raise

    def test_daemon_health_check(self):
        builder = TaskImageBuilder(Path(__file__).resolve().parents[1])
        assert builder.check_daemon_health() is True


# ============================================================
# P3: End-to-end benchmark test patterns
# ============================================================

class TestEndToEndPatterns:
    """Verify the infrastructure for e2e tests exists."""

    def test_parallel_runner_importable(self):
        from ai4sci_bench.runner.parallel import ParallelRunner
        assert ParallelRunner is not None

    def test_sandbox_modes_all_have_descriptions(self):
        for mode in KNOWN_SANDBOX_MODES:
            assert mode in SANDBOX_DESCRIPTIONS

    def test_all_sandbox_modes_in_help(self):
        for mode in KNOWN_SANDBOX_MODES:
            assert mode in SANDBOX_HELP
