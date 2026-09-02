"""OS-level sandbox execution using Docker containers."""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from ai4sci_bench.runner.proc_util import GRACEFUL_SHUTDOWN_SECONDS
from ai4sci_bench.runner.task_image import TaskImageBuilder

logger = logging.getLogger(__name__)

# Auth file paths for each agent CLI.
# Only specific files are mounted — never entire home directories.
CLAUDE_AUTH_PATHS = [
    Path.home() / ".claude" / ".credentials.json",  # OAuth tokens
    Path.home() / ".claude" / "settings.json",  # Configuration
]

CODEX_AUTH_PATHS = [
    Path.home() / ".codex" / "auth.json",
]

# pi stores OAuth tokens / API keys in ~/.pi/agent/auth.json (single file;
# models.json is only mounted when the adapter generates a temp config dir).
PI_AUTH_PATHS = [
    Path.home() / ".pi" / "agent" / "auth.json",
]

# opencode stores provider credentials in ~/.local/share/opencode/auth.json.
OPENCODE_AUTH_PATHS = [
    Path.home() / ".local" / "share" / "opencode" / "auth.json",
]

# Optional proxy vars forwarded into CLI agent containers (Codex/Claude API access).
PROXY_ENV_KEYS = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
    "no_proxy",
    "NO_PROXY",
)


class OSSandbox:
    """Run code or agent commands inside a Docker container with task-scoped images.

    Supports two execution modes:
    - execute_python(): Run a single Python file (used by direct_llm adapter)
    - run_agent(): Run an arbitrary command (used by CLI adapters like claude_code_cli, codex_cli)
    """

    DEFAULT_MEMORY_LIMIT = "8g"
    DEFAULT_CPU_LIMIT = "4.0"
    DEFAULT_PIDS_LIMIT = 4096
    DEFAULT_TMPFS_SIZE = "2g"
    DEFAULT_SHM_SIZE = "1g"
    DEFAULT_SHM_SIZE_GPU = "4g"

    # Whitelist of environment variables injected into the container.
    # The container does NOT inherit the host environment.
    ENV_WHITELIST: dict[str, str] = {
        "PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": "/home/agent",
        "VIRTUAL_ENV": "/opt/venv",
        "LANG": "C.UTF-8",
        "AI4SCI_SANDBOX": "1",
    }

    def __init__(
        self,
        repo_root: Path,
        *,
        image_builder: TaskImageBuilder | None = None,
        memory_limit: str | None = None,
        cpu_limit: str | None = None,
        enable_workspace_audit: bool = False,
        readonly_data: bool = False,
        egress_whitelist_port: int | None = None,
    ):
        self.repo_root = repo_root
        self.image_builder = image_builder or TaskImageBuilder(repo_root)
        self.memory_limit = memory_limit or self.DEFAULT_MEMORY_LIMIT
        self.cpu_limit = cpu_limit or self.DEFAULT_CPU_LIMIT
        # `docker run --cpus N` is rejected when N exceeds the host's available
        # CPUs (e.g. DEFAULT_CPU_LIMIT 4.0 on a 2-CPU host breaks --sandbox os
        # entirely). Clamp to the host CPU count so the sandbox stays usable.
        _host_cpus = os.cpu_count() or 1
        if float(self.cpu_limit) > _host_cpus:
            self.cpu_limit = f"{float(_host_cpus):.2f}"
        self.enable_workspace_audit = enable_workspace_audit
        self.readonly_data = readonly_data
        self.egress_whitelist_port = egress_whitelist_port

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_python(
        self,
        task_metadata: dict[str, Any],
        *,
        workspace: Path,
        code_file: str,
        timeout: int,
        allow_external_tools: bool = False,
    ) -> tuple[bool, str, str | None, str | None, str | None]:
        """Execute one Python file in the task image."""
        image = self.image_builder.ensure_image(task_metadata)
        image_identity = self.image_builder.get_image_identity(image)
        container_name = f"ai4sci-os-{uuid.uuid4().hex[:12]}"

        # Report daemon-down failures directly for long-running containers.
        if timeout >= 3600:
            if not self.image_builder.check_daemon_health():
                return False, "Docker daemon is not responsive (heartbeat failed)", None, None, image_identity

        # Fail-closed before building/running the container command.
        write_err = self._verify_workspace_writable(workspace)
        if write_err:
            return False, f"Infrastructure failure: {write_err}", None, None, image_identity

        # Workspace audit: snapshot before
        pre_snapshot = self._snapshot_workspace(workspace) if self.enable_workspace_audit else None

        cmd = self._build_docker_cmd(
            container_name=container_name,
            image=image,
            workspace=workspace,
            exec_args=["python", f"/workspace/{Path(code_file).name}"],
            requires_network=bool(task_metadata.get("difficulty", {}).get("requires_network")),
            requires_gpu=bool(task_metadata.get("difficulty", {}).get("requires_gpu")),
            allow_external_tools=allow_external_tools,
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout + 30,
            )
            log = result.stdout + ("\n" + result.stderr if result.stderr else "")

            # Workspace audit: diff after
            if pre_snapshot is not None:
                audit = self._diff_workspace(pre_snapshot, workspace)
                if audit:
                    logger.info("Workspace audit: %s", audit)

            return (
                result.returncode == 0,
                log,
                result.stdout,
                result.stderr,
                image_identity,
            )
        except subprocess.TimeoutExpired as exc:
            self._stop_container(container_name)
            return False, f"OS sandbox execution timed out ({timeout}s)", exc.stdout or "", exc.stderr or "", image_identity
        except Exception as exc:
            self._kill_container(container_name)
            return False, f"OS sandbox execution error: {exc}", None, None, image_identity
        finally:
            self._remove_container(container_name)

    def run_agent(
        self,
        task_metadata: dict[str, Any],
        *,
        agent_cmd: list[str],
        workspace: Path,
        timeout: int,
        agent_type: str | None = None,
        extra_env: dict[str, str] | None = None,
        allow_external_tools: bool = False,
        extra_mounts: list[str] | None = None,
        stdin_input: str | None = None,
    ) -> tuple[bool, str, str | None, str | None, str | None]:
        """Run an arbitrary agent command inside the task container.

        This is the general-purpose entry point for CLI adapters (claude_code_cli,
        codex_cli) to execute agent processes inside the OS sandbox.

        Args:
            task_metadata: Task metadata dict (from task.yaml).
            agent_cmd: The command to execute inside the container (e.g. ["claude", "--print", ...]).
            workspace: Host path to the workspace directory (mounted as /workspace).
            timeout: Timeout in seconds for the agent execution.
            agent_type: e.g. "claude_code", "codex", "pi", "opencode", or None.
                Controls which auth credentials are mounted into the container.
            extra_env: Additional environment variables to inject (e.g. API keys).
            allow_external_tools: Whether the user allowed external tools (web search etc.).
                Network is only enabled when BOTH this and task's requires_network are True.

        Returns:
            Tuple of (success, log, stdout, stderr, image_identity).
        """
        image = self.image_builder.ensure_image(task_metadata, agent_type=agent_type)
        image_identity = self.image_builder.get_image_identity(image)
        container_name = f"ai4sci-agent-{uuid.uuid4().hex[:12]}"

        # Report daemon-down failures directly for long-running containers.
        if timeout >= 3600:
            if not self.image_builder.check_daemon_health():
                return False, "Docker daemon is not responsive (heartbeat failed)", None, None, image_identity

        # Fail-closed before auth preparation or container command construction.
        write_err = self._verify_workspace_writable(workspace)
        if write_err:
            return False, f"Infrastructure failure: {write_err}", None, None, image_identity

        # Workspace audit: snapshot before
        pre_snapshot = self._snapshot_workspace(workspace) if self.enable_workspace_audit else None

        requires_network = bool(task_metadata.get("difficulty", {}).get("requires_network"))
        requires_gpu = bool(task_metadata.get("difficulty", {}).get("requires_gpu"))

        cmd = self._build_docker_cmd(
            container_name=container_name,
            image=image,
            workspace=workspace,
            exec_args=agent_cmd,
            requires_network=requires_network,
            requires_gpu=requires_gpu,
            agent_type=agent_type,
            extra_env=extra_env,
            allow_external_tools=allow_external_tools,
            extra_mounts=extra_mounts,
            stdin_open=stdin_input is not None,
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                input=stdin_input,
                timeout=timeout + 30,
            )
            log = result.stdout + ("\n" + result.stderr if result.stderr else "")

            # Workspace audit: diff after
            if pre_snapshot is not None:
                audit = self._diff_workspace(pre_snapshot, workspace)
                if audit:
                    logger.info("Workspace audit: %s", audit)

            return (
                result.returncode == 0,
                log,
                result.stdout,
                result.stderr,
                image_identity,
            )
        except subprocess.TimeoutExpired as exc:
            self._stop_container(container_name)
            return False, f"OS sandbox agent execution timed out ({timeout}s)", exc.stdout or "", exc.stderr or "", image_identity
        except Exception as exc:
            self._kill_container(container_name)
            return False, f"OS sandbox agent execution error: {exc}", None, None, image_identity
        finally:
            self._remove_container(container_name)

    # ------------------------------------------------------------------
    # Auth mount helpers
    # ------------------------------------------------------------------

    def _prepare_auth_mounts(self, agent_type: str | None) -> list[str]:
        """Build read-only volume mount args for agent authentication files.

        Only mounts specific credential files — never entire home directories.
        Missing files are silently skipped (agent may use API key env vars instead).
        """
        if agent_type == "claude_code":
            return self._mount_claude_auth()
        elif agent_type == "codex":
            return self._mount_codex_auth()
        elif agent_type == "pi":
            return self._mount_generic_auth(PI_AUTH_PATHS, "/home/agent/.pi/agent")
        elif agent_type == "opencode":
            return self._mount_generic_auth(
                OPENCODE_AUTH_PATHS, "/home/agent/.local/share/opencode"
            )
        # mimo_code, kimi_code, agy, openhands: auth via env vars (extra_env),
        # no host credential files to mount.
        return []

    def _mount_claude_auth(self) -> list[str]:
        """Mount Claude Code credential files into /home/agent/.claude/.

        On macOS, extracts fresh OAuth credentials from the system keychain
        before each container run (the host CLI auto-refreshes tokens via
        keychain, but Docker containers cannot access the keychain).
        """
        # Ensure fresh credentials are available as a file
        self._refresh_claude_credentials()

        mounts: list[str] = []
        for host_path in CLAUDE_AUTH_PATHS:
            if host_path.exists():
                container_path = f"/home/agent/.claude/{host_path.name}"
                mounts += ["-v", f"{host_path}:{container_path}:ro"]
                logger.debug("Mounting Claude auth: %s -> %s", host_path, container_path)
            else:
                logger.debug("Claude auth file not found, skipping: %s", host_path)
        return mounts

    def _refresh_claude_credentials(self) -> None:
        """Extract fresh OAuth credentials from macOS keychain into .credentials.json.

        Claude Code on macOS stores OAuth tokens in the system keychain under
        service names like 'Claude Code-credentials-<hash>'. Tokens expire
        frequently, so we find the most recently refreshed (longest remaining
        validity) entry and write it to ~/.claude/.credentials.json for the
        Docker container to use.

        On non-macOS platforms, this is a no-op (credentials must be set up
        manually, e.g. via 'claude setup-token').
        """
        if platform.system() != "Darwin":
            return

        creds_path = Path.home() / ".claude" / ".credentials.json"

        # Check if existing file has a valid (non-expired) token
        if creds_path.exists():
            try:
                data = json.loads(creds_path.read_text())
                expires_at = data.get("claudeAiOauth", {}).get("expiresAt", 0)
                # Consider valid if >2 minutes remaining
                if expires_at > int(time.time() * 1000) + 120_000:
                    logger.debug("Existing credentials still valid, skipping refresh")
                    return
            except (json.JSONDecodeError, OSError):
                pass

        # Discover all Claude Code credential entries in keychain
        try:
            result = subprocess.run(
                ["security", "dump-keychain"],
                capture_output=True, text=True, timeout=10,
            )
        except Exception as exc:
            logger.warning("Cannot read macOS keychain: %s", exc)
            return

        suffixes: set[str] = set()
        for line in result.stdout.split("\n"):
            m = re.search(r"Claude Code-credentials-([a-f0-9]+)", line)
            if m:
                suffixes.add(m.group(1))

        if not suffixes:
            logger.warning(
                "No Claude Code credentials found in macOS keychain. "
                "Run 'claude login' or 'claude setup-token' first."
            )
            return

        # Find the entry with the longest remaining validity
        now_ms = int(time.time() * 1000)
        best_cred: str | None = None
        best_expiry: int = 0

        for suffix in suffixes:
            svc = f"Claude Code-credentials-{suffix}"
            try:
                r = subprocess.run(
                    ["security", "find-generic-password", "-s", svc, "-w"],
                    capture_output=True, text=True, timeout=5,
                )
                if r.returncode != 0:
                    continue
                raw = r.stdout.strip()
                data = json.loads(raw)
                exp = data.get("claudeAiOauth", {}).get("expiresAt", 0)
                if exp > best_expiry:
                    best_expiry = exp
                    best_cred = raw
            except Exception:
                continue

        if best_cred is None:
            logger.warning("Could not extract any Claude credentials from keychain")
            return

        if best_expiry <= now_ms:
            logger.warning(
                "All keychain tokens are expired. Triggering refresh via host CLI..."
            )
            # Make a minimal API call on the host to force token refresh
            try:
                subprocess.run(
                    ["claude", "--print", "--max-turns", "1", "--output-format", "json",
                     "--", "reply ok"],
                    capture_output=True, text=True, timeout=30,
                )
            except Exception as exc:
                logger.warning("Failed to trigger token refresh: %s", exc)
                return

            # Re-scan keychain for the freshly refreshed token
            best_cred = None
            best_expiry = 0
            now_ms = int(time.time() * 1000)
            for suffix in suffixes:
                svc = f"Claude Code-credentials-{suffix}"
                try:
                    r = subprocess.run(
                        ["security", "find-generic-password", "-s", svc, "-w"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if r.returncode != 0:
                        continue
                    raw = r.stdout.strip()
                    data = json.loads(raw)
                    exp = data.get("claudeAiOauth", {}).get("expiresAt", 0)
                    if exp > best_expiry:
                        best_expiry = exp
                        best_cred = raw
                except Exception:
                    continue

            # Also check for NEW entries created by the refresh
            try:
                result2 = subprocess.run(
                    ["security", "dump-keychain"],
                    capture_output=True, text=True, timeout=10,
                )
                new_suffixes: set[str] = set()
                for line in result2.stdout.split("\n"):
                    m = re.search(r"Claude Code-credentials-([a-f0-9]+)", line)
                    if m:
                        new_suffixes.add(m.group(1))
                for suffix in new_suffixes - suffixes:
                    svc = f"Claude Code-credentials-{suffix}"
                    try:
                        r = subprocess.run(
                            ["security", "find-generic-password", "-s", svc, "-w"],
                            capture_output=True, text=True, timeout=5,
                        )
                        if r.returncode != 0:
                            continue
                        raw = r.stdout.strip()
                        data = json.loads(raw)
                        exp = data.get("claudeAiOauth", {}).get("expiresAt", 0)
                        if exp > best_expiry:
                            best_expiry = exp
                            best_cred = raw
                    except Exception:
                        continue
            except Exception:
                pass

        if best_cred is None or best_expiry <= now_ms:
            logger.warning("Could not obtain valid Claude credentials for Docker container")
            return

        # Write fresh credentials
        try:
            creds_path.write_text(best_cred)
            remaining_min = (best_expiry - now_ms) / 1000 / 60
            logger.info(
                "Wrote fresh Claude credentials to %s (valid for %.0f min)",
                creds_path, remaining_min,
            )
        except OSError as exc:
            logger.warning("Failed to write credentials file: %s", exc)

    def _mount_codex_auth(self) -> list[str]:
        """Mount CodeX credential files read-only into /home/agent/.codex/."""
        mounts: list[str] = []
        for host_path in CODEX_AUTH_PATHS:
            if host_path.exists():
                container_path = f"/home/agent/.codex/{host_path.name}"
                mounts += ["-v", f"{host_path}:{container_path}:ro"]
                logger.debug("Mounting CodeX auth: %s -> %s", host_path, container_path)
            else:
                logger.debug("CodeX auth file not found, skipping: %s", host_path)
        return mounts

    def _mount_generic_auth(self, auth_paths: list[Path], container_dir: str) -> list[str]:
        """Mount specific credential files read-only into ``container_dir``."""
        mounts: list[str] = []
        for host_path in auth_paths:
            if host_path.exists():
                container_path = f"{container_dir}/{host_path.name}"
                mounts += ["-v", f"{host_path}:{container_path}:ro"]
                logger.debug("Mounting auth file: %s -> %s", host_path, container_path)
            else:
                logger.debug("Auth file not found, skipping: %s", host_path)
        return mounts

    # ------------------------------------------------------------------
    # Docker command construction
    # ------------------------------------------------------------------

    def _build_docker_cmd(
        self,
        *,
        container_name: str,
        image: str,
        workspace: Path,
        exec_args: list[str],
        requires_network: bool,
        requires_gpu: bool,
        agent_type: str | None = None,
        extra_env: dict[str, str] | None = None,
        allow_external_tools: bool = False,
        extra_mounts: list[str] | None = None,
        stdin_open: bool = False,
    ) -> list[str]:
        cmd = [
            "docker",
            "run",
        ]
        if stdin_open:
            cmd.append("-i")
        cmd += [
            "--name",
            container_name,
            "--rm",
        ]

        # Fine-grained read-only mounts: data/, prompt.md, task_info.json as :ro
        # Output directory stays :rw.  When readonly_data is False (default),
        # mount the entire workspace as :rw for backward compatibility.
        if self.readonly_data:
            # Mount workspace root as :rw (agent needs to write output files)
            cmd += ["-v", f"{workspace.resolve()}:/workspace:rw"]
            # Overlay read-only mounts for specific input files
            data_dir = workspace / "data"
            if data_dir.is_dir():
                cmd += ["-v", f"{data_dir.resolve()}:/workspace/data:ro"]
            prompt_file = workspace / "prompt.md"
            if prompt_file.is_file():
                cmd += ["-v", f"{prompt_file.resolve()}:/workspace/prompt.md:ro"]
            task_info_file = workspace / "task_info.json"
            if task_info_file.is_file():
                cmd += ["-v", f"{task_info_file.resolve()}:/workspace/task_info.json:ro"]
        else:
            cmd += ["-v", f"{workspace.resolve()}:/workspace:rw"]

        cmd += [
            "-w",
            "/workspace",
            "--memory",
            self.memory_limit,
            "--cpus",
            self.cpu_limit,
            "--pids-limit",
            str(self.DEFAULT_PIDS_LIMIT),
            "--tmpfs",
            f"/tmp:size={self.DEFAULT_TMPFS_SIZE}",
            "--shm-size",
            self.DEFAULT_SHM_SIZE_GPU if requires_gpu else self.DEFAULT_SHM_SIZE,
        ]

        # On Linux, align the container process UID/GID with the host workspace
        # owner so that bind-mounted /workspace is actually writable from inside
        # the container.  macOS/Windows Docker Desktop already handle UID
        # remapping via VirtioFS/gRPC-FUSE, so this is Linux-only.
        if platform.system() == "Linux":
            host_uid = os.getuid()
            host_gid = os.getgid()
            if host_uid != 0:
                cmd += ["--user", f"{host_uid}:{host_gid}"]
            else:
                logger.warning(
                    "Running as root (uid=0) with --sandbox os. The container "
                    "will use the image's USER agent (uid=1000). If /workspace "
                    "is owned by root, the agent will not be able to write "
                    "output files. Either chown the workspace to uid 1000 or "
                    "use --sandbox task instead."
                )

        # Environment variable whitelist — do not inherit host env
        for key, value in self.ENV_WHITELIST.items():
            cmd += ["-e", f"{key}={value}"]

        # Extra env vars (e.g. API keys for direct_llm)
        if extra_env:
            for key, value in extra_env.items():
                cmd += ["-e", f"{key}={value}"]

        # Auth credential mounts (read-only, specific files only)
        cmd += self._prepare_auth_mounts(agent_type)

        # Adapter-specific extra mounts (e.g. Kimi config dir)
        if extra_mounts:
            cmd += extra_mounts

        # Network control:
        # - CLI agents (claude_code, codex) always need network for API calls,
        #   even when external tools are disabled (web search etc. are controlled
        #   separately via --disallowed-tools).
        # - For other agent types (e.g. direct_llm running code), network uses
        #   dual opt-in: both task's requires_network AND allow_external_tools.
        needs_api_network = agent_type in (
            "claude_code", "codex", "mimo_code", "kimi_code", "agy", "openhands",
            "pi", "opencode",
        )
        if needs_api_network:
            for key, value in self._host_proxy_env().items():
                cmd += ["-e", f"{key}={value}"]
            docker_network = os.environ.get("AI4SCI_DOCKER_NETWORK", "").strip()
            if docker_network:
                cmd += ["--network", docker_network]
                logger.info(
                    "Using Docker network mode %r for CLI agent (AI4SCI_DOCKER_NETWORK)",
                    docker_network,
                )
        elif requires_network and allow_external_tools:
            pass  # Use default bridge network
        else:
            if requires_network and not allow_external_tools:
                logger.warning(
                    "Task declares requires_network=True but --allow-external-tools "
                    "is not set. Network access disabled. Pass --allow-external-tools "
                    "to enable network for this task."
                )
            cmd += ["--network", "none"]

        if requires_gpu:
            cmd += ["--gpus", "all"]

        cmd += [image] + exec_args
        return cmd

    @staticmethod
    def _host_proxy_env() -> dict[str, str]:
        """Collect proxy-related env vars from the host for CLI agent containers."""
        return {
            key: value
            for key in PROXY_ENV_KEYS
            if (value := os.environ.get(key, "").strip())
        }

    # ------------------------------------------------------------------
    # Container lifecycle helpers
    # ------------------------------------------------------------------

    def _stop_container(
        self, container_name: str, grace: int = GRACEFUL_SHUTDOWN_SECONDS
    ) -> None:
        """Gracefully stop a container: SIGTERM → wait grace → SIGKILL.

        ``docker stop -t N`` sends SIGTERM to container PID 1, waits up to N
        seconds, then SIGKILL — the in-container analogue of the host-side
        SIGTERM-first timeout handling.  This lets a CLI agent (e.g. Claude
        Code) running as PID 1 flush credentials / session state before being
        killed, instead of the immediate SIGKILL of ``docker kill``.
        """
        try:
            subprocess.run(
                ["docker", "stop", "-t", str(grace), container_name],
                capture_output=True,
                timeout=grace + 10,
            )
        except Exception:
            # Graceful stop failed/hung — fall back to an immediate kill.
            self._kill_container(container_name)

    def _kill_container(self, container_name: str) -> None:
        try:
            subprocess.run(
                ["docker", "kill", container_name],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass

    def _remove_container(self, container_name: str) -> None:
        try:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Workspace write probe
    # ------------------------------------------------------------------

    @staticmethod
    def _verify_workspace_writable(workspace: Path) -> str | None:
        """Quick probe: create and remove a temp file in the workspace.

        Returns None on success, or an error message string on failure.
        This catches UID/permission mismatches *before* we spend API
        dollars on agent turns that can never write output files.
        """
        probe = workspace / ".ai4sci_write_probe"
        try:
            probe.write_text("probe")
            probe.unlink()
            return None
        except OSError as exc:
            return (
                f"Workspace {workspace} is not writable: {exc}. "
                "This usually means the container UID does not match the "
                "workspace owner. Check --sandbox mode and host UID."
            )

    # ------------------------------------------------------------------
    # Workspace audit helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot_workspace(workspace: Path) -> dict[str, float]:
        """Take a snapshot of workspace file paths and modification times."""
        snapshot: dict[str, float] = {}
        if not workspace.is_dir():
            return snapshot
        for path in workspace.rglob("*"):
            if path.is_file():
                rel = str(path.relative_to(workspace))
                try:
                    snapshot[rel] = path.stat().st_mtime
                except OSError:
                    pass
        return snapshot

    @staticmethod
    def _diff_workspace(
        pre: dict[str, float],
        workspace: Path,
    ) -> dict[str, list[str]]:
        """Compare workspace before/after to find created, deleted, and modified files."""
        post: dict[str, float] = {}
        if workspace.is_dir():
            for path in workspace.rglob("*"):
                if path.is_file():
                    rel = str(path.relative_to(workspace))
                    try:
                        post[rel] = path.stat().st_mtime
                    except OSError:
                        pass

        created = sorted(set(post) - set(pre))
        deleted = sorted(set(pre) - set(post))
        modified = sorted(
            f for f in set(pre) & set(post) if pre[f] != post[f]
        )

        result: dict[str, list[str]] = {}
        if created:
            result["files_created"] = created
        if deleted:
            result["files_deleted"] = deleted
        if modified:
            result["files_modified"] = modified
        return result

    # ------------------------------------------------------------------
    # Interactive shell (debug)
    # ------------------------------------------------------------------

    def shell(
        self,
        task_metadata: dict[str, Any],
        *,
        workspace: Path,
        allow_external_tools: bool = False,
    ) -> list[str]:
        """Build a ``docker run -it`` command for interactive debugging.

        Returns the command list (caller should exec it or print it).
        """
        image = self.image_builder.ensure_image(task_metadata)
        requires_network = bool(task_metadata.get("difficulty", {}).get("requires_network"))
        requires_gpu = bool(task_metadata.get("difficulty", {}).get("requires_gpu"))
        container_name = f"ai4sci-shell-{uuid.uuid4().hex[:12]}"
        cmd = self._build_docker_cmd(
            container_name=container_name,
            image=image,
            workspace=workspace,
            exec_args=["/bin/bash"],
            requires_network=requires_network,
            requires_gpu=requires_gpu,
            allow_external_tools=allow_external_tools,
        )
        # Replace --rm with -it --rm for interactive use
        idx = cmd.index("--rm")
        cmd.insert(idx, "-it")
        return cmd
