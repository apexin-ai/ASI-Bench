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

_SERVICE_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,30}$")
_SERVICE_ENV_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_SERVICE_KEYS = {
    "name", "image", "port", "command", "startup_timeout_seconds", "environment",
}


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
        services = self._task_services(task_metadata)
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

        requires_network = bool(task_metadata.get("difficulty", {}).get("requires_network"))
        requires_gpu = bool(task_metadata.get("difficulty", {}).get("requires_gpu"))
        outcome = self._run_task_container(
            task_metadata,
            services=services,
            image=image,
            image_identity=image_identity,
            container_name=container_name,
            workspace=workspace,
            exec_args=["python", f"/workspace/{Path(code_file).name}"],
            timeout=timeout,
            requires_network=requires_network,
            requires_gpu=requires_gpu,
            allow_external_tools=allow_external_tools,
            agent_type=None,
            timeout_label="OS sandbox execution",
        )
        if pre_snapshot is not None:
            audit = self._diff_workspace(pre_snapshot, workspace)
            if audit:
                logger.info("Workspace audit: %s", audit)
        return outcome

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
        services = self._task_services(task_metadata)
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
        outcome = self._run_task_container(
            task_metadata,
            services=services,
            image=image,
            image_identity=image_identity,
            container_name=container_name,
            workspace=workspace,
            exec_args=agent_cmd,
            timeout=timeout,
            requires_network=requires_network,
            requires_gpu=requires_gpu,
            allow_external_tools=allow_external_tools,
            agent_type=agent_type,
            extra_env=extra_env,
            extra_mounts=extra_mounts,
            stdin_input=stdin_input,
            timeout_label="OS sandbox agent execution",
        )
        if pre_snapshot is not None:
            audit = self._diff_workspace(pre_snapshot, workspace)
            if audit:
                logger.info("Workspace audit: %s", audit)
        return outcome

    def _run_task_container(
        self,
        task_metadata: dict[str, Any],
        *,
        services: list[dict[str, Any]],
        image: str,
        image_identity: str,
        container_name: str,
        workspace: Path,
        exec_args: list[str],
        timeout: int,
        requires_network: bool,
        requires_gpu: bool,
        allow_external_tools: bool,
        agent_type: str | None,
        extra_env: dict[str, str] | None = None,
        extra_mounts: list[str] | None = None,
        stdin_input: str | None = None,
        timeout_label: str,
    ) -> tuple[bool, str, str | None, str | None, str | None]:
        service_network: str | None = None
        service_containers: list[str] = []
        try:
            service_env = dict(extra_env or {})
            if services:
                service_network = self._create_task_service_network()
                service_env.update(
                    self._start_task_services(services, service_network, service_containers)
                )
            external_network = (
                self._agent_external_network(
                    agent_type=agent_type,
                    requires_network=requires_network,
                    allow_external_tools=allow_external_tools,
                )
                if services else None
            )
            cmd = self._build_docker_cmd(
                container_name=container_name,
                image=image,
                workspace=workspace,
                exec_args=exec_args,
                requires_network=requires_network,
                requires_gpu=requires_gpu,
                agent_type=agent_type,
                extra_env=service_env,
                allow_external_tools=allow_external_tools,
                extra_mounts=extra_mounts,
                stdin_open=stdin_input is not None,
                docker_action="create" if services else "run",
                docker_network_override=service_network,
            )
            if services:
                created = subprocess.run(
                    cmd, capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=30,
                )
                if created.returncode != 0:
                    raise RuntimeError(
                        f"Could not create agent container: {(created.stderr or created.stdout).strip()}"
                    )
                if external_network:
                    self._docker_control(
                        ["docker", "network", "connect", external_network, container_name],
                        timeout=20,
                    )
                cmd = ["docker", "start", "--attach"]
                if stdin_input is not None:
                    cmd.append("--interactive")
                cmd.append(container_name)
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
            return result.returncode == 0, log, result.stdout, result.stderr, image_identity
        except subprocess.TimeoutExpired as exc:
            self._stop_container(container_name)
            return False, f"{timeout_label} timed out ({timeout}s)", exc.stdout or "", exc.stderr or "", image_identity
        except Exception as exc:
            self._kill_container(container_name)
            return False, f"{timeout_label} error: {exc}", None, None, image_identity
        finally:
            self._remove_container(container_name)
            for service_container in reversed(service_containers):
                self._remove_container(service_container)
            if service_network:
                self._remove_task_service_network(service_network)

    @staticmethod
    def _task_services(task_metadata: dict[str, Any]) -> list[dict[str, Any]]:
        runtime = task_metadata.get("runtime", {})
        services = runtime.get("services", []) if isinstance(runtime, dict) else []
        if not isinstance(services, list):
            raise ValueError("runtime.services must be a list")
        if len(services) > 2:
            raise ValueError("runtime.services supports at most 2 services")

        parsed: list[dict[str, Any]] = []
        names: set[str] = set()
        for index, service in enumerate(services):
            if not isinstance(service, dict):
                raise ValueError(f"runtime.services[{index}] must be a mapping")
            unknown = set(service) - _SERVICE_KEYS
            if unknown:
                raise ValueError(
                    f"runtime.services[{index}] has unsupported fields: {', '.join(sorted(unknown))}"
                )
            name, image, port = service.get("name"), service.get("image"), service.get("port")
            if not isinstance(name, str) or not _SERVICE_NAME_RE.fullmatch(name):
                raise ValueError(f"runtime.services[{index}].name must be a DNS-safe lowercase name")
            if name in names:
                raise ValueError(f"runtime.services contains duplicate name {name!r}")
            names.add(name)
            if not isinstance(image, str) or not image.strip() or any(c.isspace() for c in image):
                raise ValueError(f"runtime.services[{index}].image must be a non-empty image reference")
            if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
                raise ValueError(f"runtime.services[{index}].port must be an integer from 1 to 65535")
            command = service.get("command")
            if command is not None and (
                not isinstance(command, list)
                or not command
                or any(not isinstance(arg, str) or "\0" in arg for arg in command)
            ):
                raise ValueError(f"runtime.services[{index}].command must be a non-empty string list")
            startup_timeout = service.get("startup_timeout_seconds", 60)
            if isinstance(startup_timeout, bool) or not isinstance(startup_timeout, int) or not 1 <= startup_timeout <= 300:
                raise ValueError(
                    f"runtime.services[{index}].startup_timeout_seconds must be from 1 to 300"
                )
            environment = service.get("environment", {})
            if not isinstance(environment, dict) or any(
                not isinstance(key, str)
                or not _SERVICE_ENV_RE.fullmatch(key)
                or key.startswith("ASIBENCH_SERVICE_")
                or not isinstance(value, str)
                or "\0" in value
                for key, value in environment.items()
            ):
                raise ValueError(
                    f"runtime.services[{index}].environment must map non-reserved variable names to strings"
                )
            parsed.append({
                "name": name,
                "image": image.strip(),
                "port": port,
                "command": command,
                "startup_timeout_seconds": startup_timeout,
                "environment": environment,
            })
        return parsed

    def _create_task_service_network(self) -> str:
        network = f"ai4sci-task-{uuid.uuid4().hex[:12]}"
        self._docker_control(
            ["docker", "network", "create", "--driver", "bridge", "--internal", network],
            timeout=30,
        )
        return network

    @staticmethod
    def _agent_external_network(
        *, agent_type: str | None, requires_network: bool, allow_external_tools: bool,
    ) -> str | None:
        needs_api_network = agent_type in {
            "claude_code", "codex", "mimo_code", "kimi_code", "agy", "openhands",
            "pi", "opencode",
        }
        if needs_api_network:
            configured = os.environ.get("AI4SCI_DOCKER_NETWORK", "").strip()
            if configured in {"host", "none"}:
                raise ValueError(
                    "runtime.services cannot be combined with AI4SCI_DOCKER_NETWORK="
                    f"{configured}; use a connectable bridge network"
                )
            return configured or "bridge"
        if requires_network and allow_external_tools:
            return "bridge"
        return None

    def _start_task_services(
        self,
        services: list[dict[str, Any]],
        network: str,
        containers: list[str],
    ) -> dict[str, str]:
        environment: dict[str, str] = {}
        for service in services:
            name = service["name"]
            container = f"ai4sci-svc-{uuid.uuid4().hex[:12]}"
            cmd = [
                "docker", "run", "--detach", "--name", container,
                "--network", network, "--network-alias", name,
                "--memory", "1g", "--cpus", "1.0", "--pids-limit", "512",
            ]
            for key, value in service["environment"].items():
                cmd += ["-e", f"{key}={value}"]
            cmd.append(service["image"])
            if service["command"] is not None:
                cmd.extend(service["command"])
            result = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=60,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Could not start task service {name!r}: {(result.stderr or result.stdout).strip()}"
                )
            containers.append(container)
            self._wait_for_task_service(service, container)
            environment[f"ASIBENCH_SERVICE_{name.upper().replace('-', '_')}_URL"] = (
                f"http://{name}:{service['port']}"
            )
        return environment

    def _wait_for_task_service(self, service: dict[str, Any], container: str) -> None:
        deadline = time.monotonic() + service["startup_timeout_seconds"]
        while time.monotonic() < deadline:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{if .State.Health}}{{.State.Health.Status}}{{else}}missing{{end}}", container],
                capture_output=True, text=True, timeout=10,
            )
            status = result.stdout.strip() if result.returncode == 0 else ""
            if status == "healthy":
                return
            if status == "missing":
                raise RuntimeError(
                    f"Task service {service['name']!r} image must declare a Docker HEALTHCHECK"
                )
            if status == "unhealthy":
                raise RuntimeError(f"Task service {service['name']!r} failed its Docker HEALTHCHECK")
            running = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", container],
                capture_output=True, text=True, timeout=10,
            )
            if running.returncode == 0 and running.stdout.strip() == "false":
                raise RuntimeError(f"Task service {service['name']!r} exited during startup")
            time.sleep(0.5)
        raise RuntimeError(
            f"Task service {service['name']!r} did not become healthy within "
            f"{service['startup_timeout_seconds']}s"
        )

    @staticmethod
    def _docker_control(command: list[str], *, timeout: int) -> None:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            details = (result.stderr or result.stdout).strip()
            raise RuntimeError(details or f"Docker command failed: {command[1:3]}")

    @staticmethod
    def _remove_task_service_network(network: str) -> None:
        try:
            subprocess.run(
                ["docker", "network", "rm", network],
                capture_output=True, text=True, timeout=20,
            )
        except Exception:
            logger.warning("Could not remove task service network %s", network)

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
        docker_action: str = "run",
        docker_network_override: str | None = None,
    ) -> list[str]:
        cmd = [
            "docker",
            docker_action,
        ]
        if stdin_open:
            cmd.append("-i")
        cmd += [
            "--name",
            container_name,
        ]
        if docker_action == "run":
            cmd.append("--rm")

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
        if docker_network_override:
            cmd += ["--network", docker_network_override]
        elif needs_api_network:
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
