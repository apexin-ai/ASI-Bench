"""Docker image build/cache helpers for OS sandbox execution."""

from __future__ import annotations

import hashlib
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ai4sci_bench.runner.task_env import TaskEnvironmentManager

logger = logging.getLogger(__name__)

# Minimum tools that must be present in any custom image.
REQUIRED_IMAGE_TOOLS = ("python3", "uv", "git", "curl", "node")

AGENT_INSTALL_COMMANDS: dict[str | None, list[str]] = {
    "claude_code": ["npm install -g @anthropic-ai/claude-code"],
    "codex":       ["npm install -g @openai/codex"],
    "kimi_code":   ["npm install -g @moonshot-ai/kimi-code"],
    "mimo_code":   ["npm install -g @mimo-ai/cli"],
    # Native adapters are verified against these exact CLI schemas. Floating
    # npm dist-tags have already served incompatible flag sets in production.
    "pi":          ["npm install -g @earendil-works/pi-coding-agent@0.84.3"],
    "opencode":    ["npm install -g opencode-ai@1.17.15"],
    "openhands":   [],  # mounted from host venv, no install needed
    "agy":         [],  # standalone binary, must be pre-installed on host
}


class TaskImageBuilder:
    """Build and cache Docker images keyed by task runtime requirements."""

    PYTHON_BASE_IMAGE = "python:3.12-slim"
    # v10: Node 22 is required by the pinned pi 0.84.3 CLI (node:fs.globSync).
    # v9: preserve /opt/venv/bin in login shells spawned by Codex command execution.
    # v8: chmod 0777 on /home/agent and /tmp/agent-auth so that --user host_uid
    # (set by OSSandbox on Linux to fix bind-mount UID mismatch) can still read
    # auth mounts and let CLIs write session state under HOME.
    BASE_IMAGE_SCHEMA_VERSION = 10

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.env_manager = TaskEnvironmentManager(repo_root)

    def ensure_docker_available(self) -> None:
        """Fail closed when Docker is unavailable."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Docker CLI not found; --sandbox os requires Docker.") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Timed out while checking Docker; --sandbox os requires a healthy Docker daemon.") from exc

        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            suffix = f" Details: {details}" if details else ""
            raise RuntimeError(
                "Docker is not available; --sandbox os requires a running Docker daemon."
                f"{suffix}"
            )

    def ensure_base_image(self) -> str:
        """Ensure the shared minimal base image exists.

        The base image includes:
        - uv (fast Python package manager)
        - System tools: git, curl, ca-certificates, build-essential
        - Directory structure: /opt/venv, /opt/ai4sci-bench, /home/agent, /tmp/agent-auth
        - ENV: PATH, VIRTUAL_ENV, AI4SCI_SANDBOX=1, HOME=/home/agent, LANG=C.UTF-8
        """
        self.ensure_docker_available()
        tag = self._base_image_tag()
        if self._image_exists(tag):
            return tag

        dockerfile = (
            f"FROM {self.PYTHON_BASE_IMAGE}\n"
            "\n"
            "# System tools\n"
            "RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
            "    git curl ca-certificates build-essential \\\n"
            "    && rm -rf /var/lib/apt/lists/*\n"
            "\n"
            "# Install Node.js (agent CLIs need it)\n"
            "RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \\\n"
            "    && apt-get install -y --no-install-recommends nodejs \\\n"
            "    && rm -rf /var/lib/apt/lists/*\n"
            "\n"
            "# Install uv\n"
            "RUN curl -LsSf https://astral.sh/uv/install.sh | sh \\\n"
            '    && mv /root/.local/bin/uv /usr/local/bin/uv \\\n'
            '    && mv /root/.local/bin/uvx /usr/local/bin/uvx\n'
            "\n"
            "# Create virtual environment with uv\n"
            "RUN uv venv /opt/venv\n"
            "\n"
            "# Keep task virtualenv first even in login shells used by CLI tools\n"
            "RUN printf '%s\\n' \\\n"
            "    'export PATH=\"/opt/venv/bin:/usr/local/bin:/usr/bin:/bin\"' \\\n"
            "    'export VIRTUAL_ENV=\"/opt/venv\"' \\\n"
            "    > /etc/profile.d/ai4sci-venv.sh\n"
            "\n"
            "# Create non-root user (Claude CLI refuses bypassPermissions as root).\n"
            "# Directories under HOME are made world-writable so the container can\n"
            "# also be launched with `--user <host_uid>:<host_gid>` (used on Linux\n"
            "# to align container UID with the bind-mounted workspace owner) and\n"
            "# still let the CLI write session state under /home/agent.\n"
            "RUN useradd -m -s /bin/bash -u 1000 agent \\\n"
            "    && mkdir -p /opt/ai4sci-bench /home/agent /tmp/agent-auth /workspace \\\n"
            "    && mkdir -p /home/agent/.claude/session-env /home/agent/.claude/sessions \\\n"
            "    && mkdir -p /home/agent/.codex /home/agent/.kimi-code \\\n"
            "    && mkdir -p /home/agent/.local/share/mimocode \\\n"
            "    && chown -R agent:agent /opt/venv /home/agent /tmp/agent-auth /workspace \\\n"
            "    && chmod -R 0777 /home/agent /tmp/agent-auth\n"
            "\n"
            "# Environment variables\n"
            'ENV PATH="/opt/venv/bin:/usr/local/bin:$PATH"\n'
            'ENV VIRTUAL_ENV="/opt/venv"\n'
            'ENV AI4SCI_SANDBOX="1"\n'
            'ENV HOME="/home/agent"\n'
            'ENV LANG="C.UTF-8"\n'
            "\n"
            "USER agent\n"
            "WORKDIR /workspace\n"
            'LABEL ai4sci.base_image="true"\n'
        )
        self._build_image(tag, dockerfile)
        return tag

    def ensure_image(
        self,
        task_metadata: dict[str, Any],
        agent_type: str | None = None,
    ) -> str:
        """Ensure the runtime image for one task + agent combination exists.

        Image selection priority:
        1. ``runtime.image`` — use a pre-built custom image directly
        2. ``runtime.dockerfile`` — build a custom image from the given Dockerfile
        3. Base image + agent CLI overlay + ``runtime.packages`` overlay (default)

        When ``agent_type`` is given, the agent's CLI tool is installed into
        the image on demand (e.g. ``npm install -g @moonshot-ai/kimi-code``
        for ``agent_type="kimi_code"``). This avoids bloating the base image
        with every possible agent CLI.
        """
        # 1. Custom image specified directly
        custom_image = task_metadata.get("runtime", {}).get("image") if isinstance(task_metadata.get("runtime"), dict) else None
        if custom_image:
            self.ensure_docker_available()
            if not self._image_exists(custom_image):
                raise RuntimeError(
                    f"Custom runtime image '{custom_image}' not found. "
                    "Pull or build it before running with --sandbox os."
                )
            return custom_image

        # 2. Custom Dockerfile
        custom_dockerfile = task_metadata.get("runtime", {}).get("dockerfile") if isinstance(task_metadata.get("runtime"), dict) else None
        if custom_dockerfile:
            task_dir = task_metadata.get("_task_dir")
            if task_dir is None:
                raise RuntimeError("Cannot resolve runtime.dockerfile without _task_dir in metadata")
            dockerfile_path = Path(task_dir) / custom_dockerfile
            if not dockerfile_path.exists():
                raise RuntimeError(f"runtime.dockerfile '{custom_dockerfile}' not found at {dockerfile_path}")
            tag = self._custom_dockerfile_tag(dockerfile_path)
            if not self._image_exists(tag):
                self._build_image_from_file(tag, dockerfile_path, Path(task_dir))
            return tag

        # 3. Default: base image + agent CLI + task packages overlay
        base_tag = self.ensure_base_image()
        runtime_packages = list(task_metadata.get("_runtime_packages", []))
        agent_install_cmds = AGENT_INSTALL_COMMANDS.get(agent_type, [])

        if not runtime_packages and not agent_install_cmds:
            return base_tag

        tag = self._task_agent_image_tag(task_metadata, agent_type)
        if self._image_exists(tag):
            return tag

        lines = [f"FROM {base_tag}\n", "USER root\n"]
        if agent_install_cmds:
            lines.append(f"# Install {agent_type} CLI\n")
            for cmd in agent_install_cmds:
                lines.append(f"RUN {cmd}\n")
        if runtime_packages:
            packages = " ".join(runtime_packages)
            lines.append(f"RUN uv pip install {packages}\n")
        lines.append("RUN chown -R agent:agent /opt/venv /home/agent\n")
        lines.append("USER agent\n")
        lines.append(
            f'LABEL ai4sci.task.cache_key="{self.env_manager.compute_cache_key(task_metadata)}"\n'
            f'LABEL ai4sci.agent_type="{agent_type or "none"}"\n'
        )
        self._build_image(tag, "".join(lines))
        return tag

    def get_image_identity(self, image: str) -> str:
        """Return a stable identity string for provenance."""
        self.ensure_docker_available()
        result = subprocess.run(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode == 0:
            identity = result.stdout.strip()
            if identity:
                return identity
        return image

    # ------------------------------------------------------------------
    # Image validation helpers
    # ------------------------------------------------------------------

    def validate_image(self, image: str) -> list[str]:
        """Validate that an image contains required tools and a writable /workspace.

        Returns a list of error strings (empty = valid).
        """
        self.ensure_docker_available()
        errors: list[str] = []

        # Check required tools
        for tool in REQUIRED_IMAGE_TOOLS:
            result = subprocess.run(
                ["docker", "run", "--rm", image, "which", tool],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                errors.append(f"Image '{image}' is missing required tool: {tool}")

        # Check /workspace is writable
        result = subprocess.run(
            ["docker", "run", "--rm", image, "sh", "-c", "touch /workspace/.probe && rm /workspace/.probe"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            errors.append(f"Image '{image}': /workspace is not writable")

        return errors

    def check_nvidia_toolkit(self) -> bool:
        """Return True if NVIDIA Container Toolkit is available."""
        try:
            result = subprocess.run(
                ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:12.0.0-base-ubuntu22.04", "nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except Exception:
            return False

    def check_nvidia_toolkit_installed(self) -> bool:
        """Return True if nvidia-container-toolkit is installed (without pulling images)."""
        try:
            result = subprocess.run(
                ["nvidia-container-cli", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clean_images(self, *, include_base: bool = False) -> list[str]:
        """Remove cached ai4sci-bench Docker images.

        Returns list of removed image tags.
        """
        self.ensure_docker_available()
        removed: list[str] = []

        # Find all ai4sci-bench-task images
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", "ai4sci-bench-task"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                tag = line.strip()
                if tag and tag != "<none>:<none>":
                    self._remove_image(tag)
                    removed.append(tag)

        if include_base:
            result = subprocess.run(
                ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", "ai4sci-bench-base"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    tag = line.strip()
                    if tag and tag != "<none>:<none>":
                        self._remove_image(tag)
                        removed.append(tag)

        return removed

    # ------------------------------------------------------------------
    # Docker daemon health
    # ------------------------------------------------------------------

    def check_daemon_health(self) -> bool:
        """Return True if the Docker daemon is responsive (heartbeat check)."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _image_exists(self, tag: str) -> bool:
        result = subprocess.run(
            ["docker", "image", "inspect", tag],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return result.returncode == 0

    def _build_image(self, tag: str, dockerfile: str) -> None:
        dockerfile_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="\n",
                suffix=".Dockerfile",
                delete=False,
            ) as tmp:
                tmp.write(dockerfile)
                dockerfile_path = Path(tmp.name)

            result = subprocess.run(
                ["docker", "build", "-f", str(dockerfile_path), "-t", tag, str(self.repo_root)],
                capture_output=True,
                text=True,
                timeout=1800,
            )
        finally:
            if dockerfile_path is not None:
                try:
                    dockerfile_path.unlink(missing_ok=True)
                except Exception:
                    logger.warning("Failed to remove temporary Dockerfile: %s", dockerfile_path)
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"Failed to build Docker image {tag}: {details}")

    def _build_image_from_file(self, tag: str, dockerfile_path: Path, context_dir: Path) -> None:
        """Build an image from an existing Dockerfile on disk."""
        result = subprocess.run(
            ["docker", "build", "-f", str(dockerfile_path), "-t", tag, str(context_dir)],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"Failed to build custom Docker image {tag}: {details}")

    def _remove_image(self, tag: str) -> None:
        """Remove a Docker image by tag."""
        try:
            subprocess.run(
                ["docker", "rmi", tag],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception:
            logger.warning("Failed to remove image: %s", tag)

    def _base_image_tag(self) -> str:
        payload = f"{self.PYTHON_BASE_IMAGE}|{self.BASE_IMAGE_SCHEMA_VERSION}".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:12]
        return f"ai4sci-bench-base:{digest}"

    def _task_image_tag(self, task_metadata: dict[str, Any]) -> str:
        cache_key = self.env_manager.compute_cache_key(task_metadata)
        return f"ai4sci-bench-task:{cache_key}"

    def _task_agent_image_tag(self, task_metadata: dict[str, Any], agent_type: str | None) -> str:
        cache_key = self.env_manager.compute_cache_key(task_metadata)
        agent_suffix = agent_type or "base"
        install_identity = "\n".join(AGENT_INSTALL_COMMANDS.get(agent_type, []))
        payload = (
            f"{self._base_image_tag()}|{cache_key}|{agent_suffix}|{install_identity}"
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:12]
        return f"ai4sci-bench-task:{digest}"

    def _custom_dockerfile_tag(self, dockerfile_path: Path) -> str:
        """Derive a tag from the Dockerfile content hash."""
        content = dockerfile_path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()[:12]
        return f"ai4sci-bench-custom:{digest}"
