"""CodeX CLI adapter — runs OpenAI CodeX as the evaluated agent."""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

from ai4sci_bench.adapters.subprocess_base import (
    SubprocessAgentAdapter,
    collect_output_files,
    safe_run_key,
)
from ai4sci_bench.core.types import AgentOutput, CostInfo, RunStatus, TaskInstance, ToolMode
from ai4sci_bench.runner.os_sandbox import OSSandbox

logger = logging.getLogger(__name__)

OS_SANDBOX_PYTHON = "/opt/venv/bin/python"
CODEX_AUTH_FILENAMES = ("auth.json",)

CODEX_RESTRICTED_DISABLE_FEATURES = (
    "plugins",
    "tool_call_mcp_elicitation",
    "skill_mcp_dependency_install",
    "image_generation",
    "codex_hooks",
    "computer_use",
)


class CodexCLIAdapter(SubprocessAgentAdapter):
    """Run OpenAI CodeX CLI as the evaluated agent.

    CodeX autonomously reads files, writes code, executes it,
    debugs failures, and produces output files — similar to Claude Code.

    Supports four authentication modes:

    1. **Local login** (default): No ``api_key`` / ``api_base`` — uses
       Codex CLI's own local auth (``codex auth login``).
    2. **OpenAI API Key**: Only ``api_key`` — injects ``OPENAI_API_KEY``
       and uses the official OpenAI endpoint.
    3. **Native provider** (recommended for third-party): ``provider`` +
       ``api_key`` + ``api_base`` — generates a temporary ``CODEX_HOME``
       config that uses Codex CLI's native provider support (e.g.
       ``openrouter``, ``openai-compatible``). No proxy needed.
    4. **litellm proxy** (fallback): ``api_key`` + ``api_base`` without
       ``provider`` — starts a local litellm proxy. Note: this mode has
       known limitations with Codex's WebSocket protocol.

    For third-party APIs (DeepSeek, OpenRouter, etc.), mode 3 is strongly
    recommended. Users can also set ``CODEX_HOME`` env var externally to
    point to a pre-configured Codex config directory.
    """

    VALID_EFFORT_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh")

    NATIVE_CODEX_PROVIDERS = frozenset({
        "openrouter", "openai-compatible", "openai", "anthropic",
        "gemini", "ollama", "mistral",
    })

    def __init__(
        self,
        model: str = "gpt-5.6-sol",
        full_auto: bool = True,
        timeout_seconds: int = 10800,
        allow_external_tools: bool = False,
        tool_mode: str | ToolMode | None = None,
        effort: str = "medium",
        api_key: str | None = None,
        api_base: str | None = None,
        provider: str | None = None,
        codex_home: str | None = None,
        supports_image_input: bool = False,
    ):
        super().__init__(
            timeout_seconds=timeout_seconds,
            supported_sandbox_modes=("none", "task", "os", "linux_ns"),
        )
        if effort not in self.VALID_EFFORT_LEVELS:
            raise ValueError(
                f"Invalid effort '{effort}' for codex_cli. "
                f"Valid levels: {', '.join(self.VALID_EFFORT_LEVELS)}"
            )
        self.model = model
        self.full_auto = full_auto
        self.allow_external_tools = allow_external_tools
        self.tool_mode = self._resolve_tool_mode(tool_mode, allow_external_tools)
        self.effort = effort
        self.api_key = api_key
        self.api_base = api_base
        self.provider = provider
        self.codex_home = codex_home
        self.supports_image_input = supports_image_input
        self._uses_native_provider = (
            provider is not None and api_key is not None and api_base is not None
        )
        self._uses_proxy = (
            api_key is not None
            and api_base is not None
            and not self._uses_native_provider
        )
        self._os_sandbox: OSSandbox | None = None
        self._sandbox_image_identity: str | None = None
        self._proxy: object | None = None
        self._proxy_lock = threading.Lock()
        self._temp_codex_home: str | None = None

    @staticmethod
    def _resolve_tool_mode(
        tool_mode: str | ToolMode | None,
        allow_external_tools: bool,
    ) -> ToolMode:
        if tool_mode is not None:
            if isinstance(tool_mode, str):
                return ToolMode(tool_mode)
            return tool_mode
        return ToolMode.SEARCH if allow_external_tools else ToolMode.RESTRICTED

    # ── Native provider config ────────────────────────────────

    def _ensure_codex_home(self) -> str:
        """Create or return the CODEX_HOME directory path.

        If ``codex_home`` was explicitly provided, use it directly.
        If ``_uses_native_provider`` is True, generate a temporary
        CODEX_HOME with config.toml pointing to the target provider.
        """
        if self.codex_home:
            return self.codex_home
        if self._temp_codex_home:
            return self._temp_codex_home

        tmpdir = tempfile.mkdtemp(prefix="codex_home_")
        os.chmod(tmpdir, 0o700)

        config_content = self._generate_codex_config()
        config_path = os.path.join(tmpdir, "config.toml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)

        self._temp_codex_home = tmpdir
        logger.info(
            "codex_cli: generated CODEX_HOME at %s (provider=%s)",
            tmpdir, self.provider,
        )
        return tmpdir

    def _generate_codex_config(self) -> str:
        """Generate a Codex config.toml for the native provider."""
        provider = self.provider or "openrouter"
        provider_name = provider.replace("-", " ").title()
        base_url = self.api_base or ""
        token = self.api_key or ""

        return (
            f'model_provider = "{provider}"\n'
            f'\n[model_providers.{provider}]\n'
            f'name = "{provider_name}"\n'
            f'base_url = "{base_url}"\n'
            f'experimental_bearer_token = "{token}"\n'
        )

    # ── Proxy lifecycle (fallback for non-native providers) ───

    def _ensure_proxy(self) -> str:
        """Start the litellm OpenAI proxy (once) and return its local URL."""
        with self._proxy_lock:
            if self._proxy is not None:
                return self._proxy.local_url  # type: ignore[union-attr]
            from ai4sci_bench.adapters.api_proxy import LiteLLMOpenAIProxy
            self._proxy = LiteLLMOpenAIProxy(
                model=self.model,
                api_base=self.api_base,
                api_key=self.api_key,
                supports_image_input=self.supports_image_input,
            )
            return self._proxy.start()  # type: ignore[union-attr]

    def _build_api_env(self) -> dict[str, str]:
        """Build env vars for API authentication."""
        env: dict[str, str] = {}
        if self._uses_native_provider or self.codex_home:
            codex_home = self._ensure_codex_home()
            env["CODEX_HOME"] = codex_home
            logger.info("codex_cli: using CODEX_HOME=%s", codex_home)
        elif self._uses_proxy:
            proxy_url = self._ensure_proxy()
            env["OPENAI_API_KEY"] = "sk-proxy-placeholder"
            env["OPENAI_BASE_URL"] = proxy_url
            logger.info("codex_cli: using litellm proxy at %s", proxy_url)
        elif self.api_key:
            env["OPENAI_API_KEY"] = self.api_key
        return env

    # ── Adapter lifecycle ──────────────────────────────────────

    def setup(self, config: dict) -> None:
        super().setup(config)
        if self.sandbox == "os":
            self._os_sandbox = OSSandbox(self.repo_root)

    def teardown(self) -> None:
        if self._proxy is not None:
            self._proxy.stop()  # type: ignore[union-attr]
            self._proxy = None
        if self._temp_codex_home is not None:
            shutil.rmtree(self._temp_codex_home, ignore_errors=True)
            self._temp_codex_home = None

    # ── Solve ──────────────────────────────────────────────────

    def solve(self, task_instance: TaskInstance) -> AgentOutput:
        """Execute agent — delegates to OSSandbox when sandbox == 'os'."""
        if self.sandbox != "os":
            if self._should_use_windows_task_bridge():
                output = self._solve_with_windows_task_bridge(task_instance)
            else:
                output = super().solve(task_instance)
            if output.cost is None and output.raw_stdout:
                output.cost = self._extract_usage_from_jsonl(output.raw_stdout)
            return output

        eff_timeout = self._get_effective_timeout(task_instance)
        workspace = task_instance.workspace_dir
        cmd = self._build_os_agent_cmd(workspace, python_executable=OS_SANDBOX_PYTHON)

        extra_env: dict[str, str] | None = self._build_api_env() or None

        t0 = time.time()
        assert self._os_sandbox is not None
        success, log, raw_stdout, raw_stderr, image_identity = (
            self._os_sandbox.run_agent(
                task_metadata=task_instance.metadata,
                agent_cmd=cmd,
                workspace=workspace,
                timeout=eff_timeout,
                agent_type="codex",
                allow_external_tools=self.allow_external_tools,
                extra_env=extra_env,
            )
        )
        elapsed = time.time() - t0
        self._sandbox_image_identity = image_identity

        produced_files = collect_output_files(workspace, task_instance)
        parsed_log = self._parse_log(raw_stdout or "") if raw_stdout else log

        if "timed out" in log:
            status = RunStatus.TIMEOUT
        elif success:
            status = RunStatus.COMPLETED
        else:
            status = RunStatus.FAILED

        return AgentOutput(
            instance_id=task_instance.instance_id,
            output_dir=workspace,
            code_files=[f for f in produced_files if f.endswith(".py")],
            data_files=[f for f in produced_files if not f.endswith(".py")],
            log=parsed_log,
            execution_time_seconds=elapsed,
            status=status,
            error_message=None if success else log,
            raw_stdout=raw_stdout,
            raw_stderr=raw_stderr,
            raw_stdout_format="jsonl" if raw_stdout else None,
            cost=self._extract_usage_from_jsonl(raw_stdout) if raw_stdout else None,
        )

    def _should_use_windows_task_bridge(self) -> bool:
        return self.sandbox == "task" and self._is_windows_platform()

    @staticmethod
    def _is_windows_platform() -> bool:
        return os.name == "nt"

    @property
    def sandbox_image_identity(self) -> str | None:
        return self._sandbox_image_identity

    def _solve_with_windows_task_bridge(
        self, task_instance: TaskInstance
    ) -> AgentOutput:
        workspace = task_instance.workspace_dir
        bridge_workspace = self._create_windows_task_bridge(task_instance)
        try:
            bridge_instance = replace(task_instance, workspace_dir=bridge_workspace)
            bridge_output = super().solve(bridge_instance)
            self._sync_expected_outputs(
                bridge_workspace=bridge_workspace,
                target_workspace=workspace,
                task_instance=task_instance,
            )
            produced_files = collect_output_files(workspace, task_instance)

            return replace(
                bridge_output,
                output_dir=workspace,
                code_files=[f for f in produced_files if f.endswith(".py")],
                data_files=[f for f in produced_files if not f.endswith(".py")],
            )
        finally:
            self._cleanup_bridge_workspace(bridge_workspace)

    def _create_windows_task_bridge(self, task_instance: TaskInstance) -> Path:
        bridge_root = self.repo_root / ".ai4sci-bench" / "codex_windows_task_bridge"
        bridge_root.mkdir(parents=True, exist_ok=True)
        bridge_workspace = (
            bridge_root
            / f"{task_instance.prompt_level.value}_{task_instance.instance_id[:32]}_{time.time_ns():x}"
        )
        bridge_workspace.mkdir(parents=True, exist_ok=False)
        self._copy_workspace_inputs(
            task_instance.workspace_dir,
            bridge_workspace,
            task_instance,
        )
        return bridge_workspace

    def _copy_workspace_inputs(
        self,
        source_workspace: Path,
        bridge_workspace: Path,
        task_instance: TaskInstance,
    ) -> None:
        excluded_roots = {
            Path(spec["name"]).parts[0]
            for spec in task_instance.metadata.get("output", {}).get("files", [])
            if Path(spec["name"]).parts
        }
        for item in source_workspace.iterdir():
            if item.name in excluded_roots:
                continue
            dst = bridge_workspace / item.name
            if item.is_dir():
                shutil.copytree(item, dst, copy_function=shutil.copy)
            else:
                shutil.copy(item, dst)

    def _sync_expected_outputs(
        self,
        *,
        bridge_workspace: Path,
        target_workspace: Path,
        task_instance: TaskInstance,
    ) -> None:
        for spec in task_instance.metadata.get("output", {}).get("files", []):
            rel_path = Path(spec["name"])
            src = bridge_workspace / rel_path
            if not src.exists():
                continue
            dst = target_workspace / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    def _cleanup_bridge_workspace(self, bridge_workspace: Path) -> None:
        if not bridge_workspace.exists():
            return

        def _retry_with_write_permission(func, path, exc_info):
            del exc_info
            os.chmod(path, stat.S_IWRITE)
            func(path)

        try:
            shutil.rmtree(bridge_workspace, onerror=_retry_with_write_permission)
        except OSError:
            # Best-effort cleanup only: the benchmark result should not fail
            # just because Windows denies bridge-workspace deletion.
            pass

    def _apply_tool_isolation(self, cmd: list[str]) -> None:
        """Append tool-isolation flags based on the resolved ToolMode."""
        if self.tool_mode == ToolMode.UNRESTRICTED:
            return
        # --ignore-user-config prevents user-configured MCP servers from
        # being discovered and connected; --disable plugins alone is NOT
        # sufficient (verified: Appendix F.3).
        # Also ignore ambient execpolicy rules so benchmark runs are not
        # silently tightened by host- or repo-local Codex policies.
        cmd += ["--ignore-user-config", "--ignore-rules"]
        for feature in CODEX_RESTRICTED_DISABLE_FEATURES:
            cmd += ["--disable", feature]
        if self.tool_mode == ToolMode.RESTRICTED:
            cmd += ["--config", 'web_search="disabled"']

    def _build_os_agent_cmd(
        self,
        workspace,
        *,
        python_executable: str | None = None,
    ) -> list[str]:
        """Build the codex CLI command for execution inside a Docker container.

        Uses --dangerously-bypass-approvals-and-sandbox to skip approvals and
        disable bwrap sandbox (Docker provides isolation). This flag already
        implies auto-approval so --full-auto / --sandbox must NOT be added
        (the two are mutually exclusive in Codex CLI >= 0.118).
        """
        prompt = self._build_prompt(workspace, None, python_executable=python_executable)
        cmd = [
            "codex",
            "exec",
            "--model", self.model,
            "-c", f'model_reasoning_effort="{self.effort}"',
            "--cd", "/workspace",
            "--skip-git-repo-check",
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
        ]
        self._apply_tool_isolation(cmd)
        cmd += ["--", prompt]
        return cmd

    def _build_command(self, task_instance: TaskInstance, task_env):
        workspace = task_instance.workspace_dir
        prompt = self._build_prompt(workspace, task_env)

        codex_bin = "codex.cmd" if os.name == "nt" else "codex"
        # Always resolve to absolute path to avoid double-resolution when
        # both subprocess cwd= and --cd are set (fixes #15).
        abs_workspace = str(workspace.resolve())
        cmd = [
            codex_bin,
            "exec",
            "--model", self.model,
            "-c", f'model_reasoning_effort="{self.effort}"',
            "--cd", abs_workspace,
            "--skip-git-repo-check",
            "--json",
        ]
        # Under linux_ns and os, the benchmark already provides the outer
        # isolation layer, so Codex should not nest its own sandbox.
        #
        # Under sandbox=none the benchmark explicitly requested host-mode
        # execution. Leaving Codex in its default approval flow here can
        # degrade non-interactive runs into an effectively read-only session,
        # which prevents writing outputs even though the benchmark itself is
        # not enforcing isolation. In that host-mode case, match the outer
        # benchmark contract and bypass Codex's inner approval/sandbox layer.
        if self.sandbox in ("none", "linux_ns"):
            cmd += ["--dangerously-bypass-approvals-and-sandbox"]
        else:
            cmd += ["--sandbox", "workspace-write"]
        self._apply_tool_isolation(cmd)
        # Pass prompt via stdin to avoid Windows command-line length limits
        # (fixes #12). The sentinel "-" tells codex to read from stdin.
        cmd += ["--", "-"]
        return cmd

    def _get_stdin_input(self, task_instance=None, task_env=None) -> str | None:
        """Return prompt text to pipe via stdin (fixes #12 Windows cmd length).

        Reads directly from workspace (thread-safe); no shared mutable state.
        """
        if task_instance is None:
            return None
        return self._build_prompt(task_instance.workspace_dir, task_env)

    def _build_run_env(self, task_instance: TaskInstance, task_env):
        """Override: inject API env vars for modes 2 & 3, or strip key for mode 1.

        When tool_mode is not UNRESTRICTED, also isolate HOME so the
        Codex process cannot see ambient config, MCP servers, or skills.
        """
        env = super()._build_run_env(task_instance, task_env) or os.environ.copy()
        api_env = self._build_api_env()
        if api_env:
            env.update(api_env)
        else:
            env.pop("OPENAI_API_KEY", None)
        if self.tool_mode != ToolMode.UNRESTRICTED:
            api_codex_home = api_env.get("CODEX_HOME") if api_env else None
            isolated_home = self._prepare_isolated_codex_home(
                task_instance, api_codex_home=api_codex_home,
            )
            env["HOME"] = str(isolated_home)
            env["USERPROFILE"] = str(isolated_home)
            env["CODEX_HOME"] = str(isolated_home / ".codex")
            env["XDG_CONFIG_HOME"] = str(isolated_home / ".config")
        return env

    def _prepare_isolated_codex_home(
        self,
        task_instance: TaskInstance,
        *,
        api_codex_home: str | None = None,
    ) -> Path:
        """Create a minimal Codex home that excludes ambient user config.

        When *api_codex_home* is set (native-provider mode created a temp
        CODEX_HOME with config.toml), its config is copied into the
        isolated dir so the provider routing is preserved.
        """
        home = self.repo_root / ".ai4sci-bench" / "codex_home" / safe_run_key(task_instance.run_key)
        codex_dir = home / ".codex"
        config_dir = home / ".config"
        codex_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(home, 0o700)
            os.chmod(codex_dir, 0o700)
            os.chmod(config_dir, 0o700)
        except OSError:
            pass

        source_codex_dir = self._host_codex_dir()
        for filename in CODEX_AUTH_FILENAMES:
            src = source_codex_dir / filename
            if not src.is_file():
                continue
            dst = codex_dir / filename
            shutil.copy2(src, dst)
            try:
                os.chmod(dst, 0o600)
            except OSError:
                pass

        if api_codex_home:
            api_config = Path(api_codex_home) / "config.toml"
            if api_config.is_file():
                shutil.copy2(api_config, codex_dir / "config.toml")

        return home

    @staticmethod
    def _host_codex_dir() -> Path:
        """Return the real host Codex auth directory, not an isolated HOME."""
        codex_home = os.environ.get("CODEX_HOME")
        if codex_home:
            return Path(codex_home)
        home = os.environ.get("USERPROFILE") if os.name == "nt" else os.environ.get("HOME")
        if not home:
            home = str(Path.home())
        return Path(home) / ".codex"

    def _parse_log(self, stdout: str) -> str:
        return self._parse_codex_log(stdout)

    def _raw_stdout_format(self) -> str:
        return "jsonl"

    _TOOL_USE_VALUE_KEYS = (
        "file_path",
        "path",
        "command",
        "pattern",
        "url",
        "notebook_path",
    )

    def _parse_codex_log(self, stdout: str) -> str:
        """Parse JSONL output from ``codex exec --json`` into a summary log.

        Captures assistant/user messages, tool calls (including key
        argument values like file paths / commands), tool outputs,
        errors and rate-limit signals. Raw stdout is still saved
        verbatim so nothing is lost — this is the fast-triage view.
        """
        lines: list[str] = []
        turn_counter = 0
        first_ts: float | None = None
        stdout_text = self._coerce_text(stdout)
        for raw_line in stdout_text.splitlines():
            line = self._coerce_text(raw_line)
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                lines.append(line)
                continue

            etype = event.get("type", "")
            if etype == "message" and event.get("role") == "assistant":
                turn_counter += 1
                ts = event.get("timestamp")
                try:
                    ts_float = float(ts) if ts is not None else None
                except (ValueError, TypeError):
                    ts_float = None
                if ts_float is not None:
                    if first_ts is None:
                        first_ts = ts_float
                    relative = ts_float - first_ts
                    lines.append(f"=== Turn {turn_counter} [+{relative:.1f}s] ===")
                else:
                    lines.append(f"=== Turn {turn_counter} ===")
            if etype == "message":
                lines.extend(self._format_codex_message(event))
            elif etype in ("function_call", "tool_use"):
                lines.append(self._format_codex_tool_call(event))
            elif etype in ("function_call_output", "tool_result"):
                output = event.get("output", event.get("content", ""))
                output_text = self._coerce_text(output)
                if output_text:
                    lines.append(f"[ToolResult] {output_text}")
            elif etype == "error":
                msg = event.get("message", str(event))
                msg_text = self._coerce_text(msg)
                lines.append(f"[Error] {msg_text}")
            elif etype in ("rate_limit_event", "rate_limit"):
                msg = event.get("message") or event
                lines.append(f"[rate_limit] {self._coerce_text(msg)}")
            elif etype in ("reasoning", "thinking"):
                text = self._coerce_text(event.get("text") or event.get("content") or "")
                if text:
                    lines.append(f"[thinking] {text}")
            elif etype:
                lines.append(f"[{etype}] {stripped}")
            else:
                lines.append(line)
        return "\n".join(lines)

    def _format_codex_message(self, event: dict) -> list[str]:
        role = event.get("role", "")
        content = event.get("content", "")
        out: list[str] = []
        if isinstance(content, (str, bytes, bytearray)):
            text = self._coerce_text(content)
            if text:
                out.append(f"[{role}] {text}")
            return out
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text = self._coerce_text(block.get("text", ""))
                    if text:
                        out.append(f"[{role}] {text}")
                elif btype == "tool_use":
                    out.append(self._format_codex_tool_call(block))
                elif btype == "tool_result":
                    inner = block.get("content", "")
                    if isinstance(inner, list):
                        parts = [
                            self._coerce_text(item.get("text", ""))
                            for item in inner
                            if isinstance(item, dict) and item.get("type") == "text"
                        ]
                        text = "\n".join(parts)
                    else:
                        text = self._coerce_text(inner)
                    label = "[ToolResult:err]" if block.get("is_error") else "[ToolResult]"
                    out.append(f"{label} {text}")
        return out

    def _format_codex_tool_call(self, event: dict) -> str:
        name = self._coerce_text(event.get("name", "unknown"))
        inputs = event.get("input") or event.get("arguments") or {}
        if isinstance(inputs, (bytes, bytearray, str)):
            # Codex sometimes serializes arguments as a JSON string.
            if isinstance(inputs, (bytes, bytearray)):
                inputs_str = self._coerce_text(inputs)
            else:
                inputs_str = inputs
            try:
                parsed = json.loads(inputs_str)
            except (json.JSONDecodeError, ValueError):
                trimmed = inputs_str[:200]
                return f"[Tool] {name}({trimmed})"
            inputs = parsed if isinstance(parsed, dict) else {}
        if not isinstance(inputs, dict):
            return f"[Tool] {name}"

        highlights: list[str] = []
        for key in self._TOOL_USE_VALUE_KEYS:
            if key in inputs:
                val = self._coerce_text(inputs[key])
                if len(val) > 200:
                    val = val[:200] + "..."
                highlights.append(f"{key}={val}")
        if highlights:
            return f"[Tool] {name}({', '.join(highlights)})"
        return f"[Tool] {name}({list(inputs.keys())})"

    def _coerce_text(self, value: object) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def _build_prompt(
        self,
        workspace: Path,
        task_env,
        *,
        python_executable: str | None = None,
    ) -> str:
        prompt = (workspace / "prompt.md").read_text(encoding="utf-8")
        if task_env is None and python_executable is None:
            return prompt

        helper_name = self._write_workspace_python_helper(
            workspace,
            task_env,
            python_executable=python_executable,
        )
        helper_path = workspace / helper_name
        python_desc = (
            f"`{python_executable}`"
            if python_executable is not None
            else "`python` via the sandboxed PATH provided to this run"
        )
        if os.name == "nt":
            helper_ref = str(helper_path.resolve())
            runtime_note = (
                "\n\n"
                "Runtime note for this workspace:\n"
                f"- Use `{helper_ref}` for all Python execution in this task workspace.\n"
                f"- The helper runs {python_desc}.\n"
                "- The helper first switches into its own directory, so relative paths like `data/...` stay anchored to the workspace even if the shell starts elsewhere.\n"
                "- Do not rely on bare `python` or `python3` from an ad-hoc shell, because they may resolve to the wrong environment.\n"
                f'- In PowerShell, invoke it as `& "{helper_ref}" -c "import numpy, taichi"`'
                f' or `& "{helper_ref}" simulation.py`.\n'
            )
        else:
            runtime_note = (
                "\n\n"
                "Runtime note for this workspace:\n"
                f"- Use `./{helper_name}` for all Python execution in this task workspace.\n"
                f"- The helper runs {python_desc}.\n"
                "- Do not rely on bare `python` or `python3` from an ad-hoc shell, because they may resolve to the wrong environment.\n"
                f"- For quick checks, prefer commands like `./{helper_name} -c \"import numpy, taichi\"`"
                f" or `./{helper_name} simulation.py`.\n"
            )
        return prompt + runtime_note

    def _write_workspace_python_helper(
        self,
        workspace: Path,
        task_env,
        *,
        python_executable: str | None = None,
    ) -> str:
        python_cmd = python_executable or "python"

        if os.name == "nt":
            helper_name = "_ai4sci_task_python.cmd"
            helper_path = workspace / helper_name
            helper_path.write_text(
                "@echo off\r\n"
                "cd /d \"%~dp0\"\r\n"
                f'"{python_cmd}" %*\r\n',
                encoding="utf-8",
            )
            return helper_name

        helper_name = "_ai4sci_task_python"
        helper_path = workspace / helper_name
        exec_target = "python" if python_cmd == "python" else f'"{python_cmd}"'
        helper_path.write_text(
            "#!/bin/sh\n"
            f'exec {exec_target} "$@"\n',
            encoding="utf-8",
        )
        helper_path.chmod(helper_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return helper_name

    @staticmethod
    def _extract_usage_from_jsonl(stdout: str) -> CostInfo | None:
        """Extract token usage from Codex JSONL output."""
        for line in stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            usage = event.get("usage")
            if not usage or not isinstance(usage, dict):
                continue
            input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
            output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
            if input_tokens or output_tokens:
                return CostInfo(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                )
        return None
