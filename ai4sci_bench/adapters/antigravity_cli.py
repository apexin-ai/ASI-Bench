"""Antigravity CLI adapter — runs Google's Antigravity (agy) as the evaluated agent."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

from ai4sci_bench.adapters.subprocess_base import (
    SubprocessAgentAdapter,
    collect_output_files,
)
from ai4sci_bench.core.types import AgentOutput, CostInfo, RunStatus, TaskInstance, ToolMode

logger = logging.getLogger(__name__)


class AntigravityCLIAdapter(SubprocessAgentAdapter):
    """Run Google Antigravity CLI (agy) as the evaluated agent.

    Antigravity CLI autonomously reads files, writes code, executes it,
    debugs failures, and produces output files — Google's terminal-based
    coding agent, the successor to Gemini CLI.

    Supports three authentication modes:

    1. **Local login** (default): No ``api_key`` / ``api_base`` — uses
       Google account auth or ``GEMINI_API_KEY`` from environment.
    2. **Gemini API Key**: Only ``api_key`` — injects ``GEMINI_API_KEY``
       env var, uses Google's official Gemini endpoint.
    3. **Third-party API**: ``api_base`` is set — requires ``api_key``
       and ``api_protocol`` (``openai`` or ``anthropic``). Starts a
       local litellm proxy that translates between the protocol ``agy``
       speaks and the target backend. ``model`` is the bare model name
       as the target service understands it.
    """

    def __init__(
        self,
        model: str = "gemini-3.5-flash",
        timeout_seconds: int = 10800,
        allow_external_tools: bool = False,
        tool_mode: str | ToolMode | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_protocol: str | None = None,
    ):
        super().__init__(
            timeout_seconds=timeout_seconds,
            supported_sandbox_modes=("none", "task", "os", "linux_ns"),
        )
        if api_base is not None:
            raise ValueError(
                "AntigravityCLIAdapter does not support third-party API endpoints. "
                "agy 1.0.12 only supports Google authentication (GEMINI_API_KEY or "
                "Google account login). For third-party models, use "
                "ClaudeCodeCLIAdapter or OpenHandsAdapter instead."
            )
        self.model = model
        self.allow_external_tools = allow_external_tools
        self.tool_mode = self._resolve_tool_mode(tool_mode, allow_external_tools)
        self.api_key = api_key
        self.api_base = None
        self.api_protocol = None
        self._uses_proxy = False
        self._proxy: object | None = None
        self._proxy_lock = threading.Lock()
        self._temp_config_dir: str | None = None
        self._os_sandbox: object | None = None
        self._sandbox_image_identity: str | None = None

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

    # ── Proxy lifecycle ───────────────────────────────────────

    def _ensure_proxy(self) -> str:
        with self._proxy_lock:
            if self._proxy is not None:
                return self._proxy.local_url  # type: ignore[union-attr]
            from ai4sci_bench.adapters.api_proxy import LiteLLMProxy, resolve_litellm_model
            litellm_model = resolve_litellm_model(self.model, self.api_protocol)
            self._proxy = LiteLLMProxy(
                model=litellm_model,
                api_base=self.api_base,
                api_key=self.api_key,
            )
            return self._proxy.start()  # type: ignore[union-attr]

    def _ensure_config_dir(self) -> str:
        """Create a temp config dir for agy with custom settings."""
        if self._temp_config_dir:
            return self._temp_config_dir

        tmpdir = tempfile.mkdtemp(prefix="agy_config_")
        os.chmod(tmpdir, 0o700)

        settings = {}
        if self._uses_proxy:
            proxy_url = self._ensure_proxy()
            settings["apiBaseUrl"] = proxy_url
            logger.info("antigravity_cli: using litellm proxy at %s", proxy_url)
        elif self.api_base:
            settings["apiBaseUrl"] = self.api_base

        settings_path = os.path.join(tmpdir, "settings.json")
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)

        self._temp_config_dir = tmpdir
        logger.info("antigravity_cli: generated config at %s", tmpdir)
        return tmpdir

    def _build_api_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        if self._uses_proxy:
            self._ensure_config_dir()
            env["GEMINI_API_KEY"] = "sk-proxy-placeholder"
            if self._temp_config_dir:
                env["ANTIGRAVITY_CONFIG_DIR"] = self._temp_config_dir
            logger.info("antigravity_cli: using proxy mode")
        elif self.api_key:
            env["GEMINI_API_KEY"] = self.api_key
        return env

    # ── Adapter lifecycle ──────────────────────────────────────

    def setup(self, config: dict) -> None:
        super().setup(config)
        if self.sandbox == "os":
            from ai4sci_bench.runner.os_sandbox import OSSandbox
            self._os_sandbox = OSSandbox(self.repo_root)

    def teardown(self) -> None:
        if self._proxy is not None:
            self._proxy.stop()  # type: ignore[union-attr]
            self._proxy = None
        if self._temp_config_dir is not None:
            shutil.rmtree(self._temp_config_dir, ignore_errors=True)
            self._temp_config_dir = None

    # ── Env override ───────────────────────────────────────────

    def _build_run_env(self, task_instance, task_env) -> dict[str, str] | None:
        base_env = super()._build_run_env(task_instance, task_env)
        api_env = self._build_api_env()
        if not api_env:
            return base_env
        env = dict(base_env) if base_env else dict(os.environ)
        env.update(api_env)
        return env

    # ── Solve ──────────────────────────────────────────────────

    def solve(self, task_instance: TaskInstance) -> AgentOutput:
        if self.sandbox != "os":
            output = super().solve(task_instance)
            if output.cost is None and output.raw_stdout:
                output.cost = self._extract_usage_from_jsonl(output.raw_stdout)
            return output

        eff_timeout = self._get_effective_timeout(task_instance)
        workspace = task_instance.workspace_dir
        cmd = self._build_os_agent_cmd(workspace)
        extra_env: dict[str, str] | None = self._build_api_env() or None

        t0 = time.time()
        assert self._os_sandbox is not None
        success, log, raw_stdout, raw_stderr, image_identity = (
            self._os_sandbox.run_agent(
                task_metadata=task_instance.metadata,
                agent_cmd=cmd,
                workspace=workspace,
                timeout=eff_timeout,
                agent_type="agy",
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
            raw_stdout_format="text" if raw_stdout else None,
            cost=self._extract_usage_from_jsonl(raw_stdout) if raw_stdout else None,
        )

    def _build_os_agent_cmd(self, workspace) -> list[str]:
        """Build agy command for Docker execution."""
        return [
            "agy",
            "-p",
            "--model", self.model,
            "--dangerously-skip-permissions",
        ]

    def _build_command(self, task_instance: TaskInstance, task_env):
        cmd = [
            "agy",
            "-p",
            "--model", self.model,
            "--dangerously-skip-permissions",
        ]
        return cmd

    def _get_stdin_input(self, task_instance=None, task_env=None) -> str | None:
        if task_instance is None:
            return None
        prompt_path = task_instance.workspace_dir / "prompt.md"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return None

    def _parse_log(self, stdout: str) -> str:
        return self._parse_agy_log(stdout)

    def _raw_stdout_format(self) -> str:
        return "text"

    # ── Log parsing ────────────────────────────────────────────

    _TOOL_USE_VALUE_KEYS = (
        "file_path", "path", "command", "pattern", "url",
    )

    def _parse_agy_log(self, stdout: str) -> str:
        lines: list[str] = []
        turn_counter = 0
        first_ts: float | None = None
        for line in stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                lines.append(line)
                continue

            etype = event.get("type", "")

            # Claude Code style events
            if etype == "assistant":
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
                lines.extend(self._format_assistant_blocks(event))
            # Codex / generic message style events
            elif etype == "message" and event.get("role") == "assistant":
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
                lines.extend(self._format_message_blocks(event))
            elif etype == "message":
                lines.extend(self._format_message_blocks(event))
            elif etype == "user":
                lines.extend(self._format_user_blocks(event))
            elif etype == "system":
                subtype = event.get("subtype") or ""
                msg = event.get("message") or event.get("text") or ""
                label = f"[system:{subtype}]" if subtype else "[system]"
                if msg:
                    lines.append(f"{label} {str(msg)}")
            elif etype in ("function_call", "tool_use"):
                lines.append(self._format_tool_use(event))
            elif etype in ("function_call_output", "tool_result"):
                output = event.get("output", event.get("content", ""))
                if output:
                    lines.append(f"[ToolResult] {str(output)}")
            elif etype in ("error", "rate_limit_event", "rate_limit"):
                msg = event.get("message") or event.get("error") or event
                label = "[error]" if etype == "error" else "[rate_limit]"
                lines.append(f"{label} {str(msg)}")
            elif etype in ("reasoning", "thinking"):
                text = event.get("text") or event.get("content") or ""
                if text:
                    lines.append(f"[thinking] {str(text)}")
            elif etype == "result":
                subtype = event.get("subtype") or ""
                status_bits = []
                if subtype:
                    status_bits.append(subtype)
                if "is_error" in event:
                    status_bits.append(f"is_error={event['is_error']}")
                if "num_turns" in event:
                    status_bits.append(f"turns={event['num_turns']}")
                lines.append(f"[result] {' '.join(status_bits)}".rstrip())
            elif etype:
                lines.append(f"[{etype}] {stripped[:500]}")
        return "\n".join(lines)

    def _format_assistant_blocks(self, event: dict) -> list[str]:
        out: list[str] = []
        for block in event.get("message", {}).get("content", []):
            btype = block.get("type")
            if btype == "text":
                text = str(block.get("text", ""))
                out.append(f"[Agy] {text}")
            elif btype == "tool_use":
                out.append(self._format_tool_use(block))
            elif btype == "thinking":
                text = str(block.get("thinking", ""))
                if text:
                    out.append(f"[thinking] {text}")
        return out

    def _format_message_blocks(self, event: dict) -> list[str]:
        role = event.get("role", "")
        content = event.get("content", "")
        out: list[str] = []
        if isinstance(content, str):
            if content:
                out.append(f"[{role}] {content}")
            return out
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text = str(block.get("text", ""))
                    if text:
                        out.append(f"[{role}] {text}")
                elif btype == "tool_use":
                    out.append(self._format_tool_use(block))
                elif btype == "tool_result":
                    inner = block.get("content", "")
                    if isinstance(inner, list):
                        parts = [
                            str(item.get("text", ""))
                            for item in inner
                            if isinstance(item, dict) and item.get("type") == "text"
                        ]
                        text = "\n".join(parts)
                    else:
                        text = str(inner)
                    label = "[ToolResult:err]" if block.get("is_error") else "[ToolResult]"
                    out.append(f"{label} {text}")
        return out

    def _format_user_blocks(self, event: dict) -> list[str]:
        out: list[str] = []
        content = event.get("message", {}).get("content", [])
        if not isinstance(content, list):
            return out
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                inner = block.get("content", "")
                if isinstance(inner, list):
                    parts = []
                    for item in inner:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(str(item.get("text", "")))
                    text = "\n".join(parts)
                else:
                    text = str(inner)
                is_error = block.get("is_error")
                label = "[ToolResult:err]" if is_error else "[ToolResult]"
                out.append(f"{label} {text}")
        return out

    def _format_tool_use(self, block: dict) -> str:
        name = block.get("name", "unknown")
        inputs = block.get("input", {}) or block.get("arguments", {}) or {}
        if isinstance(inputs, str):
            try:
                inputs = json.loads(inputs)
            except (json.JSONDecodeError, ValueError):
                return f"[Tool] {name}({inputs[:200]})"
        if not isinstance(inputs, dict):
            return f"[Tool] {name}"
        highlights: list[str] = []
        for key in self._TOOL_USE_VALUE_KEYS:
            if key in inputs:
                val = str(inputs[key])
                if len(val) > 200:
                    val = val[:200] + "..."
                highlights.append(f"{key}={val}")
        if highlights:
            return f"[Tool] {name}({', '.join(highlights)})"
        return f"[Tool] {name}({list(inputs.keys())})"

    # ── Usage extraction ───────────────────────────────────────

    @staticmethod
    def _extract_usage_from_jsonl(stdout: str) -> CostInfo | None:
        for line in stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "result":
                usage = event.get("usage")
                if usage and isinstance(usage, dict):
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                    return CostInfo(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                    )
            usage = event.get("usage")
            if usage and isinstance(usage, dict):
                input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
                output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
                if input_tokens or output_tokens:
                    return CostInfo(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                    )
        return None
