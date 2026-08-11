"""MiMo Code CLI adapter — runs Xiaomi's MiMo Code as the evaluated agent."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

from ai4sci_bench.adapters.subprocess_base import (
    SubprocessAgentAdapter,
    collect_output_files,
)
from ai4sci_bench.core.types import AgentOutput, CostInfo, RunStatus, TaskInstance, ToolMode

logger = logging.getLogger(__name__)

MIMO_PROXY_DUMMY_MODEL = "openai/gpt-5.5"


class MiMoCodeCLIAdapter(SubprocessAgentAdapter):
    """Run MiMo Code CLI (Xiaomi) as the evaluated agent.

    Supports two authentication modes:

    1. **Local login** (default): No ``api_key`` / ``api_base`` — uses
       MiMo Code's built-in auth.
    2. **Official endpoint** (``api_key`` + ``api_base``): starts a
       local proxy that translates the Responses API to the upstream's
       wire protocol. Only official MiMo endpoints are supported;
       the protocol is auto-detected from the URL.

    Supported endpoints::

        https://api.xiaomimimo.com/v1                     (OpenAI protocol)
        https://token-plan-cn.xiaomimimo.com/v1           (OpenAI protocol, Coding Plan)
        https://token-plan-cn.xiaomimimo.com/anthropic    (Anthropic protocol)
    """

    OFFICIAL_MIMO_ENDPOINTS: dict[str, str] = {
        "https://token-plan-cn.xiaomimimo.com/anthropic": "anthropic",
        "https://token-plan-cn.xiaomimimo.com/v1":        "openai",
        "https://api.xiaomimimo.com/v1":                  "openai",
    }

    def __init__(
        self,
        model: str = "xiaomi/mimo-v2.5-pro",
        timeout_seconds: int = 10800,
        allow_external_tools: bool = False,
        tool_mode: str | ToolMode | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        supports_image_input: bool = False,
        **kwargs,
    ):
        super().__init__(
            timeout_seconds=timeout_seconds,
            supported_sandbox_modes=("none", "task", "os", "linux_ns"),
        )
        if api_base is not None and api_key is None:
            raise ValueError("api_base was given but api_key is missing.")

        self.model = model
        self.allow_external_tools = allow_external_tools
        self.tool_mode = self._resolve_tool_mode(tool_mode, allow_external_tools)
        self.api_key = api_key
        self.api_base = api_base
        self.supports_image_input = supports_image_input

        if api_base is not None and api_key is not None:
            self._resolved_base, self._resolved_protocol = self._resolve_endpoint(api_base)
            self._uses_proxy = True
        else:
            self._resolved_base = None
            self._resolved_protocol = None
            self._uses_proxy = False

        self._proxy: object | None = None
        self._proxy_lock = threading.Lock()
        self._os_sandbox: object | None = None
        self._sandbox_image_identity: str | None = None

    @classmethod
    def _resolve_endpoint(cls, api_base: str) -> tuple[str, str]:
        """Resolve (base_url, protocol) by matching against official endpoints."""
        normalized = api_base.rstrip("/")

        if normalized in cls.OFFICIAL_MIMO_ENDPOINTS:
            return normalized, cls.OFFICIAL_MIMO_ENDPOINTS[normalized]

        known = "\n  ".join(sorted(cls.OFFICIAL_MIMO_ENDPOINTS.keys()))
        raise ValueError(
            f"api_base {api_base!r} is not a supported MiMo endpoint. "
            f"Supported endpoints:\n  {known}"
        )

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
        """Start the litellm proxy (once) and return its local URL."""
        with self._proxy_lock:
            if self._proxy is not None:
                return self._proxy.local_url  # type: ignore[union-attr]
            from ai4sci_bench.adapters.api_proxy import (
                LiteLLMOpenAIProxy,
                resolve_litellm_model,
            )
            # Prefix the model with the resolved wire protocol so
            # litellm.responses() routes to the right backend.
            litellm_model = resolve_litellm_model(self.model, self._resolved_protocol)
            # OpenAI endpoints speak Responses API natively → passthrough.
            # Anthropic endpoints need translation via litellm.responses().
            needs_translation = self._resolved_protocol != "openai"
            self._proxy = LiteLLMOpenAIProxy(
                model=litellm_model,
                api_base=self._resolved_base,
                api_key=self.api_key,
                translate_responses=needs_translation,
                supports_image_input=self.supports_image_input,
            )
            return self._proxy.start()  # type: ignore[union-attr]

    def _build_api_env(self) -> dict[str, str]:
        """Build env vars for API authentication."""
        env: dict[str, str] = {}
        if self._uses_proxy:
            proxy_url = self._ensure_proxy()
            env["OPENAI_API_KEY"] = "sk-proxy-placeholder"
            env["OPENAI_BASE_URL"] = proxy_url
            logger.info("mimo_code_cli: using litellm proxy at %s", proxy_url)
        elif self.api_key:
            env["OPENAI_API_KEY"] = self.api_key
            if self.api_base:
                env["OPENAI_BASE_URL"] = self.api_base
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
        if self._uses_proxy:
            os.environ.setdefault("AI4SCI_DOCKER_NETWORK", "host")

        t0 = time.time()
        assert self._os_sandbox is not None
        success, log, raw_stdout, raw_stderr, image_identity = (
            self._os_sandbox.run_agent(
                task_metadata=task_instance.metadata,
                agent_cmd=cmd,
                workspace=workspace,
                timeout=eff_timeout,
                agent_type="mimo_code",
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

    def _build_os_agent_cmd(self, workspace) -> list[str]:
        """Build mimo command for Docker execution."""
        prompt = (workspace / "prompt.md").read_text(encoding="utf-8")
        cli_model = MIMO_PROXY_DUMMY_MODEL if self._uses_proxy else self.model
        return [
            "mimo", "run",
            "--format", "json",
            "--dangerously-skip-permissions",
            "-m", cli_model,
            "--", prompt,
        ]

    def _build_command(self, task_instance: TaskInstance, task_env):
        workspace = task_instance.workspace_dir
        prompt = (workspace / "prompt.md").read_text(encoding="utf-8")
        cli_model = MIMO_PROXY_DUMMY_MODEL if self._uses_proxy else self.model
        cmd = [
            "mimo",
            "run",
            "--format", "json",
            "--dangerously-skip-permissions",
            "-m", cli_model,
            "--", prompt,
        ]
        return cmd

    def _parse_log(self, stdout: str) -> str:
        return self._parse_mimo_log(stdout)

    def _raw_stdout_format(self) -> str:
        return "jsonl"

    # ── Log parsing ────────────────────────────────────────────

    _TOOL_USE_VALUE_KEYS = (
        "file_path", "path", "command", "pattern", "url",
    )

    def _parse_mimo_log(self, stdout: str) -> str:
        lines: list[str] = []
        turn_counter = 0
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

            if etype == "assistant":
                turn_counter += 1
                lines.append(f"=== Turn {turn_counter} ===")
                lines.extend(self._format_assistant_blocks(event))
            elif etype == "message" and event.get("role") == "assistant":
                turn_counter += 1
                lines.append(f"=== Turn {turn_counter} ===")
                lines.extend(self._format_message_blocks(event))
            elif etype == "message":
                lines.extend(self._format_message_blocks(event))
            elif etype == "user":
                lines.extend(self._format_user_blocks(event))
            elif etype == "system":
                msg = event.get("message") or event.get("text") or ""
                if msg:
                    lines.append(f"[system] {str(msg)}")
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
                out.append(f"[MiMo] {str(block.get('text', ''))}")
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
