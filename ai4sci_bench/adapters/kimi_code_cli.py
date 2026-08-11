"""Kimi Code CLI adapter — runs Moonshot AI's Kimi Code as the evaluated agent."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
import warnings
from pathlib import Path

from ai4sci_bench.adapters.subprocess_base import (
    SubprocessAgentAdapter,
    collect_output_files,
)
from ai4sci_bench.core.types import AgentOutput, CostInfo, RunStatus, TaskInstance, ToolMode

logger = logging.getLogger(__name__)


class KimiCodeCLIAdapter(SubprocessAgentAdapter):
    """Run Kimi Code CLI as the evaluated agent.

    Kimi Code CLI autonomously reads files, writes code, executes it,
    debugs failures, and produces output files — similar to Claude Code.

    Three authentication modes:

    1. **Local login** (default): No ``api_key`` / ``api_base`` — uses
       Kimi CLI's own local auth (``kimi /login``).
    2. **Moonshot API key only**: Only ``api_key`` — injects
       ``MOONSHOT_API_KEY``; Kimi CLI hits its built-in default endpoint.
    3. **Explicit endpoint** (``api_key`` + ``api_base`` [+ optional
       ``api_protocol``]): generates a temporary Kimi Code config dir
       with ``config.toml`` and points Kimi CLI at it via
       ``KIMI_CODE_HOME``. No proxy.

       * If ``api_base`` matches one of Moonshot's official endpoints in
         :data:`OFFICIAL_KIMI_ENDPOINTS`, the wire protocol is resolved
         automatically and ``api_protocol`` may be omitted.
       * For any other (third-party) ``api_base``, ``api_protocol`` is
         **required** — one of ``openai`` / ``anthropic`` (or a Kimi CLI
         native type, see :data:`VALID_API_PROTOCOLS`).

    The legacy ``provider`` parameter is a deprecated alias of
    ``api_protocol`` and will be removed in a future release.
    """

    # Official Moonshot / Kimi endpoints. Normalized: no trailing slash.
    # Value is the Kimi CLI config.toml ``type`` (i.e. which client the CLI
    # instantiates). Native ``kimi`` type is preferred over ``openai`` on
    # Moonshot's own endpoints for Moonshot-specific optimizations.
    OFFICIAL_KIMI_ENDPOINTS: dict[str, str] = {
        "https://api.moonshot.cn/v1":     "kimi",       # regular pay-as-you-go, CN
        "https://api.moonshot.ai/v1":     "kimi",       # regular pay-as-you-go, global
        "https://api.kimi.com/coding/v1": "kimi",       # Coding Plan, OpenAI-shape
        "https://api.kimi.com/coding":    "anthropic",  # Coding Plan, Anthropic-shape
    }

    # Values accepted by Kimi CLI's config.toml ``type`` field.
    VALID_API_PROTOCOLS: frozenset[str] = frozenset({
        "kimi", "openai", "openai_responses", "anthropic",
        "google-genai", "vertexai",
    })

    def __init__(
        self,
        model: str = "kimi-k2.7",
        timeout_seconds: int = 10800,
        allow_external_tools: bool = False,
        tool_mode: str | ToolMode | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_protocol: str | None = None,
        kimi_home: str | None = None,
        provider: str | None = None,  # deprecated alias of api_protocol
    ):
        super().__init__(
            timeout_seconds=timeout_seconds,
            supported_sandbox_modes=("none", "task", "os", "linux_ns"),
        )
        if provider is not None:
            warnings.warn(
                "KimiCodeCLIAdapter: 'provider' is deprecated; use 'api_protocol' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if api_protocol is None:
                api_protocol = provider
            elif api_protocol != provider:
                raise ValueError(
                    f"Both 'api_protocol' ({api_protocol!r}) and legacy 'provider' "
                    f"({provider!r}) were given and disagree. Use only 'api_protocol'."
                )

        if api_protocol is not None and api_protocol not in self.VALID_API_PROTOCOLS:
            raise ValueError(
                f"Invalid api_protocol {api_protocol!r}. "
                f"Valid values: {sorted(self.VALID_API_PROTOCOLS)}"
            )

        if api_base is None and api_key is None and api_protocol is not None:
            raise ValueError(
                "api_protocol was given but neither api_base nor api_key. "
                "api_protocol is only meaningful together with an explicit endpoint."
            )
        if api_base is not None and api_key is None:
            raise ValueError("api_base was given but api_key is missing.")
        if api_base is None and api_protocol is not None:
            raise ValueError(
                "api_protocol was given but api_base is missing — "
                "api_protocol needs an endpoint to apply to."
            )

        self.model = model
        self.allow_external_tools = allow_external_tools
        self.tool_mode = self._resolve_tool_mode(tool_mode, allow_external_tools)
        self.api_key = api_key
        self.api_base = api_base
        self.api_protocol = api_protocol
        self.kimi_home = kimi_home

        if api_base is not None and api_key is not None:
            self._resolved_base, self._resolved_type = self._resolve_endpoint(
                api_base, api_protocol
            )
            self._uses_native_provider = True
        else:
            self._resolved_base = None
            self._resolved_type = None
            self._uses_native_provider = False

        self._temp_kimi_home: str | None = None
        self._os_sandbox: object | None = None
        self._sandbox_image_identity: str | None = None

    @property
    def provider(self) -> str | None:
        """Backwards-compat: legacy ``.provider`` maps to ``.api_protocol``."""
        return self.api_protocol

    @classmethod
    def _resolve_endpoint(
        cls,
        api_base: str,
        api_protocol: str | None,
    ) -> tuple[str, str]:
        """Resolve (base_url, cli_type) from user input.

        * If ``api_protocol`` is given, it wins. For ``anthropic`` protocol,
          a trailing ``/v1`` on ``api_base`` is stripped (Kimi CLI's
          Anthropic client appends ``/v1/messages`` itself).
        * Otherwise, the URL must be present in
          :data:`OFFICIAL_KIMI_ENDPOINTS`; the corresponding type is used.
        * Third-party endpoints without ``api_protocol`` raise ``ValueError``.
        """
        normalized = api_base.rstrip("/")

        if api_protocol is not None:
            if api_protocol == "anthropic" and normalized.endswith("/v1"):
                normalized = normalized[: -len("/v1")]
            return normalized, api_protocol

        if normalized in cls.OFFICIAL_KIMI_ENDPOINTS:
            return normalized, cls.OFFICIAL_KIMI_ENDPOINTS[normalized]

        known = sorted(cls.OFFICIAL_KIMI_ENDPOINTS.keys())
        raise ValueError(
            f"api_base {api_base!r} is not a recognized official Moonshot/Kimi "
            f"endpoint. For third-party endpoints, api_protocol is required "
            f"(one of: {sorted(cls.VALID_API_PROTOCOLS)}). "
            f"Recognized official endpoints: {known}."
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

    # ── Native config file ────────────────────────────────────

    def _ensure_kimi_home(self) -> str:
        """Return the ``KIMI_CODE_HOME`` dir, generating config.toml if needed."""
        if self.kimi_home:
            # User-provided persistent dir: use as-is, don't overwrite their config.
            return self.kimi_home
        if self._temp_kimi_home:
            return self._temp_kimi_home

        tmpdir = tempfile.mkdtemp(prefix="kimi_home_")
        os.chmod(tmpdir, 0o700)

        config_content = self._generate_kimi_config()
        config_path = os.path.join(tmpdir, "config.toml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)

        self._temp_kimi_home = tmpdir
        logger.info(
            "kimi_code_cli: generated config at %s (type=%s, base_url=%s)",
            tmpdir, self._resolved_type, self._resolved_base,
        )
        return tmpdir

    def _generate_kimi_config(self) -> str:
        cli_type = self._resolved_type or "openai"
        base_url = self._resolved_base or ""
        token = self.api_key or ""

        return (
            f'[providers.bench]\n'
            f'type = "{cli_type}"\n'
            f'base_url = "{base_url}"\n'
            f'api_key = "{token}"\n'
            f'\n'
            f'[models.bench-model]\n'
            f'provider = "bench"\n'
            f'model = "{self.model}"\n'
            f'max_context_size = 262144\n'
            f'\n'
            f'default_model = "bench-model"\n'
        )

    # ── Env building ──────────────────────────────────────────

    _CONTAINER_KIMI_HOME = "/home/agent/.kimi-code"

    def _build_api_env(self) -> dict[str, str]:
        """Build env vars for host (non-os) execution."""
        env: dict[str, str] = {}
        if self._uses_native_provider or self.kimi_home:
            kimi_home = self._ensure_kimi_home()
            env["KIMI_CODE_HOME"] = kimi_home
            logger.info("kimi_code_cli: using KIMI_CODE_HOME=%s", kimi_home)
        elif self.api_key:
            env["MOONSHOT_API_KEY"] = self.api_key
        return env

    def _build_os_api_env(self) -> dict[str, str]:
        """Build env vars for Docker (os sandbox) execution.

        KIMI_CODE_HOME must point to the container-internal path, not the
        host temp dir. The actual directory is volume-mounted separately
        via _build_os_extra_mounts().
        """
        env: dict[str, str] = {}
        if self._uses_native_provider or self.kimi_home:
            self._ensure_kimi_home()
            env["KIMI_CODE_HOME"] = self._CONTAINER_KIMI_HOME
        elif self.api_key:
            env["MOONSHOT_API_KEY"] = self.api_key
        return env

    def _build_os_extra_mounts(self) -> list[str]:
        """Build extra Docker volume mounts for os sandbox.

        Kimi CLI needs KIMI_CODE_HOME to be writable (it creates sessions/
        and logs/ on startup). We mount the host config dir as rw so that
        config.toml is visible and subdirectories can be created.
        """
        mounts: list[str] = []
        if self._uses_native_provider or self.kimi_home:
            host_kimi_home = self._ensure_kimi_home()
            mounts += ["-v", f"{host_kimi_home}:{self._CONTAINER_KIMI_HOME}:rw"]
        return mounts

    # ── Adapter lifecycle ──────────────────────────────────────

    def setup(self, config: dict) -> None:
        super().setup(config)
        if self.sandbox == "os":
            from ai4sci_bench.runner.os_sandbox import OSSandbox
            self._os_sandbox = OSSandbox(self.repo_root)

    def teardown(self) -> None:
        if self._temp_kimi_home is not None:
            shutil.rmtree(self._temp_kimi_home, ignore_errors=True)
            self._temp_kimi_home = None

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
        extra_env: dict[str, str] | None = self._build_os_api_env() or None
        extra_mounts: list[str] = self._build_os_extra_mounts()

        t0 = time.time()
        assert self._os_sandbox is not None
        success, log, raw_stdout, raw_stderr, image_identity = (
            self._os_sandbox.run_agent(
                task_metadata=task_instance.metadata,
                agent_cmd=cmd,
                workspace=workspace,
                timeout=eff_timeout,
                agent_type="kimi_code",
                allow_external_tools=self.allow_external_tools,
                extra_env=extra_env,
                extra_mounts=extra_mounts,
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
        """Build kimi command for Docker execution."""
        prompt = (workspace / "prompt.md").read_text(encoding="utf-8")
        model_arg = "bench-model" if (self._uses_native_provider or self.kimi_home) else self.model
        return [
            "kimi",
            "-m", model_arg,
            "--output-format", "stream-json",
            "-p", prompt,
        ]

    def _build_command(self, task_instance: TaskInstance, task_env):
        workspace = task_instance.workspace_dir
        prompt = (workspace / "prompt.md").read_text(encoding="utf-8")
        model_arg = "bench-model" if (self._uses_native_provider or self.kimi_home) else self.model
        cmd = [
            "kimi",
            "-m", model_arg,
            "--output-format", "stream-json",
            "-p", prompt,
        ]
        return cmd

    def _parse_log(self, stdout: str) -> str:
        return self._parse_kimi_log(stdout)

    def _raw_stdout_format(self) -> str:
        return "jsonl"

    # ── Log parsing ────────────────────────────────────────────

    _TOOL_USE_VALUE_KEYS = (
        "file_path", "path", "command", "pattern", "url",
    )

    def _parse_kimi_log(self, stdout: str) -> str:
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
            # Codex style events
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
                out.append(f"[Kimi] {text}")
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
            # Check result event (Claude style)
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
            # Check any event with usage (Codex style)
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
