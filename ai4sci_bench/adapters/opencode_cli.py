"""opencode CLI adapter — runs opencode (sst/opencode) as the evaluated agent.

Facts verified against opencode v1.17.15 (re-verify when opencode changes):

- Non-interactive run: ``opencode run [message..]``; ``--format json``
  streams raw JSON events to stdout. Piped stdin is merged into the prompt,
  so the benchmark prompt is fed via stdin (avoids Windows ~8KB argv limits,
  #27).
- JSON event types on stdout (each ``{"type": <name>, "timestamp": ...,
  "sessionID": ..., <payload>}``):
  - ``text`` / ``reasoning`` — completed assistant text/reasoning parts
    (``part.text``);
  - ``tool_use`` — tool part in final state (``part.tool``, ``part.state``
    with ``status``/``input``/``output``/``error``);
  - ``step_start`` / ``step_finish`` — model step boundaries;
    ``step_finish`` parts carry ``tokens`` and ``cost``;
  - ``error`` — terminal session error (``error.name`` + ``error.data``).
- Permission auto-approval: ``--auto`` replies to permission requests so
  headless long tasks are not blocked.
- Model selection: ``-m provider/model``; reasoning effort via
  ``--variant`` (provider-specific, e.g. ``high`` / ``max`` / ``minimal``).
- Auth: ``opencode auth login`` writes ``~/.local/share/opencode/auth.json``;
  standard provider environment variables (``ANTHROPIC_API_KEY``, ...) are
  honored too.
- Custom endpoints: ``opencode.json`` (or inline via the
  ``OPENCODE_CONFIG_CONTENT`` env var) registers providers via AI SDK
  packages — ``@ai-sdk/openai-compatible`` (Chat Completions),
  ``@ai-sdk/openai`` (Responses), ``@ai-sdk/anthropic`` (Messages).
- Unlike pi, opencode exits non-zero on terminal API errors, but the JSON
  ``error`` events are still parsed for robustness (fake-success detection).
"""

from __future__ import annotations

import json
import logging
import os
import time

from ai4sci_bench.adapters.subprocess_base import (
    SubprocessAgentAdapter,
    collect_output_files,
)
from ai4sci_bench.core.types import AgentOutput, CostInfo, RunStatus, TaskInstance, ToolMode

logger = logging.getLogger(__name__)

# Tools disabled in RESTRICTED mode. opencode's network-capable built-in
# tools are webfetch/websearch; everything else is local coding tooling.
_OPENCODE_WEB_TOOLS = ("webfetch", "websearch")


class OpenCodeCLIAdapter(SubprocessAgentAdapter):
    """Run the opencode CLI as the evaluated agent.

    Three authentication modes:

    1. **Local login** (default): No ``api_key`` / ``api_base`` — uses
       opencode's own auth state (``opencode auth login`` or provider env
       vars already present in the harness environment).
    2. **Official API key only**: Only ``api_key`` — injects the provider's
       standard environment variable (e.g. ``ANTHROPIC_API_KEY``). The
       provider is taken from ``provider`` if given, else from the
       ``provider/`` prefix of ``model``, else ``anthropic``.
    3. **Explicit endpoint** (``api_key`` + ``api_base`` + ``api_protocol``):
       generates an inline opencode config (``OPENCODE_CONFIG_CONTENT``)
       registering a ``bench`` provider backed by the matching AI SDK
       package. No proxy.

       ``api_protocol`` is one of ``openai`` (Chat Completions via
       ``@ai-sdk/openai-compatible``), ``openai_responses`` (Responses API
       via ``@ai-sdk/openai``) or ``anthropic`` (Messages API via
       ``@ai-sdk/anthropic``).

    Effort maps to opencode's ``--variant`` flag (provider-specific
    reasoning effort); unlike other adapters the default is ``None`` (flag
    not passed) because variants are model-specific.
    """

    VALID_EFFORT_LEVELS = ("minimal", "low", "medium", "high", "max")
    VERIFIED_CLI_VERSION = "1.17.15"
    CLI_BINARY = "opencode"

    VALID_API_PROTOCOLS: frozenset[str] = frozenset({
        "openai", "openai_responses", "anthropic",
    })

    _API_PROTOCOL_TO_NPM: dict[str, str] = {
        "openai": "@ai-sdk/openai-compatible",
        "openai_responses": "@ai-sdk/openai",
        "anthropic": "@ai-sdk/anthropic",
    }

    _PROVIDER_ENV_VARS: dict[str, str] = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GEMINI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "groq": "GROQ_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "xai": "XAI_API_KEY",
        "zai": "ZAI_API_KEY",
        "opencode": "OPENCODE_API_KEY",
    }

    def __init__(
        self,
        model: str = "anthropic/claude-opus-4-6",
        timeout_seconds: int = 10800,
        allow_external_tools: bool = False,
        tool_mode: str | ToolMode | None = None,
        effort: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        api_base: str | None = None,
        api_base_env: str | None = None,
        api_protocol: str | None = None,
        provider: str | None = None,
    ):
        super().__init__(
            timeout_seconds=timeout_seconds,
            supported_sandbox_modes=("none", "task", "os", "linux_ns"),
        )
        if effort is not None and effort not in self.VALID_EFFORT_LEVELS:
            raise ValueError(
                f"Invalid effort '{effort}' for opencode_cli. "
                f"Valid levels: {', '.join(self.VALID_EFFORT_LEVELS)}"
            )
        if api_protocol is not None and api_protocol not in self.VALID_API_PROTOCOLS:
            raise ValueError(
                f"Invalid api_protocol {api_protocol!r}. "
                f"Valid values: {sorted(self.VALID_API_PROTOCOLS)}"
            )
        resolved_api_key = api_key if api_key is not None else self._read_env_secret(api_key_env)
        resolved_api_base = api_base if api_base is not None else self._read_env_secret(api_base_env)
        if resolved_api_base is not None and resolved_api_key is None:
            raise ValueError("api_base was given but api_key is missing.")
        if resolved_api_base is None and api_protocol is not None:
            raise ValueError(
                "api_protocol was given but api_base is missing — "
                "api_protocol needs an endpoint to apply to."
            )

        self.model = model
        self.allow_external_tools = allow_external_tools
        self.tool_mode = self._resolve_tool_mode(tool_mode, allow_external_tools)
        self.effort = effort
        self.api_key = resolved_api_key
        self.api_base = resolved_api_base
        self.api_key_env = api_key_env
        self.api_base_env = api_base_env
        self.api_protocol = api_protocol
        self.provider = provider

        self._uses_native_provider = self.api_base is not None and self.api_key is not None
        if not self._uses_native_provider and self.api_key is not None:
            self._resolved_provider = self._resolve_key_only_provider(provider, model)
            if self._resolved_provider not in self._PROVIDER_ENV_VARS:
                raise ValueError(
                    f"Unknown provider {self._resolved_provider!r} for api_key-only "
                    f"mode. Known providers: {sorted(self._PROVIDER_ENV_VARS)}. "
                    f"For other providers use api_base + api_protocol (mode 3) "
                    f"instead."
                )
        else:
            self._resolved_provider = None

        self._os_sandbox: object | None = None
        self._sandbox_image_identity: str | None = None

    @staticmethod
    def _read_env_secret(env_name: str | None) -> str | None:
        if not env_name:
            return None
        value = os.environ.get(env_name)
        if value is None or value == "":
            raise ValueError(f"Environment variable '{env_name}' is not set")
        return value

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

    @classmethod
    def _resolve_key_only_provider(cls, provider: str | None, model: str) -> str:
        if provider:
            return provider
        if "/" in model:
            return model.split("/", 1)[0]
        return "anthropic"

    # ── Native config generation (mode 3 + tool isolation) ────

    def _build_config_content(self) -> str | None:
        """Build the inline opencode config JSON (or None if not needed)."""
        config: dict = {}
        if self._uses_native_provider:
            config["provider"] = {
                "bench": {
                    "npm": self._API_PROTOCOL_TO_NPM[self.api_protocol or "openai"],
                    "name": "ASI-Bench Target",
                    "options": {
                        "baseURL": self.api_base,
                        "apiKey": self.api_key,
                    },
                    "models": {
                        self.model: {"name": self.model},
                    },
                }
            }
        if self.tool_mode == ToolMode.RESTRICTED:
            # Restrict the default `build` agent's toolset: disable the
            # network-capable web tools. SEARCH keeps them; UNRESTRICTED
            # passes no config at all.
            tools = {name: False for name in _OPENCODE_WEB_TOOLS}
            config.setdefault("agent", {}).setdefault("build", {})["tools"] = tools
        if not config:
            return None
        return json.dumps(config)

    # ── Env building ──────────────────────────────────────────

    def _build_api_env(self) -> dict[str, str]:
        """Build env vars for API authentication (modes 2 & 3)."""
        env: dict[str, str] = {}
        if self._uses_native_provider:
            env["OPENCODE_CONFIG_CONTENT"] = self._build_config_content() or "{}"
        elif self.api_key and self._resolved_provider:
            env[self._PROVIDER_ENV_VARS[self._resolved_provider]] = self.api_key
        elif self.tool_mode == ToolMode.RESTRICTED:
            config = self._build_config_content()
            if config:
                env["OPENCODE_CONFIG_CONTENT"] = config
        return env

    # ── Adapter lifecycle ──────────────────────────────────────

    def setup(self, config: dict) -> None:
        super().setup(config)
        if self.sandbox == "os":
            from ai4sci_bench.runner.os_sandbox import OSSandbox
            self._os_sandbox = OSSandbox(self.repo_root)

    def _build_run_env(self, task_instance, task_env) -> dict[str, str] | None:
        base_env = super()._build_run_env(task_instance, task_env)
        api_env = self._build_api_env()
        if not api_env:
            return base_env
        env = dict(base_env) if base_env else dict(os.environ)
        env.update(api_env)
        return env

    # ── Command building ───────────────────────────────────────

    def _model_arg(self) -> str:
        if self._uses_native_provider:
            return f"bench/{self.model}"
        return self.model

    def _base_command(self) -> list[str]:
        oc_bin = "opencode.cmd" if os.name == "nt" else "opencode"
        cmd = [
            oc_bin,
            "run",
            "--format", "json",
            "--auto",
        ]
        if self.model:
            cmd += ["-m", self._model_arg()]
        if self.effort is not None:
            cmd += ["--variant", self.effort]
        # Prompt is fed via stdin (opencode merges piped stdin into the
        # message); no positional message args.
        return cmd

    def _build_command(self, task_instance: TaskInstance, task_env):
        return self._base_command()

    def _get_stdin_input(self, task_instance=None, task_env=None) -> str | None:
        """Feed the prompt via stdin (opencode appends piped stdin)."""
        if task_instance is None:
            return None
        prompt_path = task_instance.workspace_dir / "prompt.md"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return None

    # ── Solve ──────────────────────────────────────────────────

    def solve(self, task_instance: TaskInstance) -> AgentOutput:
        if self.sandbox != "os":
            output = super().solve(task_instance)
            if output.cost is None and output.raw_stdout:
                output.cost = self._extract_usage_from_jsonl(output.raw_stdout)
            if output.raw_stdout:
                terminal_error = self._extract_terminal_error_from_jsonl(output.raw_stdout)
                if terminal_error is not None:
                    output.status = RunStatus.FAILED
                    output.error_message = terminal_error
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
                agent_type="opencode",
                allow_external_tools=self.allow_external_tools,
                extra_env=extra_env,
                stdin_input=(workspace / "prompt.md").read_text(encoding="utf-8"),
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
        terminal_error = self._extract_terminal_error_from_jsonl(raw_stdout or "")
        if terminal_error is not None:
            status = RunStatus.FAILED

        return AgentOutput(
            instance_id=task_instance.instance_id,
            output_dir=workspace,
            code_files=[f for f in produced_files if f.endswith(".py")],
            data_files=[f for f in produced_files if not f.endswith(".py")],
            log=parsed_log,
            execution_time_seconds=elapsed,
            status=status,
            error_message=terminal_error if terminal_error is not None else (None if success else log),
            raw_stdout=raw_stdout,
            raw_stderr=raw_stderr,
            raw_stdout_format="jsonl" if raw_stdout else None,
            cost=self._extract_usage_from_jsonl(raw_stdout) if raw_stdout else None,
        )

    def _build_os_agent_cmd(self, workspace) -> list[str]:
        """Build the opencode command; Docker supplies the prompt on stdin."""
        return self._base_command()

    # ── Log parsing ────────────────────────────────────────────

    _TOOL_INPUT_VALUE_KEYS = (
        "filePath", "file_path", "path", "command", "pattern", "url", "query",
    )

    def _parse_log(self, stdout: str) -> str:
        return self._parse_opencode_log(stdout)

    def _raw_stdout_format(self) -> str:
        return "jsonl"

    def _parse_opencode_log(self, stdout: str) -> str:
        """Parse opencode ``--format json`` events into a readable summary."""
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
            part = event.get("part")
            part = part if isinstance(part, dict) else {}
            if etype == "step_start":
                turn_counter += 1
                lines.append(f"=== Turn {turn_counter} ===")
            elif etype == "text":
                text = str(part.get("text", ""))
                if text:
                    lines.append(f"[opencode] {text}")
            elif etype == "reasoning":
                text = str(part.get("text", ""))
                if text:
                    lines.append(f"[thinking] {text}")
            elif etype == "tool_use":
                lines.append(self._format_tool_part(part))
            elif etype == "step_finish":
                tokens = part.get("tokens") or {}
                cost = part.get("cost")
                bits = []
                if tokens:
                    bits.append(
                        f"tokens(in={tokens.get('input', 0)}, out={tokens.get('output', 0)})"
                    )
                if cost is not None:
                    bits.append(f"cost={cost}")
                if bits:
                    lines.append(f"[step_finish] {' '.join(bits)}")
            elif etype == "error":
                error = event.get("error", {})
                if isinstance(error, dict):
                    data = error.get("data") or {}
                    message = data.get("message") if isinstance(data, dict) else None
                    lines.append(
                        f"[error] {error.get('name', 'UnknownError')}: {message or error}"
                    )
                else:
                    lines.append(f"[error] {error}")
            elif etype:
                lines.append(f"[{etype}] {stripped[:300]}")
        return "\n".join(lines)

    def _format_tool_part(self, part: dict) -> str:
        name = part.get("tool", "unknown")
        state = part.get("state") or {}
        if not isinstance(state, dict):
            return f"[Tool] {name}"
        status = state.get("status", "")
        highlights: list[str] = []
        inputs = state.get("input")
        if isinstance(inputs, dict):
            for key in self._TOOL_INPUT_VALUE_KEYS:
                if key in inputs:
                    val = str(inputs[key])
                    if len(val) > 200:
                        val = val[:200] + "..."
                    highlights.append(f"{key}={val}")
        suffix = f"({', '.join(highlights)})" if highlights else ""
        prefix = "[Tool]"
        if status == "error":
            prefix = "[Tool:err]"
        elif status == "completed":
            prefix = "[Tool]"
        return f"{prefix} {name}{suffix} status={status or 'unknown'}"

    # ── Usage + fake-success extraction ────────────────────────

    @staticmethod
    def _extract_usage_from_jsonl(stdout: str) -> CostInfo | None:
        """Sum per-step token usage across ``step_finish`` events.

        opencode reports tokens per model step (``input``/``output``/
        ``reasoning``/``cache``); reasoning tokens are real generation cost
        and are included in the output total.
        """
        total_input = 0
        total_output = 0
        found = False
        for line in stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "step_finish":
                continue
            part = event.get("part")
            tokens = part.get("tokens") if isinstance(part, dict) else None
            if not isinstance(tokens, dict):
                continue
            found = True
            total_input += int(tokens.get("input", 0) or 0)
            total_output += int(tokens.get("output", 0) or 0)
            total_output += int(tokens.get("reasoning", 0) or 0)
        if not found:
            return None
        return CostInfo(
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_input + total_output,
        )

    @staticmethod
    def _extract_terminal_error_from_jsonl(stdout: str) -> str | None:
        """Return the last terminal ``error`` event, if any.

        opencode already exits non-zero on terminal API errors, but the
        event stream is the authoritative signal (and covers cases where
        the harness inspects raw output after an exit-0 run).
        """
        last_error: str | None = None
        for line in stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "error":
                continue
            error = event.get("error", {})
            if isinstance(error, dict):
                data = error.get("data") or {}
                message = data.get("message") if isinstance(data, dict) else None
                last_error = (
                    f"opencode error {error.get('name', 'UnknownError')}: "
                    f"{message or error}"
                )
            else:
                last_error = f"opencode error: {error}"
        return last_error
