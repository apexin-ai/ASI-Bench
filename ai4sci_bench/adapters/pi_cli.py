"""pi CLI adapter — runs the pi coding agent (earendil-works) as the evaluated agent.

Facts verified against pi v0.84.3 (re-verify when pi's CLI changes):

- Non-interactive print mode: ``pi -p``; JSONL event stream on stdout:
  ``pi --mode json``. Piped stdin is merged into the initial prompt, so the
  benchmark prompt is fed via stdin (avoids Windows ~8KB argv limits, #27).
- Headless trust flags: ``--no-approve`` (ignore project-local files).
- Built-in tools: ``read, bash, edit, write, grep, find, ls`` (plus
  ``powershell`` on Windows).
- Thinking levels: ``off, minimal, low, medium, high, xhigh, max`` — used as
  the effort axis via ``--thinking``.
- Sessions are saved under ``~/.pi/agent/sessions/``; ``--no-session`` runs
  ephemeral (the JSON event stream is our trajectory, no session files needed).
- Config dir override: ``PI_CODING_AGENT_DIR`` (default ``~/.pi/agent``).
- Auth resolution order: CLI ``--api-key`` > ``auth.json`` entry > provider
  environment variable > ``models.json`` provider keys. Credential file:
  ``~/.pi/agent/auth.json``.
- Custom endpoints: ``~/.pi/agent/models.json`` registers providers with
  ``baseUrl`` / ``api`` / ``apiKey`` / ``models``. Supported APIs:
  ``openai-completions``, ``anthropic-messages``, ``openai-responses``,
  ``google-generative-ai``.
- Terminal errors do NOT change the exit code: with a bad API key ``pi -p
  --mode json`` still exits 0 while the assistant message carries
  ``stopReason: "error"`` and ``errorMessage`` — hence fake-success detection.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

from ai4sci_bench.adapters.subprocess_base import (
    SubprocessAgentAdapter,
    collect_output_files,
)
from ai4sci_bench.core.types import AgentOutput, CostInfo, RunStatus, TaskInstance, ToolMode

logger = logging.getLogger(__name__)

# pi built-in coding tools (tool isolation allowlist). There is no built-in
# web search/fetch tool, so RESTRICTED and SEARCH map to the same set; network
# access (if any) would go through bash, which must stay enabled for coding.
PI_CORE_TOOLS = "read,bash,edit,write,grep,find,ls"


class PiCLIAdapter(SubprocessAgentAdapter):
    """Run the pi coding agent as the evaluated agent.

    Three authentication modes:

    1. **Local login** (default): No ``api_key`` / ``api_base`` — uses pi's
       own auth state (``~/.pi/agent/auth.json`` from ``/login`` or standard
       provider environment variables already present in the harness env).
    2. **Official API key only**: Only ``api_key`` — injects the provider's
       standard environment variable (e.g. ``ANTHROPIC_API_KEY``). The
       provider is taken from ``provider`` if given, else from the
       ``provider/`` prefix of ``model``, else ``anthropic``.
    3. **Explicit endpoint** (``api_key`` + ``api_base`` + ``api_protocol``):
       generates a temporary pi config dir containing ``models.json`` that
       registers a ``bench`` provider and points ``PI_CODING_AGENT_DIR`` at
       it. No proxy.

       ``api_protocol`` is one of ``openai`` / ``anthropic`` /
       ``openai_responses`` / ``google`` (mapped to pi's
       ``openai-completions`` / ``anthropic-messages`` /
       ``openai-responses`` / ``google-generative-ai`` API types).
       For the ``anthropic`` protocol a trailing ``/v1`` on ``api_base`` is
       stripped because pi's Anthropic client appends ``/v1/messages``.

    Effort maps to pi's ``--thinking`` levels; ``model`` uses pi's
    ``provider/model`` syntax (mode 3 rewrites it to ``bench/<model>``).
    """

    VALID_EFFORT_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")

    VALID_API_PROTOCOLS: frozenset[str] = frozenset({
        "openai", "anthropic", "openai_responses", "google",
    })

    _API_PROTOCOL_TO_PI_API: dict[str, str] = {
        "openai": "openai-completions",
        "anthropic": "anthropic-messages",
        "openai_responses": "openai-responses",
        "google": "google-generative-ai",
    }

    # Standard provider env vars honored by pi (docs/providers.md).
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
        "kimi-coding": "KIMI_API_KEY",
    }

    _CONTAINER_PI_CONFIG_DIR = "/home/agent/.pi-bench-config"

    def __init__(
        self,
        model: str = "anthropic/claude-opus-4-6",
        timeout_seconds: int = 10800,
        allow_external_tools: bool = False,
        tool_mode: str | ToolMode | None = None,
        effort: str = "medium",
        api_key: str | None = None,
        api_key_env: str | None = None,
        api_base: str | None = None,
        api_base_env: str | None = None,
        api_protocol: str | None = None,
        pi_config_dir: str | None = None,
        provider: str | None = None,
        reasoning: bool = True,
    ):
        super().__init__(
            timeout_seconds=timeout_seconds,
            supported_sandbox_modes=("none", "task", "os", "linux_ns"),
        )
        if effort not in self.VALID_EFFORT_LEVELS:
            raise ValueError(
                f"Invalid effort '{effort}' for pi_cli. "
                f"Valid levels: {', '.join(self.VALID_EFFORT_LEVELS)}"
            )
        if api_protocol is not None and api_protocol not in self.VALID_API_PROTOCOLS:
            raise ValueError(
                f"Invalid api_protocol {api_protocol!r}. "
                f"Valid values: {sorted(self.VALID_API_PROTOCOLS)}"
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
        self.effort = effort
        self.api_key = api_key if api_key is not None else self._read_env_secret(api_key_env)
        self.api_base = api_base if api_base is not None else self._read_env_secret(api_base_env)
        self.api_key_env = api_key_env
        self.api_base_env = api_base_env
        self.api_protocol = api_protocol
        self.pi_config_dir = pi_config_dir
        self.provider = provider
        self.reasoning = reasoning

        self._uses_native_provider = api_base is not None and self.api_key is not None
        if not self._uses_native_provider and self.api_key is not None:
            # Key-only mode: the key must map to a known provider env var.
            self._resolved_provider = self._resolve_key_only_provider(
                provider, model
            )
            if self._resolved_provider not in self._PROVIDER_ENV_VARS:
                raise ValueError(
                    f"Unknown provider {self._resolved_provider!r} for api_key-only "
                    f"mode. Known providers: {sorted(self._PROVIDER_ENV_VARS)}. "
                    f"For other providers use api_base + api_protocol (mode 3) "
                    f"instead."
                )
        else:
            self._resolved_provider = None

        self._temp_pi_config_dir: str | None = None
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

    # ── Native config file (mode 3) ───────────────────────────

    def _normalize_anthropic_base(self, api_base: str) -> str:
        base = api_base.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return base

    def _generate_models_json(self) -> str:
        assert self.api_protocol is not None
        base_url = self.api_base or ""
        if self.api_protocol == "anthropic":
            base_url = self._normalize_anthropic_base(base_url)
        config = {
            "providers": {
                "bench": {
                    "name": "ASI-Bench Target",
                    "baseUrl": base_url,
                    "api": self._API_PROTOCOL_TO_PI_API[self.api_protocol],
                    "apiKey": self.api_key or "",
                    "models": [
                        {
                            "id": self.model,
                            "name": self.model,
                            "reasoning": self.reasoning,
                            "input": ["text"],
                            "contextWindow": 262144,
                            "maxTokens": 32768,
                        }
                    ],
                }
            }
        }
        return json.dumps(config, indent=2)

    def _ensure_pi_config_dir(self) -> str:
        """Return the ``PI_CODING_AGENT_DIR``, generating models.json if needed."""
        if self.pi_config_dir:
            return self.pi_config_dir
        if self._temp_pi_config_dir:
            return self._temp_pi_config_dir

        tmpdir = tempfile.mkdtemp(prefix="pi_config_")
        os.chmod(tmpdir, 0o700)
        config_path = os.path.join(tmpdir, "models.json")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(self._generate_models_json())

        self._temp_pi_config_dir = tmpdir
        logger.info(
            "pi_cli: generated models.json at %s (api=%s, baseUrl=%s)",
            tmpdir,
            self._API_PROTOCOL_TO_PI_API.get(self.api_protocol or ""),
            self.api_base,
        )
        return tmpdir

    # ── Env building ──────────────────────────────────────────

    def _build_api_env(self) -> dict[str, str]:
        """Build env vars for API authentication (modes 2 & 3)."""
        env: dict[str, str] = {}
        if self._uses_native_provider or self.pi_config_dir:
            env["PI_CODING_AGENT_DIR"] = self._ensure_pi_config_dir()
        elif self.api_key and self._resolved_provider:
            env[self._PROVIDER_ENV_VARS[self._resolved_provider]] = self.api_key
        return env

    def _build_os_api_env(self) -> dict[str, str]:
        """Env vars for Docker (os sandbox): config dir uses container path."""
        env: dict[str, str] = {}
        if self._uses_native_provider or self.pi_config_dir:
            self._ensure_pi_config_dir()
            env["PI_CODING_AGENT_DIR"] = self._CONTAINER_PI_CONFIG_DIR
        elif self.api_key and self._resolved_provider:
            env[self._PROVIDER_ENV_VARS[self._resolved_provider]] = self.api_key
        return env

    def _build_os_extra_mounts(self) -> list[str]:
        """Extra Docker mounts: the generated pi config dir (rw; pi may write)."""
        mounts: list[str] = []
        if self._uses_native_provider or self.pi_config_dir:
            host_dir = self._ensure_pi_config_dir()
            mounts += ["-v", f"{host_dir}:{self._CONTAINER_PI_CONFIG_DIR}:rw"]
        return mounts

    # ── Adapter lifecycle ──────────────────────────────────────

    def setup(self, config: dict) -> None:
        super().setup(config)
        if self.sandbox == "os":
            from ai4sci_bench.runner.os_sandbox import OSSandbox
            self._os_sandbox = OSSandbox(self.repo_root)

    def teardown(self) -> None:
        if self._temp_pi_config_dir is not None:
            shutil.rmtree(self._temp_pi_config_dir, ignore_errors=True)
            self._temp_pi_config_dir = None

    def _build_run_env(self, task_instance, task_env) -> dict[str, str] | None:
        base_env = super()._build_run_env(task_instance, task_env)
        api_env = self._build_api_env()
        if not api_env:
            return base_env
        env = dict(base_env) if base_env else dict(os.environ)
        env.update(api_env)
        return env

    # ── Command building ───────────────────────────────────────

    def _apply_tool_isolation(self, cmd: list[str]) -> None:
        if self.tool_mode == ToolMode.UNRESTRICTED:
            return
        # pi has no built-in web tools; RESTRICTED and SEARCH share the same
        # coding-tool allowlist (see PI_CORE_TOOLS note).
        cmd += ["--tools", PI_CORE_TOOLS]

    def _model_arg(self) -> str:
        if self._uses_native_provider or self.pi_config_dir:
            return f"bench/{self.model}"
        return self.model

    def _base_command(self) -> list[str]:
        pi_bin = "pi.cmd" if os.name == "nt" else "pi"
        cmd = [
            pi_bin,
            "-p",
            "--mode", "json",
            "--no-approve",
            "--no-session",
            "--thinking", self.effort,
            "--model", self._model_arg(),
        ]
        self._apply_tool_isolation(cmd)
        # Prompt is fed via stdin (see _get_stdin_input); trailing "--" would
        # make pi treat stdin as message text, so no positional args at all.
        return cmd

    def _build_os_agent_cmd(self, workspace) -> list[str]:
        """Build the pi command for execution inside a Docker container.

        The container has no usable stdin (``docker run`` without ``-i``), so
        the prompt goes in argv behind ``--`` — same trade-off as the claude
        and kimi adapters in os mode.
        """
        prompt = (workspace / "prompt.md").read_text(encoding="utf-8")
        return self._base_command() + ["--", prompt]

    def _build_command(self, task_instance: TaskInstance, task_env):
        return self._base_command()

    def _get_stdin_input(self, task_instance=None, task_env=None) -> str | None:
        """Feed the prompt via stdin (pi merges piped stdin into the prompt).

        Avoids Windows ~8KB argv limits (#27) and shell-quoting issues; the
        prompt never appears in the process argv or the provenance metadata.
        """
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
                agent_type="pi",
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

    # ── Log parsing ────────────────────────────────────────────

    _TOOL_USE_VALUE_KEYS = (
        "path", "filePath", "file_path", "command", "pattern", "url", "query",
    )

    def _parse_log(self, stdout: str) -> str:
        return self._parse_pi_log(stdout)

    def _raw_stdout_format(self) -> str:
        return "jsonl"

    def _parse_pi_log(self, stdout: str) -> str:
        """Parse pi ``--mode json`` events into a human-readable summary."""
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
            if etype == "session":
                lines.append(f"[session] id={event.get('id', '')}")
            elif etype == "agent_start":
                lines.append("[agent_start]")
            elif etype == "turn_start":
                turn_counter += 1
                lines.append(f"=== Turn {turn_counter} ===")
            elif etype == "message_end":
                message = event.get("message", {})
                if isinstance(message, dict) and message.get("role") == "assistant":
                    lines.extend(self._format_assistant_message(message))
            elif etype == "tool_execution_start":
                lines.append(self._format_tool_call(
                    event.get("toolName", "unknown"), event.get("args")
                ))
            elif etype == "tool_execution_end":
                result = event.get("result")
                text = result.get("output", "") if isinstance(result, dict) else str(result or "")
                text = str(text)
                if len(text) > 500:
                    text = text[:500] + "..."
                label = "[ToolResult:err]" if event.get("isError") else "[ToolResult]"
                if text:
                    lines.append(f"{label} {text}")
            elif etype == "turn_end":
                message = event.get("message", {})
                if isinstance(message, dict) and message.get("stopReason") not in (None, "pending"):
                    stop = message.get("stopReason")
                    if stop not in ("stop", "end", "toolUse"):
                        err = message.get("errorMessage") or ""
                        lines.append(f"[turn_end] stopReason={stop} {err}".rstrip())
            elif etype == "agent_end":
                pass  # final message list duplicates message_end content
            elif etype in ("message_update", "message_start"):
                pass  # streaming/duplicate events — message_end is authoritative
            elif etype in ("error",):
                lines.append(f"[error] {event.get('error') or stripped[:500]}")
            elif etype:
                lines.append(f"[{etype}] {stripped[:300]}")
        return "\n".join(lines)

    def _format_assistant_message(self, message: dict) -> list[str]:
        out: list[str] = []
        for block in message.get("content", []) or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = str(block.get("text", ""))
                if text:
                    out.append(f"[pi] {text}")
            elif btype == "thinking":
                text = str(block.get("thinking", ""))
                if text:
                    out.append(f"[thinking] {text}")
            elif btype == "toolCall":
                # Actual execution is surfaced via tool_execution_* events;
                # skip here to avoid double counting.
                pass
        stop = message.get("stopReason")
        if stop == "error":
            out.append(f"[error] {message.get('errorMessage') or 'unknown error'}")
        elif stop == "aborted":
            out.append("[error] agent aborted")
        return out

    def _format_tool_call(self, name: str, args) -> str:
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                return f"[Tool] {name}({args[:200]})"
        if not isinstance(args, dict):
            return f"[Tool] {name}"
        highlights: list[str] = []
        for key in self._TOOL_USE_VALUE_KEYS:
            if key in args:
                val = str(args[key])
                if len(val) > 200:
                    val = val[:200] + "..."
                highlights.append(f"{key}={val}")
        if highlights:
            return f"[Tool] {name}({', '.join(highlights)})"
        return f"[Tool] {name}({list(args.keys())})"

    # ── Usage + fake-success extraction ────────────────────────

    @staticmethod
    def _iter_assistant_messages(stdout: str):
        """Yield final assistant messages from ``message_end`` events."""
        for line in stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "message_end":
                continue
            message = event.get("message", {})
            if isinstance(message, dict) and message.get("role") == "assistant":
                yield message

    @classmethod
    def _extract_usage_from_jsonl(cls, stdout: str) -> CostInfo | None:
        """Sum per-message token usage across all assistant messages.

        pi reports usage per assistant message (``input``/``output``/
        ``cacheRead``/``cacheWrite``); the session total is the sum.
        """
        total_input = 0
        total_output = 0
        found = False
        for message in cls._iter_assistant_messages(stdout):
            usage = message.get("usage")
            if not isinstance(usage, dict):
                continue
            found = True
            total_input += int(usage.get("input", 0) or 0)
            total_output += int(usage.get("output", 0) or 0)
        if not found:
            return None
        return CostInfo(
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_input + total_output,
        )

    @classmethod
    def _extract_terminal_error_from_jsonl(cls, stdout: str) -> str | None:
        """Detect terminal errors encoded in an otherwise exit-0 pi run.

        pi exits 0 even when the API fails; the final assistant message then
        carries ``stopReason: "error"`` (+ ``errorMessage``) or the agent was
        aborted mid-run. Both must map to FAILED, not a scored solution.
        """
        last_error: str | None = None
        for message in cls._iter_assistant_messages(stdout):
            stop = message.get("stopReason")
            if stop == "error":
                last_error = (
                    f"pi agent error: {message.get('errorMessage') or 'unknown error'}"
                )
            elif stop == "aborted":
                last_error = "pi agent run was aborted before completion"
        return last_error
