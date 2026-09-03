"""Claude Code CLI adapter — runs Claude Code as the evaluated agent."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path

from ai4sci_bench.adapters.subprocess_base import (
    SubprocessAgentAdapter,
    collect_output_files,
    safe_run_key,
)
from ai4sci_bench.core.types import AgentOutput, CostInfo, RunStatus, TaskInstance, ToolMode
from ai4sci_bench.runner.os_sandbox import OSSandbox

logger = logging.getLogger(__name__)

CLAUDE_CORE_TOOLS = (
    "Bash,Read,Write,Edit,Glob,Grep,TodoWrite,Agent,"
    "NotebookEdit,ToolSearch,Monitor,TaskOutput,TaskStop,"
    "EnterWorktree,ExitWorktree"
)
CLAUDE_SEARCH_TOOLS = CLAUDE_CORE_TOOLS + ",WebSearch,WebFetch"

# Auth/config files copied into the isolated per-run Claude home for
# host-side (non-OS-sandbox) execution. Mirrors the read-only auth mounts
# used by the OS sandbox (runner/os_sandbox.py CLAUDE_AUTH_PATHS) so host
# and container runs see the same ambient config surface. Everything else
# under the real HOME — session transcripts, ~/.claude.json project
# history, todos, auto-memory files, user-scoped MCP servers, skills — is
# deliberately NOT copied: that is the cross-instance memory surface.
CLAUDE_AUTH_FILENAMES = (".credentials.json", "settings.json")


class ClaudeCodeCLIAdapter(SubprocessAgentAdapter):
    """Run Claude Code CLI as the evaluated agent.

    Claude Code autonomously reads files, writes code, executes it,
    debugs failures, and produces output files.

    Supports three authentication modes:

    1. **Local login** (default): No ``api_key`` / ``api_base`` — uses
       ``~/.claude/`` session (Claude Max / Pro subscription).
    2. **Anthropic API Key**: Only ``api_key`` — injects
       ``ANTHROPIC_API_KEY`` and uses the official Anthropic endpoint.
    3. **Third-party API**: ``api_base`` is set — requires ``api_key``
       and ``api_protocol`` (``openai`` or ``anthropic``).

       * ``api_protocol='anthropic'`` — the target already speaks the
         Anthropic Messages API (``/v1/messages``). Claude points straight
         at it via ``ANTHROPIC_BASE_URL`` with **no** litellm proxy. A
         trailing ``/v1`` on ``api_base`` is stripped because the Claude
         CLI appends its own ``/v1/messages``.
       * ``api_protocol='openai'`` — starts a local litellm proxy that
         translates Anthropic → OpenAI for the target service.

       ``model`` is the bare model name as the target service understands
       it (e.g. ``z-ai/glm-5.2``, ``moonshotai/kimi-k2.7-code``).
    """

    VALID_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        timeout_seconds: int = 10800,
        permission_mode: str | None = None,
        allow_external_tools: bool = False,
        tool_mode: str | ToolMode | None = None,
        effort: str = "medium",
        api_key: str | None = None,
        api_key_env: str | None = None,
        api_base: str | None = None,
        api_base_env: str | None = None,
        api_protocol: str | None = None,
        supports_image_input: bool = False,
    ):
        super().__init__(
            timeout_seconds=timeout_seconds,
            supported_sandbox_modes=("none", "task", "os", "linux_ns"),
        )
        if effort not in self.VALID_EFFORT_LEVELS:
            raise ValueError(
                f"Invalid effort '{effort}' for claude_code_cli. "
                f"Valid levels: {', '.join(self.VALID_EFFORT_LEVELS)}"
            )
        self.model = model
        self.permission_mode = permission_mode
        self.allow_external_tools = allow_external_tools
        self.tool_mode = self._resolve_tool_mode(tool_mode, allow_external_tools)
        self.effort = effort
        self.api_key = api_key if api_key is not None else self._read_env_secret(api_key_env)
        self.api_base = api_base if api_base is not None else self._read_env_secret(api_base_env)
        self.api_key_env = api_key_env
        self.api_base_env = api_base_env
        self.api_protocol = api_protocol
        self.supports_image_input = supports_image_input
        if self.api_base is not None and api_protocol is None:
            raise ValueError(
                "api_protocol is required when api_base is set. "
                "Use 'openai' or 'anthropic' to specify the protocol that api_base speaks."
            )
        if api_protocol is not None and api_protocol not in ("openai", "anthropic"):
            raise ValueError(
                f"Invalid api_protocol '{api_protocol}'. Use 'openai' or 'anthropic'."
            )
        # Anthropic-protocol targets usually speak Messages natively, so we
        # talk to them directly. TokenRouter needs small model-specific
        # compatibility rewrites for some native routes; those use a lightweight
        # passthrough proxy, not litellm.
        self._uses_proxy = self.api_base is not None and api_protocol != "anthropic"
        self._uses_tokenrouter_openai_chat_proxy = self._should_use_tokenrouter_openai_chat_proxy(
            model=model,
            api_base=self.api_base,
        )
        self._uses_anthropic_rewrite_proxy = self._should_use_anthropic_rewrite_proxy(
            model=model,
            api_base=self.api_base,
            api_protocol=api_protocol,
        )
        self._os_sandbox: OSSandbox | None = None
        self._sandbox_image_identity: str | None = None
        self._proxy: object | None = None
        self._proxy_lock = threading.Lock()

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

    @staticmethod
    def _should_use_anthropic_rewrite_proxy(
        model: str,
        api_base: str | None,
        api_protocol: str | None,
    ) -> bool:
        if api_base is None or api_protocol != "anthropic":
            return False
        base = api_base.lower()
        model_l = model.lower()
        return "tokenrouter" in base and (
            "claude" in model_l
        )

    @staticmethod
    def _should_use_tokenrouter_openai_chat_proxy(
        model: str,
        api_base: str | None,
    ) -> bool:
        if api_base is None:
            return False
        base = api_base.lower()
        model_l = model.lower()
        return "tokenrouter" in base and (
            "gemini" in model_l or "deepseek" in model_l
        )

    # ── Proxy lifecycle ───────────────────────────────────────

    def _ensure_proxy(self) -> str:
        """Start the litellm proxy (once) and return its local URL."""
        with self._proxy_lock:
            if self._proxy is not None:
                return self._proxy.local_url  # type: ignore[union-attr]
            if self._uses_tokenrouter_openai_chat_proxy:
                from ai4sci_bench.adapters.api_proxy import TokenRouterOpenAIChatProxy
                self._proxy = TokenRouterOpenAIChatProxy(
                    model=self.model,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    effort=self.effort,
                    supports_image_input=self.supports_image_input,
                )
                return self._proxy.start()  # type: ignore[union-attr]
            from ai4sci_bench.adapters.api_proxy import LiteLLMProxy, resolve_litellm_model
            litellm_model = resolve_litellm_model(self.model, self.api_protocol)
            self._proxy = LiteLLMProxy(
                model=litellm_model,
                api_base=self.api_base,
                api_key=self.api_key,
                supports_image_input=self.supports_image_input,
            )
            return self._proxy.start()  # type: ignore[union-attr]

    def _ensure_anthropic_rewrite_proxy(self) -> str:
        """Start the Anthropic passthrough/rewrite proxy and return its URL."""
        with self._proxy_lock:
            if self._proxy is not None:
                return self._proxy.local_url  # type: ignore[union-attr]
            if self.api_base is None:
                raise RuntimeError("api_base is required for Anthropic rewrite proxy")
            from ai4sci_bench.adapters.api_proxy import AnthropicRewriteProxy
            self._proxy = AnthropicRewriteProxy(
                model=self.model,
                api_base=self.api_base,
                api_key=self.api_key,
                effort=self.effort,
                supports_image_input=self.supports_image_input,
            )
            return self._proxy.start()  # type: ignore[union-attr]

    @staticmethod
    def _normalize_anthropic_base(api_base: str) -> str:
        """Return ``api_base`` fit for ``ANTHROPIC_BASE_URL``.

        The Claude CLI / Anthropic SDK append ``/v1/messages`` to
        ``ANTHROPIC_BASE_URL``. Third-party services usually advertise an
        OpenAI-style base that already ends in ``/v1`` (e.g.
        ``https://api.tokenrouter.com/v1``); left as-is that would resolve to
        ``/v1/v1/messages`` (404). Strip a single trailing ``/v1`` so the
        base is the host root.
        """
        base = api_base.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return base

    def _build_api_env(self) -> dict[str, str]:
        """Build env vars for API authentication (modes 2 & 3)."""
        env: dict[str, str] = {}
        if self._uses_tokenrouter_openai_chat_proxy:
            proxy_url = self._ensure_proxy()
            env["ANTHROPIC_BASE_URL"] = proxy_url
            env["ANTHROPIC_API_KEY"] = "sk-proxy-placeholder"
            logger.info("claude_code_cli: using TokenRouter OpenAI chat proxy at %s", proxy_url)
        elif self._uses_proxy:
            proxy_url = self._ensure_proxy()
            env["ANTHROPIC_BASE_URL"] = proxy_url
            # Claude CLI requires a non-empty key; the proxy ignores it.
            env["ANTHROPIC_API_KEY"] = "sk-proxy-placeholder"
            logger.info("claude_code_cli: using litellm proxy at %s", proxy_url)
        elif self._uses_anthropic_rewrite_proxy:
            proxy_url = self._ensure_anthropic_rewrite_proxy()
            env["ANTHROPIC_BASE_URL"] = proxy_url
            # Claude CLI requires a non-empty key; the proxy forwards api_key.
            env["ANTHROPIC_API_KEY"] = "sk-proxy-placeholder"
            logger.info("claude_code_cli: using anthropic rewrite proxy at %s", proxy_url)
        elif self.api_base is not None:
            # Mode 3, anthropic protocol: target speaks the Messages API
            # natively — point Claude straight at it, no proxy translation.
            base_url = self._normalize_anthropic_base(self.api_base)
            env["ANTHROPIC_BASE_URL"] = base_url
            if self.api_key:
                env["ANTHROPIC_API_KEY"] = self.api_key
            logger.info("claude_code_cli: direct anthropic endpoint at %s", base_url)
        elif self.api_key:
            env["ANTHROPIC_API_KEY"] = self.api_key
        return env

    # ── Adapter lifecycle ──────────────────────────────────────

    def setup(self, config: dict) -> None:
        super().setup(config)
        self.permission_mode = config.get("permission_mode", self.permission_mode)
        if self.sandbox == "os":
            self._os_sandbox = OSSandbox(self.repo_root)

    def teardown(self) -> None:
        if self._proxy is not None:
            self._proxy.stop()  # type: ignore[union-attr]
            self._proxy = None

    # ── Env override ───────────────────────────────────────────

    def _build_run_env(self, task_instance, task_env) -> dict[str, str] | None:
        base_env = super()._build_run_env(task_instance, task_env)
        api_env = self._build_api_env()
        env = dict(base_env) if base_env else dict(os.environ)
        if api_env:
            env.update(api_env)
        if self.tool_mode != ToolMode.UNRESTRICTED:
            isolated_home = self._prepare_isolated_claude_home(task_instance)
            env["HOME"] = str(isolated_home)
            env["USERPROFILE"] = str(isolated_home)
            # Overwrite any ambient CLAUDE_CONFIG_DIR so the child cannot
            # escape the isolated home via a pre-set environment variable.
            env["CLAUDE_CONFIG_DIR"] = str(isolated_home / ".claude")
        return env

    def _prepare_isolated_claude_home(self, task_instance) -> Path:
        """Create a minimal per-run Claude home that excludes ambient state.

        Host-side runs (sandbox none/task/linux_ns) must not see the real
        ``HOME``: Claude Code writes session transcripts, project history,
        todos and auto-memory under it, and reads user-level config
        (``~/.claude.json`` MCP servers, hooks, skills) from it. Running
        instances sequentially with the ambient HOME would let one
        instance's state — or the user's personal config — influence the
        next, contaminating benchmark results.

        Only the credential/config files mirrored by the OS sandbox auth
        mounts (``CLAUDE_AUTH_FILENAMES``) are copied in; everything else
        starts empty for this instance + prompt level run.
        """
        home = (
            self.repo_root
            / ".ai4sci-bench"
            / "claude_home"
            / safe_run_key(task_instance.run_key)
        )
        claude_dir = home / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(home, 0o700)
            os.chmod(claude_dir, 0o700)
        except OSError:
            pass

        source_claude_dir = self._host_claude_dir()
        for filename in CLAUDE_AUTH_FILENAMES:
            src = source_claude_dir / filename
            if not src.is_file():
                continue
            dst = claude_dir / filename
            shutil.copy2(src, dst)
            try:
                os.chmod(dst, 0o600)
            except OSError:
                pass

        logger.info(
            "claude_code_cli: using isolated per-run HOME at %s", home,
        )
        return home

    @staticmethod
    def _host_claude_dir() -> Path:
        """Return the real host Claude config dir (honors CLAUDE_CONFIG_DIR)."""
        config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
        if config_dir:
            return Path(config_dir)
        return Path.home() / ".claude"

    # ── Solve ──────────────────────────────────────────────────

    def solve(self, task_instance: TaskInstance) -> AgentOutput:
        """Execute agent — delegates to OSSandbox when sandbox == 'os'."""
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
        if self._uses_proxy or self._uses_anthropic_rewrite_proxy or self._uses_tokenrouter_openai_chat_proxy:
            os.environ.setdefault("AI4SCI_DOCKER_NETWORK", "host")

        t0 = time.time()
        assert self._os_sandbox is not None
        success, log, raw_stdout, raw_stderr, image_identity = (
            self._os_sandbox.run_agent(
                task_metadata=task_instance.metadata,
                agent_cmd=cmd,
                workspace=workspace,
                timeout=eff_timeout,
                agent_type="claude_code",
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

    def _apply_tool_isolation(self, cmd: list[str]) -> None:
        """Append tool-isolation flags based on the resolved ToolMode."""
        if self.tool_mode == ToolMode.UNRESTRICTED:
            return
        tools = CLAUDE_SEARCH_TOOLS if self.tool_mode == ToolMode.SEARCH else CLAUDE_CORE_TOOLS
        cmd += [
            "--tools", tools,
            "--strict-mcp-config",
            "--disable-slash-commands",
        ]

    def _build_os_agent_cmd(self, workspace) -> list[str]:
        """Build the claude CLI command for execution inside a Docker container.

        Forces bypassPermissions so the agent doesn't block on non-interactive
        write-file approval inside the container.
        """
        prompt = (workspace / "prompt.md").read_text(encoding="utf-8")
        cmd = [
            "claude",
            "--model", self.model,
            "--effort", self.effort,
            "--output-format", "stream-json",
            "--print",
            "--verbose",
            "--permission-mode", "bypassPermissions",
        ]
        self._apply_tool_isolation(cmd)
        cmd += ["--", prompt]
        return cmd

    def _build_command(self, task_instance: TaskInstance, task_env):
        workspace = task_instance.workspace_dir
        prompt = (workspace / "prompt.md").read_text(encoding="utf-8")

        claude_bin = "claude.cmd" if os.name == "nt" else "claude"
        cmd = [
            claude_bin,
            "--model", self.model,
            "--effort", self.effort,
            "--output-format", "stream-json",
            "--print",
            "--verbose",
        ]
        if self.permission_mode:
            cmd += ["--permission-mode", self.permission_mode]
        self._apply_tool_isolation(cmd)
        # Pass prompt via stdin rather than argv so Windows cmd.exe's
        # ~8 KB command-line limit (surfaced as "参数太长" / "The command line
        # is too long") cannot kill launches of long B1 prompts (fixes #27).
        # `claude --print` reads from stdin when no trailing prompt arg is given.
        return cmd

    _PROXY_AGENT_PREFIX = (
        "You are an autonomous coding agent. You MUST complete the task below by "
        "writing Python code, executing it, and producing the required output files. "
        "Do NOT stop after just reading or exploring data. Follow these steps:\n"
        "1. Read the task requirements and input data\n"
        "2. Write a complete Python solution script\n"
        "3. Execute the script using the Bash tool\n"
        "4. Verify that all required output files were created\n"
        "5. If any step fails, debug and retry until the output files exist\n\n"
        "TASK:\n"
    )

    def _get_stdin_input(self, task_instance=None, task_env=None) -> str | None:
        """Return prompt text to pipe via stdin (fixes #27 Windows cmd length).

        Reads directly from workspace (thread-safe); no shared mutable state.
        For third-party models via proxy, prepends agentic instructions so the
        model understands it must write and execute code, not just explore.
        """
        if task_instance is None:
            return None
        prompt_path = task_instance.workspace_dir / "prompt.md"
        if prompt_path.exists():
            prompt = prompt_path.read_text(encoding="utf-8")
            if self._uses_proxy or self._uses_tokenrouter_openai_chat_proxy or (
                self._uses_anthropic_rewrite_proxy
                and "claude" not in self.model.lower()
            ):
                prompt = self._PROXY_AGENT_PREFIX + prompt
            return prompt
        return None

    @staticmethod
    def _extract_terminal_error_from_jsonl(stdout: str) -> str | None:
        """Return a terminal Claude Code error encoded in an otherwise-0 exit.

        Claude Code can exit with status 0 while the JSONL ``result`` event
        reports an API error, or while the final result is the harness sentinel
        ``[Tool use interrupted]``. Both cases mean the agent trajectory did not
        complete cleanly and should not be treated as a normal scored solution.
        """
        terminal: dict | None = None
        for line in stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "result":
                terminal = event
        if terminal is None:
            return None

        result = str(terminal.get("result") or "")
        api_status = terminal.get("api_error_status")
        if terminal.get("is_error") or api_status:
            if api_status:
                return f"Claude Code API error {api_status}: {result}".strip()
            return f"Claude Code reported is_error=true: {result}".strip()
        if result.strip() == "[Tool use interrupted]":
            return "Claude Code ended with interrupted tool use"
        if ClaudeCodeCLIAdapter._looks_like_terminal_pseudo_tool_call(result):
            return "Claude Code ended with a plain-text pseudo tool call"
        return None

    @staticmethod
    def _looks_like_terminal_pseudo_tool_call(result: str) -> bool:
        stripped = result.strip()
        if not stripped:
            return False
        pseudo_markers = (
            "Input JSON:",
            "Tool name:",
            "Tool id:",
            "[Tool call ",
        )
        if "tool" not in stripped.lower():
            return False
        if "use the provided tool-calling interface" in stripped:
            return True
        return any(marker in stripped for marker in pseudo_markers)

    def _parse_log(self, stdout: str) -> str:
        return self._parse_claude_log(stdout)

    def _raw_stdout_format(self) -> str:
        return "jsonl"

    # Keys whose values are worth surfacing as tool_use arguments. Kept
    # small on purpose — the raw sidecar has the full JSON.
    _TOOL_USE_VALUE_KEYS = (
        "file_path",
        "path",
        "command",
        "pattern",
        "url",
        "notebook_path",
    )

    def _parse_claude_log(self, stdout: str) -> str:
        """Parse stream-json output into a human-readable summary log.

        The raw stdout is persisted separately via ``raw_stdout``; this
        summary is optimized for quick post-mortem from the result JSON.
        It captures assistant text, tool_use (with key arg values),
        tool_result, system, errors and rate_limit events so debugging
        does not require cracking open the raw sidecar for every issue.
        """
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
            elif etype == "user":
                lines.extend(self._format_user_blocks(event))
            elif etype == "system":
                subtype = event.get("subtype") or ""
                msg = event.get("message") or event.get("text") or ""
                label = f"[system:{subtype}]" if subtype else "[system]"
                if msg:
                    lines.append(f"{label} {str(msg)}")
                else:
                    lines.append(label)
            elif etype in ("error", "rate_limit_event", "rate_limit"):
                msg = event.get("message") or event.get("error") or event
                label = "[error]" if etype == "error" else "[rate_limit]"
                lines.append(f"{label} {str(msg)}")
            elif etype == "result":
                # The final turn summary; keep short status fields.
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
                lines.append(f"[unknown:{etype}] {stripped[:500]}")
        return "\n".join(lines)

    def _format_assistant_blocks(self, event: dict) -> list[str]:
        out: list[str] = []
        for block in event.get("message", {}).get("content", []):
            btype = block.get("type")
            if btype == "text":
                text = str(block.get("text", ""))
                out.append(f"[Claude] {text}")
            elif btype == "tool_use":
                out.append(self._format_tool_use(block))
            elif btype == "thinking":
                text = str(block.get("thinking", ""))
                if text:
                    out.append(f"[thinking] {text}")
        return out

    def _format_user_blocks(self, event: dict) -> list[str]:
        """Surface tool_result events (which stream back as user messages)."""
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
        inputs = block.get("input", {}) or {}
        if not isinstance(inputs, dict):
            return f"[Tool] {name}"
        # Prefer known keys whose values identify *what* was touched.
        highlights: list[str] = []
        for key in self._TOOL_USE_VALUE_KEYS:
            if key in inputs:
                val = str(inputs[key])
                if len(val) > 200:
                    val = val[:200] + "..."
                highlights.append(f"{key}={val}")
        if highlights:
            return f"[Tool] {name}({', '.join(highlights)})"
        # Fall back to just the argument names.
        return f"[Tool] {name}({list(inputs.keys())})"

    @staticmethod
    def _extract_usage_from_jsonl(stdout: str) -> CostInfo | None:
        """Extract token usage from the result event in Claude Code JSONL."""
        for line in stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "result":
                continue
            usage = event.get("usage")
            if not usage or not isinstance(usage, dict):
                continue
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            return CostInfo(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )
        return None
