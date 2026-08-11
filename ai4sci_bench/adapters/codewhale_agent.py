"""CodeWhale adapter — runs CodeWhale (Rust coding agent) via its app-server HTTP API.

CodeWhale is a terminal-native coding agent that speaks the OpenAI Chat
Completions protocol. It provides a full agentic loop with tool calling
(shell, file read/write, git, web search, sub-agents, etc.).

This adapter launches ``codewhale app-server`` as a background process,
communicates via its HTTP API (``POST /prompt`` for single-shot agentic
execution), and collects results.

For non-OpenAI-compatible backends (Anthropic, Gemini, etc.), a local
``LiteLLMOpenAIProxy`` is started to translate the OpenAI Chat Completions
protocol to the target API.

Requires the ``codewhale`` binary to be installed (via npm, cargo, or
direct download).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from ai4sci_bench.adapters.subprocess_base import collect_output_files
from ai4sci_bench.core.agent_interface import AgentAdapter
from ai4sci_bench.core.types import AgentOutput, CostInfo, RunStatus, TaskInstance

logger = logging.getLogger(__name__)


class CodeWhaleAdapter(AgentAdapter):
    """Run CodeWhale as the evaluated agent via its app-server HTTP API.

    CodeWhale is a Rust-based coding agent that supports DeepSeek V4 and
    other OpenAI-compatible providers. It provides a rich tool ecosystem
    including shell execution, file editing, git, web search, and sub-agents.

    Supports three authentication modes:

    1. **Default**: No ``api_key`` / ``api_base`` — uses CodeWhale's default
       config (DeepSeek with ``DEEPSEEK_API_KEY`` env var).
    2. **Direct API**: ``api_base`` with ``api_protocol='openai'`` —
       passed directly to CodeWhale via config.toml.
    3. **Third-party via litellm**: ``api_base`` with
       ``api_protocol`` != ``openai`` — starts a local litellm proxy
       to translate OpenAI Chat Completions to the target protocol.

    When ``api_base`` is provided, ``api_protocol`` is required.
    """

    CODEWHALE_PROVIDERS = frozenset({
        "deepseek", "deepseek-cn", "nvidia-nim", "openai", "atlascloud",
        "wanjie-ark", "volcengine", "openrouter", "xiaomi-mimo", "novita",
        "fireworks", "siliconflow", "siliconflow-CN", "arcee", "moonshot",
        "sglang", "vllm", "ollama", "huggingface",
    })

    def __init__(
        self,
        model: str = "deepseek-v4-pro",
        api_key: str | None = None,
        api_base: str | None = None,
        api_protocol: str | None = None,
        provider: str | None = None,
        timeout_seconds: int = 10800,
        reasoning_effort: str = "high",
        approval_policy: str = "never",
        allow_external_tools: bool = False,
        tool_mode: str | None = None,
        supports_image_input: bool = False,
    ):
        from ai4sci_bench.core.types import ToolMode
        if tool_mode is not None:
            self.tool_mode = ToolMode(tool_mode) if isinstance(tool_mode, str) else tool_mode
        else:
            self.tool_mode = ToolMode.SEARCH if allow_external_tools else ToolMode.RESTRICTED
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.api_protocol = api_protocol
        if api_base is not None and api_protocol is None:
            raise ValueError(
                "api_protocol is required when api_base is set. "
                "Use 'openai' or 'anthropic' to specify the protocol that api_base speaks."
            )
        self.provider = provider or self._infer_provider(model, api_base)
        self.timeout_seconds = timeout_seconds
        self.reasoning_effort = reasoning_effort
        self.approval_policy = approval_policy
        self.supports_image_input = supports_image_input
        self._uses_proxy = api_base is not None and api_protocol != "openai"
        self._proxy: object | None = None
        self._proxy_lock = threading.Lock()
        self.repo_root = Path(__file__).resolve().parents[2]

    @staticmethod
    def _infer_provider(model: str, api_base: str | None) -> str:
        if model.startswith("openrouter/"):
            return "openrouter"
        if api_base and "deepseek" in api_base:
            return "deepseek"
        if api_base and "openrouter" in api_base:
            return "openrouter"
        if api_base and "openai" in api_base:
            return "openai"
        return "deepseek"

    def _ensure_proxy(self) -> str:
        with self._proxy_lock:
            if self._proxy is not None:
                return self._proxy.local_url  # type: ignore[union-attr]
            from ai4sci_bench.adapters.api_proxy import LiteLLMOpenAIProxy, resolve_litellm_model
            litellm_model = resolve_litellm_model(self.model, self.api_protocol)
            self._proxy = LiteLLMOpenAIProxy(
                model=litellm_model,
                api_base=self.api_base,
                api_key=self.api_key,
                supports_image_input=self.supports_image_input,
            )
            return self._proxy.start()  # type: ignore[union-attr]

    def setup(self, config: dict) -> None:
        self.timeout_seconds = int(config.get("timeout", self.timeout_seconds))
        self.repo_root = Path(config.get("repo_root", self.repo_root))

    def teardown(self) -> None:
        if self._proxy is not None:
            self._proxy.stop()  # type: ignore[union-attr]
            self._proxy = None

    def _generate_features_section(self) -> str:
        """Build the ``[features]`` TOML section for tool isolation."""
        from ai4sci_bench.core.types import ToolMode
        if self.tool_mode == ToolMode.UNRESTRICTED:
            return ""
        lines = ["\n[features]"]
        lines.append("mcp = false")
        if self.tool_mode == ToolMode.RESTRICTED:
            lines.append("web_search = false")
        return "\n".join(lines) + "\n"

    def _generate_config(self, workspace: Path) -> Path:
        """Generate a temporary config.toml for the app-server."""
        provider_default_urls = {
            "deepseek": "https://api.deepseek.com/beta",
            "deepseek-cn": "https://api.deepseek.com/beta",
            "openai": "https://api.openai.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
        }
        effective_base_url = self.api_base or provider_default_urls.get(self.provider, "https://api.deepseek.com/beta")
        effective_api_key = self.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        effective_model = self.model
        effective_provider = self.provider

        if self._uses_proxy:
            proxy_url = self._ensure_proxy()
            effective_base_url = proxy_url
            effective_api_key = "sk-proxy-placeholder"
            effective_provider = "openai"
            effective_model = "auto"
            logger.info("codewhale: using litellm proxy at %s", proxy_url)
        elif not effective_model.startswith("deepseek"):
            # CodeWhale binary only accepts DeepSeek model IDs or "auto".
            # For non-DeepSeek models via OpenAI-compatible endpoints, use "auto"
            # so CodeWhale sends the model field as-is to the endpoint.
            effective_model = "auto"

        config_content = (
            f'provider = "{effective_provider}"\n'
            f'default_text_model = "{effective_model}"\n'
            f'reasoning_effort = "{self.reasoning_effort}"\n'
            f'approval_policy = "{self.approval_policy}"\n'
            f'sandbox_mode = "danger-full-access"\n'
            f'cost_currency = "usd"\n'
            f'\n[providers.{effective_provider}]\n'
            f'api_key = "{effective_api_key}"\n'
            f'base_url = "{effective_base_url}"\n'
        )
        config_content += self._generate_features_section()

        config_path = workspace / ".codewhale_config.toml"
        config_path.write_text(config_content, encoding="utf-8")
        return config_path

    def solve(self, task_instance: TaskInstance) -> AgentOutput:
        codewhale_bin = shutil.which("codewhale")
        if codewhale_bin is None:
            return AgentOutput(
                instance_id=task_instance.instance_id,
                output_dir=task_instance.workspace_dir,
                code_files=[], data_files=[],
                log="codewhale binary not found in PATH",
                execution_time_seconds=0.0,
                status=RunStatus.FAILED,
                error_message="codewhale not installed. Install via: npm install -g codewhale",
            )

        workspace = task_instance.workspace_dir
        raw_prompt = (workspace / "prompt.md").read_text(encoding="utf-8")
        eff_timeout = self._get_effective_timeout(task_instance)

        workspace_prefix = (
            f"IMPORTANT: Your working directory is {workspace}. "
            f"ALL files you create or read must use paths relative to or inside this directory. "
            f"Input data files are in {workspace}/data/. "
            f"Write all output files (code, results, etc.) directly in {workspace}/."
            "\n\n"
        )
        prompt = workspace_prefix + raw_prompt

        config_path = self._generate_config(workspace)

        env = os.environ.copy()
        env["CODEWHALE_CONFIG_PATH"] = str(config_path)
        if self.api_key:
            provider_env_map = {
                "openrouter": "OPENROUTER_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
                "openai": "OPENAI_API_KEY",
                "nvidia-nim": "NVIDIA_NIM_API_KEY",
            }
            env_key = provider_env_map.get(self.provider)
            if env_key:
                env[env_key] = self.api_key

        cmd = [
            codewhale_bin, "exec",
            "--auto",
            "--output-format", "stream-json",
            "--config", str(config_path),
        ]
        cmd.append(prompt)

        t0 = time.time()
        run_error = None
        timed_out = False
        stdout_out = ""
        stderr_out = ""

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=eff_timeout,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            stdout_out = proc.stdout or ""
            stderr_out = proc.stderr or ""
            if proc.returncode != 0:
                run_error = f"codewhale exec exited with code {proc.returncode}"
                if stderr_out:
                    run_error += f": {stderr_out[:1000]}"
        except subprocess.TimeoutExpired as e:
            timed_out = True
            run_error = f"CodeWhale timed out after {eff_timeout}s"
            stdout_out = (e.stdout or b"").decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr_out = (e.stderr or b"").decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        except Exception as e:
            run_error = f"CodeWhale error: {type(e).__name__}: {e}"
            logger.exception("CodeWhale agent failed")
        finally:
            try:
                config_path.unlink(missing_ok=True)
            except Exception:
                pass

        elapsed = time.time() - t0

        produced_files = collect_output_files(workspace, task_instance)
        code_files = [f for f in produced_files if f.endswith(".py")]
        data_files = [f for f in produced_files if not f.endswith(".py")]

        result = self._parse_stream_json(stdout_out)
        human_log = self._build_human_log(result, elapsed, run_error)

        if timed_out:
            status = RunStatus.TIMEOUT
        elif run_error:
            status = RunStatus.FAILED
        else:
            status = RunStatus.COMPLETED

        cost_info = self._extract_cost(result)

        return AgentOutput(
            instance_id=task_instance.instance_id,
            output_dir=workspace,
            code_files=code_files,
            data_files=data_files,
            log=human_log,
            execution_time_seconds=elapsed,
            status=status,
            error_message=run_error,
            raw_model_output=stdout_out[:50000] if stdout_out else None,
            raw_model_output_format="stream-json" if stdout_out else None,
            cost=cost_info,
        )

    @staticmethod
    def _parse_stream_json(stdout: str) -> dict[str, Any]:
        events: list[dict] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return {"events": events, "raw_line_count": len(stdout.splitlines())}

    def _extract_cost(self, result: dict[str, Any]) -> CostInfo | None:
        events = result.get("events", [])
        input_tokens = 0
        output_tokens = 0
        for ev in events:
            usage = ev.get("usage") if isinstance(ev, dict) else None
            if usage and isinstance(usage, dict):
                input_tokens += usage.get("input_tokens", 0) or 0
                output_tokens += usage.get("output_tokens", 0) or 0
        if input_tokens or output_tokens:
            return CostInfo(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )
        return None

    def _build_human_log(
        self, result: dict[str, Any], elapsed: float, error: str | None,
    ) -> str:
        lines: list[str] = []
        lines.append(f"[CodeWhale] model={self.model} provider={self.provider}")
        lines.append(f"[CodeWhale] elapsed={elapsed:.1f}s")

        if error:
            lines.append(f"[Error] {error}")

        output = result.get("output", "")
        model_used = result.get("model", "")
        if model_used:
            lines.append(f"[CodeWhale] model_used={model_used}")

        events = result.get("events", [])
        tool_calls = []
        exec_commands = []
        errors = []

        for ev in events:
            if not isinstance(ev, dict):
                continue
            etype = ev.get("event", "")
            if etype == "tool_call_start":
                tool_calls.append(ev.get("tool_name", "unknown"))
            elif etype == "exec_command_begin":
                exec_commands.append(ev.get("command", "")[:200])
            elif etype == "error":
                errors.append(ev.get("message", "")[:300])

        if tool_calls:
            lines.append(f"[CodeWhale] tool_calls={len(tool_calls)}: {', '.join(tool_calls[:20])}")
        if exec_commands:
            lines.append(f"[CodeWhale] exec_commands={len(exec_commands)}")
            for i, cmd in enumerate(exec_commands[:10], 1):
                lines.append(f"  [{i}] {cmd}")

        if output:
            preview = output[:2000]
            if len(output) > 2000:
                preview += f"... [{len(output) - 2000} more chars]"
            lines.append("")
            lines.append("[Final Output]")
            lines.append(preview)

        if errors:
            lines.append("")
            lines.append(f"[Errors ({len(errors)})]")
            for i, err in enumerate(errors, 1):
                lines.append(f"  [{i}] {err}")

        return "\n".join(lines)
