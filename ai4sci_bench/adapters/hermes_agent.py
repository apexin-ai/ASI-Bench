"""Hermes Agent adapter — runs Nous Research Hermes Agent as the evaluated agent.

Hermes Agent is a Python-based agentic framework that uses the OpenAI Chat
Completions API with tool calling. It supports multiple LLM backends via
its ``base_url`` / ``api_key`` configuration.

For non-OpenAI-compatible backends (Anthropic, Gemini, etc.), this adapter
starts a local ``LiteLLMOpenAIProxy`` that translates OpenAI Chat Completions
to the target API via litellm.

Requires ``hermes-agent`` to be installed in the Python environment.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from ai4sci_bench.adapters.subprocess_base import collect_output_files
from ai4sci_bench.core.agent_interface import AgentAdapter
from ai4sci_bench.core.types import AgentOutput, CostInfo, RunStatus, TaskInstance

logger = logging.getLogger(__name__)


class HermesAgentAdapter(AgentAdapter):
    """Run Hermes Agent (Nous Research) as the evaluated agent.

    Hermes provides a full agentic loop with tool calling: the model can
    read files, write code, execute it, observe errors, and iterate.

    Supports three authentication modes:

    1. **Default**: No ``api_key`` / ``api_base`` — uses Hermes defaults
       (OpenRouter with ``OPENROUTER_API_KEY`` env var).
    2. **Direct API**: Only ``api_key`` (and optionally ``api_base``) —
       passed directly to Hermes when the endpoint speaks OpenAI format.
    3. **Third-party via litellm**: ``api_base`` is set with
       ``api_protocol`` != ``openai`` — starts a local litellm proxy
       to translate OpenAI Chat Completions to the target protocol.

    When ``api_base`` is provided, ``api_protocol`` is required.
    """

    SEARCH_DISABLED_TOOLSETS = [
        "browser", "image_gen", "video_gen", "video", "x_search",
        "moa", "tts", "delegation", "cronjob", "session_search",
        "memory", "context_engine",
    ]
    RESTRICTED_DISABLED_TOOLSETS = SEARCH_DISABLED_TOOLSETS + ["web"]

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-6",
        api_key: str | None = None,
        api_base: str | None = None,
        api_protocol: str | None = None,
        timeout_seconds: int = 10800,
        max_iterations: int = 999999,
        enabled_toolsets: list[str] | None = None,
        disabled_toolsets: list[str] | None = None,
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
        self.timeout_seconds = timeout_seconds
        self.max_iterations = max_iterations
        self.enabled_toolsets = enabled_toolsets
        self.disabled_toolsets = disabled_toolsets
        self.supports_image_input = supports_image_input
        self._uses_proxy = api_base is not None and api_protocol != "openai"
        self._proxy: object | None = None
        self._proxy_lock = threading.Lock()
        self.repo_root = Path(__file__).resolve().parents[2]

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

    def _resolve_hermes_params(self) -> dict[str, Any]:
        """Build kwargs for AIAgent constructor."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_iterations": self.max_iterations,
            "quiet_mode": True,
            "save_trajectories": False,
        }
        if self.enabled_toolsets:
            kwargs["enabled_toolsets"] = self.enabled_toolsets
        disabled = list(self.disabled_toolsets or [])
        from ai4sci_bench.core.types import ToolMode
        if self.tool_mode == ToolMode.RESTRICTED:
            disabled.extend(t for t in self.RESTRICTED_DISABLED_TOOLSETS if t not in disabled)
        elif self.tool_mode == ToolMode.SEARCH:
            disabled.extend(t for t in self.SEARCH_DISABLED_TOOLSETS if t not in disabled)
        if disabled:
            kwargs["disabled_toolsets"] = disabled

        if self._uses_proxy:
            proxy_url = self._ensure_proxy()
            kwargs["base_url"] = proxy_url
            kwargs["api_key"] = "sk-proxy-placeholder"
            logger.info("hermes_agent: using litellm proxy at %s", proxy_url)
        else:
            if self.api_base:
                kwargs["base_url"] = self.api_base
            if self.api_key:
                kwargs["api_key"] = self.api_key

        return kwargs

    def setup(self, config: dict) -> None:
        self.timeout_seconds = int(config.get("timeout", self.timeout_seconds))
        self.repo_root = Path(config.get("repo_root", self.repo_root))

    def teardown(self) -> None:
        if self._proxy is not None:
            self._proxy.stop()  # type: ignore[union-attr]
            self._proxy = None

    def solve(self, task_instance: TaskInstance) -> AgentOutput:
        try:
            from run_agent import AIAgent
        except ImportError as e:
            return AgentOutput(
                instance_id=task_instance.instance_id,
                output_dir=task_instance.workspace_dir,
                code_files=[], data_files=[],
                log=f"hermes-agent not installed: {e}",
                execution_time_seconds=0.0,
                status=RunStatus.FAILED,
                error_message=f"ImportError: {e}",
            )

        workspace = task_instance.workspace_dir
        raw_prompt = (workspace / "prompt.md").read_text(encoding="utf-8")
        eff_timeout = self._get_effective_timeout(task_instance)

        workspace_prefix = (
            f"IMPORTANT: Your working directory is {workspace}. "
            f"ALL files you create or read must use paths relative to or inside this directory. "
            f"Before starting, run `cd {workspace}` in the terminal. "
            f"Input data files are in {workspace}/data/. "
            f"Write all output files (code, results, etc.) directly in {workspace}/."
            "\n\n"
        )
        prompt = workspace_prefix + raw_prompt

        hermes_kwargs = self._resolve_hermes_params()

        t0 = time.time()
        run_error = None
        timed_out = False
        result: dict[str, Any] = {}

        try:
            agent = AIAgent(**hermes_kwargs)
            result = agent.run_conversation(prompt)
        except TimeoutError:
            timed_out = True
            run_error = f"Hermes timed out after {eff_timeout}s"
        except Exception as e:
            run_error = f"Hermes error: {type(e).__name__}: {e}"
            logger.exception("Hermes agent failed")

        elapsed = time.time() - t0

        produced_files = collect_output_files(workspace, task_instance)
        code_files = [f for f in produced_files if f.endswith(".py")]
        data_files = [f for f in produced_files if not f.endswith(".py")]

        human_log = self._build_human_log(result, elapsed)

        structured_log = json.dumps({
            "messages_count": len(result.get("messages", [])),
            "api_calls": result.get("api_calls", 0),
            "completed": result.get("completed", False),
            "failed": result.get("failed", False),
            "model": result.get("model", self.model),
            "elapsed_s": round(elapsed, 2),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
        }, ensure_ascii=False, default=str)

        if timed_out:
            status = RunStatus.TIMEOUT
        elif run_error or result.get("failed", False):
            status = RunStatus.FAILED
        elif result.get("completed", False):
            status = RunStatus.COMPLETED
        else:
            status = RunStatus.COMPLETED if not run_error else RunStatus.FAILED

        input_tokens = result.get("input_tokens", 0) or 0
        output_tokens = result.get("output_tokens", 0) or 0
        total_tokens = input_tokens + output_tokens
        cost_info = CostInfo(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        ) if total_tokens > 0 else None

        return AgentOutput(
            instance_id=task_instance.instance_id,
            output_dir=workspace,
            code_files=code_files,
            data_files=data_files,
            log=human_log,
            execution_time_seconds=elapsed,
            status=status,
            error_message=run_error or (result.get("error") if result.get("failed") else None),
            raw_model_output=structured_log,
            raw_model_output_format="json",
            cost=cost_info,
        )

    def _build_human_log(self, result: dict[str, Any], elapsed: float) -> str:
        lines: list[str] = []
        lines.append(f"[Hermes] model={self.model} max_iter={self.max_iterations}")
        lines.append(
            f"[Hermes] elapsed={elapsed:.1f}s "
            f"api_calls={result.get('api_calls', 0)} "
            f"completed={result.get('completed', False)}"
        )

        input_t = result.get("input_tokens", 0) or 0
        output_t = result.get("output_tokens", 0) or 0
        if input_t or output_t:
            lines.append(f"[Hermes] tokens: input={input_t} output={output_t}")

        cost = result.get("estimated_cost_usd")
        if cost:
            lines.append(f"[Hermes] estimated_cost=${cost:.4f}")

        final = result.get("final_response")
        if final:
            preview = final[:2000]
            if len(final) > 2000:
                preview += f"... [{len(final) - 2000} more chars]"
            lines.append("")
            lines.append("[Final Response]")
            lines.append(preview)

        error = result.get("error")
        if error:
            lines.append("")
            lines.append(f"[Error] {error}")

        return "\n".join(lines)
