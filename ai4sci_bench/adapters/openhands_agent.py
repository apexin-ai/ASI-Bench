"""OpenHands SDK adapter — runs OpenHands agent via SDK in a subprocess.

The OpenHands SDK (``openhands.sdk``) lives in a dedicated virtualenv
(``.venv-openhands``) with its own dependencies. This adapter launches a
helper script under that venv, communicating via a JSON result file.

Supports any litellm-compatible model via ``model``, ``api_key``, ``api_base``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

from ai4sci_bench.adapters.subprocess_base import collect_output_files
from ai4sci_bench.core.agent_interface import AgentAdapter
from ai4sci_bench.core.types import AgentOutput, CostInfo, RunStatus, TaskInstance

logger = logging.getLogger(__name__)

_HELPER_SCRIPT = textwrap.dedent(r'''
import json, os, sys, time, traceback

os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"

config_path = sys.argv[1]
with open(config_path) as f:
    cfg = json.load(f)

workspace = cfg["workspace"]
prompt = cfg["prompt"]
result_path = cfg["result_path"]
model = cfg["model"]
api_key = cfg.get("api_key")
api_base = cfg.get("api_base")
max_iterations = cfg.get("max_iterations", 999999)
temperature = cfg.get("temperature")
max_output_tokens = cfg.get("max_output_tokens", 16384)
enable_browser = cfg.get("enable_browser", False)

result = {
    "status": "failed",
    "error": None,
    "elapsed": 0,
    "tool_calls": [],
    "events_count": 0,
    "input_tokens": 0,
    "output_tokens": 0,
}

try:
    from pydantic import SecretStr
    import openhands.tools  # noqa: F401  — registers TerminalTool, FileEditorTool etc.
    from openhands.sdk import LLM, Agent, Tool, LocalConversation

    llm_kwargs = {"model": model, "drop_params": True, "caching_prompt": False}
    if temperature is not None:
        llm_kwargs["temperature"] = temperature
    if max_output_tokens:
        llm_kwargs["max_output_tokens"] = max_output_tokens
    if api_key:
        llm_kwargs["api_key"] = SecretStr(api_key)
    if api_base:
        llm_kwargs["base_url"] = api_base

    llm = LLM(**llm_kwargs)

    tools = [
        Tool(name="terminal"),
        Tool(name="file_editor"),
        Tool(name="task_tracker"),
    ]
    if enable_browser:
        from openhands.tools.browser_use import BrowserToolSet
        tools.append(Tool(name=BrowserToolSet.name))
    agent = Agent(llm=llm, tools=tools)

    events_log = []
    tool_calls = []

    def on_event(event):
        etype = type(event).__name__
        entry = {"type": etype, "t": time.time()}
        if hasattr(event, "tool_call") and event.tool_call:
            tc = event.tool_call
            name = getattr(tc, "name", "") or getattr(tc, "function", {}).get("name", "")
            entry["tool"] = name
            tool_calls.append({"tool": name})
        if hasattr(event, "tool_name"):
            entry["tool_name"] = str(event.tool_name)
        if hasattr(event, "thought") and event.thought:
            try:
                thought_text = " ".join(t.text for t in event.thought if hasattr(t, "text") and t.text)
            except Exception:
                thought_text = str(event.thought)
            if thought_text:
                entry["thought"] = thought_text[:2000]
        if hasattr(event, "reasoning_content") and event.reasoning_content:
            entry["reasoning"] = str(event.reasoning_content)[:2000]
        if hasattr(event, "action") and event.action:
            try:
                entry["action"] = str(event.action)[:2000]
            except Exception:
                pass
        if hasattr(event, "observation") and event.observation:
            try:
                from openhands.sdk.llm import content_to_str
                obs_text = "".join(content_to_str(event.observation.to_llm_content))
                if obs_text:
                    entry["observation"] = obs_text[:2000]
            except Exception:
                entry["observation"] = str(event.observation)[:2000]
        events_log.append(entry)

    conv = LocalConversation(
        agent=agent,
        workspace=workspace,
        max_iteration_per_run=max_iterations,
        delete_on_close=False,
        visualizer=None,
        callbacks=[on_event],
    )

    t0 = time.time()
    conv.send_message(prompt)
    conv.run()
    elapsed = time.time() - t0

    stats = conv.conversation_stats
    input_tokens = 0
    output_tokens = 0
    if stats:
        input_tokens = getattr(stats, "input_tokens", 0) or 0
        output_tokens = getattr(stats, "output_tokens", 0) or 0

    try:
        conv.close()
    except Exception:
        pass

    result.update({
        "status": "completed",
        "elapsed": round(elapsed, 2),
        "tool_calls": tool_calls[:200],
        "events_count": len(events_log),
        "events_log": events_log[:500],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    })

except Exception as e:
    result["error"] = f"{type(e).__name__}: {e}"
    result["traceback"] = traceback.format_exc()

with open(result_path, "w") as f:
    json.dump(result, f, default=str)
''')


class OpenHandsAdapter(AgentAdapter):
    """Run an OpenHands SDK agent in a dedicated virtualenv subprocess.

    Parameters
    ----------
    model : str
        litellm model identifier (e.g. ``openai/gpt-4o``, ``anthropic/claude-opus-4-6``).
    api_base : str or None
        Custom API endpoint URL.
    api_key : str or None
        API key for the LLM service.
    timeout_seconds : int
        Wall-clock timeout for the agent subprocess.
    max_iterations : int
        Maximum agent iterations (tool calls).
    temperature : float
        Sampling temperature.
    max_output_tokens : int
        Maximum output tokens per LLM call.
    openhands_venv : str or None
        Path to the OpenHands virtualenv. Auto-detected if None.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        api_base: str | None = None,
        api_key: str | None = None,
        api_protocol: str | None = None,
        timeout_seconds: int = 10800,
        max_iterations: int = 999999,
        temperature: float | None = None,
        max_output_tokens: int = 16384,
        openhands_venv: str | None = None,
        allow_external_tools: bool = False,
        tool_mode: str | None = None,
    ):
        from ai4sci_bench.core.types import ToolMode
        if tool_mode is not None:
            self.tool_mode = ToolMode(tool_mode) if isinstance(tool_mode, str) else tool_mode
        else:
            self.tool_mode = ToolMode.SEARCH if allow_external_tools else ToolMode.RESTRICTED
        self.api_protocol = api_protocol
        if api_base is not None and api_protocol is None:
            raise ValueError(
                "api_protocol is required when api_base is set. "
                "Use 'openai' or 'anthropic' to specify the protocol that api_base speaks."
            )
        # OpenHands SDK uses litellm internally, which routes by provider
        # prefix (e.g. "openai/gpt-4o"). When api_base is set with
        # api_protocol, always prepend the protocol so litellm routes to
        # the custom endpoint — model names like "z-ai/glm-5.2" contain
        # a slash but that's part of the model id, not a litellm provider.
        if api_protocol is not None and api_base is not None:
            self.model = f"{api_protocol}/{model}"
        else:
            self.model = model
        self.api_base = api_base
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        self.timeout_seconds = timeout_seconds
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._openhands_venv = openhands_venv
        self.allow_external_tools = allow_external_tools
        self.repo_root = Path(__file__).resolve().parents[2]
        self.sandbox: str = "none"
        self._os_sandbox: object | None = None
        self._sandbox_image_identity: str | None = None

    @property
    def sandbox_image_identity(self) -> str | None:
        """Expose the Docker image used for OS-sandbox provenance."""
        return self._sandbox_image_identity

    def setup(self, config: dict) -> None:
        self.timeout_seconds = int(config.get("timeout", self.timeout_seconds))
        self.repo_root = Path(config.get("repo_root", self.repo_root))
        self.sandbox = config.get("sandbox", "none")
        if self.sandbox == "os":
            from ai4sci_bench.runner.os_sandbox import OSSandbox
            self._os_sandbox = OSSandbox(self.repo_root)

    @staticmethod
    def _find_site_packages(venv_root: Path) -> str | None:
        """Find the site-packages directory in a venv."""
        for sp in sorted(venv_root.glob("lib/python*/site-packages")):
            if sp.is_dir():
                return str(sp)
        return None

    def _find_openhands_python(self) -> str | None:
        """Find the Python binary in the OpenHands virtualenv."""
        if self._openhands_venv:
            p = Path(self._openhands_venv) / "bin" / "python"
            return str(p) if p.exists() else None

        candidates = [
            self.repo_root / ".venv-openhands" / "bin" / "python",
            self.repo_root.parent / ".venv-openhands" / "bin" / "python",
            Path.home() / "ASI-Bench" / ".venv-openhands" / "bin" / "python",
            Path.home() / ".venv-openhands" / "bin" / "python",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    def _build_task_prompt(self, workspace: Path) -> str:
        raw_prompt = (workspace / "prompt.md").read_text(encoding="utf-8")
        return (
            f"IMPORTANT: Your working directory is {workspace}. "
            f"ALL files you create or read must use paths relative to or inside this directory. "
            f"Before starting, run `cd {workspace}` in the terminal. "
            f"Input data files are in {workspace}/data/. "
            f"Write all output files (code, results, etc.) directly in {workspace}/."
            "\n\n" + raw_prompt
        )

    def solve(self, task_instance: TaskInstance) -> AgentOutput:
        if self.sandbox == "os":
            return self._solve_os_sandbox(task_instance)
        return self._solve_host(task_instance)

    _CONTAINER_OH_VENV = "/opt/venv-openhands"

    def _solve_os_sandbox(self, task_instance: TaskInstance) -> AgentOutput:
        """Run OpenHands inside the OS sandbox (Docker container).

        The host .venv-openhands is volume-mounted into the container at
        /opt/venv-openhands. The helper script and config JSON are written
        into the workspace (mounted as /workspace) so the container can
        execute them.
        """
        eff_timeout = self._get_effective_timeout(task_instance)
        workspace = task_instance.workspace_dir

        host_oh_python = self._find_openhands_python()
        if not host_oh_python:
            return self._fail(task_instance, "OpenHands venv not found for os sandbox.")
        host_oh_venv = Path(host_oh_python).parent.parent.resolve()
        host_site_packages = self._find_site_packages(host_oh_venv)
        if not host_site_packages:
            return self._fail(task_instance, f"Cannot find site-packages in {host_oh_venv}")

        helper_path = workspace / ".oh_runner.py"
        helper_path.write_text(_HELPER_SCRIPT, encoding="utf-8")

        prompt = self._build_task_prompt(workspace)
        from ai4sci_bench.core.types import ToolMode
        config = {
            "workspace": "/workspace",
            "prompt": prompt,
            "result_path": "/workspace/.oh_result.json",
            "model": self.model,
            "api_key": self.api_key,
            "api_base": self.api_base,
            "max_iterations": self.max_iterations,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "enable_browser": self.tool_mode == ToolMode.SEARCH,
        }
        config_path = workspace / ".oh_config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        # Use the container's own Python (/opt/venv/bin/python), mount only
        # the host venv's site-packages so OpenHands SDK is importable.
        container_python = "/opt/venv/bin/python"
        cmd = [container_python, "/workspace/.oh_runner.py", "/workspace/.oh_config.json"]

        extra_env: dict[str, str] = {}
        if self.api_key:
            extra_env["OPENAI_API_KEY"] = self.api_key
        if self.api_base:
            extra_env["OPENAI_BASE_URL"] = self.api_base
        extra_env["OPENHANDS_SUPPRESS_BANNER"] = "1"

        # Mount the OpenHands site-packages into the container's venv
        container_site_packages = "/opt/venv/lib/python3.12/site-packages"
        extra_mounts = ["-v", f"{host_site_packages}:{container_site_packages}:ro"]

        t0 = time.time()
        assert self._os_sandbox is not None
        success, log, raw_stdout, raw_stderr, image_identity = (
            self._os_sandbox.run_agent(
                task_metadata=task_instance.metadata,
                agent_cmd=cmd,
                workspace=workspace,
                timeout=eff_timeout,
                agent_type="openhands",
                allow_external_tools=self.allow_external_tools,
                extra_env=extra_env or None,
                extra_mounts=extra_mounts,
            )
        )
        elapsed = time.time() - t0
        self._sandbox_image_identity = image_identity

        produced_files = collect_output_files(workspace, task_instance)

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
            log=log,
            execution_time_seconds=elapsed,
            status=status,
            error_message=None if success else log,
            raw_stdout=raw_stdout,
            raw_stderr=raw_stderr,
        )

    def _solve_host(self, task_instance: TaskInstance) -> AgentOutput:
        """Original host-mode solve logic."""
        workspace = task_instance.workspace_dir
        eff_timeout = self._get_effective_timeout(task_instance)

        oh_python = self._find_openhands_python()
        if not oh_python:
            return self._fail(
                task_instance,
                "OpenHands venv not found. Expected at .venv-openhands/. "
                "Install via: uv venv .venv-openhands && source .venv-openhands/bin/activate "
                "&& pip install openhands-ai",
            )

        check = subprocess.run(
            [oh_python, "-c", "from openhands.sdk import LLM, LocalConversation"],
            capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace",
        )
        if check.returncode != 0:
            return self._fail(
                task_instance,
                f"OpenHands SDK not importable in {oh_python}: {check.stderr.strip()[:500]}",
            )

        prompt = self._build_task_prompt(workspace)

        tmpdir = tempfile.mkdtemp(prefix="openhands_")
        try:
            helper_path = os.path.join(tmpdir, "oh_runner.py")
            with open(helper_path, "w") as f:
                f.write(_HELPER_SCRIPT)

            result_path = os.path.join(tmpdir, "result.json")

            from ai4sci_bench.core.types import ToolMode
            config = {
                "workspace": str(workspace),
                "prompt": prompt,
                "result_path": result_path,
                "model": self.model,
                "api_key": self.api_key,
                "api_base": self.api_base,
                "max_iterations": self.max_iterations,
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "enable_browser": self.tool_mode == ToolMode.SEARCH,
            }
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f)

            env = os.environ.copy()
            env["OPENHANDS_SUPPRESS_BANNER"] = "1"
            env.setdefault("PYTHONUNBUFFERED", "1")

            t0 = time.time()
            timed_out = False
            stderr_out = ""

            try:
                proc = subprocess.run(
                    [oh_python, helper_path, config_path],
                    capture_output=True,
                    text=True,
                    timeout=eff_timeout,
                    cwd=str(workspace),
                    env=env,
                    encoding="utf-8",
                    errors="replace",
                )
                stderr_out = proc.stderr or ""
            except subprocess.TimeoutExpired as e:
                timed_out = True
                stderr_out = (e.stderr or b"").decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")

            elapsed = time.time() - t0

            sdk_result = self._load_result(result_path)

            if timed_out:
                status = RunStatus.TIMEOUT
                run_error = f"OpenHands timed out after {eff_timeout}s"
            elif sdk_result.get("status") == "completed":
                status = RunStatus.COMPLETED
                run_error = None
            else:
                status = RunStatus.FAILED
                run_error = sdk_result.get("error") or "OpenHands failed (unknown error)"

            tool_calls = sdk_result.get("tool_calls", [])
            input_tokens = sdk_result.get("input_tokens", 0)
            output_tokens = sdk_result.get("output_tokens", 0)

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        produced_files = collect_output_files(workspace, task_instance)
        code_files = [f for f in produced_files if f.endswith(".py")]
        data_files = [f for f in produced_files if not f.endswith(".py")]

        human_log = self._build_human_log(
            tool_calls, elapsed, status, run_error, stderr_out, sdk_result,
        )

        structured_log = json.dumps({
            "sdk_result": sdk_result,
            "stderr_tail": stderr_out[-3000:] if stderr_out else "",
            "model": self.model,
        }, ensure_ascii=False, default=str)

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
            error_message=run_error,
            raw_model_output=structured_log,
            raw_model_output_format="json",
            cost=cost_info,
        )

    def _fail(self, task_instance: TaskInstance, msg: str) -> AgentOutput:
        return AgentOutput(
            instance_id=task_instance.instance_id,
            output_dir=task_instance.workspace_dir,
            code_files=[], data_files=[],
            log=msg,
            execution_time_seconds=0.0,
            status=RunStatus.FAILED,
            error_message=msg,
        )

    @staticmethod
    def _load_result(path: str) -> dict[str, Any]:
        try:
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {"status": "failed", "error": "Result file not found or unreadable"}

    def _build_human_log(
        self,
        tool_calls: list[dict],
        elapsed: float,
        status: RunStatus,
        run_error: str | None,
        stderr: str,
        sdk_result: dict,
    ) -> str:
        lines: list[str] = []
        lines.append(f"[OpenHands SDK] model={self.model} max_iter={self.max_iterations}")
        lines.append(f"[OpenHands SDK] elapsed={elapsed:.1f}s tool_calls={len(tool_calls)} status={status.value}")
        lines.append("")

        for i, tc in enumerate(tool_calls[:50], 1):
            lines.append(f"  Step {i}: {tc.get('tool', '?')}")

        if run_error:
            lines.append("")
            lines.append(f"Error: {run_error}")

        tb = sdk_result.get("traceback", "")
        if tb:
            lines.append("")
            lines.append("Traceback:")
            lines.append(tb[-2000:])

        if stderr:
            err_lines = [l for l in stderr.splitlines() if "error" in l.lower() or "traceback" in l.lower()]
            if err_lines:
                lines.append("")
                lines.append(f"Stderr errors ({len(err_lines)}):")
                for el in err_lines[:10]:
                    lines.append(f"  {el[:200]}")

        return "\n".join(lines)
