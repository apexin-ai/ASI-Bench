"""Direct LLM API adapter — single-turn, no agentic tools."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai4sci_bench.adapters.subprocess_base import collect_output_files, get_task_environment
from ai4sci_bench.core.agent_interface import AgentAdapter
from ai4sci_bench.core.types import AgentOutput, CostInfo, RunStatus, TaskInstance
from ai4sci_bench.runner.linux_ns_sandbox import LinuxNSSandbox
from ai4sci_bench.runner.os_sandbox import OSSandbox
from ai4sci_bench.runner.sandbox_support import validate_sandbox_mode
from ai4sci_bench.runner.task_env import TaskEnvironmentManager
from ai4sci_bench.trajectory.call_observability import ModelCallRecord


def _plain_metadata(value: Any) -> dict[str, Any]:
    """Return provider metadata only when LiteLLM exposes a real mapping."""
    return value if isinstance(value, dict) else {}


def _string_metadata(value: Any) -> str | None:
    """Avoid persisting MagicMock/object reprs as provider identifiers."""
    return str(value) if isinstance(value, (str, int)) and str(value) else None


def _response_transport_metadata(response: Any) -> dict[str, Any]:
    """Extract safe LiteLLM/provider response metadata without secrets."""
    hidden = _plain_metadata(getattr(response, "_hidden_params", None))
    headers: dict[str, Any] = {}
    for candidate in (
        hidden.get("additional_headers"),
        hidden.get("headers"),
        getattr(response, "additional_headers", None),
    ):
        if isinstance(candidate, dict):
            headers.update(candidate)
    lowered = {str(key).lower(): value for key, value in headers.items()}
    request_id = next(
        (
            _string_metadata(lowered.get(key))
            for key in ("x-request-id", "request-id", "x-amzn-requestid", "cf-ray")
            if lowered.get(key) is not None
        ),
        None,
    )
    status = next(
        (
            value
            for value in (
                hidden.get("status_code"),
                hidden.get("http_status"),
                getattr(response, "status_code", None),
            )
            if isinstance(value, int)
        ),
        None,
    )
    return {
        "request_id": request_id,
        "http_status": status,
        "headers_received": True if headers or status is not None else None,
    }


class DirectLLMAdapter(AgentAdapter):
    """Call an LLM API directly to generate code (single-turn, no agentic tools).

    Primary use: no-tool baseline to measure intrinsic LLM capability.
    Uses litellm for unified API access.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        system_prompt: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_protocol: str | None = None,
    ):
        self.api_protocol = api_protocol
        if api_base is not None and api_protocol is None:
            raise ValueError(
                "api_protocol is required when api_base is set. "
                "Use 'openai' or 'anthropic' to specify the protocol that api_base speaks."
            )
        if api_protocol is not None:
            from ai4sci_bench.adapters.api_proxy import resolve_litellm_model
            self.model = resolve_litellm_model(model, api_protocol)
        else:
            self.model = model
        self.system_prompt = system_prompt or (
            "You are an expert scientific programmer. "
            "When asked to write code, output ONLY a single Python code block. "
            "Do not include any explanation outside the code block."
        )
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = 10800
        self.sandbox = "none"
        self.repo_root = Path(__file__).resolve().parents[2]
        self.task_env_manager: TaskEnvironmentManager | None = None
        self.sandbox_image_identity: str | None = None

    def setup(self, config: dict) -> None:
        self.timeout = int(config.get("timeout", self.timeout))
        self.sandbox = config.get("sandbox", "none")
        validate_sandbox_mode(
            self.sandbox,
            supported_modes=("none", "task", "os", "linux_ns"),
            component=self.__class__.__name__,
        )
        self.repo_root = Path(config.get("repo_root", self.repo_root))
        self.task_env_manager = (
            TaskEnvironmentManager(self.repo_root)
            if self.sandbox in ("task", "linux_ns")
            else None
        )

    def solve(self, task_instance: TaskInstance) -> AgentOutput:
        import litellm

        prompt = (task_instance.workspace_dir / "prompt.md").read_text(encoding="utf-8")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        structured_log: list[dict[str, Any]] = []
        model_call_records: list[dict[str, Any]] = []
        call_record = ModelCallRecord(
            call_index=1,
            adapter="direct_llm",
            provider=self.model.split("/", 1)[0] if "/" in self.model else None,
            model=self.model,
            protocol=self.api_protocol,
            endpoint=self.api_base,
            transport_observability="provider_client_response_metadata",
            instance_id=task_instance.instance_id,
            attempted=False,
            request_started=False,
            request_sent=None,
            boundary_observed=True,
            boundary_kind="provider_client_call",
            evidence_status="partial",
        )
        t0 = time.time()
        try:
            structured_log.append({
                "step": "prompt",
                "system_prompt": self.system_prompt,
                "user_prompt": prompt,
                "model": self.model,
                "timestamp_ms": int((time.time() - t0) * 1000),
            })

            kwargs: dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.api_base:
                kwargs["api_base"] = self.api_base

            api_t0 = time.time()
            call_record.attempted = True
            call_record.request_started = True
            call_record.started_at = datetime.now(timezone.utc).isoformat()
            call_record.request_started_at = call_record.started_at
            response = litellm.completion(model=self.model, messages=messages, **kwargs)
            api_latency_ms = int((time.time() - api_t0) * 1000)
            content = response.choices[0].message.content
            choice = response.choices[0]
            transport = _response_transport_metadata(response)
            call_record.request_sent = True
            call_record.response_received = True
            call_record.stream_opened = False
            call_record.stream_ended = None
            call_record.http_headers_received = transport["headers_received"]
            call_record.http_status = transport["http_status"]
            call_record.request_id = transport["request_id"]
            call_record.call_finished = True
            tool_calls = getattr(choice.message, "tool_calls", None)
            if content:
                call_record.first_token_received = True
                call_record.content_state = "nonempty"
                call_record.outcome = "complete_nonempty_response"
            elif tool_calls:
                call_record.content_state = "omitted"
                call_record.outcome = "complete_tool_only_response"
            else:
                call_record.first_token_received = False
                call_record.content_state = "empty"
                call_record.outcome = "complete_empty_response"
            call_record.response_id = _string_metadata(getattr(response, "id", None))
            call_record.finish_reason = _string_metadata(
                getattr(choice, "finish_reason", None)
            )
            call_record.provider_status = "completed"
            if call_record.finish_reason in {
                "incomplete", "length", "max_output", "max_output_tokens", "max_tokens",
            }:
                call_record.error_type = "incomplete_response"
                call_record.outcome = "incomplete_response"
                call_record.provider_status = "incomplete"
            call_record.ended_at = datetime.now(timezone.utc).isoformat()
            call_record.request_ended_at = call_record.ended_at
            call_record.response_started_at = call_record.ended_at
            call_record.response_ended_at = call_record.ended_at
            call_record.duration_ms = api_latency_ms
            call_record.evidence_status = "complete"
            call_record.event_count = 1
            call_record.first_event_sequence = 1
            call_record.last_event_sequence = 1
            call_record.event_sequences = [1]

            usage_dict = {}
            cost_info = None
            if hasattr(response, "usage") and response.usage:
                usage_dict = {
                    "input_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                    "output_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                }
                cost_info = CostInfo(
                    input_tokens=usage_dict["input_tokens"],
                    output_tokens=usage_dict["output_tokens"],
                    total_tokens=usage_dict["input_tokens"] + usage_dict["output_tokens"],
                )
                call_record.input_tokens = usage_dict["input_tokens"]
                call_record.output_tokens = usage_dict["output_tokens"]
                call_record.token_count = (
                    usage_dict["input_tokens"] + usage_dict["output_tokens"]
                )
            model_call_records.append(call_record.to_dict())

            structured_log.append({
                "step": "llm_response",
                "content": content or "",
                "content_length": len(content) if content else 0,
                "usage": usage_dict,
                "latency_ms": api_latency_ms,
                "timestamp_ms": int((time.time() - t0) * 1000),
            })

            code_files = []
            for f in task_instance.metadata.get("output", {}).get("files", []):
                if f.get("type") == "code":
                    code_files.append(f["name"])
            if not code_files:
                code_files = ["simulation.py"]

            code_file = code_files[0]
            expected_output_files = [
                f["name"] for f in task_instance.metadata.get("output", {}).get("files", [])
            ]

            blocks = self._extract_code_blocks(content)
            candidates = [
                self._unwrap_file_writer_wrapper(block, code_file) or block.strip()
                for block in blocks
            ] if blocks else [content.strip()]

            scored = [
                (c, self._score_code_candidate(
                    c, code_file=code_file,
                    expected_output_files=expected_output_files or [],
                ))
                for c in candidates
            ]
            selected_idx = max(range(len(scored)), key=lambda i: scored[i][1]) if scored else 0
            code = candidates[selected_idx] if candidates else content.strip()

            structured_log.append({
                "step": "code_extraction",
                "num_candidates": len(candidates),
                "candidate_scores": [s for _, s in scored],
                "selected_index": selected_idx,
                "selected_code": code,
                "selected_code_length": len(code),
                "timestamp_ms": int((time.time() - t0) * 1000),
            })

            (task_instance.workspace_dir / code_file).write_text(code, encoding="utf-8")

            exec_t0 = time.time()
            success, log, raw_stdout, raw_stderr = self._execute(task_instance, code_file)
            exec_duration_ms = int((time.time() - exec_t0) * 1000)
            elapsed = time.time() - t0

            structured_log.append({
                "step": "code_execution",
                "code_file": code_file,
                "exit_code": 0 if success else 1,
                "duration_ms": exec_duration_ms,
                "stdout_length": len(raw_stdout) if raw_stdout else 0,
                "stderr_length": len(raw_stderr) if raw_stderr else 0,
                "timestamp_ms": int((time.time() - t0) * 1000),
            })

            structured_log_json = json.dumps(structured_log, ensure_ascii=False)

            return AgentOutput(
                instance_id=task_instance.instance_id,
                output_dir=task_instance.workspace_dir,
                code_files=code_files,
                data_files=self._collect_data_files(task_instance),
                log=log,
                execution_time_seconds=elapsed,
                status=RunStatus.COMPLETED if success else RunStatus.FAILED,
                raw_stdout=raw_stdout,
                raw_stderr=raw_stderr,
                raw_stdout_format="log" if raw_stdout is not None else None,
                raw_model_output=structured_log_json,
                raw_model_output_format="json",
                cost=cost_info,
                model_call_records=model_call_records,
            )
        except Exception as e:
            elapsed = time.time() - t0
            if not model_call_records:
                call_record.call_finished = True
                call_record.ended_at = datetime.now(timezone.utc).isoformat()
                call_record.request_ended_at = call_record.ended_at
                call_record.error_class = type(e).__name__
                call_record.error_message = str(e)
                status_code = getattr(e, "status_code", None)
                call_record.http_status = status_code if isinstance(status_code, int) else None
                call_record.request_id = _string_metadata(getattr(e, "request_id", None))
                request_sent = getattr(e, "request_sent", None)
                if isinstance(request_sent, bool):
                    call_record.request_sent = request_sent
                if call_record.http_status is not None or call_record.request_id is not None:
                    call_record.request_sent = True
                    call_record.http_headers_received = call_record.http_status is not None
                error_name = type(e).__name__.lower()
                if call_record.attempted is False:
                    call_record.error_type = "client_error"
                    call_record.outcome = "no_request_created"
                elif isinstance(e, TimeoutError) or "timeout" in error_name:
                    call_record.error_type = "timeout"
                    call_record.timeout_phase = "response"
                    call_record.outcome = "timeout"
                elif call_record.http_status is not None:
                    call_record.error_type = "provider_error"
                    call_record.outcome = "provider_error"
                elif call_record.request_sent is True:
                    call_record.error_type = "transport_error"
                    call_record.outcome = "transport_error"
                else:
                    call_record.error_type = "client_error"
                    call_record.outcome = "request_send_state_unknown"
                call_record.evidence_status = "complete"
                model_call_records.append(call_record.to_dict())
            structured_log.append({
                "step": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp_ms": int((time.time() - t0) * 1000),
            })
            structured_log_json = json.dumps(structured_log, ensure_ascii=False)
            return AgentOutput(
                instance_id=task_instance.instance_id,
                output_dir=task_instance.workspace_dir,
                code_files=[],
                data_files=[],
                log=str(e),
                execution_time_seconds=elapsed,
                status=RunStatus.FAILED,
                error_message=str(e),
                raw_model_output=structured_log_json,
                raw_model_output_format="json",
                model_call_records=model_call_records,
            )

    def _extract_code(
        self,
        content: str,
        *,
        code_file: str = "simulation.py",
        expected_output_files: list[str] | None = None,
    ) -> str:
        """Extract the most runnable Python code from an LLM response."""
        blocks = self._extract_code_blocks(content)
        if not blocks:
            return content.strip()

        candidates = [
            self._unwrap_file_writer_wrapper(block, code_file) or block.strip()
            for block in blocks
        ]
        return max(
            candidates,
            key=lambda candidate: self._score_code_candidate(
                candidate,
                code_file=code_file,
                expected_output_files=expected_output_files or [],
            ),
        )

    def _extract_code_blocks(self, content: str) -> list[str]:
        """Return code blocks, preferring explicit python fences."""
        python_blocks = re.findall(r"```python\s*\n(.*?)```", content, re.DOTALL)
        if python_blocks:
            return [block.strip() for block in python_blocks if block.strip()]

        generic_blocks = re.findall(r"```\s*\n(.*?)```", content, re.DOTALL)
        return [block.strip() for block in generic_blocks if block.strip()]

    def _unwrap_file_writer_wrapper(self, code: str, code_file: str) -> str | None:
        """Unwrap code that only writes the real program into the target file."""
        try:
            module = ast.parse(code)
        except SyntaxError:
            return None

        assigned_strings: dict[str, str] = {}
        for node in module.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                assigned_strings[node.targets[0].id] = node.value.value

        for node in module.body:
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            if not isinstance(call.func, ast.Attribute) or call.func.attr != "write_text":
                continue
            if not call.args or not isinstance(call.args[0], ast.Name):
                continue
            path_name = self._extract_path_literal(call.func.value)
            if path_name != code_file:
                continue
            embedded = assigned_strings.get(call.args[0].id)
            if embedded and embedded.strip():
                return embedded.strip()
        return None

    def _extract_path_literal(self, node: ast.AST) -> str | None:
        """Resolve simple Path('file.py') expressions used by wrapper scripts."""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path":
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                return node.args[0].value
        return None

    def _score_code_candidate(
        self,
        code: str,
        *,
        code_file: str,
        expected_output_files: list[str],
    ) -> int:
        """Prefer candidates that look like the final runnable solver."""
        score = 0
        if "if __name__ == \"__main__\"" in code:
            score += 3
        if "np.save(" in code:
            score += 4
        if any(output_file in code for output_file in expected_output_files):
            score += 2
        if f'Path("{code_file}")' in code or f"Path('{code_file}')" in code:
            score -= 6
        if ".write_text(" in code and "code =" in code:
            score -= 6
        return score

    def _execute(self, task_instance: TaskInstance, code_file: str) -> tuple[bool, str, str | None, str | None]:
        """Execute the generated code in the workspace."""
        timeout = self._get_effective_timeout(task_instance)
        workspace = task_instance.workspace_dir
        if self.sandbox == "os":
            sandbox = OSSandbox(self.repo_root)
            success, log, raw_stdout, raw_stderr, image_identity = sandbox.execute_python(
                task_instance.metadata,
                workspace=workspace,
                code_file=code_file,
                timeout=timeout,
            )
            self.sandbox_image_identity = image_identity
            return success, log, raw_stdout, raw_stderr

        if self.sandbox == "linux_ns":
            sandbox = LinuxNSSandbox()
            task_env = self._get_task_environment(task_instance)
            success, log, raw_stdout, raw_stderr = sandbox.execute_python(
                workspace=workspace,
                code_file=code_file,
                timeout=timeout,
                python_executable=(
                    str(task_env.python_executable) if task_env else "python3"
                ),
                extra_env=task_env.build_subprocess_env() if task_env else None,
            )
            return success, log, raw_stdout, raw_stderr

        task_env = self._get_task_environment(task_instance)
        command = [str(task_env.python_executable), code_file] if task_env else ["python3", code_file]
        try:
            result = subprocess.run(
                command,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=task_env.build_subprocess_env() if task_env else None,
            )
            log = result.stdout + "\n" + result.stderr
            return result.returncode == 0, log, result.stdout, result.stderr
        except subprocess.TimeoutExpired as e:
            return False, f"Execution timed out ({timeout}s)", e.stdout or "", e.stderr or ""
        except Exception as e:
            return False, str(e), None, None

    def _collect_data_files(self, task_instance: TaskInstance) -> list[str]:
        """Collect data files produced in the workspace."""
        data_files = []
        for f in task_instance.metadata.get("output", {}).get("files", []):
            if f.get("type") == "data":
                fpath = task_instance.workspace_dir / f["name"]
                if fpath.exists():
                    data_files.append(f["name"])
        return data_files

    def _get_task_environment(self, task_instance: TaskInstance):
        return get_task_environment(self.task_env_manager, task_instance)
