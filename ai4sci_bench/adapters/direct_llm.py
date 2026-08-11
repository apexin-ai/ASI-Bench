"""Direct LLM API adapter — single-turn, no agentic tools."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from ai4sci_bench.adapters.subprocess_base import collect_output_files, get_task_environment
from ai4sci_bench.core.agent_interface import AgentAdapter
from ai4sci_bench.core.types import AgentOutput, CostInfo, RunStatus, TaskInstance
from ai4sci_bench.runner.linux_ns_sandbox import LinuxNSSandbox
from ai4sci_bench.runner.os_sandbox import OSSandbox
from ai4sci_bench.runner.sandbox_support import validate_sandbox_mode
from ai4sci_bench.runner.task_env import TaskEnvironmentManager


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
            response = litellm.completion(model=self.model, messages=messages, **kwargs)
            api_latency_ms = int((time.time() - api_t0) * 1000)
            content = response.choices[0].message.content

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
            )
        except Exception as e:
            elapsed = time.time() - t0
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
