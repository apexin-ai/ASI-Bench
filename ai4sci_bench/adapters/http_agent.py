"""HTTP API agent adapter — communicates with agents via HTTP endpoints."""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

from ai4sci_bench.core.agent_interface import AgentAdapter
from ai4sci_bench.core.logger import get_logger
from ai4sci_bench.core.types import AgentOutput, RunStatus, TaskInstance

logger = get_logger(__name__)


class HTTPAgentAdapter(AgentAdapter):
    """Communicate with an agent via HTTP API.

    The agent server must implement:
      POST /solve
        Body: {
          "task_id": str,
          "prompt_level": str,
          "prompt": str,           # contents of prompt.md
          "task_info": dict,       # agent-facing contents of task_info.json
          "input_files": {         # base64-encoded input files
            "filename": "base64data",
            ...
          }
        }
        Response: {
          "status": "completed" | "failed",
          "code_files": {"filename": "content", ...},
          "data_files": {"filename": "base64data", ...},
          "log": str,
          "error_message": str | null
        }

    Config example:
        adapter = HTTPAgentAdapter(
            base_url="http://localhost:8080",
            timeout=10800,
        )
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 10800,
        headers: dict[str, str] | None = None,
        api_key: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = headers or {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.headers.setdefault("Content-Type", "application/json")

    def solve(self, task_instance: TaskInstance) -> AgentOutput:
        workspace = task_instance.workspace_dir
        prompt = (workspace / "prompt.md").read_text(encoding="utf-8")

        # Read task_info.json if exists
        task_info_path = workspace / "task_info.json"
        task_info = {}
        if task_info_path.exists():
            task_info = json.loads(task_info_path.read_text(encoding="utf-8"))

        # Encode input files
        import base64
        input_files = {}
        data_dir = workspace / "data"
        if data_dir.exists():
            for f in data_dir.iterdir():
                if f.is_file():
                    input_files[f.name] = base64.b64encode(f.read_bytes()).decode("ascii")

        # NOTE: instance_id is intentionally omitted from the payload.
        # It encodes parameters and seed (e.g. "task__grid255_seed42"),
        # which would leak evaluation-sensitive information to remote agents.
        payload = {
            "task_id": task_instance.task_id,
            "prompt_level": task_instance.prompt_level.value,
            "prompt": prompt,
            "task_info": task_info,
            "input_files": input_files,
        }

        eff_timeout = self._get_effective_timeout(task_instance)

        t0 = time.time()
        try:
            response = self._post("/solve", payload, timeout=eff_timeout)
            elapsed = time.time() - t0

            # Write output files to workspace
            code_files = []
            for name, content in response.get("code_files", {}).items():
                (workspace / name).write_text(content, encoding="utf-8")
                code_files.append(name)

            data_files = []
            for name, b64data in response.get("data_files", {}).items():
                (workspace / name).write_bytes(base64.b64decode(b64data))
                data_files.append(name)

            status_str = response.get("status", "completed")
            status = RunStatus.COMPLETED if status_str == "completed" else RunStatus.FAILED

            trajectory_data = response.get("trajectory")
            raw_stdout = response.get("raw_stdout")
            raw_stdout_format = response.get("raw_stdout_format")

            if trajectory_data and not raw_stdout:
                import json as _json
                if isinstance(trajectory_data, list):
                    raw_stdout = "\n".join(
                        _json.dumps(step, ensure_ascii=False) for step in trajectory_data
                    )
                    raw_stdout_format = "jsonl"

            return AgentOutput(
                instance_id=task_instance.instance_id,
                output_dir=workspace,
                code_files=code_files,
                data_files=data_files,
                log=response.get("log", ""),
                execution_time_seconds=elapsed,
                status=status,
                error_message=response.get("error_message"),
                raw_stdout=raw_stdout,
                raw_stderr=response.get("raw_stderr"),
                raw_stdout_format=raw_stdout_format,
            )

        except urllib.error.URLError as e:
            elapsed = time.time() - t0
            return AgentOutput(
                instance_id=task_instance.instance_id,
                output_dir=workspace,
                code_files=[],
                data_files=[],
                log=f"HTTP request failed: {e}",
                execution_time_seconds=elapsed,
                status=RunStatus.FAILED,
                error_message=str(e),
            )

        except Exception as e:
            elapsed = time.time() - t0
            return AgentOutput(
                instance_id=task_instance.instance_id,
                output_dir=workspace,
                code_files=[],
                data_files=[],
                log=f"HTTP agent error: {e}",
                execution_time_seconds=elapsed,
                status=RunStatus.FAILED,
                error_message=str(e),
            )

    def _post(
        self,
        endpoint: str,
        payload: dict[str, Any],
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Send a POST request and return the JSON response."""
        url = f"{self.base_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers=self.headers, method="POST")

        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
