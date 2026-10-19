"""Client for the isolated Harbor verifier execution sidecar."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ai4sci_bench.runner.submission_sandbox import SubmissionRunResult


class HarborExecutorError(RuntimeError):
    """The isolated Harbor execution service is unavailable or rejected work."""


def _post(payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    endpoint = os.environ.get("ASIBENCH_HARBOR_EXECUTOR_URL", "").rstrip("/")
    token = os.environ.get("ASIBENCH_HARBOR_EXECUTOR_TOKEN", "")
    if not endpoint or not token:
        raise HarborExecutorError("Harbor executor URL and token are required")
    request = urllib.request.Request(
        endpoint + "/run",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout + 30) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HarborExecutorError(
            f"Harbor executor returned HTTP {exc.code}: {detail}"
        ) from exc
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise HarborExecutorError(f"Harbor executor request failed: {exc}") from exc
    if not isinstance(result, dict) or result.get("ok") is not True:
        message = result.get("error", "invalid executor response") if isinstance(result, dict) else "invalid executor response"
        raise HarborExecutorError(str(message))
    return result


def _encoded_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise HarborExecutorError(f"Executor input is missing or unsafe: {path.name}")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def run_solver(spec: Any, *, input_name: str, seed: int, task_kind: str) -> SubmissionRunResult:
    """Run one solver in the verifier's isolated Compose sidecar."""
    if task_kind not in {"tsp", "max3sat"}:
        raise HarborExecutorError(f"Unsupported solver kind: {task_kind}")
    if len(spec.readonly_inputs) != 1:
        raise HarborExecutorError("Solver execution requires exactly one private input")
    submission = Path(spec.submission_dir)
    solver_path = submission / "solver.py"
    private_input = spec.readonly_inputs[0].host_path
    payload = _post(
        {
            "kind": "solver",
            "task_kind": task_kind,
            "seed": seed,
            "timeout_seconds": int(spec.timeout_seconds),
            "files": {
                "solver.py": _encoded_file(solver_path),
                input_name: _encoded_file(private_input),
            },
        },
        timeout=float(spec.timeout_seconds),
    )
    artifacts = tuple(
        (name, base64.b64decode(value, validate=True))
        for name, value in payload.get("artifacts", {}).items()
    )
    return SubmissionRunResult(
        launched=True,
        timed_out=bool(payload.get("timed_out")),
        exit_code=payload.get("exit_code"),
        elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
        stdout_tail=str(payload.get("stdout_tail", "")),
        stderr_tail=str(payload.get("stderr_tail", "")),
        output_artifacts=artifacts,
        image_identity="harbor-isolated-executor",
        infrastructure_error=payload.get("infrastructure_error"),
    )


def run_ngspice(work_dir: Path, netlist_name: str, timeout_seconds: int):
    """Run one generated evaluator bench in the isolated executor sidecar."""
    files: dict[str, str] = {}
    for path in sorted(work_dir.rglob("*")):
        if path.is_symlink():
            raise HarborExecutorError(f"Unsafe ngspice input: {path.relative_to(work_dir)}")
        if path.is_file():
            relative = path.relative_to(work_dir).as_posix()
            files[relative] = _encoded_file(path)
    payload = _post(
        {
            "kind": "ngspice",
            "netlist": netlist_name,
            "timeout_seconds": timeout_seconds,
            "files": files,
        },
        timeout=float(timeout_seconds),
    )
    import subprocess

    for relative, encoded in payload.get("artifacts", {}).items():
        target = Path(relative)
        if target.is_absolute() or ".." in target.parts or target.is_symlink():
            raise HarborExecutorError("Executor returned an unsafe output path")
        destination = work_dir / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(base64.b64decode(encoded, validate=True))

    return subprocess.CompletedProcess(
        args=["ngspice", "-b", netlist_name],
        returncode=int(payload.get("exit_code", 1)),
        stdout=str(payload.get("stdout_tail", "")),
        stderr=str(payload.get("stderr_tail", "")),
    )
