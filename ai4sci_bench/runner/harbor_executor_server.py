"""Network-isolated second-stage execution service for Harbor verifier tasks."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import http.server
import json
import os
import re
import resource
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path("/opt/harbor-executor")
MAX_BODY = 64 * 1024 * 1024
MAX_FILE = 32 * 1024 * 1024
TAIL_BYTES = 64 * 1024
TOKEN = os.environ["ASIBENCH_HARBOR_EXECUTOR_TOKEN"]
REQUEST_SLOT = threading.BoundedSemaphore(1)


def _tail(value: bytes) -> str:
    return value[-TAIL_BYTES:].decode("utf-8", errors="replace")


def _safe_relative(name: str) -> Path:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != name
    ):
        raise ValueError(f"unsafe input path: {name!r}")
    return Path(*path.parts)


def _decode_files(payload: dict[str, Any], root: Path) -> None:
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("files must be a non-empty object")
    for name, encoded in files.items():
        if not isinstance(name, str) or not isinstance(encoded, str):
            raise ValueError("file names and contents must be strings")
        relative = _safe_relative(name)
        data = base64.b64decode(encoded, validate=True)
        if len(data) > MAX_FILE:
            raise ValueError(f"input file is too large: {name}")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _limits(timeout: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (timeout + 5, timeout + 5))
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE, MAX_FILE))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_NPROC, (512, 512))


def _execute(command: list[str], *, cwd: Path, timeout: int) -> tuple[bool, int | None, bytes, bytes]:
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": str(cwd),
                "TMPDIR": str(cwd),
                "LANG": "C.UTF-8",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            preexec_fn=lambda: _limits(timeout),
            extra_groups=(),
            user=65534,
            group=65534,
        )
        timed_out = False
        try:
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                exit_code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                exit_code = process.wait()
        _kill_untrusted_processes()
        stdout.seek(0, os.SEEK_END)
        stdout_size = stdout.tell()
        stdout.seek(max(0, stdout_size - TAIL_BYTES))
        stdout_tail = stdout.read(TAIL_BYTES)
        stderr.seek(0, os.SEEK_END)
        stderr_size = stderr.tell()
        stderr.seek(max(0, stderr_size - TAIL_BYTES))
        stderr_tail = stderr.read(TAIL_BYTES)
    return timed_out, exit_code, stdout_tail, stderr_tail


def _kill_untrusted_processes() -> None:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            status = (entry / "status").read_text(encoding="ascii")
            uid_line = next(line for line in status.splitlines() if line.startswith("Uid:"))
            if int(uid_line.split()[1]) == 65534:
                os.kill(int(entry.name), signal.SIGKILL)
        except (OSError, StopIteration, ValueError, ProcessLookupError):
            continue


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    kind = payload.get("kind")
    timeout = payload.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 3600:
        raise ValueError("timeout_seconds must be an integer from 1 to 3600")
    with tempfile.TemporaryDirectory(prefix="harbor-executor-") as temporary:
        root = Path(temporary)
        root.chmod(0o711)
        run_root = root / "run"
        run_root.mkdir(mode=0o700)
        _decode_files(payload, run_root)
        initial_files = {
            path.relative_to(run_root).as_posix()
            for path in run_root.rglob("*")
            if path.is_file()
        }
        run_root.chmod(0o777)
        for path in run_root.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o666)
        output_name = None
        if kind == "solver":
            task_kind = payload.get("task_kind")
            expected = {
                "tsp": ("instance.tsp", "tour.npy", 271828182, 3600),
                "max3sat": ("instance.cnf", "assignment.npy", 314159265, 1800),
            }
            if task_kind not in expected:
                raise ValueError("unknown solver task_kind")
            input_name, output_name, required_seed, required_timeout = expected[task_kind]
            seed = payload.get("seed")
            if seed != required_seed or timeout != required_timeout:
                raise ValueError("solver resource contract does not match task contract")
            if set(path.name for path in run_root.iterdir()) != {"solver.py", input_name}:
                raise ValueError("solver request must contain exactly solver.py and its private input")
            output_name = "tour.npy" if task_kind == "tsp" else "assignment.npy"
            command = [
                "python", "solver.py", "--input", input_name,
                "--output", output_name,
                "--seed", str(seed), "--time-limit-seconds", str(timeout),
            ]
        elif kind == "ngspice":
            netlist = payload.get("netlist")
            relative = _safe_relative(netlist) if isinstance(netlist, str) else None
            if relative is None or len(relative.parts) != 1 or not (run_root / relative).is_file():
                raise ValueError("netlist must name a root-level uploaded file")
            command = ["ngspice", "-b", netlist]
        else:
            raise ValueError("kind must be solver or ngspice")
        started = time.monotonic()
        try:
            timed_out, exit_code, stdout, stderr = _execute(
                command, cwd=run_root, timeout=timeout
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                "ok": True,
                "timed_out": False,
                "exit_code": None,
                "elapsed_seconds": time.monotonic() - started,
                "stdout_tail": "",
                "stderr_tail": "",
                "artifacts": {},
                "infrastructure_error": str(exc),
            }
        artifacts = {}
        if output_name:
            output = run_root / output_name
            if output.is_file() and not output.is_symlink() and output.stat().st_size <= MAX_FILE:
                artifacts[output_name] = base64.b64encode(output.read_bytes()).decode("ascii")
        elif kind == "ngspice":
            for output in sorted(run_root.rglob("*")):
                relative = output.relative_to(run_root).as_posix()
                if (
                    relative not in initial_files
                    and output.is_file()
                    and not output.is_symlink()
                    and output.stat().st_size <= MAX_FILE
                ):
                    artifacts[relative] = base64.b64encode(output.read_bytes()).decode("ascii")
        return {
            "ok": True,
            "timed_out": timed_out,
            "exit_code": exit_code,
            "elapsed_seconds": time.monotonic() - started,
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
            "artifacts": artifacts,
            "infrastructure_error": None,
        }


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        body = b"ok\n"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/run" or not hmac.compare_digest(
            self.headers.get("Authorization", ""), f"Bearer {TOKEN}"
        ):
            self.send_error(403)
            return
        if not REQUEST_SLOT.acquire(blocking=False):
            self.send_error(429)
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if not 0 < length <= MAX_BODY:
                raise ValueError("request body is too large or empty")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            result = _run(payload)
            status = 200
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            status = 400
        finally:
            REQUEST_SLOT.release()
        body = json.dumps(result, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


if __name__ == "__main__":
    http.server.ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
