"""Stdlib-only MCP preflight/launcher, executed inside the instance container.

This file is mounted, not imported from a host checkout. Diagnostics never
include server stdout, stderr, env values, headers, or exception text.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).parent
BASE_ENV = {"PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"}


def server_environment(entry):
    env = dict(BASE_ENV)
    env.update(entry.get("env", {}))
    # Never inherit agent credentials, proxy variables, or its HOME/config.
    home = tempfile.mkdtemp(prefix="mcp-home-")
    env.update(HOME=home, TMPDIR=home)
    return env


def message(method, request_id=None):
    value = {"jsonrpc": "2.0", "method": method}
    if request_id is not None:
        value["id"] = request_id
    if method == "initialize":
        value["params"] = {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "asibench-preflight", "version": "1"}}
    elif method == "tools/list":
        value["params"] = {}
    return value


def validate_reply(reply, request_id):
    if not isinstance(reply, dict) or reply.get("jsonrpc") != "2.0" or reply.get("id") != request_id or "error" in reply or not isinstance(reply.get("result"), dict):
        raise RuntimeError("invalid MCP response")
    result = reply["result"]
    if request_id == 1 and (not isinstance(result.get("capabilities"), dict) or not isinstance(result.get("protocolVersion"), str) or not isinstance(result.get("serverInfo"), dict)):
        raise RuntimeError("invalid initialize response")
    if request_id == 2 and not isinstance(result.get("tools"), list):
        raise RuntimeError("invalid tools/list response")


def probe_stdio(entry):
    env = server_environment(entry)
    try:
        process = subprocess.Popen(
            [entry["command"], *entry.get("args", [])],
            cwd=entry.get("cwd", "/workspace"), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, start_new_session=True,
        )
    except Exception:
        shutil.rmtree(env["HOME"], ignore_errors=True)
        raise
    deadline = time.monotonic() + entry["sandbox"].get("startup_timeout_seconds", 30)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    pending = b""

    def send(value):
        process.stdin.write(json.dumps(value).encode() + b"\n")
        process.stdin.flush()

    def receive(request_id):
        nonlocal pending
        while True:
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                reply = json.loads(line)
                if reply.get("id") == request_id:
                    validate_reply(reply, request_id)
                    return
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise TimeoutError("MCP startup timeout")
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk or len(pending) + len(chunk) > 4 * 1024 * 1024:
                raise RuntimeError("MCP invalid stream")
            pending += chunk

    try:
        send(message("initialize", 1))
        receive(1)
        send(message("notifications/initialized"))
        send(message("tools/list", 2))
        receive(2)
    finally:
        selector.close()
        # Kill the whole probe process group, including solver children.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()
        shutil.rmtree(env["HOME"], ignore_errors=True)


def probe_http(entry):
    """Probe streamable HTTP; service HEALTHCHECK alone is not MCP readiness."""
    timeout = entry["sandbox"].get("startup_timeout_seconds", 30)
    deadline = time.monotonic() + timeout
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

    def send(value):
        request = urllib.request.Request(entry["url"], data=json.dumps(value).encode(), headers=headers)
        with opener.open(request, timeout=max(0.1, deadline - time.monotonic())) as response:
            if response.headers.get("Mcp-Session-Id"):
                headers["Mcp-Session-Id"] = response.headers["Mcp-Session-Id"]
            if "id" not in value:
                return
            if "text/event-stream" in response.headers.get("Content-Type", ""):
                size = 0
                data = []
                while True:
                    line = response.readline(4 * 1024 * 1024 + 1)
                    if not line:
                        break
                    size += len(line)
                    if size > 4 * 1024 * 1024 or time.monotonic() > deadline:
                        raise TimeoutError("MCP startup timeout")
                    if line in (b"\n", b"\r\n") and data:
                        reply = json.loads(b"\n".join(data))
                        data.clear()
                        if reply.get("id") == value["id"]:
                            validate_reply(reply, value["id"])
                            return reply["result"]
                    elif line.startswith(b"data:"):
                        data.append(line[5:].strip())
                raise RuntimeError("MCP response missing")
            payload = response.read(4 * 1024 * 1024 + 1)
            reply = json.loads(payload)
            validate_reply(reply, value["id"])
            return reply["result"]
    try:
        initialized = send(message("initialize", 1))
        headers["MCP-Protocol-Version"] = initialized["protocolVersion"]
        send(message("notifications/initialized"))
        send(message("tools/list", 2))
    finally:
        if "Mcp-Session-Id" in headers:
            try:
                opener.open(urllib.request.Request(entry["url"], headers=headers, method="DELETE"), timeout=2).close()
            except Exception:
                pass


def main():
    servers = json.loads((ROOT / "source.json").read_text())
    if sys.argv[1] == "server":
        entry = servers[sys.argv[2]]
        env = server_environment(entry)
        os.chdir(entry.get("cwd", "/workspace"))
        os.execve(entry["command"], [entry["command"], *entry.get("args", [])], env)
    def startup_timeout(signum, frame):
        raise TimeoutError("MCP startup timeout")

    signal.signal(signal.SIGALRM, startup_timeout)
    for name, entry in servers.items():
        try:
            # A wall-clock limit also bounds slowly streaming HTTP bodies.
            signal.setitimer(signal.ITIMER_REAL, entry["sandbox"].get("startup_timeout_seconds", 30))
            if "command" in entry:
                probe_stdio(entry)
            else:
                probe_http(entry)
        except Exception:
            print(f"ASIBENCH_MCP_STARTUP_FAILED: {name}", file=sys.stderr)
            return 78
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    command = sys.argv[sys.argv.index("--") + 1:]
    os.execvp(command[0], command)


if __name__ == "__main__":
    sys.exit(main())
