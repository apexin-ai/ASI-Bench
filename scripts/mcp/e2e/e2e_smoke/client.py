"""Minimal stdlib-only MCP stdio client for E2E smoke tests.

Deliberately does not use the MCP Python SDK: it reads the server's raw stdout
line by line so that non-JSON output (e.g. a library printing to stdout, which
corrupts the stdio transport for some clients) is recorded verbatim instead of
being swallowed by an SDK.
"""
from __future__ import annotations

import itertools
import json
import queue
import subprocess
import threading
import time
from pathlib import Path

PROTOCOL_VERSION = "2025-03-26"


class MCPError(RuntimeError):
    pass


class StdioMCP:
    def __init__(self, command: str, args: list[str], *, cwd: Path, env: dict[str, str], stderr_path: Path):
        self._stderr = stderr_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [command, *args], cwd=str(cwd), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        self._ids = itertools.count(1)
        self._inbox: queue.Queue = queue.Queue()
        self.non_json_stdout: list[str] = []
        self.notifications: list[dict] = []
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self.non_json_stdout.append(line[:500])
                continue
            if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
                self.non_json_stdout.append(line[:500])
                continue
            self._inbox.put(msg)
        self._inbox.put(None)  # EOF sentinel

    def _send(self, payload: dict) -> None:
        if self.proc.poll() is not None:
            raise MCPError(f"server exited with code {self.proc.returncode}")
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

    def notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})})

    def request(self, method: str, params: dict | None = None, timeout: float = 120.0) -> dict:
        rid = next(self._ids)
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MCPError(f"timeout after {timeout:.0f}s waiting for {method}")
            try:
                msg = self._inbox.get(timeout=remaining)
            except queue.Empty:
                continue
            if msg is None:
                raise MCPError(f"server closed stdout (exit={self.proc.poll()}) while waiting for {method}")
            if "method" in msg and "id" in msg:  # server -> client request
                reply = {"result": {}} if msg["method"] == "ping" else {
                    "error": {"code": -32601, "message": "not supported by smoke client"}}
                self._send({"jsonrpc": "2.0", "id": msg["id"], **reply})
                continue
            if "method" in msg:
                self.notifications.append(msg)
                continue
            if msg.get("id") == rid:
                return msg

    def initialize(self, timeout: float = 180.0) -> dict:
        resp = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "asibench-mcp-e2e-smoke", "version": "1"},
        }, timeout=timeout)
        if "error" in resp:
            raise MCPError(f"initialize failed: {resp['error']}")
        self.notify("notifications/initialized")
        return resp["result"]

    def list_tools(self) -> list[dict]:
        tools, cursor, seen = [], None, set()
        while True:
            resp = self.request("tools/list", {"cursor": cursor} if cursor else {})
            if "error" in resp:
                raise MCPError(f"tools/list failed: {resp['error']}")
            tools.extend(resp["result"].get("tools", []))
            cursor = resp["result"].get("nextCursor")
            if not cursor:
                return tools
            if cursor in seen:
                raise MCPError("tools/list repeated a pagination cursor")
            seen.add(cursor)

    def call_tool(self, name: str, arguments: dict, timeout: float = 300.0) -> dict:
        """Return the raw JSON-RPC response (result or error) for tools/call."""
        return self.request("tools/call", {"name": name, "arguments": arguments}, timeout=timeout)

    def close(self) -> int | None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        finally:
            self._stderr.close()
        return self.proc.returncode


def text_of(result: dict) -> str:
    """Concatenate text content blocks of a tools/call result."""
    return "".join(b.get("text", "") for b in result.get("content", []) if b.get("type") == "text")
