"""Shared helpers for the per-server ``smoke_<id>.py`` scripts (stdlib only)."""
from __future__ import annotations

import importlib.metadata as md
import platform
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from stdio_client import MCPError, StdioMCP, text_of  # noqa: E402


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, level: str, name: str, status: str, detail: str = "", **data) -> None:
        self.checks.append({"level": level, "name": name, "status": status, "detail": detail, **data})
        print(f"[{status:<4}] {level} {name}" + (f" — {detail}" if detail else ""), flush=True)

    @property
    def failed(self) -> bool:
        return any(c["status"] == "FAIL" for c in self.checks)


class Caller:
    """tools/call wrapper: records FAILs for transport/tool errors and stdout lines per tool."""

    def __init__(self, client: StdioMCP, report: Report) -> None:
        self.client = client
        self.report = report
        self.stdout_by_tool: Counter = Counter()

    def __call__(self, check: str, tool: str, arguments: dict, *, allow_error: bool = False) -> dict | None:
        before = len(self.client.non_json_stdout)
        try:
            resp = self.client.call_tool(tool, arguments)
        except MCPError as exc:
            self.report.add("L1", check, "FAIL", str(exc))
            return None
        finally:
            self.stdout_by_tool[tool] += len(self.client.non_json_stdout) - before
        if "error" in resp:
            self.report.add("L1", check, "FAIL", f"JSON-RPC error: {resp['error']}")
            return None
        result = resp["result"]
        if result.get("isError") and not allow_error:
            self.report.add("L1", check, "FAIL", f"tool error: {text_of(result)[:300]}")
            return None
        return result


def package_versions(packages: tuple[str, ...]) -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in packages:
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


__all__ = ["Caller", "MCPError", "Report", "StdioMCP", "package_versions", "text_of"]
