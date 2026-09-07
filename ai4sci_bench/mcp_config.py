"""Validated MCP configuration and the bundled science-tool catalog."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_SERVER_FIELDS = frozenset({"type", "command", "args", "env", "cwd", "url", "headers"})
_STRING_MAP_FIELDS = ("env", "headers")


def load_mcp_config(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a portable ``mcpServers`` JSON file and fail closed on ambiguity."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file() or config_path.is_symlink():
        raise ValueError(f"MCP config must be a regular file: {config_path}")
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read MCP config {config_path}: {exc}") from exc
    if not isinstance(document, dict) or set(document) != {"mcpServers"}:
        raise ValueError("MCP config must contain only a top-level 'mcpServers' object")
    servers = document["mcpServers"]
    if not isinstance(servers, dict) or not servers:
        raise ValueError("MCP config must define at least one server")

    normalized: dict[str, dict[str, Any]] = {}
    for name, raw in servers.items():
        if not isinstance(name, str) or not _SERVER_NAME_RE.fullmatch(name):
            raise ValueError(
                f"Invalid MCP server name {name!r}; use letters, digits, '_' or '-'"
            )
        if not isinstance(raw, dict):
            raise ValueError(f"MCP server '{name}' must be an object")
        unknown = sorted(set(raw) - _SERVER_FIELDS)
        if unknown:
            raise ValueError(
                f"MCP server '{name}' has unsupported field(s): {', '.join(unknown)}"
            )
        has_command = isinstance(raw.get("command"), str) and bool(raw["command"])
        has_url = isinstance(raw.get("url"), str) and bool(raw["url"])
        if has_command == has_url:
            raise ValueError(
                f"MCP server '{name}' must define exactly one of 'command' or 'url'"
            )
        entry = dict(raw)
        if has_command:
            if "type" in entry and entry["type"] not in ("stdio", None):
                raise ValueError(f"MCP server '{name}' command transport must be 'stdio'")
            entry.pop("type", None)
            args = entry.get("args", [])
            if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
                raise ValueError(f"MCP server '{name}' args must be a list of strings")
            entry["args"] = args
            cwd = entry.get("cwd")
            if cwd is not None and not isinstance(cwd, str):
                raise ValueError(f"MCP server '{name}' cwd must be a string")
        else:
            if entry.get("type") not in (None, "http", "streamable-http", "sse"):
                raise ValueError(f"MCP server '{name}' has an unsupported HTTP transport")
            entry.pop("type", None)
            if any(field in entry for field in ("args", "cwd")):
                raise ValueError(f"MCP server '{name}' URL transport cannot define args/cwd")
        for field in _STRING_MAP_FIELDS:
            value = entry.get(field)
            if value is not None and (
                not isinstance(value, dict)
                or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items())
            ):
                raise ValueError(f"MCP server '{name}' {field} must map strings to strings")
        normalized[name] = entry
    return normalized


def claude_mcp_tool_patterns(servers: dict[str, dict[str, Any]]) -> str:
    """Return Claude Code allowlist patterns for explicit MCP servers."""
    return ",".join(f"mcp__{name}__*" for name in sorted(servers))


def codex_mcp_toml(servers: dict[str, dict[str, Any]]) -> str:
    """Render portable server definitions into Codex ``config.toml`` syntax."""
    lines: list[str] = []
    for name, entry in servers.items():
        lines.append(f"[mcp_servers.{name}]")
        if "command" in entry:
            lines.append(f"command = {_toml_string(entry['command'])}")
            lines.append("args = [" + ", ".join(_toml_string(x) for x in entry["args"]) + "]")
            if entry.get("cwd"):
                lines.append(f"cwd = {_toml_string(entry['cwd'])}")
        else:
            lines.append(f"url = {_toml_string(entry['url'])}")
        if entry.get("headers"):
            rendered = ", ".join(
                f"{_toml_key(key)} = {_toml_string(value)}"
                for key, value in entry["headers"].items()
            )
            lines.append(f"http_headers = {{ {rendered} }}")
        for field in ("env",):
            values = entry.get(field)
            if not values:
                continue
            lines.append(f"[mcp_servers.{name}.{field}]")
            for key, value in values.items():
                lines.append(f"{_toml_key(key)} = {_toml_string(value)}")
        lines.append("")
    return "\n".join(lines)


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_key(value: str) -> str:
    return value if re.fullmatch(r"[A-Za-z0-9_-]+", value) else _toml_string(value)


def science_mcp_catalog_path() -> Path:
    return Path(__file__).with_name("data") / "science_mcp_catalog.json"


def load_science_mcp_catalog() -> dict[str, dict[str, Any]]:
    """Load the shipped catalog of operator-installed scientific MCP servers."""
    data = json.loads(science_mcp_catalog_path().read_text(encoding="utf-8"))
    return data["servers"]
