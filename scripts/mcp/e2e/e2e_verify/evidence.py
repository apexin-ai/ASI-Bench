"""Evidence of what an agent did, from the persisted run artefacts.

One parser per harness log format turns the raw stdout into :class:`Evidence`;
the normalised trajectory is the adapter-neutral fallback.

``asibench run`` saves the raw stdout sanitized: user events (Claude tool
results) are redacted and absolute host paths become ``<abs_path>``. The
trajectory is extracted from the unsanitized stdout, so :func:`load_evidence`
fills in tool results and the as-executed text of shell commands from it. The parser is chosen
from the result's ``agent_name`` (:data:`HARNESSES`); an unknown agent falls
back to trying each format in turn. To support another harness, add a parser
to :data:`PARSERS` and map its adapter class in :data:`HARNESSES`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

SCRUBBED = "<abs_path>"
"""What ``asibench run`` leaves in a persisted artefact in place of an absolute host path."""


@dataclass
class ToolCall:
    id: str | None = None
    name: str = ""
    input: Any = None                   # dict of tool arguments; None = not observable
    result_text: str | None = None      # None = no result observed
    is_error: bool | None = None
    content_types: list[str] | None = None   # content block types; None = not observable
    media_types: list[str] | None = None     # image media types


@dataclass
class Command:
    """One shell command: as persisted (paths may be ``<abs_path>``) and, when the
    trajectory has it, as executed."""
    id: str | None
    text: str
    raw: str | None = None

    @property
    def scrubbed(self) -> bool:
        return SCRUBBED in self.text and self.raw is None


@dataclass
class Evidence:
    source: str
    mcp_servers: dict[str, str] | None = None  # name -> status; None = not observable
    tools_offered: list[str] | None = None
    calls: list[ToolCall] = field(default_factory=list)
    commands: list[Command] = field(default_factory=list)
    assistant_text: list[str] = field(default_factory=list)
    permission_mode: str | None = None
    final_result: dict | None = None

    @property
    def bash_commands(self) -> list[str]:
        """Shell commands as persisted."""
        return [c.text for c in self.commands]


def _json_lines(path: Path):
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(b.get("text", "")) for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _block_types(content) -> tuple[list[str] | None, list[str] | None]:
    """Content block types and image media types of a tool_result payload."""
    if isinstance(content, str):
        return (["text"], []) if content != "<redacted>" else (None, None)
    if not isinstance(content, list):
        return None, None
    types, media = [], []
    for block in content:
        if not isinstance(block, dict):
            continue
        types.append(str(block.get("type", "")))
        if block.get("type") == "image":
            source = block.get("source") if isinstance(block.get("source"), dict) else {}
            media.append(str(source.get("media_type") or block.get("media_type") or block.get("mimeType") or ""))
    return types, media


# --------------------------------------------------------------------------
# Claude Code stream-json
# --------------------------------------------------------------------------

def _content_blocks(event: dict) -> list[dict]:
    """Content blocks of an assistant/user event, tolerating string payloads."""
    message = event.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    return [block for block in content if isinstance(block, dict)] if isinstance(content, list) else []


def parse_claude_stream(path: Path) -> Evidence | None:
    """Parse Claude Code ``--output-format stream-json``; None if nothing in it is such an event."""
    ev = Evidence(source=f"claude stream-json ({path.name})")
    by_id: dict[str, ToolCall] = {}
    for event in _json_lines(path):
        etype = event.get("type")
        if etype == "system" and event.get("subtype") == "init":
            servers = event.get("mcp_servers") or []
            ev.mcp_servers = {s.get("name"): s.get("status") for s in servers if isinstance(s, dict)}
            ev.tools_offered = [t for t in event.get("tools", []) if isinstance(t, str)]
            ev.permission_mode = event.get("permissionMode")
        elif etype == "assistant":
            for block in _content_blocks(event):
                if block.get("type") == "tool_use":
                    call = ToolCall(block.get("id"), block.get("name", ""), block.get("input") or {})
                    ev.calls.append(call)
                    if call.id:
                        by_id[call.id] = call
                    if call.name == "Bash" and isinstance(call.input, dict):
                        ev.commands.append(Command(call.id, str(call.input.get("command", ""))))
                elif block.get("type") == "text":
                    ev.assistant_text.append(str(block.get("text", "")))
        elif etype == "user":
            for block in _content_blocks(event):
                if block.get("type") == "tool_result":
                    call = by_id.get(block.get("tool_use_id"))
                    if call is not None:
                        call.result_text = _text_of(block.get("content"))
                        call.is_error = bool(block.get("is_error"))
                        call.content_types, call.media_types = _block_types(block.get("content"))
        elif etype == "result":
            ev.final_result = {k: event.get(k) for k in ("subtype", "is_error", "num_turns")}
            ev.final_result["result"] = str(event.get("result") or "")[:1000]
    if ev.mcp_servers is None and not ev.calls and ev.final_result is None:
        return None
    return ev


# --------------------------------------------------------------------------
# Codex exec --json
# --------------------------------------------------------------------------

def _codex_arguments(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def parse_codex_stream(path: Path) -> Evidence | None:
    """Parse ``codex exec --json`` output; None if the file is not such a log.

    MCP calls are ``mcp_tool_call`` items carrying server, tool, arguments and
    (on ``item.completed``) result or error. They are named
    ``mcp__<server>__<tool>`` here, as in Claude Code. There is no event listing
    the connected servers or offered tools. Codex's own ``list_mcp_resources`` /
    ``list_mcp_resource_templates`` calls appear under the server name too; they
    are kept in the tool sequence but never match a required tool.
    """
    ev = Evidence(source=f"codex exec JSONL ({path.name})")
    by_id: dict[str, ToolCall] = {}
    recognised = False

    def call_for(item: dict, name: str, inputs) -> tuple[ToolCall, bool]:
        call = by_id.get(item.get("id")) if item.get("id") else None
        if call is not None:
            return call, False
        call = ToolCall(item.get("id"), name, inputs)
        ev.calls.append(call)
        if call.id:
            by_id[call.id] = call
        return call, True

    for event in _json_lines(path):
        etype = event.get("type")
        if etype in ("thread.started", "turn.started"):
            recognised = True
        elif etype in ("turn.completed", "turn.failed"):
            recognised = True
            error = event.get("error")
            message = error.get("message") if isinstance(error, dict) else error
            ev.final_result = {"subtype": etype, "is_error": etype == "turn.failed", "num_turns": None,
                               "result": str(message or "")[:1000]}
        item = event.get("item")
        if etype not in ("item.started", "item.updated", "item.completed") or not isinstance(item, dict):
            continue
        recognised = True
        kind, done = item.get("type"), etype == "item.completed"
        if kind == "agent_message" and done:
            ev.assistant_text.append(str(item.get("text", "")))
        elif kind == "command_execution":
            command = str(item.get("command", ""))
            call, created = call_for(item, "command_execution", {"command": command})
            if created:
                ev.commands.append(Command(call.id, command))
            if done:
                call.result_text = str(item.get("aggregated_output") or "")
                call.is_error = item.get("exit_code") not in (None, 0)
                call.content_types, call.media_types = ["text"], []
        elif kind in ("web_search", "web_fetch"):
            details = {k: v for k, v in item.items() if k not in ("id", "type", "status")}
            call, _created = call_for(item, kind, details)
            if done:
                call.input = details
                call.is_error = item.get("status") == "failed"
                call.result_text = ""
                call.content_types, call.media_types = [], []
        elif kind == "mcp_tool_call":
            name = f"mcp__{item.get('server', '')}__{item.get('tool', '')}"
            call, _created = call_for(item, name, _codex_arguments(item.get("arguments")))
            if not done:
                continue
            result, error = item.get("result"), item.get("error")
            if error is not None or item.get("status") == "failed" or not isinstance(result, dict):
                call.is_error = True
                call.result_text = str(error.get("message", error) if isinstance(error, dict) else error or "")
                continue
            content = result.get("content")
            call.is_error = bool(result.get("is_error") or result.get("isError"))
            call.result_text = _text_of(content)
            call.content_types, call.media_types = _block_types(content)
            structured = result.get("structured_content")
            if not call.result_text and structured is not None:
                call.result_text = json.dumps(structured)
    return ev if recognised else None


# --------------------------------------------------------------------------
# Normalised trajectory (fallback, and source of redacted tool results)
# --------------------------------------------------------------------------

def _trajectory_steps(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    steps = data.get("steps", []) if isinstance(data, dict) else data  # run writes a bare list
    return [s for s in steps if isinstance(s, dict)] if isinstance(steps, list) else []


def _apply_result(call: ToolCall, step: dict) -> None:
    meta = step.get("metadata") or {}
    call.result_text = step.get("content", "")
    call.is_error = bool(meta.get("is_error"))
    call.content_types = meta.get("content_types")
    call.media_types = meta.get("image_media_types")


def parse_trajectory(path: Path) -> Evidence:
    """Adapter-neutral fallback: tool names and results, but no tool inputs."""
    ev = Evidence(source=f"trajectory ({path.name})")
    by_id: dict[str, ToolCall] = {}
    for step in _trajectory_steps(path):
        meta = step.get("metadata") or {}
        if step.get("step_type") == "tool_call":
            call = ToolCall(meta.get("tool_call_id"), meta.get("tool_name", ""), None)
            ev.calls.append(call)
            if call.id:
                by_id[call.id] = call
            command = (meta.get("key_args") or {}).get("command")
            if command:
                ev.commands.append(Command(call.id, str(command), str(command)))
        elif step.get("step_type") == "tool_result":
            call = by_id.get(meta.get("tool_call_id"))
            if call is not None:
                _apply_result(call, step)
    return ev


def _step_id(meta: dict) -> str | None:
    return meta.get("tool_call_id") or meta.get("item_id")


def enrich_commands_from_trajectory(ev: Evidence, path: Path) -> int:
    """Attach the as-executed text of shell commands (trajectory ``key_args.command``,
    matched by call id) to the persisted, path-scrubbed ones; return how many matched."""
    raw = {}
    for step in _trajectory_steps(path):
        meta = step.get("metadata") or {}
        command = (meta.get("key_args") or {}).get("command")
        if step.get("step_type") == "tool_call" and command and _step_id(meta):
            raw[_step_id(meta)] = str(command)
    matched = 0
    for command in ev.commands:
        if command.id in raw:
            command.raw = raw[command.id]
            matched += 1
    return matched


def enrich_results_from_trajectory(ev: Evidence, path: Path) -> int:
    """Fill tool results missing or scrubbed in the persisted stream; return how many were filled.

    `asibench run` redacts the content of every user-role event in the saved
    stream-json (prompt protection), which also removes tool_result payloads, and
    replaces absolute host paths with ``<abs_path>`` everywhere. The Codex JSONL
    keeps its MCP results, so a tool returning a path (``mol_to_sdf``) survives
    persistence only as ``<abs_path>``. The trajectory is extracted from the
    unsanitized stream and keeps the result text and content block types, keyed by
    the same tool_call_id: it is the source for both cases.
    """
    steps = {}
    for step in _trajectory_steps(path):
        meta = step.get("metadata") or {}
        if step.get("step_type") == "tool_result" and meta.get("tool_call_id"):
            steps[meta["tool_call_id"]] = step
    filled = 0
    for call in ev.calls:
        lost = call.result_text in (None, "<redacted>") or SCRUBBED in (call.result_text or "")
        if lost and call.id in steps:
            _apply_result(call, steps[call.id])
            filled += 1
    return filled


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

PARSERS: dict[str, Callable[[Path], Evidence | None]] = {
    "claude": parse_claude_stream,
    "codex": parse_codex_stream,
}

# EvalResult.agent_name (the adapter class) -> raw stdout format.
HARNESSES: dict[str, str] = {
    "ClaudeCodeCLIAdapter": "claude",
    "CodexCLIAdapter": "codex",
}


def parse_stdout(path: Path, agent_name: str | None) -> Evidence | None:
    """Parse a raw stdout log with the agent's parser, or by trying each known format."""
    harness = HARNESSES.get(agent_name or "")
    for name in ([harness] if harness else PARSERS):
        ev = PARSERS[name](path)
        if ev is not None:
            return ev
    return None


def load_evidence(result_path: Path, result: dict) -> Evidence | None:
    """Evidence for one run result: the raw stdout (tool results the stream lost filled
    in from the trajectory), else the trajectory alone; None if neither exists."""
    run_dir = result_path.parent
    agent = result.get("agent_output") or {}
    stdout_file, traj_file = agent.get("raw_stdout_file"), agent.get("trajectory_file")
    traj = run_dir / traj_file if traj_file and (run_dir / traj_file).is_file() else None
    ev = None
    if stdout_file and (run_dir / stdout_file).is_file():
        ev = parse_stdout(run_dir / stdout_file, result.get("agent_name"))
    if ev is None:
        return parse_trajectory(traj) if traj else None
    if traj:
        filled = enrich_results_from_trajectory(ev, traj)
        if filled:
            ev.source += f" + {filled} tool result(s) from {traj_file}"
        enrich_commands_from_trajectory(ev, traj)
    return ev
