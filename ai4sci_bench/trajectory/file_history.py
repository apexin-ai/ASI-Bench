"""Extract file modification history from JSONL sidecar files."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class FileVersion:
    """One version of a file in the agent's modification history."""

    version: int
    action: str  # "write" | "edit" | "notebook_edit" | "delete" | "file_change"
    step_index: int
    timestamp_ms: int | None
    content_hash: str
    content_length: int
    diff_summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _coerce_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _timestamp_ms(event: dict[str, Any]) -> int | None:
    ts_raw = event.get("timestamp")
    if ts_raw is None:
        return None
    try:
        return int(float(ts_raw) * 1000)
    except (TypeError, ValueError):
        return None


def _codex_file_change_action(kind: str) -> str:
    normalized = kind.lower()
    if normalized in {"add", "create", "write"}:
        return "write"
    if normalized in {"delete", "remove"}:
        return "delete"
    if normalized in {"modify", "update", "edit", "patch"}:
        return "edit"
    return "file_change"


def _codex_file_change_content(change: dict[str, Any]) -> str:
    for key in (
        "content",
        "new_content",
        "new_text",
        "after",
        "diff",
        "patch",
        "old_content",
        "old_text",
        "before",
    ):
        if key in change and change[key] is not None:
            return _coerce_text(change[key])
    return json.dumps(change, sort_keys=True, ensure_ascii=False)


def _codex_file_change_summary(change: dict[str, Any]) -> str | None:
    kind = _coerce_text(change.get("kind", change.get("type", "file_change")))
    path = _coerce_text(change.get("path", ""))
    if not kind and not path:
        return None
    return f"{kind}: {path}" if path else kind


def extract_file_versions(jsonl_text: str) -> dict[str, list[dict[str, Any]]]:
    """Extract file modification history from JSONL text.

    Returns a dict mapping file paths to lists of FileVersion dicts.
    """
    file_versions: dict[str, list[FileVersion]] = {}
    step_idx = 0

    for line in jsonl_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            step_idx += 1
            continue

        etype = event.get("type", "")
        ts_ms = _timestamp_ms(event)

        if etype == "assistant":
            for block in event.get("message", {}).get("content", []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_use":
                    continue
                tool_name = block.get("name", "")
                inputs = block.get("input")
                if not isinstance(inputs, dict):
                    continue

                if tool_name == "Write":
                    fp = inputs.get("file_path", "")
                    content = inputs.get("content", "")
                    if not fp:
                        continue
                    versions = file_versions.setdefault(fp, [])
                    versions.append(FileVersion(
                        version=len(versions) + 1,
                        action="write",
                        step_index=step_idx,
                        timestamp_ms=ts_ms,
                        content_hash=_content_hash(content),
                        content_length=len(content),
                    ))

                elif tool_name == "Edit":
                    fp = inputs.get("file_path", "")
                    old_string = inputs.get("old_string", "")
                    new_string = inputs.get("new_string", "")
                    if not fp:
                        continue
                    versions = file_versions.setdefault(fp, [])
                    diff_summary = None
                    if old_string and new_string:
                        old_preview = old_string[:50].replace("\n", "\\n")
                        new_preview = new_string[:50].replace("\n", "\\n")
                        diff_summary = f"'{old_preview}' → '{new_preview}'"
                    combined = f"{old_string}|{new_string}"
                    versions.append(FileVersion(
                        version=len(versions) + 1,
                        action="edit",
                        step_index=step_idx,
                        timestamp_ms=ts_ms,
                        content_hash=_content_hash(combined),
                        content_length=len(new_string),
                        diff_summary=diff_summary,
                    ))

                elif tool_name == "NotebookEdit":
                    fp = inputs.get("notebook_path", inputs.get("file_path", ""))
                    content = str(inputs.get("new_source", inputs.get("content", "")))
                    if not fp:
                        continue
                    versions = file_versions.setdefault(fp, [])
                    versions.append(FileVersion(
                        version=len(versions) + 1,
                        action="notebook_edit",
                        step_index=step_idx,
                        timestamp_ms=ts_ms,
                        content_hash=_content_hash(content),
                        content_length=len(content),
                    ))

        elif etype == "item.completed":
            item = event.get("item", {})
            if not isinstance(item, dict) or item.get("type") != "file_change":
                step_idx += 1
                continue

            changes = item.get("changes", [])
            if not isinstance(changes, list):
                step_idx += 1
                continue

            for change in changes:
                if not isinstance(change, dict):
                    continue
                fp = change.get("path") or change.get("file_path")
                if not fp:
                    continue
                fp_text = _coerce_text(fp)
                kind = _coerce_text(change.get("kind", change.get("type", "")))
                action = _codex_file_change_action(kind)
                content = _codex_file_change_content(change)
                versions = file_versions.setdefault(fp_text, [])
                versions.append(FileVersion(
                    version=len(versions) + 1,
                    action=action,
                    step_index=step_idx,
                    timestamp_ms=ts_ms,
                    content_hash=_content_hash(content),
                    content_length=len(content),
                    diff_summary=_codex_file_change_summary(change),
                ))

        step_idx += 1

    return {fp: [v.to_dict() for v in versions] for fp, versions in file_versions.items()}


def extract_file_versions_from_file(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Extract file versions from a JSONL sidecar file."""
    return extract_file_versions(path.read_text(encoding="utf-8"))
