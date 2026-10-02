#!/usr/bin/env python3
"""Verify that agent runs of MCP E2E fake tasks really went through the MCP tools.

`asibench score` only compares output files with references. This verifier
reads the persisted run artefacts (raw Claude Code stream-json or Codex
``exec --json`` JSONL, or the normalised trajectory as a fallback) and checks,
per result:

  mcp_connected     the MCP server was connected and every required tool offered
                    (Codex has no server list: every required tool must have
                    returned a result)
  tool_called       every required tool was called (mcp__<server>__<tool>)
  tool_correct      per required tool, a successful call returned a result that
                    matches the reference (value, JSON field or image type); WARN
                    if the inputs were not the reference inputs verbatim
  tool_chain        (only if configured) a call's inputs are exactly the result
                    of an earlier tool call, e.g. plot(scan output)
  answer_from_tool  each configured output-file value equals a value a tool
                    returned and matches the reference
  no_bypass         no Bash command or produced source file installs/imports
                    the backend directly, and no listed non-MCP tool (e.g.
                    WebFetch) reached it (suspicious commands/tools are WARN)

Per-task expectations come from ``e2e_check.json`` in the task directory
(examples in examples/mcp-e2e-tasks/mcp_e2e/*/). Schema 1 (one tool, one
scalar) is normalised to schema 2:

  server, reference_file, prediction_file
  calls     list of {name, tool, inputs_from_reference?, inputs_from_call?,
            result, optional?, group?}; ``result.format`` is ``number``,
            ``json`` + ``key``, or ``image`` + ``media_type``
  answers   list of {prediction_key, from_call + result_key or from_calls,
            reference_key, abs_tol}
  server_tools                     tools the server must offer (extra = WARN)
  bypass_patterns / suspicious_patterns
                                   regexes over Bash commands and produced
                                   source files; FAIL / WARN
  bypass_tools / suspicious_tools  non-MCP (or forbidden MCP) tool name ->
                                   regex over its input; FAIL / WARN

Results are numeric by default; a call result, chain or answer with
``extract`` uses one of the named EXTRACTORS (``arxiv_ids``, ``term_counts``)
and compares canonical values with ``match`` = equal / subset / member; an
answer with ``"merge_calls": true`` merges the values of all calls first.
A call spec with ``"optional": true`` is only judged if called; optional specs
sharing a ``"group"`` count as one requirement (at least one of them must be
called and correct), e.g. reading a state with either of two tools. A numeric
answer may list several sources in ``from_calls``; a source with ``select``
takes one element of a list field (``reduce`` max/min, ``argmax_of``/``argmin_of``
another field, or ``where_key`` equal to a reference value, e.g. R at a given
wavelength of a returned spectrum). Chained inputs that are not
numbers (e.g. a ``session_id``) compare as exact strings, or with
``"compare": "geometry"`` as atom lists within ``abs_tol`` (Angstrom). Result
and answer keys may be dotted paths into nested JSON (``result.zpe.value``);
a literal key containing dots wins. Stdlib only.

Usage::

    python3 scripts/mcp/e2e/verify_run.py --results-dir out/ \
        --instances-dir instances/ --tasks-dir examples/mcp-e2e-tasks
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

CHECK_ORDER = ("mcp_connected", "tool_called", "tool_correct", "tool_chain", "answer_from_tool", "no_bypass")
SKIP_MARKERS = (".trajectory.", ".agent_model_output.", ".model_calls.", "local_score_")
SEVERITY = {"PASS": 0, "WARN": 1, "FAIL": 2}


@dataclass
class Evidence:
    source: str
    mcp_servers: dict[str, str] | None = None  # name -> status; None = unknown
    tools_offered: list[str] | None = None
    # {id, name, input, result_text, is_error, content_types, media_types}; None = not observable
    calls: list[dict] = field(default_factory=list)
    bash_commands: list[str] = field(default_factory=list)
    assistant_text: list[str] = field(default_factory=list)
    permission_mode: str | None = None
    final_result: dict | None = None


def _new_call(call_id, name, inputs) -> dict:
    return {"id": call_id, "name": name, "input": inputs, "result_text": None, "is_error": None,
            "content_types": None, "media_types": None}


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


def _content_blocks(event: dict) -> list[dict]:
    """Return content blocks of an assistant/user event, tolerating string payloads."""
    message = event.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    return [block for block in content if isinstance(block, dict)] if isinstance(content, list) else []


def parse_claude_stream(path: Path) -> Evidence:
    ev = Evidence(source=f"claude stream-json ({path.name})")
    by_id: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if etype == "system" and event.get("subtype") == "init":
            servers = event.get("mcp_servers") or []
            ev.mcp_servers = {s.get("name"): s.get("status") for s in servers if isinstance(s, dict)}
            ev.tools_offered = [t for t in event.get("tools", []) if isinstance(t, str)]
            ev.permission_mode = event.get("permissionMode")
        elif etype == "assistant":
            for block in _content_blocks(event):
                if block.get("type") == "tool_use":
                    call = _new_call(block.get("id"), block.get("name", ""), block.get("input") or {})
                    ev.calls.append(call)
                    if call["id"]:
                        by_id[call["id"]] = call
                    if call["name"] == "Bash" and isinstance(call["input"], dict):
                        ev.bash_commands.append(str(call["input"].get("command", "")))
                elif block.get("type") == "text":
                    ev.assistant_text.append(str(block.get("text", "")))
        elif etype == "user":
            for block in _content_blocks(event):
                if block.get("type") == "tool_result":
                    call = by_id.get(block.get("tool_use_id"))
                    if call is not None:
                        call["result_text"] = _text_of(block.get("content"))
                        call["is_error"] = bool(block.get("is_error"))
                        call["content_types"], call["media_types"] = _block_types(block.get("content"))
        elif etype == "result":
            ev.final_result = {k: event.get(k) for k in ("subtype", "is_error", "num_turns")}
            ev.final_result["result"] = str(event.get("result") or "")[:1000]
    return ev


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
    by_id: dict[str, dict] = {}
    recognised = False

    def call_for(item: dict, name: str, inputs) -> tuple[dict, bool]:
        call = by_id.get(item.get("id")) if item.get("id") else None
        if call is not None:
            return call, False
        call = _new_call(item.get("id"), name, inputs)
        ev.calls.append(call)
        if call["id"]:
            by_id[call["id"]] = call
        return call, True

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
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
                ev.bash_commands.append(command)
            if done:
                call["result_text"] = str(item.get("aggregated_output") or "")
                call["is_error"] = item.get("exit_code") not in (None, 0)
                call["content_types"], call["media_types"] = ["text"], []
        elif kind in ("web_search", "web_fetch"):
            details = {k: v for k, v in item.items() if k not in ("id", "type", "status")}
            call, _created = call_for(item, kind, details)
            if done:
                call["input"] = details
                call["is_error"] = item.get("status") == "failed"
                call["result_text"] = ""
                call["content_types"], call["media_types"] = [], []
        elif kind == "mcp_tool_call":
            name = f"mcp__{item.get('server', '')}__{item.get('tool', '')}"
            call, _created = call_for(item, name, _codex_arguments(item.get("arguments")))
            if not done:
                continue
            result, error = item.get("result"), item.get("error")
            if error is not None or item.get("status") == "failed" or not isinstance(result, dict):
                call["is_error"] = True
                call["result_text"] = str(error.get("message", error) if isinstance(error, dict) else error or "")
                continue
            content = result.get("content")
            call["is_error"] = bool(result.get("is_error") or result.get("isError"))
            call["result_text"] = _text_of(content)
            call["content_types"], call["media_types"] = _block_types(content)
            structured = result.get("structured_content")
            if not call["result_text"] and structured is not None:
                call["result_text"] = json.dumps(structured)
    return ev if recognised else None


def _trajectory_steps(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    steps = data.get("steps", []) if isinstance(data, dict) else data  # run writes a bare list
    return [s for s in steps if isinstance(s, dict)] if isinstance(steps, list) else []


def _result_fields(step: dict) -> dict:
    meta = step.get("metadata") or {}
    return {"result_text": step.get("content", ""), "is_error": bool(meta.get("is_error")),
            "content_types": meta.get("content_types"), "media_types": meta.get("image_media_types")}


def parse_trajectory(path: Path) -> Evidence:
    """Adapter-neutral fallback: tool names and results, but no tool inputs."""
    ev = Evidence(source=f"trajectory ({path.name})")
    by_id: dict[str, dict] = {}
    for step in _trajectory_steps(path):
        meta = step.get("metadata") or {}
        if step.get("step_type") == "tool_call":
            call = _new_call(meta.get("tool_call_id"), meta.get("tool_name", ""), None)
            ev.calls.append(call)
            if call["id"]:
                by_id[call["id"]] = call
            command = (meta.get("key_args") or {}).get("command")
            if command:
                ev.bash_commands.append(str(command))
        elif step.get("step_type") == "tool_result":
            call = by_id.get(meta.get("tool_call_id"))
            if call is not None:
                call.update(_result_fields(step))
    return ev


def enrich_results_from_trajectory(ev: Evidence, path: Path) -> int:
    """Fill tool results missing from the persisted stream.

    `asibench run` redacts the content of every user-role event in the saved
    stream-json (prompt protection), which also removes tool_result payloads.
    The trajectory is extracted from the unredacted stream and keeps the result
    text and content block types, keyed by the same tool_call_id.
    """
    results = {}
    for step in _trajectory_steps(path):
        meta = step.get("metadata") or {}
        if step.get("step_type") == "tool_result" and meta.get("tool_call_id"):
            results[meta["tool_call_id"]] = _result_fields(step)
    filled = 0
    for call in ev.calls:
        if call["result_text"] in (None, "<redacted>") and call["id"] in results:
            call.update(results[call["id"]])
            filled += 1
    return filled


# --------------------------------------------------------------------------
# Values
# --------------------------------------------------------------------------

def _float(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _values(value):
    """A finite float, a list of finite floats, or None. Accepts numeric strings and JSON-encoded lists."""
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, list):
        numbers = [_float(v) for v in value]
        return numbers if numbers and all(n is not None for n in numbers) else None
    return _float(value)


def _diff(a, b) -> float:
    """Max absolute difference between two scalars or two equal-length lists; inf otherwise."""
    if a is None or b is None:
        return math.inf
    if isinstance(a, list) != isinstance(b, list):
        return math.inf
    if isinstance(a, list):
        return max((abs(x - y) for x, y in zip(a, b)), default=0.0) if len(a) == len(b) else math.inf
    return abs(a - b)


def _same(a, b) -> bool:
    """Equal up to printing precision: the value was copied, not recomputed."""
    scale = max((abs(v) for v in (a if isinstance(a, list) else [a])), default=0.0) if a is not None else 0.0
    return _diff(a, b) <= max(1e-9, 1e-12 * scale)


def _same_input(given, expected) -> bool:
    """Numbers compare numerically (0.9 == "0.9"); anything else as stripped strings."""
    a, b = _values(given), _values(expected)
    if a is not None and b is not None:
        return _diff(a, b) <= 1e-12
    return str(given).strip() == str(expected).strip()


def _same_link(given, source) -> bool:
    """A chained input equals an earlier result: numbers to printing precision, other
    scalars (e.g. a session id) as identical non-empty strings."""
    a, b = _values(given), _values(source)
    if a is not None and b is not None:
        return _same(a, b)
    if isinstance(given, (dict, list)) or isinstance(source, (dict, list)) or given is None or source is None:
        return False
    return str(given).strip() != "" and str(given).strip() == str(source).strip()


def _field(data, key):
    """Field ``key`` of a JSON object; a dotted key walks nested objects
    (``result.final_energy.value``). A literal key containing dots wins."""
    if not isinstance(data, dict) or not isinstance(key, str):
        return None
    if key in data:
        return data[key]
    node = data
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _geometry(text) -> list[tuple[str, tuple[float, float, float]]] | None:
    """Atom lines ``Symbol x y z`` of a geometry string; other lines (an XYZ atom count
    or comment, a psi4 ``charge multiplicity`` header, ``symmetry c1``) are ignored."""
    if not isinstance(text, str):
        return None
    atoms = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 4 or not fields[0].isalpha():
            continue
        coords = [_float(v) for v in fields[1:]]
        if any(c is None for c in coords):
            continue
        atoms.append((fields[0].capitalize(), tuple(coords)))
    return atoms or None


def _same_geometry(given, source, abs_tol: float) -> bool:
    """Same atoms in the same order, every Cartesian coordinate within ``abs_tol``
    (an agent may reformat or round the geometry it passes on)."""
    a, b = _geometry(given), _geometry(source)
    if a is None or b is None or len(a) != len(b) or [s for s, _ in a] != [s for s, _ in b]:
        return False
    return all(abs(x - y) <= abs_tol for (_, p), (_, q) in zip(a, b) for x, y in zip(p, q))


def _link_comparator(link: dict):
    """How a chained input is compared with the earlier result: ``"compare": "geometry"``
    (with ``abs_tol``, default 1e-4 Angstrom) or, by default, ``_same_link``."""
    if link.get("compare") == "geometry":
        tol = float(link.get("abs_tol", 1e-4))
        return lambda given, source: _same_geometry(given, source, tol)
    return _same_link


def _unwrap_structured(data):
    """Undo FastMCP's structured-output wrapper.

    A FastMCP tool with a return annotation declares an ``outputSchema`` and
    returns ``structuredContent = {"result": <value>}``; clients that show the
    structured content (Claude Code does) put that object in the tool result
    instead of the text block. ``<value>`` is the original string, or for a tool
    returning content blocks a list of ``{"type": "text", "text": ...}`` blocks.
    Anything else is returned unchanged.
    """
    if not (isinstance(data, dict) and set(data) == {"result"}):
        return data
    inner = data["result"]
    if isinstance(inner, list) and inner and all(isinstance(b, dict) and "type" in b for b in inner):
        inner = "\n".join(str(b.get("text", "")) for b in inner if b.get("type") == "text")
    if isinstance(inner, str):
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            return inner
    return inner


def _parsed_result(call: dict):
    """The call's result text parsed as JSON (structured-output wrapper removed), or None."""
    try:
        return _unwrap_structured(json.loads(call.get("result_text") or ""))
    except (json.JSONDecodeError, TypeError):
        return None


def _result_value(call: dict, key: str | None):
    """Numeric value of a text result, or of field `key` of a JSON-object result."""
    text = call.get("result_text")
    if text is None:
        return None
    if key is None:
        value = _values(text)
        if value is None:
            unwrapped = _parsed_result(call)
            value = _values(unwrapped) if isinstance(unwrapped, (str, int, float, list)) else None
        return value
    data = _parsed_result(call)
    return _values(_field(data, key))


def _source_value(call: dict, source: dict, reference: dict):
    """The value an answer may be copied from: field `result_key` of the call's result, or,
    with `select`, one element of that (list) field: {"reduce": "max"|"min"}, {"argmax_of": key}
    / {"argmin_of": key} (element at the extreme of another list field of the same result), or
    {"where_key": key, "equals_reference_key": ref} (element where list field `key` equals the
    reference value, e.g. R at a given wavelength of a returned spectrum)."""
    select = source.get("select")
    if not select:
        return _result_value(call, source.get("result_key"))
    data = _json_result(call)
    values = _values(_field(data, source.get("result_key"))) if data else None
    if not isinstance(values, list):
        return None
    if select.get("reduce") in ("max", "min"):
        return max(values) if select["reduce"] == "max" else min(values)
    for mode, pick in (("argmax_of", max), ("argmin_of", min)):
        if select.get(mode):
            other = _values(_field(data, select[mode]))
            if not isinstance(other, list) or len(other) != len(values):
                return None
            return values[other.index(pick(other))]
    if select.get("where_key"):
        keys = _values(_field(data, select["where_key"]))
        target = _float(reference.get(select.get("equals_reference_key")))
        if not isinstance(keys, list) or len(keys) != len(values) or target is None:
            return None
        hits = [v for k, v in zip(keys, values) if abs(k - target) <= 1e-9 * max(1.0, abs(target))]
        return hits[0] if hits else None
    return None


def _json_result(call: dict) -> dict | None:
    data = _parsed_result(call)
    return data if isinstance(data, dict) else None


# --------------------------------------------------------------------------
# Named extractors for non-numeric results: (extract from parsed tool JSON,
# canonicalise a prediction/reference/input value). Both return None when the
# value has the wrong shape.
# --------------------------------------------------------------------------

def _arxiv_id(value) -> str | None:
    """'1103.0291' from '1103.0291v1', 'arXiv:1103.0291' or an abs/pdf URL (old-style IDs kept intact)."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = re.sub(r"^https?://(export\.)?arxiv\.org/(abs|pdf)/", "", value.strip())
    text = re.sub(r"\.pdf$", "", text).removeprefix("arXiv:")
    return re.sub(r"v\d+$", "", text) or None


def _canon_ids(value):
    if isinstance(value, list):
        ids = [_arxiv_id(v) for v in value]
        return ids if ids and all(ids) else None
    return _arxiv_id(value)


def _extract_arxiv_ids(data):
    if isinstance(data, dict) and isinstance(data.get("data"), list):  # SMCP truncation wrapper
        data = data["data"]
    if not isinstance(data, list) or not all(isinstance(p, dict) for p in data):
        return None
    return [p.get("url") for p in data]


def _canon_counts(value):
    if not isinstance(value, dict) or not value:
        return None
    out = {}
    for key, count in value.items():
        number = _float(count)
        if number is None or number != int(number) or number < 0:
            return None
        out[str(key).strip().lower()] = int(number)
    return out


def _extract_term_counts(data):
    if not isinstance(data, dict) or data.get("status") == "error" or not isinstance(data.get("snippets"), list):
        return None
    counts: dict[str, int] = {}
    for snippet in data["snippets"]:
        if isinstance(snippet, dict) and isinstance(snippet.get("term"), str):
            counts[snippet["term"]] = counts.get(snippet["term"], 0) + 1
    return counts


EXTRACTORS = {
    "arxiv_ids": (_extract_arxiv_ids, _canon_ids),
    "term_counts": (_extract_term_counts, _canon_counts),
}


def _extracted(call: dict, name: str):
    """Canonical extracted value of a call's JSON result, or None."""
    extract, canon = EXTRACTORS[name]
    data = _parsed_result(call)
    if data is None:
        return None
    raw = extract(data)
    return None if raw is None else canon(raw)


def _matches(value, ref, mode: str) -> bool:
    """equal: identical canonical values; subset: every key of a dict value has the
    reference count; member: the reference scalar is in the value list."""
    if value is None or ref is None:
        return False
    if mode == "member":
        return isinstance(value, list) and ref in value
    if mode == "subset":
        return isinstance(value, dict) and isinstance(ref, dict) and bool(value) and \
            all(k in ref and ref[k] == v for k, v in value.items())
    return value == ref


# --------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------

def normalize_spec(spec: dict) -> dict:
    """Return a schema-2 spec; schema 1 is one numeric-result tool and one scalar answer."""
    if int(spec.get("schema_version", 1)) >= 2:
        return spec
    tol = float(spec["abs_tol"])
    return {
        "schema_version": 2,
        "server": spec["server"],
        "reference_file": spec["reference_file"],
        "prediction_file": spec["prediction_file"],
        "calls": [{"name": spec["tool"], "tool": spec["tool"],
                   "inputs_from_reference": spec.get("tool_inputs_from_reference") or {},
                   "result": {"format": "number", "reference_key": spec["reference_key"], "abs_tol": tol}}],
        "answers": [{"prediction_key": spec["prediction_key"], "from_call": spec["tool"],
                     "reference_key": spec["reference_key"], "abs_tol": tol}],
        "bypass_patterns": spec.get("bypass_patterns", []),
        "suspicious_patterns": spec.get("suspicious_patterns", []),
    }


def _is_target(name: str, server: str, tool: str) -> bool:
    return name == f"mcp__{server}__{tool}" or (server in name and name.endswith(tool))


def _worst(statuses) -> str:
    return max(statuses, key=lambda s: SEVERITY[s], default="PASS")


def _judge_call(call: dict, call_spec: dict, reference: dict) -> dict:
    """Assess one actual tool call against its spec: inputs and result."""
    report = {"id": call["id"], "is_error": call["is_error"], "result_text": (call["result_text"] or "")[:300]}
    inputs_ok = None
    if isinstance(call["input"], dict) and call_spec.get("inputs_from_reference"):
        inputs_ok = all(_same_input(call["input"].get(arg), reference.get(ref_key))
                        for arg, ref_key in call_spec["inputs_from_reference"].items())
    report["inputs_match_reference"] = inputs_ok
    result_spec = call_spec.get("result") or {}
    fmt = result_spec.get("format", "number")
    if call["is_error"] or (call["result_text"] is None and fmt != "image"):
        report["result_ok"] = False
    elif fmt == "image":
        if call["content_types"] is None:
            report["result_ok"] = None  # not observable in this evidence
        else:
            wanted = result_spec.get("media_type")
            images = call["media_types"] or []
            report["media_types"] = images
            report["result_ok"] = bool(images) and (wanted is None or wanted in images)
    elif result_spec.get("extract"):
        name = result_spec["extract"]
        value = _extracted(call, name)
        ref_value = EXTRACTORS[name][1](reference.get(result_spec.get("reference_key")))
        report["extracted"] = value
        report["result_ok"] = _matches(value, ref_value, result_spec.get("match", "equal"))
    else:
        value = _result_value(call, result_spec.get("key") if fmt == "json" else None)
        ref_value = _values(reference.get(result_spec.get("reference_key")))
        err = _diff(value, ref_value)
        report["abs_error"] = None if math.isinf(err) else err
        report["result_ok"] = err <= float(result_spec.get("abs_tol", 0.0))
    return report


def _requirements(spec: dict) -> list[list[dict]]:
    """Call specs grouped into requirements: a required spec alone, optional specs by `group`
    (a group is met if any member is). Optional specs without a group are no requirement."""
    groups: dict[str, list[dict]] = {}
    out = []
    for cs in spec["calls"]:
        if not cs.get("optional"):
            out.append([cs])
        elif cs.get("group"):
            if cs["group"] not in groups:
                groups[cs["group"]] = []
                out.append(groups[cs["group"]])
            groups[cs["group"]].append(cs)
    return out


def _label(requirement: list[dict]) -> str:
    return "|".join(cs["tool"] for cs in requirement)


def _check_calls(ev: Evidence, spec: dict, reference: dict, checks: dict) -> dict[str, list[dict]]:
    server = spec["server"]
    by_spec = {cs["name"]: [c for c in ev.calls if _is_target(c["name"], server, cs["tool"])] for cs in spec["calls"]}

    missing = [_label(req) for req in _requirements(spec) if not any(by_spec[cs["name"]] for cs in req)]
    counts = ", ".join(f"{cs['tool']}×{len(by_spec[cs['name']])}" for cs in spec["calls"])
    checks["tool_called"] = {"status": "FAIL" if missing else "PASS",
                             "detail": (f"not called: {missing}; " if missing else "") + counts}

    per_call, judged = {}, {}
    for cs in spec["calls"]:
        reports = [_judge_call(c, cs, reference) for c in by_spec[cs["name"]]]
        good = [r for r in reports if r["result_ok"]]
        verbatim = [r for r in good if r["inputs_match_reference"] is not False]
        if not reports and cs.get("optional"):
            status, detail = None, "not called (optional)"
        elif verbatim:
            status, detail = "PASS", f"{len(good)} correct call(s)"
        elif good:
            status, detail = "WARN", "result correct but inputs differ from reference"
        elif any(r["result_ok"] is None for r in reports):
            status, detail = "WARN", "result type not observable in this evidence"
        else:
            status, detail = "FAIL", "no successful call returned the expected result"
        judged[cs["name"]] = status
        per_call[cs["name"]] = {"tool": cs["tool"], "status": status or "SKIP", "detail": detail, "calls": reports}
    statuses, covered = [], set()
    for cs in spec["calls"]:  # ungrouped optional specs count only if called
        if cs.get("optional") and not cs.get("group") and judged[cs["name"]] is not None:
            statuses.append(judged[cs["name"]])
    for req in _requirements(spec):  # a group takes its best called member
        called = [judged[cs["name"]] for cs in req if judged[cs["name"]] is not None]
        best = min(called, key=lambda s: SEVERITY[s]) if called else "FAIL"
        statuses.append(best)
        if len(req) > 1 and best == "PASS":
            covered.update(cs["name"] for cs in req)
    failing = [f"{n}: {c['detail']}" for n, c in per_call.items()
               if c["status"] not in ("PASS", "SKIP") and n not in covered]
    checks["tool_correct"] = {"status": _worst(statuses),
                              "detail": "; ".join(failing) or "all required tools returned expected results",
                              "per_call": per_call}
    return by_spec


def _check_chain(spec: dict, by_spec: dict[str, list[dict]], checks: dict) -> None:
    chained = [cs for cs in spec["calls"] if cs.get("inputs_from_call")
               and not (cs.get("optional") and not by_spec.get(cs["name"]))]
    if not chained:
        return
    statuses, details = [], []
    for cs in chained:
        link = cs["inputs_from_call"]
        if link.get("extract"):
            status, detail = _chain_by_extract(cs["name"], link, by_spec)
            statuses.append(status)
            details.append(detail)
            continue
        sources = [data for c in by_spec.get(link["call"], []) if not c["is_error"]
                   for data in [_json_result(c)] if data is not None]
        consumers = by_spec.get(cs["name"], [])
        if not consumers or not sources:
            statuses.append("FAIL")
            details.append(f"{cs['name']}←{link['call']}: no {'consumer' if not consumers else 'source'} call")
            continue
        if all(not isinstance(c["input"], dict) for c in consumers):
            statuses.append("WARN")
            details.append(f"{cs['name']}←{link['call']}: tool inputs not observable in this evidence")
            continue
        same = _link_comparator(link)
        linked = any(isinstance(c["input"], dict) and all(
            same(c["input"].get(arg), _field(src, key)) for arg, key in link["map"].items())
            for c in consumers for src in sources)
        statuses.append("PASS" if linked else "FAIL")
        details.append(f"{cs['name']}←{link['call']}: " + ("inputs equal an earlier result" if linked else
                       f"inputs {sorted(link['map'])} do not equal any {link['call']} result"))
    checks["tool_chain"] = {"status": _worst(statuses), "detail": "; ".join(details)}


def _chain_by_extract(name: str, link: dict, by_spec: dict[str, list[dict]]) -> tuple[str, str]:
    """A consumer argument (any of link['args']) is one of the values extracted from an earlier result."""
    canon = EXTRACTORS[link["extract"]][1]
    pool = []
    for call in by_spec.get(link["call"], []):
        if not call["is_error"]:
            value = _extracted(call, link["extract"])
            pool.extend(value if isinstance(value, list) else [value] if value is not None else [])
    consumers = by_spec.get(name, [])
    label = f"{name}←{link['call']}"
    if not consumers or not pool:
        return "FAIL", f"{label}: no {'consumer' if not consumers else 'source'} call"
    if all(not isinstance(c["input"], dict) for c in consumers):
        return "WARN", f"{label}: tool inputs not observable in this evidence"
    used = [canon(c["input"].get(arg)) for c in consumers if isinstance(c["input"], dict)
            for arg in link["args"] if c["input"].get(arg) is not None]
    linked = [u for u in used if u is not None and u in pool]
    if linked:
        return "PASS", f"{label}: {link['args']} = {linked[0]!r}, returned by {link['call']}"
    return "FAIL", f"{label}: {link['args']} values {used} are not among the {len(pool)} {link['call']} results"


def _check_answers(spec: dict, by_spec: dict[str, list[dict]], reference: dict, pred_path: Path,
                   out: dict, checks: dict) -> None:
    try:
        prediction = json.loads(pred_path.read_text(encoding="utf-8"))
        if not isinstance(prediction, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as exc:
        checks["answer_from_tool"] = {"status": "FAIL", "detail": f"{spec['prediction_file']} missing or unreadable ({exc})"}
        out["prediction_value"] = None
        return
    statuses, details = [], []
    for k, answer in enumerate(spec["answers"]):
        if answer.get("extract"):
            status, detail, pred, ref = _answer_by_extract(answer, prediction, reference, by_spec)
            if k == 0:
                out["prediction_value"], out["reference_value"] = pred, ref
            statuses.append(status)
            details.append(detail)
            continue
        pred = _values(prediction.get(answer["prediction_key"]))
        ref = _values(reference.get(answer["reference_key"]))
        sources = answer.get("from_calls") or [{"call": answer["from_call"], "result_key": answer.get("result_key")}]
        returned = [_source_value(c, src, reference) for src in sources
                    for c in by_spec.get(src["call"], []) if not c["is_error"]]
        from_tool = any(_same(pred, v) for v in returned if v is not None)
        err = _diff(pred, ref)
        correct = err <= float(answer["abs_tol"])
        if k == 0:
            out["prediction_value"], out["reference_value"] = pred, ref
        statuses.append("PASS" if from_tool and correct else "FAIL")
        details.append(f"{answer['prediction_key']}: " + ("missing/not numeric" if pred is None else
                       f"|d|={err:.2e} vs reference, equals a tool-returned value: {from_tool}"))
    checks["answer_from_tool"] = {"status": _worst(statuses), "detail": "; ".join(details)}


def _answer_by_extract(answer: dict, prediction: dict, reference: dict, by_spec: dict[str, list[dict]]):
    name = answer["extract"]
    canon = EXTRACTORS[name][1]
    pred = canon(prediction.get(answer["prediction_key"]))
    ref = canon(reference.get(answer["reference_key"]))
    returned = [v for c in by_spec.get(answer["from_call"], []) if not c["is_error"]
                for v in [_extracted(c, name)] if v is not None]
    if answer.get("merge_calls") and returned:
        merged: dict = {}
        for value in returned:  # e.g. one snippet call per term
            if isinstance(value, dict):
                merged.update(value)
        returned = [merged]
    mode = answer.get("match", "equal")
    if mode == "member":
        from_tool = pred is not None and any(_matches(v, pred, "member") for v in returned)
    else:
        from_tool = pred is not None and any(v == pred for v in returned)
    correct = pred is not None and pred == ref
    detail = f"{answer['prediction_key']}: " + ("missing or malformed" if pred is None else
                                                f"matches reference: {correct}, equals a tool-returned value: {from_tool}")
    return ("PASS" if from_tool and correct else "FAIL"), detail, pred, ref


def _load_evidence(result_path: Path, agent: dict) -> Evidence | None:
    run_dir = result_path.parent
    stdout_file = agent.get("raw_stdout_file")
    traj_file = agent.get("trajectory_file")
    ev = None
    if stdout_file and (run_dir / stdout_file).is_file():
        ev = parse_claude_stream(run_dir / stdout_file)
        if ev.mcp_servers is None and not ev.calls and ev.final_result is None:
            ev = parse_codex_stream(run_dir / stdout_file)  # None: not a Codex log either
    if ev is None and traj_file and (run_dir / traj_file).is_file():
        ev = parse_trajectory(run_dir / traj_file)
    elif ev is not None and traj_file and (run_dir / traj_file).is_file():
        filled = enrich_results_from_trajectory(ev, run_dir / traj_file)
        if filled:
            ev.source += f" + {filled} tool result(s) from {traj_file}"
    return ev


def verify_one(result_path: Path, result: dict, instances_dir: Path, tasks_dir: Path) -> dict:
    task_id = result["task_id"]
    task_dir = tasks_dir / Path(*task_id.split("."))
    spec_path = task_dir / "e2e_check.json"
    out = {"result_file": str(result_path), "task_id": task_id, "instance_id": result.get("instance_id"),
           "prompt_level": result.get("prompt_level"), "attempt": result.get("attempt"),
           "run_status": result.get("status"), "checks": {}, "notes": []}
    checks = out["checks"]
    if not spec_path.is_file():
        out["verdict"] = "SKIP"
        out["notes"].append(f"no e2e_check.json in {task_dir}")
        return out
    spec = normalize_spec(json.loads(spec_path.read_text(encoding="utf-8")))
    server = spec["server"]
    reference = json.loads((instances_dir / result["instance_id"] / "reference" / spec["reference_file"])
                           .read_text(encoding="utf-8"))

    agent = result.get("agent_output") or {}
    ev = _load_evidence(result_path, agent)
    if ev is None:
        out["verdict"] = "FAIL"
        out["failure"] = "no_evidence"
        out["notes"].append("neither raw stdout nor trajectory artefact found")
        return out
    out["evidence_source"] = ev.source
    out["tool_sequence"] = [call["name"] for call in ev.calls]
    if ev.tools_offered is not None:
        out["toolsearch_offered"] = "ToolSearch" in ev.tools_offered
        out["mcp_tools_offered"] = [t for t in ev.tools_offered if t.startswith("mcp__")]
    out["tool_call_counts"] = {}
    for call in ev.calls:
        out["tool_call_counts"][call["name"]] = out["tool_call_counts"].get(call["name"], 0) + 1
    if ev.final_result:
        out["agent_final"] = ev.final_result
    out["permission_mode"] = ev.permission_mode
    out["persisted_outputs"] = agent.get("persisted_outputs")
    out["agent_status"] = agent.get("status")
    out["agent_error"] = agent.get("error_message")
    if ev.assistant_text:
        out["last_assistant_text"] = ev.assistant_text[-1][:1000]

    # 1. MCP server connected and every required tool offered
    targets = [f"mcp__{server}__{cs['tool']}" for cs in spec["calls"]]
    if ev.mcp_servers is None:
        # No server list (Codex JSONL, trajectory): a result returned by the
        # server's tool is the only proof that it was connected and offered.
        silent = [_label(req) for req in _requirements(spec)
                  if not any(_is_target(c["name"], server, cs["tool"]) and c["is_error"] is False
                             for cs in req for c in ev.calls)]
        checks["mcp_connected"] = (
            {"status": "PASS", "detail": "no server list in this evidence; every required tool returned a result"}
            if not silent else
            {"status": "WARN", "detail": f"no server list in this evidence and no successful call of {silent}"})
    else:
        status = ev.mcp_servers.get(server)
        not_offered = [t for t in targets if ev.tools_offered is None or t not in ev.tools_offered]
        checks["mcp_connected"] = {"status": "PASS" if status == "connected" and not not_offered else "FAIL",
                                   "detail": f"server status={status!r}, not offered={not_offered}",
                                   "mcp_servers": ev.mcp_servers}
        if spec.get("server_tools") and ev.tools_offered is not None:
            prefix = f"mcp__{server}__"
            offered = {t for t in ev.tools_offered if t.startswith(prefix)}
            extra = sorted(offered - {prefix + t for t in spec["server_tools"]})
            if extra:
                checks["mcp_connected"]["detail"] += (f"; server offered {len(offered)} tools, expected "
                                                      f"{len(spec['server_tools'])} (e.g. {extra[:3]})")
                if checks["mcp_connected"]["status"] == "PASS":
                    checks["mcp_connected"]["status"] = "WARN"

    # 2-4. tools called, results correct, chained inputs
    by_spec = _check_calls(ev, spec, reference, checks)
    _check_chain(spec, by_spec, checks)

    # 5. the answer file carries tool-returned values
    run_dir = result_path.parent
    outputs_dir = run_dir / ((agent.get("persisted_outputs") or {}).get("dir") or f"{result_path.stem}.outputs")
    _check_answers(spec, by_spec, reference, outputs_dir / spec["prediction_file"], out, checks)

    # 6. bypass detection over Bash commands and produced source files
    texts = [("bash", c) for c in ev.bash_commands]
    if outputs_dir.is_dir():
        for path in sorted(outputs_dir.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh", ".ipynb"}:
                texts.append((f"file:{path.relative_to(outputs_dir)}", path.read_text(encoding="utf-8", errors="replace")))
    hard = [(where, p) for where, text in texts for p in spec.get("bypass_patterns", []) if re.search(p, text)]
    soft = [(where, p) for where, text in texts for p in spec.get("suspicious_patterns", []) if re.search(p, text)]
    for call in ev.calls:  # non-MCP tools such as WebFetch / Codex web_search
        payload = json.dumps(call["input"], ensure_ascii=False)
        hard += [(f"tool:{call['name']}", p) for tool, p in spec.get("bypass_tools", {}).items()
                 if call["name"] == tool and re.search(p, payload)]
        soft += [(f"tool:{call['name']}", p) for tool, p in spec.get("suspicious_tools", {}).items()
                 if call["name"] == tool and re.search(p, payload)]
    if hard:
        checks["no_bypass"] = {"status": "FAIL", "detail": f"direct backend use: {hard[:5]}"}
    elif soft:
        checks["no_bypass"] = {"status": "WARN", "detail": f"review: {soft[:5]}"}
    else:
        checks["no_bypass"] = {"status": "PASS", "detail": f"{len(ev.bash_commands)} shell command(s) scanned"}
    out["bash_commands"] = ev.bash_commands[:50]

    failed = [name for name in CHECK_ORDER if checks.get(name, {}).get("status") == "FAIL"]
    out["verdict"] = "FAIL" if failed else "PASS"
    if failed:
        out["failure"] = failed[0]
    return out


def iter_results(results_dir: Path):
    for path in sorted(results_dir.glob("*/*.json")):
        if any(marker in path.name for marker in SKIP_MARKERS):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("task_id") and data.get("instance_id"):
            yield path, data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--instances-dir", required=True)
    parser.add_argument("--tasks-dir", default="examples/mcp-e2e-tasks")
    parser.add_argument("--report", default=None, help="default: <results-dir>/mcp_e2e_verify.json")
    args = parser.parse_args(argv)
    results_dir = Path(args.results_dir).expanduser().resolve()
    instances_dir = Path(args.instances_dir).expanduser().resolve()
    tasks_dir = Path(args.tasks_dir).expanduser().resolve()

    rows = [verify_one(p, d, instances_dir, tasks_dir) for p, d in iter_results(results_dir)]
    if not rows:
        print(f"No result JSON found under {results_dir}", file=sys.stderr)
        return 2
    for row in rows:
        marks = " ".join(f"{name}={row['checks'][name]['status']}" for name in CHECK_ORDER if name in row["checks"])
        print(f"[{row['verdict']}] {row['instance_id']} {row['prompt_level']} (run {row['run_status']}): {marks}")
        for name in CHECK_ORDER:
            check = row["checks"].get(name)
            if check and check["status"] != "PASS":
                print(f"         {name}: {check['detail']}")
        for note in row["notes"]:
            print(f"         note: {note}")
        if row["verdict"] == "FAIL":
            said = (row.get("agent_final") or {}).get("result") or row.get("last_assistant_text") or ""
            print(f"         agent status={row.get('agent_status')} permission_mode={row.get('permission_mode')}"
                  f" toolsearch_offered={row.get('toolsearch_offered')} mcp_tools_offered={row.get('mcp_tools_offered')}")
            print(f"         tool sequence: {row.get('tool_sequence')}")
            if said:
                print("         agent said: " + " ".join(said.split())[:400])
    summary = {name: sum(1 for r in rows if r["verdict"] == name) for name in ("PASS", "FAIL", "SKIP")}
    report = {"schema_version": 2, "results_dir": str(results_dir), "summary": summary, "results": rows}
    destination = Path(args.report) if args.report else results_dir / "mcp_e2e_verify.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nPASS {summary['PASS']}  FAIL {summary['FAIL']}  SKIP {summary['SKIP']}  report -> {destination}")
    return 1 if summary["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
