#!/usr/bin/env python3
"""Verify that agent runs of MCP E2E fake tasks really went through the MCP tool.

`asibench score` only compares output files with references. This verifier
reads the persisted run artefacts (raw Claude Code stream-json, or the
normalised trajectory as a fallback) and checks, per result:

  mcp_connected     the MCP server was connected and the target tool offered
  tool_called       the agent called mcp__<server>__<tool>
  tool_correct      a successful call returned a value matching the reference
                    (and, if configured, with the reference inputs verbatim)
  answer_from_tool  the output file value equals a value the tool returned
                    and matches the reference
  no_bypass         no Bash command or produced source file installs/imports
                    the backend directly (suspicious commands are WARN)

Per-task expectations come from ``e2e_check.json`` in the task directory.
Stdlib only.

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

CHECK_ORDER = ("mcp_connected", "tool_called", "tool_correct", "answer_from_tool", "no_bypass")
SKIP_MARKERS = (".trajectory.", ".agent_model_output.", ".model_calls.", "local_score_")


@dataclass
class Evidence:
    source: str
    mcp_servers: dict[str, str] | None = None  # name -> status; None = unknown
    tools_offered: list[str] | None = None
    calls: list[dict] = field(default_factory=list)  # {id,name,input,result_text,is_error}
    bash_commands: list[str] = field(default_factory=list)
    assistant_text: list[str] = field(default_factory=list)
    permission_mode: str | None = None
    final_result: dict | None = None


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(b.get("text", "")) for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


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
                    call = {"id": block.get("id"), "name": block.get("name", ""), "input": block.get("input") or {},
                            "result_text": None, "is_error": None}
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
        elif etype == "result":
            ev.final_result = {k: event.get(k) for k in ("subtype", "is_error", "num_turns")}
            ev.final_result["result"] = str(event.get("result") or "")[:1000]
    return ev


def _trajectory_steps(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    steps = data.get("steps", []) if isinstance(data, dict) else data  # run writes a bare list
    return [s for s in steps if isinstance(s, dict)] if isinstance(steps, list) else []


def parse_trajectory(path: Path) -> Evidence:
    """Adapter-neutral fallback: tool names and results, but no tool inputs."""
    ev = Evidence(source=f"trajectory ({path.name})")
    by_id: dict[str, dict] = {}
    for step in _trajectory_steps(path):
        meta = step.get("metadata") or {}
        if step.get("step_type") == "tool_call":
            call = {"id": meta.get("tool_call_id"), "name": meta.get("tool_name", ""), "input": None,
                    "result_text": None, "is_error": None}
            ev.calls.append(call)
            if call["id"]:
                by_id[call["id"]] = call
            command = (meta.get("key_args") or {}).get("command")
            if command:
                ev.bash_commands.append(str(command))
        elif step.get("step_type") == "tool_result":
            call = by_id.get(meta.get("tool_call_id"))
            if call is not None:
                call["result_text"] = step.get("content", "")
                call["is_error"] = bool(meta.get("is_error"))
    return ev


def enrich_results_from_trajectory(ev: Evidence, path: Path) -> int:
    """Fill tool results missing from the persisted stream.

    `asibench run` redacts the content of every user-role event in the saved
    stream-json (prompt protection), which also removes tool_result payloads.
    The trajectory is extracted from the unredacted stream and keeps them,
    keyed by the same tool_call_id.
    """
    results = {}
    for step in _trajectory_steps(path):
        meta = step.get("metadata") or {}
        if step.get("step_type") == "tool_result" and meta.get("tool_call_id"):
            results[meta["tool_call_id"]] = (step.get("content", ""), bool(meta.get("is_error")))
    filled = 0
    for call in ev.calls:
        if call["result_text"] in (None, "<redacted>") and call["id"] in results:
            call["result_text"], call["is_error"] = results[call["id"]]
            filled += 1
    return filled


def _float(value) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_target(name: str, server: str, tool: str) -> bool:
    return name == f"mcp__{server}__{tool}" or (server in name and name.endswith(tool))


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
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    server, tool, tol = spec["server"], spec["tool"], float(spec["abs_tol"])

    ref_path = instances_dir / result["instance_id"] / "reference" / spec["reference_file"]
    reference = json.loads(ref_path.read_text(encoding="utf-8"))
    ref_value = float(reference[spec["reference_key"]])
    out["reference_value"] = ref_value

    agent = result.get("agent_output") or {}
    run_dir = result_path.parent
    stdout_file = agent.get("raw_stdout_file")
    traj_file = agent.get("trajectory_file")
    ev = None
    if stdout_file and (run_dir / stdout_file).is_file():
        ev = parse_claude_stream(run_dir / stdout_file)
        if ev.mcp_servers is None and not ev.calls and ev.final_result is None:
            ev = None  # not a Claude stream-json log
    if ev is None and traj_file and (run_dir / traj_file).is_file():
        ev = parse_trajectory(run_dir / traj_file)
    elif ev is not None and traj_file and (run_dir / traj_file).is_file():
        filled = enrich_results_from_trajectory(ev, run_dir / traj_file)
        if filled:
            ev.source += f" + {filled} tool result(s) from {traj_file}"
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

    # 1. MCP server connected and tool offered
    target = f"mcp__{server}__{tool}"
    if ev.mcp_servers is None:
        checks["mcp_connected"] = {"status": "WARN", "detail": "init event not available in this evidence"}
    else:
        status = ev.mcp_servers.get(server)
        offered = ev.tools_offered is not None and target in ev.tools_offered
        ok = status == "connected" and offered
        checks["mcp_connected"] = {"status": "PASS" if ok else "FAIL",
                                   "detail": f"server status={status!r}, {target} offered={offered}",
                                   "mcp_servers": ev.mcp_servers}

    # 2. target tool called
    target_calls = [c for c in ev.calls if _is_target(c["name"], server, tool)]
    checks["tool_called"] = {"status": "PASS" if target_calls else "FAIL", "detail": f"{len(target_calls)} call(s)"}

    # 3. a successful call returned the reference value (with reference inputs)
    returned: list[float] = []
    call_reports = []
    for call in target_calls:
        value = _float(call["result_text"]) if call["result_text"] is not None else None
        inputs_ok = None
        if isinstance(call["input"], dict) and spec.get("tool_inputs_from_reference"):
            inputs_ok = all(str(call["input"].get(arg, "")).strip() == str(reference.get(ref_key, "")).strip()
                            for arg, ref_key in spec["tool_inputs_from_reference"].items())
        if value is not None and not call["is_error"]:
            returned.append(value)
        call_reports.append({"is_error": call["is_error"], "returned": value,
                             "abs_error": None if value is None else abs(value - ref_value),
                             "inputs_match_reference": inputs_ok,
                             "result_text": (call["result_text"] or "")[:300]})
    good = [r for r in call_reports if r["returned"] is not None and not r["is_error"] and r["abs_error"] <= tol]
    if good:
        verbatim = [r for r in good if r["inputs_match_reference"] is not False]
        checks["tool_correct"] = {"status": "PASS" if verbatim else "WARN",
                                  "detail": f"{len(good)} correct call(s)" + ("" if verbatim else
                                            "; inputs differ from reference but value within tolerance"),
                                  "calls": call_reports}
    else:
        checks["tool_correct"] = {"status": "FAIL", "detail": "no successful call returned the reference value",
                                  "calls": call_reports}

    # 4. the answer file carries a tool-returned value
    outputs_dir = run_dir / ((agent.get("persisted_outputs") or {}).get("dir") or f"{result_path.stem}.outputs")
    pred_path = outputs_dir / spec["prediction_file"]
    try:
        pred = _float(json.loads(pred_path.read_text(encoding="utf-8"))[spec["prediction_key"]])
    except (OSError, ValueError, KeyError, TypeError):
        pred = None
    out["prediction_value"] = pred
    if pred is None:
        checks["answer_from_tool"] = {"status": "FAIL", "detail": f"{spec['prediction_file']} missing or unreadable"}
    else:
        from_tool = any(abs(pred - v) <= max(1e-9, 1e-12 * abs(v)) for v in returned)
        correct = abs(pred - ref_value) <= tol
        checks["answer_from_tool"] = {
            "status": "PASS" if from_tool and correct else "FAIL",
            "detail": f"pred={pred!r} ref={ref_value!r} |d|={abs(pred - ref_value):.2e}; equals a tool-returned value: {from_tool}"}

    # 5. bypass detection over Bash commands and produced source files
    texts = [("bash", c) for c in ev.bash_commands]
    if outputs_dir.is_dir():
        for path in sorted(outputs_dir.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh", ".ipynb"}:
                texts.append((f"file:{path.relative_to(outputs_dir)}", path.read_text(encoding="utf-8", errors="replace")))
    hard = [(where, p) for where, text in texts for p in spec.get("bypass_patterns", []) if re.search(p, text)]
    soft = [(where, p) for where, text in texts for p in spec.get("suspicious_patterns", []) if re.search(p, text)]
    if hard:
        checks["no_bypass"] = {"status": "FAIL", "detail": f"direct backend use: {hard[:5]}"}
    elif soft:
        checks["no_bypass"] = {"status": "WARN", "detail": f"review: {soft[:5]}"}
    else:
        checks["no_bypass"] = {"status": "PASS", "detail": f"{len(ev.bash_commands)} Bash command(s) scanned"}
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
        marks = " ".join(f"{name}={row['checks'].get(name, {}).get('status', '-')}" for name in CHECK_ORDER)
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
    report = {"schema_version": 1, "results_dir": str(results_dir), "summary": summary, "results": rows}
    destination = Path(args.report) if args.report else results_dir / "mcp_e2e_verify.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nPASS {summary['PASS']}  FAIL {summary['FAIL']}  SKIP {summary['SKIP']}  report -> {destination}")
    return 1 if summary["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
