"""The six checks and :func:`verify_one`.

Each check is a function ``(Context) -> dict | None`` (``None`` = not
applicable) registered in :data:`CHECKS` in report order. Checks read values
through :func:`.values.read` and compare them with the shared comparators, so
they do not depend on the value types a particular server returns.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import values
from .evidence import Evidence, ToolCall, load_evidence
from .spec import AnswerSpec, CallSpec, Spec, SpecError, label, parse_spec

SEVERITY = {"PASS": 0, "WARN": 1, "FAIL": 2}
SKIP_MARKERS = (".trajectory.", ".agent_model_output.", ".model_calls.", "local_score_")


def worst(statuses) -> str:
    return max(statuses, key=lambda s: SEVERITY[s], default="PASS")


@dataclass
class Context:
    spec: Spec
    reference: dict
    evidence: Evidence
    outputs_dir: Path
    row: dict
    by_spec: dict[str, list[ToolCall]] = field(default_factory=dict)  # call spec name -> matching calls

    def tool_name(self, cs: CallSpec) -> str:
        return f"mcp__{self.spec.server}__{cs.tool}"

    def succeeded(self, name: str) -> list[ToolCall]:
        return [c for c in self.by_spec.get(name, []) if not c.is_error]


# --------------------------------------------------------------------------
# 1. mcp_connected
# --------------------------------------------------------------------------

def check_mcp_connected(ctx: Context) -> dict:
    spec, ev = ctx.spec, ctx.evidence
    if ev.mcp_servers is None:
        # No server list (Codex JSONL, trajectory): a result returned by the
        # server's tool is the only proof that it was connected and offered.
        silent = [label(req) for req in spec.requirements()
                  if not any(c.is_error is False for cs in req for c in ctx.by_spec[cs.name])]
        if not silent:
            return {"status": "PASS", "detail": "no server list in this evidence; every required tool returned a result"}
        return {"status": "WARN", "detail": f"no server list in this evidence and no successful call of {silent}"}
    status = ev.mcp_servers.get(spec.server)
    not_offered = [ctx.tool_name(cs) for cs in spec.calls
                   if ev.tools_offered is None or ctx.tool_name(cs) not in ev.tools_offered]
    check = {"status": "PASS" if status == "connected" and not not_offered else "FAIL",
             "detail": f"server status={status!r}, not offered={not_offered}", "mcp_servers": ev.mcp_servers}
    if spec.server_tools and ev.tools_offered is not None:
        prefix = f"mcp__{spec.server}__"
        offered = {t for t in ev.tools_offered if t.startswith(prefix)}
        extra = sorted(offered - {prefix + t for t in spec.server_tools})
        if extra:
            check["detail"] += (f"; server offered {len(offered)} tools, expected "
                                f"{len(spec.server_tools)} (e.g. {extra[:3]})")
            if check["status"] == "PASS":
                check["status"] = "WARN"
    return check


# --------------------------------------------------------------------------
# 2. tool_called
# --------------------------------------------------------------------------

def check_tool_called(ctx: Context) -> dict:
    missing = [label(req) for req in ctx.spec.requirements() if not any(ctx.by_spec[cs.name] for cs in req)]
    counts = ", ".join(f"{cs.tool}×{len(ctx.by_spec[cs.name])}" for cs in ctx.spec.calls)
    return {"status": "FAIL" if missing else "PASS", "detail": (f"not called: {missing}; " if missing else "") + counts}


# --------------------------------------------------------------------------
# 3. tool_correct
# --------------------------------------------------------------------------

def judge_call(call: ToolCall, cs: CallSpec, reference: dict) -> dict:
    """One actual tool call against its spec: inputs verbatim from the reference, result correct."""
    report = {"id": call.id, "is_error": call.is_error, "result_text": (call.result_text or "")[:300]}
    inputs_ok = None
    if isinstance(call.input, dict) and cs.inputs_from_reference:
        inputs_ok = all(values.same_input(call.input.get(arg), reference.get(ref_key))
                        for arg, ref_key in cs.inputs_from_reference.items())
        # An argument only available scrubbed (a host path the trajectory could not
        # restore) carries nothing to compare: leave the verdict open rather than
        # reporting inputs that "differ from the reference".
        if not inputs_ok and call.input_lost(*cs.inputs_from_reference):
            report["inputs_scrubbed"] = True
            inputs_ok = None
    report["inputs_match_reference"] = inputs_ok
    rs = cs.result
    if call.is_error or (call.result_text is None and rs.format != "image"):
        report["result_ok"] = False
    elif rs.format == "image":
        if call.content_types is None:
            report["result_ok"] = None  # not observable in this evidence
        else:
            images = call.media_types or []
            report["media_types"] = images
            report["result_ok"] = bool(images) and (rs.media_type is None or rs.media_type in images)
    else:
        value = values.read(call, rs.selector, reference)
        if rs.selector.extract:
            ref = values.canon(rs.selector.extract)(reference.get(rs.reference_key))
            report["extracted"] = value
            report["result_ok"] = values.matches(value, ref, rs.match)
        else:
            err = values.diff(value, values.numbers(reference.get(rs.reference_key)))
            report["abs_error"] = None if math.isinf(err) else err
            report["result_ok"] = err <= rs.abs_tol
        # A result the persistence replaced with placeholders (a returned host path) and
        # that no trajectory restored carries nothing to compare: a coverage gap, not a
        # wrong result.
        if not report["result_ok"] and call.result_lost:
            report["result_scrubbed"] = True
            report["result_ok"] = None
    return report


def check_tool_correct(ctx: Context) -> dict:
    spec, per_call, judged = ctx.spec, {}, {}
    for cs in spec.calls:
        reports = [judge_call(c, cs, ctx.reference) for c in ctx.by_spec[cs.name]]
        good = [r for r in reports if r["result_ok"]]
        verbatim = [r for r in good if r["inputs_match_reference"] is not False]
        if not reports and cs.optional:
            status, detail = None, "not called (optional)"
        elif verbatim:
            status, detail = "PASS", f"{len(good)} correct call(s)"
        elif good:
            status, detail = "WARN", "result correct but inputs differ from reference"
        elif any(r["result_ok"] is None for r in reports):
            status, detail = "WARN", "result not observable in this evidence"
        else:
            status, detail = "FAIL", "no successful call returned the expected result"
        judged[cs.name] = status
        per_call[cs.name] = {"tool": cs.tool, "status": status or "SKIP", "detail": detail, "calls": reports}
    statuses, covered = [], set()
    for cs in spec.calls:  # ungrouped optional specs count only if called
        if cs.optional and not cs.group and judged[cs.name] is not None:
            statuses.append(judged[cs.name])
    for req in spec.requirements():  # a group takes its best called member
        called = [judged[cs.name] for cs in req if judged[cs.name] is not None]
        best = min(called, key=lambda s: SEVERITY[s]) if called else "FAIL"
        statuses.append(best)
        if len(req) > 1 and best == "PASS":
            covered.update(cs.name for cs in req)
    failing = [f"{n}: {c['detail']}" for n, c in per_call.items()
               if c["status"] not in ("PASS", "SKIP") and n not in covered]
    return {"status": worst(statuses), "detail": "; ".join(failing) or "all required tools returned expected results",
            "per_call": per_call}


# --------------------------------------------------------------------------
# 4. tool_chain
# --------------------------------------------------------------------------

def _judge_link(ctx: Context, cs: CallSpec) -> tuple[str, str]:
    link = cs.inputs_from_call
    where = f"{cs.name}←{link.call}"
    sources = [c for c in ctx.succeeded(link.call) if all(values.readable(c, b.source) for b in link.bindings)]
    consumers = ctx.by_spec.get(cs.name, [])
    if not consumers or not sources:
        return "FAIL", f"{where}: no {'consumer' if not consumers else 'source'} call"
    compare = [(b, values.link_comparator(b)) for b in link.bindings]
    args = [arg for b in link.bindings for arg in b.args]
    observed = [c for c in consumers if isinstance(c.input, dict)]
    if not observed:
        return "WARN", f"{where}: tool inputs not observable in this evidence"
    for consumer in observed:
        for source in sources:
            if all(any(consumer.input.get(arg) is not None and same(consumer.input[arg], values.read(source, b.source))
                       for arg in b.args) for b, same in compare):
                return "PASS", f"{where}: {args} equal a {link.call} result"
    seen = [{arg: c.input.get(arg) for arg in args if c.input.get(arg) is not None} for c in observed]
    if any(values.link_input_lost(c, b) for c in observed for b in link.bindings):
        return "WARN", (f"{where}: {args} are only available scrubbed in this evidence "
                        f"({seen[:3]}), so the link is not checkable")
    return "FAIL", f"{where}: {args} do not equal any of {len(sources)} {link.call} result(s); inputs seen: {seen[:3]}"


def check_tool_chain(ctx: Context) -> dict | None:
    chained = [cs for cs in ctx.spec.calls if cs.inputs_from_call
               and not (cs.optional and not ctx.by_spec.get(cs.name))]
    if not chained:
        return None
    judged = [_judge_link(ctx, cs) for cs in chained]
    return {"status": worst(s for s, _ in judged), "detail": "; ".join(d for _, d in judged)}


# --------------------------------------------------------------------------
# 5. answer_from_tool
# --------------------------------------------------------------------------

def judge_answer(ctx: Context, answer: AnswerSpec, prediction: dict) -> tuple[str, str, object, object]:
    canon = values.canon(answer.kind)
    pred = canon(prediction.get(answer.prediction_key))
    ref = canon(ctx.reference.get(answer.reference_key))
    returned = [v for src in answer.sources for c in ctx.succeeded(src.call)
                for v in [values.read(c, src.selector, ctx.reference)] if v is not None]
    if answer.merge_calls and returned:
        merged: dict = {}
        for value in returned:  # e.g. one snippet call per term
            if isinstance(value, dict):
                merged.update(value)
        returned = [merged]
    if answer.match == "member":
        from_tool = pred is not None and any(values.matches(v, pred, "member") for v in returned)
    elif answer.kind == "number":
        from_tool = any(values.copied(pred, v) for v in returned)
    else:
        from_tool = pred is not None and any(v == pred for v in returned)
    if answer.kind == "number":
        err = values.diff(pred, ref)
        correct, versus = err <= answer.abs_tol, f"|d|={err:.2e} vs reference"
    else:
        correct = pred is not None and pred == ref
        versus = f"matches reference: {correct}"
    detail = f"{answer.prediction_key}: " + ("missing or malformed" if pred is None else
                                             f"{versus}, equals a tool-returned value: {from_tool}")
    return ("PASS" if from_tool and correct else "FAIL"), detail, pred, ref


def check_answer_from_tool(ctx: Context) -> dict:
    spec, row = ctx.spec, ctx.row
    try:
        prediction = json.loads((ctx.outputs_dir / spec.prediction_file).read_text(encoding="utf-8"))
        if not isinstance(prediction, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as exc:
        row["prediction_value"] = None
        return {"status": "FAIL", "detail": f"{spec.prediction_file} missing or unreadable ({exc})"}
    judged = [judge_answer(ctx, answer, prediction) for answer in spec.answers]
    row["prediction_value"], row["reference_value"] = judged[0][2], judged[0][3]
    return {"status": worst(s for s, *_ in judged), "detail": "; ".join(d for _, d, *_ in judged)}


# --------------------------------------------------------------------------
# 6. no_bypass
# --------------------------------------------------------------------------

def check_no_bypass(ctx: Context) -> dict:
    spec, ev = ctx.spec, ctx.evidence
    # Scan commands as executed; a persisted command whose paths were scrubbed and that
    # the trajectory cannot restore hides path-based patterns: a coverage gap (WARN).
    texts = [("bash", c.raw if c.raw is not None else c.text) for c in ev.commands]
    gaps = [c.text for c in ev.commands if c.scrubbed]
    if ctx.outputs_dir.is_dir():
        for path in sorted(ctx.outputs_dir.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh", ".ipynb"}:
                texts.append((f"file:{path.relative_to(ctx.outputs_dir)}",
                              path.read_text(encoding="utf-8", errors="replace")))
    hard = [(where, p) for where, text in texts for p in spec.bypass_patterns if re.search(p, text)]
    soft = [(where, p) for where, text in texts for p in spec.suspicious_patterns if re.search(p, text)]
    for call in ev.calls:  # non-MCP tools such as WebFetch / Codex web_search
        payload = json.dumps(call.input, ensure_ascii=False)
        hard += [(f"tool:{call.name}", p) for tool, p in spec.bypass_tools.items()
                 if call.name == tool and re.search(p, payload)]
        soft += [(f"tool:{call.name}", p) for tool, p in spec.suspicious_tools.items()
                 if call.name == tool and re.search(p, payload)]
    if hard:
        return {"status": "FAIL", "detail": f"direct backend use: {hard[:5]}"}
    if soft or gaps:
        notes = [f"review: {soft[:5]}"] if soft else []
        if gaps:
            notes.append(f"{len(gaps)} command(s) have scrubbed paths and no trajectory text, so path-based "
                         f"patterns are not checkable: {gaps[:3]}")
        return {"status": "WARN", "detail": "; ".join(notes)}
    return {"status": "PASS", "detail": f"{len(ev.commands)} shell command(s) scanned"}


CHECKS: tuple[tuple[str, Callable[[Context], dict | None]], ...] = (
    ("mcp_connected", check_mcp_connected),
    ("tool_called", check_tool_called),
    ("tool_correct", check_tool_correct),
    ("tool_chain", check_tool_chain),
    ("answer_from_tool", check_answer_from_tool),
    ("no_bypass", check_no_bypass),
)
CHECK_ORDER = tuple(name for name, _ in CHECKS)


# --------------------------------------------------------------------------
# One result
# --------------------------------------------------------------------------

def load_spec(task_dir: Path) -> Spec | None:
    path = task_dir / "e2e_check.json"
    if not path.is_file():
        return None
    return parse_spec(json.loads(path.read_text(encoding="utf-8")), str(path))


def verify_one(result_path: Path, result: dict, instances_dir: Path, tasks_dir: Path) -> dict:
    task_id = result["task_id"]
    task_dir = tasks_dir / Path(*task_id.split("."))
    row = {"result_file": str(result_path), "task_id": task_id, "instance_id": result.get("instance_id"),
           "prompt_level": result.get("prompt_level"), "attempt": result.get("attempt"),
           "run_status": result.get("status"), "checks": {}, "notes": []}
    try:
        spec = load_spec(task_dir)
    except SpecError as exc:
        row.update(verdict="FAIL", failure="invalid_spec")
        row["notes"].append(str(exc))
        return row
    if spec is None:
        row["verdict"] = "SKIP"
        row["notes"].append(f"no e2e_check.json in {task_dir}")
        return row
    reference = json.loads((instances_dir / result["instance_id"] / "reference" / spec.reference_file)
                           .read_text(encoding="utf-8"))

    agent = result.get("agent_output") or {}
    ev = load_evidence(result_path, result)
    if ev is None:
        row.update(verdict="FAIL", failure="no_evidence")
        row["notes"].append("neither raw stdout nor trajectory artefact found")
        return row
    row["evidence_source"] = ev.source
    row["tool_sequence"] = [call.name for call in ev.calls]
    if ev.tools_offered is not None:
        row["toolsearch_offered"] = "ToolSearch" in ev.tools_offered
        row["mcp_tools_offered"] = [t for t in ev.tools_offered if t.startswith("mcp__")]
    row["tool_call_counts"] = {}
    for call in ev.calls:
        row["tool_call_counts"][call.name] = row["tool_call_counts"].get(call.name, 0) + 1
    if ev.final_result:
        row["agent_final"] = ev.final_result
    row["permission_mode"] = ev.permission_mode
    row["persisted_outputs"] = agent.get("persisted_outputs")
    row["agent_status"] = agent.get("status")
    row["agent_error"] = agent.get("error_message")
    if ev.assistant_text:
        row["last_assistant_text"] = ev.assistant_text[-1][:1000]

    outputs_dir = result_path.parent / ((agent.get("persisted_outputs") or {}).get("dir")
                                        or f"{result_path.stem}.outputs")
    ctx = Context(spec, reference, ev, outputs_dir, row)
    ctx.by_spec = {cs.name: [c for c in ev.calls if c.name == ctx.tool_name(cs)] for cs in spec.calls}
    for name, check in CHECKS:
        outcome = check(ctx)
        if outcome is not None:
            row["checks"][name] = outcome
    row["bash_commands"] = ev.bash_commands[:50]
    row["raw_bash_commands"] = [c.raw if c.raw is not None else c.text for c in ev.commands][:50]

    failed = [name for name in CHECK_ORDER if row["checks"].get(name, {}).get("status") == "FAIL"]
    row["verdict"] = "FAIL" if failed else "PASS"
    if failed:
        row["failure"] = failed[0]
    return row


def iter_results(results_dir: Path):
    """``(path, result)`` for every run result JSON under ``results_dir/<task>/``."""
    for path in sorted(results_dir.glob("*/*.json")):
        if any(marker in path.name for marker in SKIP_MARKERS):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("task_id") and data.get("instance_id"):
            yield path, data
