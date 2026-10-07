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
                    WebFetch) reached it (suspicious commands/tools are WARN).
                    Commands are scanned as executed (trajectory text; the
                    saved log has host paths replaced with placeholders); a
                    scrubbed command without trajectory text is a WARN

Persistence replaces host paths with placeholders (<abs_path>, <home>,
<workspace>, <run_output_dir>, <repo_root>). Tool results, shell commands and
path-like tool arguments are restored from the trajectory by call id; a
comparison that only a placeholder made fail is reported as unobservable
(WARN), never as a wrong result or a wrong input.

Per-task expectations come from ``e2e_check.json`` in the task directory
(examples in examples/mcp-e2e-tasks/mcp_e2e/*/). It is validated strictly when
loaded: unknown keys, invalid values and references to undefined calls make
the result FAIL with ``invalid_spec``. Schema 1 (one tool, one scalar) is
normalised to schema 2:

  server, reference_file, prediction_file
  calls     list of {name, tool, inputs_from_reference?, inputs_from_call?,
            result, optional?, group?}
  answers   list of {prediction_key, reference_key, from_call (+ result_key)
            or from_calls [{call, result_key?, select?}], abs_tol}
  server_tools                     tools the server must offer (extra = WARN)
  bypass_patterns / suspicious_patterns
                                   regexes over Bash commands and produced
                                   source files; FAIL / WARN
  bypass_tools / suspicious_tools  non-MCP (or forbidden MCP) tool name ->
                                   regex over its input; FAIL / WARN

Reading a value from a tool result works the same everywhere (a call
``result``, an answer source): ``key`` is a dotted path into the JSON result
(``result.zpe.value``; a literal key containing dots wins; no key = the whole
text result), read as a number or list of numbers; ``select`` narrows a list
field to one element (``reduce`` max/min, ``argmax_of``/``argmin_of`` another
field, or ``where_key`` equal to the reference value ``equals_reference_key``,
e.g. R at a given wavelength of a returned spectrum); ``extract`` instead
applies a named extractor (``arxiv_ids``, ``term_counts``, ``file_name``,
``exported_files``, …; see e2e_verify/extractors.py) to the whole result.

``result.format`` is ``number`` (the whole text result), ``json`` (``key`` or
``extract``) or ``image`` (+ ``media_type``); numbers match within ``abs_tol``,
extracted values with ``match`` = equal / subset / superset / member
(``superset``: every element of the reference list is among the extracted ones,
for a listing that also carries entries the task does not pin). Answers are numeric
(within ``abs_tol`` of the reference and equal to a returned value to printing
precision) unless they name an ``extract``; then ``match`` member means the
prediction is one of the returned values, and ``"merge_calls": true`` merges
the values of all calls first (e.g. one snippet call per term).

``inputs_from_call`` = {call, map: {argument: source result key}} requires each
mapped argument to equal the source field: numbers to printing precision,
other values (e.g. a ``session_id``) as exact strings, or with
``"compare": "geometry"`` as atom lists within ``abs_tol`` (default 1e-4
Angstrom). {call, extract, args} instead requires one of ``args`` to be among
the values extracted from the source result.

A call spec with ``"optional": true`` is only judged if called; optional specs
sharing a ``"group"`` count as one requirement (at least one of them must be
called and correct), e.g. reading a state with either of two tools.

Usage::

    python3 scripts/mcp/e2e/verify_run.py --results-dir out/ \
        --instances-dir instances/ --tasks-dir examples/mcp-e2e-tasks
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e2e_verify import checks, evidence, extractors, spec, values  # noqa: E402,F401
from e2e_verify import (  # noqa: E402,F401
    CHECK_ORDER, SpecError, iter_results, normalize_spec, parse_codex_stream, parse_spec, verify_one,
)


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
