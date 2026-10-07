"""openroad_tiny_floorplan (session-id chain, facts read out of PTY text, a grouped query/exec
area report, Tcl ``exec`` through the session as a bypass): generator, scorer, verifier
scenarios. No MCP server, no openroad; the B1 Tcl runs under ``tclsh`` against a fake odb."""
import json
import re
import shutil
import subprocess

import pytest

from . import support
from .support import claude, codex, jsonl

TASK = support.Task("mcp_e2e.openroad_tiny_floorplan")
TASK_DIR = TASK.dir
SERVER = "openroad"

generate_gt = TASK.module("generate_gt")
scorer = TASK.module("custom_scorer")
L1 = support.smoke_module("openroad")
verify = support.verify
_status = support.statuses

CASE = generate_gt.build_case(31415)
REF = {**CASE, **generate_gt.reference(CASE)}
SID = CASE["session_id"]
TOOLS = ("cancel_orfs_job", "create_interactive_session", "get_orfs_job", "get_session_history",
         "get_session_metrics", "grep_session_output", "inspect_interactive_session", "interactive_openroad_exec",
         "interactive_openroad_query", "list_interactive_sessions", "list_report_images", "read_orfs_metrics",
         "read_report_image", "run_orfs_stage", "terminate_interactive_session")
ANSWER = {key: REF[key] for key in ("design", "session_id", "instance_count", "net_count", "io_pin_count",
                                     "die_width_dbu", "die_height_dbu", "design_area_um2",
                                     "utilization_percent", "hpwl_total_dbu", "area_report_line")}


def _b1_tcl() -> list[str]:
    return re.findall(r"```tcl\n\s*(.+?)\n\s*```", (TASK_DIR / "prompt_b1.md").read_text())


DIE_TCL, PROC_TCL = _b1_tcl()
HPWL_CALL = 'puts "HPWL_TOTAL [or_hpwl]"'


# --------------------------------------------------------------------------
# Generator and closed form
# --------------------------------------------------------------------------

def test_cases_are_deterministic_and_varied():
    assert generate_gt.build_case(31415) == CASE
    cases = [generate_gt.build_case(s) for s in range(200)]
    refs = [generate_gt.reference(c) for c in cases]
    assert len({json.dumps(c, sort_keys=True) for c in cases}) == 200
    assert {len(c["components"]) for c in cases} == {4, 5, 6, 7} and {len(c["rows"]) for c in cases} == {1, 2}
    assert {m for c in cases for _, m, _, _ in c["components"]} == {"INV", "BUF"}
    assert any(r["def_facts"][4] != f"connections={2 * len(c['components'])}" for c, r in zip(cases, refs))  # fan-out
    assert len({r["hpwl_total_dbu"] for r in refs}) > 50 and len({r["utilization_percent"] for r in refs}) > 20


def test_generator_reproduces_the_l1_fixture_validated_on_the_host():
    """With the L1 smoke's constants the renderers write its TINY_LEF / TINY_DEF byte for byte,
    and the reference equals the L1 closed form (9400 DBU, 6 u^2, 30 %) checked on real openroad."""
    assert generate_gt.render_lef(["INV"]) == L1.TINY_LEF
    case = {"design": "tiny", "die": list(L1.DIE), "rows": [list(L1.ROW_ORIGIN)], "row_sites": L1.ROW_SITES,
            "components": [[n, "INV", x, y] for n, (x, y) in L1.INSTANCES.items()],
            "pins": {k: list(v) for k, v in L1.BTERMS.items()}, "library": ["INV"],
            "nets": [[n, [list(t) for t in terms]] for n, terms in L1.NETS.items()]}
    assert generate_gt.render_def(case) == L1.TINY_DEF
    ref, cf = generate_gt.reference(case), L1.closed_form()
    assert ref["hpwl_per_net_dbu"] == cf["hpwl"] and ref["hpwl_total_dbu"] == cf["hpwl_total"] == 9400
    assert ref["core"] == list(cf["core"]) and ref["area_report_line"] == "Design area 6 u^2 30% utilization."
    assert ref["def_facts"] == ["pins=2", "components=3", "component_terminals=6", "nets=4", "connections=6"]
    assert generate_gt.MACROS["INV"] == (L1.CELL[0], L1.CELL_PINS)


def _parse_def(text: str) -> dict:
    """An independent reading of the DEF text (not the case dict the generator rendered it from)."""
    comps = {m[0]: (m[1], int(m[2]), int(m[3]))
             for m in re.findall(r"^- (\w+) (\w+) \+ PLACED \( (\d+) (\d+) \) N ;$", text, re.M)}
    pins = {m[0]: (int(m[1]), int(m[2])) for m in re.findall(r"^- (\w+) \+ NET.*?\+ PLACED \( (\d+) (\d+) \)",
                                                              text, re.M | re.S)}
    nets = {m[0]: re.findall(r"\( (\w+) (\w+) \)", m[1]) for m in re.findall(r"^- (\w+) ((?:\( \w+ \w+ \) )+)\+ USE",
                                                                             text, re.M)}
    rows = [(int(x), int(y), int(n)) for x, y, n in re.findall(r"^ROW \w+ core (\d+) (\d+) N DO (\d+)", text, re.M)]
    die = [int(v) for v in re.search(r"DIEAREA \( (\d+) (\d+) \) \( (\d+) (\d+) \)", text).groups()]
    return {"comps": comps, "pins": pins, "nets": nets, "rows": rows, "die": die}


def _parse_lef(text: str) -> dict:
    macros = {}
    for name, body in re.findall(r"^MACRO (\w+)\n(.*?)^END \1$", text, re.M | re.S):
        width = round(float(re.search(r"SIZE ([\d.]+) BY", body).group(1)) * 1000)
        rects = {p: tuple(round(float(v) * 1000) for v in r.split())
                 for p, r in re.findall(r"PIN (\w+)\n.*?RECT ([\d. ]+) ;", body, re.S)}
        macros[name] = (width, rects)
    return macros


@pytest.mark.parametrize("seed", [31415, 0, 1, 7, 42, 99, 123, 2024])
def test_reference_matches_an_independent_reading_of_the_files(tmp_path, seed):
    generate_gt.generate(tmp_path, {"seed": seed})
    ref = json.loads((tmp_path / "reference/reference.json").read_text())
    d = _parse_def((tmp_path / "data/design.def").read_text())
    macros = _parse_lef((tmp_path / "data/design.lef").read_text())
    assert set(macros) == {"INV", "BUF"} and "LAYER metal1" in (tmp_path / "data/design.lef").read_text()

    def centre(owner, pin):
        if owner == "PIN":
            x, y = d["pins"][pin]
            return x, y                                                     # shape is symmetric about the pin
        master, x, y = d["comps"][owner]
        r = macros[master][1][pin]
        assert (r[0] + r[2]) % 2 == 0 and (r[1] + r[3]) % 2 == 0
        return x + (r[0] + r[2]) // 2, y + (r[1] + r[3]) // 2

    total = 0
    for terms in d["nets"].values():
        pts = [centre(o, p) for o, p in terms]
        total += max(p[0] for p in pts) - min(p[0] for p in pts) + max(p[1] for p in pts) - min(p[1] for p in pts)
    cell = sum(macros[m][0] * 2000 for m, _, _ in d["comps"].values())
    x0, y0, sites = d["rows"][0]
    core = sites * 200 * 2000 * len(d["rows"])
    assert all(r[0] == x0 and r[2] == sites for r in d["rows"])
    assert [r[1] for r in d["rows"]] == [y0 + 2000 * k for k in range(len(d["rows"]))]   # contiguous rows
    util = 100 * cell / core
    assert abs(util - int(util) - 0.5) >= 0.05
    assert ref["hpwl_total_dbu"] == total and ref["design_area_um2"] == cell // 10 ** 6 == cell / 10 ** 6
    assert ref["utilization_percent"] == round(util)
    assert [ref["die_width_dbu"], ref["die_height_dbu"]] == d["die"][2:] and d["die"][:2] == [0, 0]
    assert ref["instance_count"] == len(d["comps"]) and ref["net_count"] == len(d["nets"])
    assert ref["def_facts"][4] == f"connections={sum(o != 'PIN' for t in d['nets'].values() for o, _ in t)}"
    # placements: on the site grid, inside a row, no overlaps; pins inside the die
    for row_y in {y for _, _, y in d["comps"].values()}:
        spans = sorted((x, x + macros[m][0]) for m, x, y in d["comps"].values() if y == row_y)
        assert all((x - x0) % 200 == 0 and x0 <= x and end <= x0 + sites * 200 for x, end in spans)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    assert all(100 <= x <= d["die"][2] - 100 and 400 <= y <= d["die"][3] - 400 for x, y in d["pins"].values())


def test_generated_files_and_reference_carry_every_key_the_checks_use(tmp_path):
    meta = generate_gt.generate(tmp_path, {"seed": 31415})
    assert meta["input_files"] == ["task.json", "design.lef", "design.def"]
    assert sorted(p.name for p in (tmp_path / "data").iterdir()) == ["design.def", "design.lef", "task.json"]
    task = json.loads((tmp_path / "data/task.json").read_text())
    assert task == {"design": CASE["design"], "lef_file": "data/design.lef", "def_file": "data/design.def",
                    "session_id": "or31415", "grep_pattern": "^Design area"}
    ref = json.loads((tmp_path / "reference/reference.json").read_text())
    spec = json.loads((TASK_DIR / "e2e_check.json").read_text())
    keys = {cs["result"]["reference_key"] for cs in spec["calls"]}
    keys |= {r for cs in spec["calls"] for r in cs.get("inputs_from_reference", {}).values()}
    keys |= {a["reference_key"] for a in spec["answers"]}
    assert keys <= set(ref) and set(scorer.NUMBERS) | set(scorer.TEXTS) <= set(ref)
    for level in ("b1", "b2", "b3", "b4"):
        text = (tmp_path / f"prompt_{level}.md").read_text()
        assert "{{" not in text and CASE["design"] in text


def test_prompts_keep_the_rules_and_name_tools_by_level():
    for level in ("b1", "b2", "b3", "b4"):
        text = (TASK_DIR / f"prompt_{level}.md").read_text()
        assert "every number in `result.json` must be one that OpenROAD printed in the session" in text
        assert "do not use Tcl `exec`" in text and "Do not run the `openroad` binary" in text
        assert "set_cmd_units -distance um" in text and "pin shapes" in text
        assert ("`create_interactive_session`" in text) == (level in ("b1", "b2"))
        assert ("proc or_hpwl" in text) == (level == "b1")
        assert "{{session_id}}" in text or level == "b4"


@pytest.mark.skipif(shutil.which("tclsh") is None, reason="tclsh not installed")
def test_b1_tcl_prints_the_reference_against_a_fake_odb(tmp_path):
    """The prompt's one-line Tcl (die box, HPWL procedure) over an odb stand-in built from the
    case: it must print the closed-form die and HPWL, so B1 asks for the right computation."""
    def rect(name, r):
        return (f"proc {name} {{m}} {{ switch $m {{ xMin {{return {r[0]}}} yMin {{return {r[1]}}} "
                f"xMax {{return {r[2]}}} yMax {{return {r[3]}}} }} }}\n")
    lines = ["namespace eval ord { proc get_db_block {} { return blk } }\n", rect("die", CASE["die"])]
    nets, k = [], 0
    for name, terms in CASE["nets"]:
        iterms, bterms = [], []
        for owner, pin in terms:
            k += 1
            x, y = generate_gt.terminal_centre(CASE, owner, pin)
            lines.append(rect(f"r{k}", (x - 50, y - 50, x + 50, y + 50)))
            lines.append(f"proc t{k} {{m}} {{ return r{k} }}\n")
            (bterms if owner == "PIN" else iterms).append(f"t{k}")
        lines.append(f"proc net_{name} {{m}} {{ switch $m {{ getITerms {{return {{{' '.join(iterms)}}}}} "
                     f"getBTerms {{return {{{' '.join(bterms)}}}}} }} }}\n")
        nets.append(f"net_{name}")
    lines.append(f"proc blk {{m}} {{ switch $m {{ getDieArea {{return die}} getNets {{return {{{' '.join(nets)}}}}} }} }}\n")
    script = tmp_path / "b1.tcl"
    script.write_text("".join(lines) + "\n".join([DIE_TCL, PROC_TCL, HPWL_CALL]) + "\n")
    out = subprocess.run(["tclsh", str(script)], capture_output=True, text=True, check=True).stdout.split("\n")
    assert out[:2] == [f"DIE 0 0 {REF['die_width_dbu']} {REF['die_height_dbu']}",
                       f"HPWL_TOTAL {REF['hpwl_total_dbu']}"]
    assert "\n" not in PROC_TCL and "{{" not in PROC_TCL and "exec" not in PROC_TCL


def test_task_meta_is_test_status_without_packages():
    import yaml
    meta = yaml.safe_load((TASK_DIR / "task_meta.yaml").read_text())
    assert meta["status"] == "test" and meta["runtime"]["packages"] == []
    assert [f["name"] for f in meta["input"]["files"]] == ["task.json", "design.lef", "design.def"]


# --------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------

def _dirs(tmp_path, prediction, reference=REF):
    return support.score_dirs(tmp_path, prediction, reference)


def test_exact_answers_score_full_and_each_field_carries_its_share(tmp_path):
    assert TASK.total(*_dirs(tmp_path / "a", ANSWER)) == pytest.approx(100.0)
    floats = {**ANSWER, "design_area_um2": float(ANSWER["design_area_um2"]), "die_width_dbu": "13000"}
    assert TASK.total(*_dirs(tmp_path / "b", floats)) == pytest.approx(100.0)
    assert TASK.total(*_dirs(tmp_path / "c", {**ANSWER, "hpwl_total_dbu": ANSWER["hpwl_total_dbu"] + 200})) == \
        pytest.approx(70.0)
    zero_area = {**ANSWER, "design_area_um2": 0, "area_report_line": "Design area 0 u^2 36% utilization."}
    assert TASK.total(*_dirs(tmp_path / "d", zero_area)) == pytest.approx(100 - 10 - 15)
    assert TASK.total(*_dirs(tmp_path / "e", {**ANSWER, "net_count": 99})) == pytest.approx(100 - 25 / 3)


@pytest.mark.parametrize("prediction", [None, "not json", [], {**ANSWER, "hpwl_total_dbu": None},
                                        {**ANSWER, "instance_count": True}, {**ANSWER, "area_report_line": ""},
                                        {**ANSWER, "utilization_percent": "36%"},
                                        {k: v for k, v in ANSWER.items() if k != "die_height_dbu"}])
def test_submission_failures_are_valid_zero_scores(tmp_path, prediction):
    from ai4sci_bench.core.scorer import get_scorer
    pred, ref = _dirs(tmp_path, prediction)
    gate = get_scorer("openroad_e2e_schema").score(pred, ref, {"weight": 1.0})
    assert gate.score == 0.0 and not gate.details.get("scorer_internal_error")
    for item in TASK.eval_config()["scoring"]:
        detail = get_scorer(item["scorer"]).score(pred, ref, {**item["config"], "weight": item["weight"]})
        assert detail.score == 0.0 and not detail.details.get("scorer_internal_error")


@pytest.mark.parametrize("reference", [None, {"design": "x"}, {**REF, "hpwl_total_dbu": 0}])
def test_missing_or_broken_reference_is_an_evaluator_failure(tmp_path, reference):
    from ai4sci_bench.core.scorer import get_scorer
    pred, ref = _dirs(tmp_path, ANSWER, reference=reference)
    for item in TASK.eval_config()["scoring"]:
        detail = get_scorer(item["scorer"]).score(pred, ref, {**item["config"], "weight": item["weight"]})
        assert detail.details["scorer_internal_error"]
        assert detail.details["failure_kind"] == "missing_evaluator_input"


# --------------------------------------------------------------------------
# Verifier scenarios
# --------------------------------------------------------------------------

class FakeServer:
    """OpenROAD-MCP v1.1.0 result shapes as recorded on the AWS host (2026-10-07): compact JSON,
    ``output`` = the PTY echo, what the command printed, a bare ``%`` prompt."""

    def __init__(self, units=False, scripted=None):
        self.units, self.count, self.printed = units, 0, []
        self.scripted = scripted or {}                    # command -> printed lines (an agent's own Tcl)

    def result(self, tool, args):
        if tool == "create_interactive_session":
            return {"session_id": args.get("session_id", "a1b2c3d4"), "created_at": "2026-10-07T11:30:10.648Z",
                    "is_alive": True, "command_count": 0, "buffer_size": 0, "uptime_seconds": 0.145,
                    "state": "active", "error": None}
        if tool == "grep_session_output":
            matches = [m for m in self.printed if re.search(args["pattern"], m[2])]
            return {"session_id": args["session_id"], "pattern": args["pattern"],
                    "matches": [{"command_number": n, "command": c, "line_number": 2, "line": line}
                                for n, c, line in matches],
                    "total_matches": len(matches), "truncated": False, "pattern_kind": "regex",
                    "searched_commands": self.count, "searched_lines": 3 * self.count, "retained_chars": 900,
                    "evicted_commands": 0, "message": None, "error": None}
        if tool == "terminate_interactive_session":
            return {"session_id": args["session_id"], "terminated": True, "was_alive": True, "force": False,
                    "error": None}
        command = args["command"]
        self.count += 1
        lines, error = self.run(command)
        for line in lines:
            self.printed.append((self.count, command, line))
        output = "\n".join([command, *lines, "%"])
        return {"output": output, "session_id": args.get("session_id"), "timestamp": "2026-10-07T11:30:10.804Z",
                "execution_time": 0.003, "command_count": self.count, "buffer_size": 131072, "truncated": False,
                "bytes_discarded": 0, "total_bytes": len(output), "error": error}

    def run(self, command):
        if command in self.scripted:
            return self.scripted[command], None
        if command.startswith("read_lef ") and "; read_def " in command:         # both files in one command
            lef, deff = (part.split(" ", 1)[1] for part in command.split("; "))
            return self.run(f"read_lef {lef}")[0] + self.run(f"read_def {deff}")[0], None
        if command.startswith("read_lef "):
            return [f"[INFO ODB-0227] LEF file: {command[9:]}, created 1 layers, 2 library cells"], None
        if command.startswith("read_def "):
            path = command[9:]
            facts = dict(f.split("=") for f in REF["def_facts"])
            return [f"[INFO ODB-0127] Reading DEF file: {path}", f"[INFO ODB-0128] Design: {CASE['design']}",
                    f"[INFO ODB-0130]     Created {facts['pins']} pins.",
                    f"[INFO ODB-0131]     Created {facts['components']} components and "
                    f"{facts['component_terminals']} component-terminals.",
                    f"[INFO ODB-0133]     Created {facts['nets']} nets and {facts['connections']} connections.",
                    f"[INFO ODB-0134] Finished DEF file: {path}"], None
        if command == "set_cmd_units -distance um":
            self.units = True
            return [], None
        if command == "report_design_area":
            area = REF["design_area_um2"] if self.units else 0
            return [f"Design area {area} u^2 {REF['utilization_percent']}% utilization."], None
        if command == DIE_TCL:
            return [f"DIE 0 0 {REF['die_width_dbu']} {REF['die_height_dbu']}"], None
        if command == HPWL_CALL:
            return [f"HPWL_TOTAL {REF['hpwl_total_dbu']}"], None
        if command.startswith("puts "):
            return [command[5:].strip('"')], None
        return [], None


def _b1_calls(workdir="/home/e2e/run/ws", area_tool="interactive_openroad_query", units=True, sid=SID,
              exec_sid=SID, hpwl=True):
    ex = "interactive_openroad_exec"
    calls = [("create_interactive_session", {"session_id": sid}),
             (ex, {"session_id": exec_sid, "command": f"read_lef {workdir}/data/design.lef"}),
             (ex, {"session_id": exec_sid, "command": f"read_def {workdir}/data/design.def"})]
    if units:
        calls.append((ex, {"session_id": exec_sid, "command": "set_cmd_units -distance um"}))
    calls += [(area_tool, {"session_id": exec_sid, "command": "report_design_area"}),
              (ex, {"session_id": exec_sid, "command": DIE_TCL})]
    if hpwl:
        calls += [(ex, {"session_id": exec_sid, "command": PROC_TCL}),
                  (ex, {"session_id": exec_sid, "command": HPWL_CALL})]
    calls += [("grep_session_output", {"session_id": exec_sid, "pattern": "^Design area"}),
              ("terminate_interactive_session", {"session_id": exec_sid})]
    return calls


def _payloads(calls, server=None):
    server = server or FakeServer()
    return [json.dumps(server.result(tool, args), separators=(",", ":")) for tool, args in calls]


def _stream(calls, extra=(), server=None):
    events = [claude.init(SERVER, ["Bash", "Read", "Write", "WebFetch", "WebSearch"]
                          + [f"mcp__{SERVER}__{t}" for t in TOOLS])]
    for k, (name, args) in enumerate(extra):
        events += claude.call(f"x{k}", name, args, "ok")
    for k, ((tool, args), text) in enumerate(zip(calls, _payloads(calls, server))):
        events += claude.call(f"c{k}", f"mcp__{SERVER}__{tool}", args, text)
    events.append(claude.result(len(calls)))
    return jsonl(events)


def _codex(calls, extra=(), server=None):
    events = list(codex.START)
    for k, (name, args) in enumerate(extra):
        events += codex.shell(f"s{k}", args["command"])
    for k, ((tool, args), text) in enumerate(zip(calls, _payloads(calls, server))):
        events += codex.mcp(f"m{k}", SERVER, tool, args, text)
    events.append(codex.done())
    return jsonl(events)


def _run(tmp_path, stream, answer=None, harness="claude", **kw):
    return TASK.verify(tmp_path, stream, reference=REF, answer=answer or ANSWER, harness=harness, **kw)


def test_genuine_b1_run_passes_every_check(tmp_path):
    row = _run(tmp_path, _stream(_b1_calls()))
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}
    per_call = row["checks"]["tool_correct"]["per_call"]
    assert per_call["area_query"]["status"] == "PASS" and per_call["area_exec"]["status"] == "FAIL"   # group: best


def test_genuine_run_in_the_workspace_survives_path_scrubbing(tmp_path):
    """read_lef / read_def name host paths: the persisted log scrubs them in the arguments and
    results, the trajectory restores both, and the facts never depended on path digits."""
    workdir = str(tmp_path / "ws")
    row = _run(tmp_path, _stream(_b1_calls(workdir=workdir)))
    assert row["verdict"] == "PASS" and set(_status(row).values()) == {"PASS"}, row["checks"]


def test_genuine_codex_run_passes_also_without_a_trajectory(tmp_path):
    row = _run(tmp_path / "a", _codex(_b1_calls()), harness="codex")
    assert row["verdict"] == "PASS" and set(_status(row).values()) == {"PASS"}, row["checks"]
    workdir = str(tmp_path / "b" / "ws")
    row = _run(tmp_path / "b", _codex(_b1_calls(workdir=workdir)), harness="codex", trajectory=False)
    assert row["verdict"] == "PASS" and _status(row)["tool_correct"] == "PASS", row["checks"]


def test_area_reported_through_the_exec_tool_satisfies_the_group(tmp_path):
    row = _run(tmp_path, _stream(_b1_calls(area_tool="interactive_openroad_exec")))
    assert row["verdict"] == "PASS" and set(_status(row).values()) == {"PASS"}, row["checks"]
    assert row["checks"]["tool_correct"]["per_call"]["area_query"]["status"] == "SKIP"


def test_area_without_micron_units_is_wrong(tmp_path):
    calls = _b1_calls(units=False)
    answer = {**ANSWER, "design_area_um2": 0,
              "area_report_line": f"Design area 0 u^2 {REF['utilization_percent']}% utilization."}
    row = _run(tmp_path, _stream(calls), answer=answer)
    assert _status(row)["tool_correct"] == "FAIL" and _status(row)["answer_from_tool"] == "FAIL"
    assert "area_query" in row["checks"]["tool_correct"]["detail"]
    assert "design_area_um2" in row["checks"]["answer_from_tool"]["detail"]


def test_an_own_session_id_only_warns_but_commands_on_another_session_break_the_chain(tmp_path):
    row = _run(tmp_path / "a", _stream(_b1_calls(sid="mine", exec_sid="mine")), answer={**ANSWER, "session_id": "mine"})
    assert row["verdict"] == "PASS" and _status(row)["tool_correct"] == "WARN", row["checks"]
    row = _run(tmp_path / "b", _stream(_b1_calls(exec_sid="a1b2c3d4")))
    assert _status(row)["tool_chain"] == "FAIL" and "read_lef←create" in row["checks"]["tool_chain"]["detail"]


def test_hpwl_computed_outside_the_session_is_not_from_the_tool(tmp_path):
    row = _run(tmp_path, _stream(_b1_calls(hpwl=False)))
    assert _status(row)["tool_correct"] == "FAIL" and "hpwl" in row["checks"]["tool_correct"]["detail"]
    assert _status(row)["answer_from_tool"] == "FAIL"
    detail = row["checks"]["answer_from_tool"]["detail"]
    assert "hpwl_total_dbu: matches reference: True, equals a tool-returned value: False" in detail
    assert "instance_count: matches reference: True, equals a tool-returned value: True" in detail


def test_answers_must_be_printed_numbers_not_conversions(tmp_path):
    converted = {**ANSWER, "die_width_dbu": ANSWER["die_width_dbu"] / 1000}
    row = _run(tmp_path, _stream(_b1_calls()), answer=converted)
    assert _status(row)["answer_from_tool"] == "FAIL" and "die_width_dbu" in row["checks"]["answer_from_tool"]["detail"]


@pytest.mark.parametrize("command,status", [
    ("exec python3 -c 'print(1)'", "FAIL"),
    ("puts [exec cat data/design.def]", "FAIL"),
    ("dict for {k v} {a 1} {exec touch x}", "FAIL"),
    ("set f [open data/design.def]", "WARN"),
    ("puts [file exists data/design.def]", "WARN"),
    ("puts [expr {3 * 2}]", "PASS"),
])
def test_tcl_exec_through_the_session_is_a_bypass(tmp_path, command, status):
    calls = _b1_calls()
    calls.insert(5, ("interactive_openroad_exec", {"session_id": SID, "command": command}))
    row = _run(tmp_path, _stream(calls))
    assert _status(row)["no_bypass"] == status, row["checks"]["no_bypass"]


@pytest.mark.parametrize("command,status", [
    ("openroad -no_init -exit run.tcl", "FAIL"),
    ("echo 'report_design_area' | /usr/bin/openroad", "FAIL"),
    ("python3 -c 'import odb'", "FAIL"),
    ("pip install opendbpy", "FAIL"),
    ("node ~/mcp/openroad/typescript/dist/main.js --transport stdio", "FAIL"),
    ("klayout -b -r count.py", "FAIL"),
    ("grep -n DIEAREA data/design.def", "WARN"),
    ("ls ~/mcp/openroad", "WARN"),
    ("cat data/task.json && pwd", "PASS"),
])
def test_shell_access_to_openroad(tmp_path, command, status):
    row = _run(tmp_path, _stream(_b1_calls(), extra=[("Bash", {"command": command})]))
    assert _status(row)["no_bypass"] == status, row["checks"]["no_bypass"]


def test_web_lookups_are_flagged_for_review(tmp_path):
    row = _run(tmp_path, _stream(_b1_calls(), extra=[("WebSearch", {"query": "OpenROAD report_design_area units"})]))
    assert row["verdict"] == "PASS" and _status(row)["no_bypass"] == "WARN"


def test_skipping_grep_is_an_uncalled_requirement(tmp_path):
    calls = [c for c in _b1_calls() if c[0] != "grep_session_output"]
    row = _run(tmp_path, _stream(calls))
    assert row["failure"] == "tool_called" and "grep" in row["checks"]["tool_called"]["detail"]


# Claude B3 on AWS (2026-10-07): both files read in one command, die and HPWL printed as
# key=value by the agent's own Tcl, a first HPWL attempt failing in band (SWIG overload).
B3_COUNTS = ("set blk [ord::get_db_block]; set d [$blk getDieArea]; puts \"counts inst=[llength [$blk getInsts]] "
             "die_w=[expr {[$d xMax]-[$d xMin]}] die_h=[expr {[$d yMax]-[$d yMin]}]\"; $t apply $r")
B3_HPWL = ("set tot 0; foreach n [$blk getNets] { foreach it [$n getITerms] { set b [$it getBBox] } }; "
           "puts \"hpwl_total=[expr {int($tot)}] raw=$tot\"")
B3_OUTPUT = {
    B3_COUNTS: [f"counts inst={REF['instance_count']} die_w={REF['die_width_dbu']} die_h={REF['die_height_dbu']}",
                "Wrong number or type of arguments for overloaded function 'dbTransform_apply'."],
    B3_HPWL: [f"net {name} pts=2 hpwl={float(h)}" for name, h in REF["hpwl_per_net_dbu"].items()]
             + [f"hpwl_total={REF['hpwl_total_dbu']} raw={float(REF['hpwl_total_dbu'])}"],
}


def _b3_calls(workdir="/tmp/ai4sci_ws_x/workspace"):
    ex = "interactive_openroad_exec"
    return [("create_interactive_session", {"session_id": SID, "cwd": workdir}),
            (ex, {"session_id": SID,
                  "command": f"read_lef {workdir}/data/design.lef; read_def {workdir}/data/design.def"}),
            (ex, {"command": B3_COUNTS, "session_id": SID}),
            (ex, {"command": B3_HPWL, "session_id": SID}),
            (ex, {"command": "set_cmd_units -distance um", "session_id": SID}),
            ("interactive_openroad_query", {"command": "report_design_area", "session_id": SID}),
            ("grep_session_output", {"session_id": SID, "pattern": "^Design area"}),
            ("terminate_interactive_session", {"session_id": SID})]


def test_genuine_b3_run_with_key_value_prints_passes(tmp_path):
    """Values printed as ``die_w=13000`` / ``hpwl_total=35200`` are printed numbers; this
    run was a false FAIL before the extractor read ``key=value`` tokens."""
    workdir = str(tmp_path / "a" / "ws")
    row = _run(tmp_path / "a", _stream(_b3_calls(workdir), server=FakeServer(scripted=B3_OUTPUT)))
    assert row["verdict"] == "PASS" and set(_status(row).values()) == {"PASS"}, row["checks"]
    row = _run(tmp_path / "b", _codex(_b3_calls(), server=FakeServer(scripted=B3_OUTPUT)), harness="codex")
    assert row["verdict"] == "PASS" and set(_status(row).values()) == {"PASS"}, row["checks"]
