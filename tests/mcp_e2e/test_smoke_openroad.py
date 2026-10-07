"""e2e_smoke/servers/openroad.py without Node or OpenROAD: the closed-form design and its LEF/DEF,
PTY output parsing, the sentinel-echo race classifier, the direct-run script, the fake ORFS tree,
gate expectations, and the result classification against stub servers."""
import base64
import importlib
import json
import os
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from . import support
from .support import StubClient, report_statuses, rpc_json, rpc_text, runner

orl = support.smoke_module("openroad")
helpers = importlib.import_module("e2e_smoke.helpers")


# --------------------------------------------------------------------------
# Closed form and fixtures
# --------------------------------------------------------------------------

def test_closed_form_of_the_tiny_design():
    cf = orl.closed_form()
    assert cf["hpwl"] == {"in": 1000, "n1": 1200, "n2": 1200, "out": 6000}
    assert cf["hpwl_total"] == 9400
    assert cf["core"] == (1000, 2000, 11000, 4000)
    assert (cf["cell_area_um2"], cf["core_area_um2"], cf["utilization_percent"]) == (6.0, 20.0, 30.0)
    assert (cf["component_terminals"], cf["connections"]) == (6, 6)
    assert orl.geometry_lines(cf) == [
        "COUNTS 3 4 2 1 1000", "DIE 0 0 12000 6000", "CORE 1000 2000 11000 4000",
        "INST u1 1000 2000", "INST u2 3000 2000", "INST u3 5000 2000",
        "HPWL in 1000", "HPWL n1 1200", "HPWL n2 1200", "HPWL out 6000", "HPWL_TOTAL 9400"]


def _um(text):
    return round(float(text) * orl.DBU)


def test_lef_and_def_text_encode_the_constants():
    lef, deff = orl.TINY_LEF, orl.TINY_DEF
    assert "DATABASE MICRONS 1000" in lef and "UNITS DISTANCE MICRONS 1000" in deff
    w, h = re.search(r"MACRO INV.*?SIZE ([\d.]+) BY ([\d.]+)", lef, re.S).groups()
    assert (_um(w), _um(h)) == orl.CELL
    site = re.search(r"SITE core.*?SIZE ([\d.]+) BY ([\d.]+)", lef, re.S).groups()
    assert tuple(_um(v) for v in site) == orl.SITE
    for pin, rect in orl.CELL_PINS.items():
        found = re.search(rf"PIN {pin}\b.*?RECT ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)", lef, re.S).groups()
        assert tuple(_um(v) for v in found) == rect
    assert tuple(map(int, re.search(r"DIEAREA \( (\d+) (\d+) \) \( (\d+) (\d+) \)", deff).groups())) == orl.DIE
    row = re.search(r"ROW ROW_0 core (\d+) (\d+) N DO (\d+) BY 1 STEP (\d+) 0", deff).groups()
    assert (int(row[0]), int(row[1])) == orl.ROW_ORIGIN and int(row[2]) == orl.ROW_SITES and int(row[3]) == orl.SITE[0]
    placed = {m[0]: (int(m[1]), int(m[2])) for m in re.findall(r"- (u\d) INV \+ PLACED \( (\d+) (\d+) \) N", deff)}
    assert placed == orl.INSTANCES
    pin_rows = re.findall(r"- (\w+) \+ NET \w+.*?PLACED \( (\d+) (\d+) \)", deff, re.S)
    pins = {m[0]: (int(m[1]), int(m[2])) for m in pin_rows}
    assert pins == orl.BTERMS
    shapes = set(re.findall(r"LAYER metal1 \( (-?\d+) (-?\d+) \) \( (-?\d+) (-?\d+) \)", deff))
    assert {tuple(map(int, s)) for s in shapes} == {orl.BTERM_SHAPE}
    nets = {}
    for name, body in re.findall(r"^- (\w+) ((?:\( \w+ \w+ \) )+)\+ USE SIGNAL ;", deff, re.M):
        nets[name] = tuple(tuple(t.split()) for t in re.findall(r"\( (\w+ \w+) \)", body))
    assert nets == orl.NETS


def test_design_area_and_read_counts_parse_openroad_lines():
    assert orl.design_area(["Design area 6 u^2 30% utilization."]) == (6.0, 30.0)
    assert orl.design_area(["Design area 0 u^2 30% utilization."]) == (0.0, 30.0)
    assert orl.design_area(["nothing"]) is None
    lines = ["[INFO ODB-0227] LEF file: /x/tiny.lef, created 1 layers, 1 library cells",
             "[INFO ODB-0130]     Created 2 pins.",
             "[INFO ODB-0131]     Created 3 components and 6 component-terminals.",
             "[INFO ODB-0133]     Created 4 nets and 6 connections."]
    assert orl.read_counts(lines) == {"layers": 1, "library_cells": 1, "pins": 2, "components": 3,
                                      "component_terminals": 6, "nets": 4, "connections": 6}


# --------------------------------------------------------------------------
# Session output parsing and the direct run
# --------------------------------------------------------------------------

def test_result_lines_strips_echo_prompts_and_blank_lines():
    assert orl.result_lines("report_design_area\nDesign area 6 u^2 30% utilization.\n%", "report_design_area") == (
        ["Design area 6 u^2 30% utilization."], [])
    assert orl.result_lines("% puts M\nM\n%", "puts M") == (["M"], [])
    assert orl.result_lines("report_units\n time 1s  \n distance 1um", "report_units") == (
        [" time 1s", " distance 1um"], [])
    assert orl.result_lines("read_lef /x\n[ERROR ORD-0001] /x does not exist.\nORD-0001", "read_lef /x") == (
        ["[ERROR ORD-0001] /x does not exist.", "ORD-0001"], [])
    assert orl.result_lines("%\nstale line\nputs M\nM", "puts M") == (["M"], ["stale line"])
    assert orl.result_lines("something else", "puts M") == (None, [])


def test_echo_damage_accepts_only_the_sentinel_race_signature():
    want = ["COUNTS 3 4 2 1 1000", "DIE 0 0 12000 6000", "CORE 1000 2000 11000 4000", "INST u1 1000 2000"]
    assert not orl.echo_damage(want, want)
    assert orl.echo_damage(["COUNTS 3 4 2 1 1000", "DIE 0 0 12000 6000", "CORE 1000 2000 11000 4000put",
                            "INST u1 1000 2000"], want)                       # echo glued to a line's end
    assert orl.echo_damage(want[1:], want)                                   # a line lost with the marker line
    assert orl.echo_damage(['bh95q4mm} -]"', *want[1:]], want)               # the nonce tail left behind
    assert orl.echo_damage(['s "[jo' + want[0], *want[1:]], want)            # echo glued to a line's start
    assert not orl.echo_damage(["COUNTS 3 4 2 1 1000", "DIE 0 0 12000 6001"], want)    # a wrong number
    assert not orl.echo_damage([*want, "extra"], want)
    assert not orl.echo_damage(list(reversed(want)), want)


def test_parse_direct_reads_marked_blocks():
    stdout = ("banner\n@@OR_L1_BEGIN 0\nv2.0-x\n@@OR_L1_END 0 0\n@@OR_L1_BEGIN 1\n\n[ERROR ORD-0001] gone.\n"
              "ORD-0001\n@@OR_L1_END 1 1\n@@OR_L1_BEGIN 2\n@@OR_L1_END 2 0\n")
    assert orl.parse_direct(stdout) == {0: (0, ["v2.0-x"]), 1: (1, ["[ERROR ORD-0001] gone.", "ORD-0001"]),
                                        2: (0, [])}
    assert orl.parse_direct("@@OR_L1_BEGIN 0\npartial\n") == {}


@pytest.mark.skipif(shutil.which("tclsh") is None, reason="tclsh not installed")
def test_direct_script_prints_like_the_interactive_shell(tmp_path):
    steps = [orl.Step("a", "query", "puts hello"), orl.Step("b", "exec", "expr {6*7}"),
             orl.Step("c", "exec", "error boom"), orl.Step("d", "exec", "set x 1; puts [incr x]")]
    script = tmp_path / "s.tcl"
    script.write_text(orl.direct_script(steps))
    out = subprocess.run(["tclsh", str(script)], capture_output=True, text=True, check=True).stdout
    assert orl.parse_direct(out) == {0: (0, ["hello"]), 1: (0, ["42"]), 2: (1, ["boom"]), 3: (0, ["2"])}


def test_session_steps_and_blocked_probes(tmp_path):
    steps = orl.session_steps(tmp_path, tmp_path / "missing.lef")
    names = [s.name for s in steps]
    assert len(set(names)) == len(names) and all(s.tool in orl.TOOLS for s in steps)
    assert names.index("set_cmd_units") < names.index("report_design_area") < orl.BLOCKED_AFTER + 1
    errors = {s.name: s.error for s in steps if s.error}
    assert errors == {"unknown_command": "Invalid command: report_no_such_thing",
                      "missing_lef": f"OpenROAD ORD-0001: {tmp_path / 'missing.lef'} does not exist."}
    # every read-only probe really is a read-only verb for the server's allowlist
    assert all(s.command.split()[0].startswith(("report_", "check_", "puts")) for s in steps if s.tool == "query")
    assert {verb for _, _, verb in orl.blocked_probes(tmp_path)} >= {"exec", "socket", "gui::show", "glob", "subst"}
    assert orl.bypass_command(tmp_path / "m").startswith("dict for ")


# --------------------------------------------------------------------------
# Fake ORFS tree, images, gates
# --------------------------------------------------------------------------

def test_flow_tree_layout(tmp_path):
    flow = orl.build_flow_tree(tmp_path / "home", tmp_path / "outside.png")
    assert flow == tmp_path / "home/OpenROAD-flow-scripts/flow"
    reports = flow / f"reports/{orl.PLATFORM}/{orl.DESIGN}/{orl.RUN}"
    assert helpers.png_size((reports / orl.SMALL_PNG[0]).read_bytes()) == orl.SMALL_PNG[1:]
    assert helpers.png_size((reports / orl.LARGE_PNG[0]).read_bytes()) == orl.LARGE_PNG[1:]
    link = reports / orl.ESCAPE_LINK
    assert link.is_symlink() and link.resolve() == (tmp_path / "outside.png").resolve()
    assert json.loads((flow / f"designs/{orl.PLATFORM}/{orl.DESIGN}/rules-base.json").read_text()) == orl.RULES
    makefile = (flow / "Makefile").read_text()
    assert makefile.count("\n\t@") == 4 and "$$$$" in makefile
    logs = flow / f"logs/{orl.PLATFORM}/{orl.DESIGN}/{orl.RUN}"
    assert sorted(p.name for p in logs.iterdir()) == [
        "2_1_floorplan.json", "2_1_floorplan.log", "2_2_floorplan_io.log", "3_3_place_gp.json"]
    assert orl.FLOORPLAN_JSON.count("floorplan__design__utilization") == 2     # a repeated key on purpose
    assert json.loads(orl.FLOORPLAN_JSON)["floorplan__design__utilization"] == orl.FLOORPLAN_METRICS[
        "floorplan__design__utilization"][-1]


def test_floorplan_log_lines_follow_the_tagged_format():
    tagged = [line.strip() for line in orl.FLOORPLAN_LOG.splitlines()]
    assert [t for t in tagged if re.match(r"\[ERROR\s+[^\]]*\]", t)] == orl.FLOORPLAN_LOG_ERRORS
    assert [t for t in tagged if re.match(r"\[WARNING\s+[^\]]*\]", t)] == orl.FLOORPLAN_LOG_WARNINGS


def test_expected_gates_for_the_fixture_rules():
    stages = [("2_1_floorplan", orl.FLOORPLAN_METRICS), ("2_2_floorplan_io", {}),
              ("3_3_place_gp", orl.PLACE_GP_METRICS)]
    gates, unmatched, summary = orl.expected_gates(stages, orl.RULES)
    assert {g["metric"]: g["status"] for g in gates} == {
        "floorplan__design__core__area": "pass", "floorplan__design__instance__count": "fail",
        "floorplan__design__utilization": "pass", "globalplace__design__hpwl": "pass"}
    util = next(g for g in gates if g["metric"] == "floorplan__design__utilization")
    assert util["value"] == 0.3 and util["ambiguous"] is True
    assert unmatched == ["route__drc_errors"]
    assert summary == {"total": 4, "pass": 3, "fail": 1, "unknown": 0, "failing_errors": 0,
                       "failing_warnings": 1, "unmatched": 1}
    assert orl.compare_gate("a", "<=", 1) is None and orl.compare_gate("a", "==", "a") is True
    assert orl.gate_digest({**gates[0], "extra": 1}) == gates[0]


def test_webp_size_reads_all_three_headers():
    vp8 = (b"RIFF\0\0\0\0WEBPVP8 \0\0\0\0" + b"\0\0\0\x9d\x01\x2a"
           + (1568).to_bytes(2, "little") + (78).to_bytes(2, "little"))
    assert orl.webp_size(vp8) == (1568, 78)
    bits = (1568 - 1) | ((78 - 1) << 14)
    vp8l = b"RIFF\0\0\0\0WEBPVP8L\0\0\0\0\x2f" + bits.to_bytes(4, "little")
    assert orl.webp_size(vp8l) == (1568, 78)
    vp8x = b"RIFF\0\0\0\0WEBPVP8X\0\0\0\0\0\0\0\0" + (1567).to_bytes(3, "little") + (77).to_bytes(3, "little")
    assert orl.webp_size(vp8x) == (1568, 78)
    with pytest.raises(ValueError):
        orl.webp_size(orl.png_image(2, 2))
    assert orl.resized_box(2000, 100) == (1568, 78.4) and orl.resized_box(24, 16) == (24, 16)


def test_make_command_matches_the_server_argv():
    assert orl.make_command("floorplan", "base", dry_run=True) == (
        "make -n floorplan DESIGN_CONFIG=./designs/l1pdk/l1design/config.mk FLOW_VARIANT=base")
    assert orl.make_command("floorplan", "l1run", overrides={"A": "1", "B": "x"}).endswith(
        "FLOW_VARIANT=l1run A=1 B=x")


def test_process_gone():
    proc = subprocess.Popen(["true"])
    proc.wait()
    assert orl.process_gone(proc.pid, limit=0)
    assert not orl.process_gone(os.getpid(), limit=0)


# --------------------------------------------------------------------------
# Classification against stub servers
# --------------------------------------------------------------------------

def _session(tmp_path, responses, **state):
    report = runner.Report()
    session = runner.Session(orl.SMOKE, None, {}, {}, tmp_path, report, tmp_path, tmp_path / "home",
                             tmp_path / "cwd", tmp_path / "tmp", {})
    session.call = runner.Caller(StubClient(responses), report)
    session.state.update(fixtures=tmp_path, **state)
    return session


def test_expect_error_classification_and_in_band_count(tmp_path):
    session = _session(tmp_path, {
        "get_orfs_job": rpc_json({"job_id": None, "error": "FlowJobNotFound", "message": "Flow job 'x' not found"}),
        "cancel_orfs_job": rpc_json({"job_id": "x", "status": "cancelled", "error": None}),
        "read_orfs_metrics": rpc_json({"error": "UnexpectedError", "message": "boom"}),
        "read_report_image": rpc_text(json.dumps({"error": "ImageNotFound", "message": "m"}), is_error=True)})
    orl.expect_error(session, "a", "get_orfs_job", {}, "FlowJobNotFound")
    orl.expect_error(session, "b", "cancel_orfs_job", {}, "FlowJobNotFound")
    orl.expect_error(session, "c", "read_orfs_metrics", {}, "StageNotFound")
    orl.expect_error(session, "d", "read_report_image", {}, "ImageNotFound")
    assert report_statuses(session.report) == {"a": "PASS", "b": "FAIL", "c": "WARN", "d": "PASS"}
    assert session.state["in_band"] == {"get_orfs_job": 1, "read_orfs_metrics": 1}
    orl.check_in_band_summary(session)
    assert report_statuses(session.report)["errors are in-band"] == "WARN"


def _exec_result(command, lines, number, error=None, sid=orl.MAIN):
    return rpc_json({"output": "\n".join([command, *lines, "%"]), "session_id": sid, "command_count": number,
                     "truncated": False, "bytes_discarded": 0, "error": error})


def test_run_step_and_compare_with_direct(tmp_path):
    step = orl.Step("report_units", "query", "report_units")
    responses = {"interactive_openroad_query": _exec_result("report_units", [" time 1s", " distance 1um"], 1)}
    session = _session(tmp_path, responses)
    executed = []
    record = orl.run_step(session, orl.MAIN, step, executed)
    assert record["number"] == 1 and record["lines"] == [" time 1s", " distance 1um"]
    orl.compare_with_direct(session, record, {0: (0, [" time 1s", " distance 1um"])}, 0)
    orl.compare_with_direct(session, record, {0: (0, [" time 1s"])}, 0)                  # an extra line
    record["lines"] = [" distance 1um"]                                                    # a line lost
    orl.compare_with_direct(session, record, {0: (0, [" time 1s", " distance 1um"])}, 0)
    record["lines"] = [" time 2s", " distance 1um"]                                        # a wrong value
    orl.compare_with_direct(session, record, {0: (0, [" time 1s", " distance 1um"])}, 0)
    assert [c["status"] for c in session.report.checks] == ["PASS", "FAIL", "WARN", "FAIL"]
    assert record.get("damaged") is True

    session = _session(tmp_path, {"interactive_openroad_query": _exec_result("puts x", ["x"], 3)})
    orl.run_step(session, orl.MAIN, orl.Step("mark", "query", "puts x"), [])    # numbered 1, server says 3
    assert [c["status"] for c in session.report.checks] == ["FAIL"]


def test_compare_with_direct_error_expectations(tmp_path):
    session = _session(tmp_path, {})
    step = orl.Step("missing_lef", "exec", "read_lef /x", error="OpenROAD ORD-0001: /x does not exist.")
    lines = ["[ERROR ORD-0001] /x does not exist.", "ORD-0001"]
    record = {"step": step, "lines": lines, "error": "OpenROAD ORD-0001: /x does not exist.", "output": ""}
    orl.compare_with_direct(session, record, {0: (1, lines)}, 0)
    orl.compare_with_direct(session, {**record, "error": None}, {0: (1, lines)}, 0)
    plain = {"step": orl.Step("dpl", "exec", "detailed_placement"), "lines": ["x"], "error": None, "output": ""}
    orl.compare_with_direct(session, plain, {0: (1, ["x"])}, 0)
    orl.compare_with_direct(session, {**plain, "error": "Error: y"}, {0: (0, ["x"])}, 0)
    orl.compare_with_direct(session, plain, {}, 0)
    assert [c["status"] for c in session.report.checks] == ["PASS", "WARN", "WARN", "WARN", "FAIL"]


def test_blocked_probe_must_not_run(tmp_path):
    def blocked(arguments):
        verb = {"exec ls": "exec"}.get(arguments["command"], "x")
        return rpc_json({"output": "", "command_count": 0, "error": f"CommandBlocked: '{verb}'"})
    session = _session(tmp_path, {"interactive_openroad_query": blocked, "interactive_openroad_exec": blocked})
    orl.check_blocked(session, orl.MAIN, 0)
    statuses = report_statuses(session.report)
    assert statuses["query[blocked exec]"] == "PASS"
    assert statuses["exec[blocked socket]"] == "WARN"       # refused, but under another verb


def test_check_lines_three_states(tmp_path):
    session = _session(tmp_path, {})
    orl.check_lines(session, "a", ["x"], ["x"], "")
    orl.check_lines(session, "b", ["xput"], ["x"], "")
    orl.check_lines(session, "c", ["y"], ["x"], "")
    orl.check_lines(session, "d", None, ["x"], "")
    assert report_statuses(session.report) == {"a": "PASS", "b": "WARN", "c": "FAIL", "d": "FAIL"}


def test_prepare_without_openroad_still_builds_the_flow_tree(tmp_path, monkeypatch):
    (tmp_path / "empty").mkdir()
    for d in ("home", "tmp"):
        (tmp_path / d).mkdir()
    monkeypatch.setattr(orl, "load_setup", lambda: SimpleNamespace(HOST_PATH=str(tmp_path / "empty")))
    report = runner.Report()
    session = runner.Session(orl.SMOKE, None, {}, {}, tmp_path, report, tmp_path, tmp_path / "home",
                             tmp_path / "cwd", tmp_path / "tmp", {})
    orl.prepare(session)
    assert report_statuses(report) == {"openroad on the host": "FAIL"}
    assert (session.state["flow"] / "Makefile").is_file() and "openroad" not in session.state
    assert (tmp_path / "fixtures/tiny.def").read_text() == orl.TINY_DEF


def test_openroad_smoke_declaration():
    assert orl.SMOKE.server == "openroad" and orl.SMOKE.prepare is not None
    assert orl.SMOKE.expected_cwd_files == ()
    assert "proc l1_geometry" in orl.PROBE_TCL and "\\\n" not in orl.PROBE_TCL
    entry = support.setup.load_manifest()["openroad"]
    assert "ORFS_FLOW_PATH" not in entry["launch"].get("env", {})      # the fake tree lives under $HOME


def test_image_bytes_round_trip():
    data = orl.png_image(3, 2)
    assert helpers.png_size(data) == (3, 2)
    assert base64.b64decode(base64.b64encode(data)) == data
