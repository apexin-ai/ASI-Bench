"""Direct (agent-free) E2E smoke test for the pinned OpenROAD-MCP server (TypeScript, tag v1.1.0).

The server is a Node program, so its ``.venv`` is empty and only runs this script::

    ~/mcp/openroad/.venv/bin/python scripts/mcp/e2e/smoke.py openroad --config ~/mcp/openroad.mcp.json

No network is needed. Host prerequisites (``node``, ``openroad``, ``make`` in
``/usr/bin:/bin``) are the manifest's ``host_requirements``.

References, all outside the server process:

* **closed form** -- a hand-written tiny LEF/DEF (3 inverters in a row, 4 nets,
  2 I/O pins; no third-party PDK). Counts, die/core boxes, placements, per-net
  HPWL from pin-shape centres, cell area and utilisation follow from the
  constants below;
* **the same openroad binary, run directly** -- ``openroad -no_init -no_splash
  -exit script.tcl`` in a separate process (no Node, no MCP, no PTY) runs every
  compared session command; the MCP result lines must equal its lines;
* **a fake ORFS flow tree** under the temporary HOME (``ORFS_FLOW_PATH`` is
  deliberately left at its ``$HOME`` default): report images, stage metrics,
  stage logs, a ``rules-base.json`` and a stub ``Makefile``, so all five ORFS
  tools return real results without OpenROAD-flow-scripts or yosys.

L1 covers all 15 tools: an interactive session that reads the design and
queries it (``interactive_openroad_query`` / ``_exec``), the allowlist (blocked
verbs never reach openroad and do not consume a command number), grep over
retained output, history, inspection, metrics and listing, session options
(``cwd``, ``env``, launch command validation, duplicate ids, per-call timeout),
termination; then report images (raw and resized), metrics with gate verdicts,
and flow runs (dry run, real run with overrides, job limit, cancel, timeout --
the run's process group must be gone afterwards).

Upstream behaviour that does not make a correctly used tool wrong is WARN: every
tool but ``read_report_image`` reports failure only in the JSON ``error`` field
(``isError`` stays false; one summary check), the read-only query tool runs
``exec`` inside a ``dict for`` body, the launch allowlist compares only the
executable's basename, and a call without ``session_id`` leaves a new session
running. Wrong numbers or text for correct inputs are FAIL.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import struct
import subprocess
import time
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..client import text_of
from ..runner import Session, Smoke, load_setup

MAIN, SIDE = "l1a", "l1b"
MARK = "OR_L1_MARK_31415"
AFTER_TIMEOUT_MARK = "OR_L1_AFTER_TIMEOUT"
SIDE_ENV = {"OR_L1_ENV": "31415"}
BUFFER_SIZE = 131072              # OPENROAD_DEFAULT_BUFFER_SIZE default (not set by the manifest)
MAX_SESSIONS = 50                 # OPENROAD_MAX_SESSIONS default
MAX_FLOW_JOBS = 2                 # OPENROAD_MAX_FLOW_JOBS default
DIRECT_TIMEOUT = 300.0

# --------------------------------------------------------------------------
# Closed-form design (DBU; 1000 DBU = 1 um)
# --------------------------------------------------------------------------

DBU = 1000
CELL = (1000, 2000)                                           # INV width, height
CELL_PINS = {"A": (50, 950, 150, 1050), "Y": (850, 950, 950, 1050)}
DIE = (0, 0, 12000, 6000)
ROW_ORIGIN, ROW_SITES, SITE = (1000, 2000), 50, (200, 2000)
INSTANCES = {"u1": (1000, 2000), "u2": (3000, 2000), "u3": (5000, 2000)}   # all N, all in the row
BTERM_SHAPE = (-100, -50, 100, 50)
BTERMS = {"in": (100, 3000), "out": (11900, 3000)}
NETS = {"in": (("PIN", "in"), ("u1", "A")), "n1": (("u1", "Y"), ("u2", "A")),
        "n2": (("u2", "Y"), ("u3", "A")), "out": (("u3", "Y"), ("PIN", "out"))}

# The exact text validated on the target host (2026-10-07); tests check it encodes the constants above.
TINY_LEF = """\
VERSION 5.8 ;
BUSBITCHARS "[]" ;
DIVIDERCHAR "/" ;

UNITS
  DATABASE MICRONS 1000 ;
END UNITS

MANUFACTURINGGRID 0.005 ;

LAYER metal1
  TYPE ROUTING ;
  DIRECTION HORIZONTAL ;
  PITCH 0.2 ;
  WIDTH 0.1 ;
  SPACING 0.1 ;
END metal1

SITE core
  CLASS CORE ;
  SYMMETRY Y ;
  SIZE 0.2 BY 2.0 ;
END core

MACRO INV
  CLASS CORE ;
  ORIGIN 0 0 ;
  SIZE 1.0 BY 2.0 ;
  SYMMETRY X Y ;
  SITE core ;
  PIN A
    DIRECTION INPUT ;
    USE SIGNAL ;
    PORT
      LAYER metal1 ;
        RECT 0.05 0.95 0.15 1.05 ;
    END
  END A
  PIN Y
    DIRECTION OUTPUT ;
    USE SIGNAL ;
    PORT
      LAYER metal1 ;
        RECT 0.85 0.95 0.95 1.05 ;
    END
  END Y
END INV

END LIBRARY
"""

TINY_DEF = """\
VERSION 5.8 ;
DIVIDERCHAR "/" ;
BUSBITCHARS "[]" ;
DESIGN tiny ;
UNITS DISTANCE MICRONS 1000 ;

DIEAREA ( 0 0 ) ( 12000 6000 ) ;

ROW ROW_0 core 1000 2000 N DO 50 BY 1 STEP 200 0 ;

COMPONENTS 3 ;
- u1 INV + PLACED ( 1000 2000 ) N ;
- u2 INV + PLACED ( 3000 2000 ) N ;
- u3 INV + PLACED ( 5000 2000 ) N ;
END COMPONENTS

PINS 2 ;
- in + NET in + DIRECTION INPUT + USE SIGNAL
  + LAYER metal1 ( -100 -50 ) ( 100 50 )
  + PLACED ( 100 3000 ) N ;
- out + NET out + DIRECTION OUTPUT + USE SIGNAL
  + LAYER metal1 ( -100 -50 ) ( 100 50 )
  + PLACED ( 11900 3000 ) N ;
END PINS

NETS 4 ;
- in ( PIN in ) ( u1 A ) + USE SIGNAL ;
- n1 ( u1 Y ) ( u2 A ) + USE SIGNAL ;
- n2 ( u2 Y ) ( u3 A ) + USE SIGNAL ;
- out ( u3 Y ) ( PIN out ) + USE SIGNAL ;
END NETS

END DESIGN
"""

# Sourced into the session (exec) and into the direct run: prints the odb view of the design
# in a fixed format. HPWL is from pin-shape centres, as in the closed form.
PROBE_TCL = """\
proc l1_centre {r} { list [expr {([$r xMin] + [$r xMax]) / 2}] [expr {([$r yMin] + [$r yMax]) / 2}] }
proc l1_by_name {a b} { string compare [$a getName] [$b getName] }
proc l1_geometry {} {
  set block [ord::get_db_block]
  set counts [list [llength [$block getInsts]] [llength [$block getNets]] [llength [$block getBTerms]]]
  puts "COUNTS [join $counts] [llength [$block getRows]] [$block getDefUnits]"
  set r [$block getDieArea]
  puts "DIE [$r xMin] [$r yMin] [$r xMax] [$r yMax]"
  set r [$block getCoreArea]
  puts "CORE [$r xMin] [$r yMin] [$r xMax] [$r yMax]"
  foreach inst [lsort -command l1_by_name [$block getInsts]] {
    set b [$inst getBBox]
    puts "INST [$inst getName] [$b xMin] [$b yMin]"
  }
  set total 0
  foreach net [lsort -command l1_by_name [$block getNets]] {
    set xs {}; set ys {}
    foreach it [$net getITerms] { set p [l1_centre [$it getBBox]]; lappend xs [lindex $p 0]; lappend ys [lindex $p 1] }
    foreach bt [$net getBTerms] { set p [l1_centre [$bt getBBox]]; lappend xs [lindex $p 0]; lappend ys [lindex $p 1] }
    set xs [lsort -integer $xs]; set ys [lsort -integer $ys]
    set h [expr {[lindex $xs end] - [lindex $xs 0] + [lindex $ys end] - [lindex $ys 0]}]
    incr total $h
    puts "HPWL [$net getName] $h"
  }
  puts "HPWL_TOTAL $total"
}
"""


def _centre(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    return (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2


def _term_centre(owner: str, pin: str) -> tuple[int, int]:
    if owner == "PIN":
        x, y = BTERMS[pin]
        s = BTERM_SHAPE
        return _centre((x + s[0], y + s[1], x + s[2], y + s[3]))
    x, y = INSTANCES[owner]
    r = CELL_PINS[pin]
    return _centre((x + r[0], y + r[1], x + r[2], y + r[3]))


def closed_form() -> dict:
    """Everything the tiny design implies, from the constants alone."""
    core = (ROW_ORIGIN[0], ROW_ORIGIN[1], ROW_ORIGIN[0] + ROW_SITES * SITE[0], ROW_ORIGIN[1] + SITE[1])
    hpwl = {}
    for net, terms in NETS.items():
        points = [_term_centre(owner, pin) for owner, pin in terms]
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        hpwl[net] = max(xs) - min(xs) + max(ys) - min(ys)
    cell_area = len(INSTANCES) * CELL[0] * CELL[1] / DBU ** 2
    core_area = (core[2] - core[0]) * (core[3] - core[1]) / DBU ** 2
    return {
        "instances": len(INSTANCES), "nets": len(NETS), "bterms": len(BTERMS), "rows": 1, "dbu": DBU,
        "component_terminals": len(INSTANCES) * len(CELL_PINS),
        "connections": sum(1 for terms in NETS.values() for owner, _ in terms if owner != "PIN"),
        "layers": 1, "library_cells": 1,
        "die": DIE, "core": core, "placements": dict(INSTANCES), "hpwl": hpwl,
        "hpwl_total": sum(hpwl.values()), "cell_area_um2": cell_area, "core_area_um2": core_area,
        "utilization_percent": 100.0 * cell_area / core_area,
    }


def geometry_lines(cf: dict) -> list[str]:
    """What ``l1_geometry`` must print for the closed form."""
    lines = [f"COUNTS {cf['instances']} {cf['nets']} {cf['bterms']} {cf['rows']} {cf['dbu']}",
             "DIE " + " ".join(map(str, cf["die"])), "CORE " + " ".join(map(str, cf["core"]))]
    lines += [f"INST {name} {x} {y}" for name, (x, y) in sorted(cf["placements"].items())]
    lines += [f"HPWL {net} {h}" for net, h in sorted(cf["hpwl"].items())]
    return lines + [f"HPWL_TOTAL {cf['hpwl_total']}"]


AREA_LINE = re.compile(r"^Design area ([0-9.]+) u\^2 ([0-9.]+)% utilization\.$")


def design_area(lines: list[str]) -> tuple[float, float] | None:
    """(area um^2, utilisation %) from report_design_area lines."""
    for line in lines:
        m = AREA_LINE.match(line.strip())
        if m:
            return float(m.group(1)), float(m.group(2))
    return None


def read_counts(lines: list[str]) -> dict:
    """Counts from the ODB INFO lines of read_lef / read_def."""
    text = "\n".join(lines)
    patterns = {"layers": r"created (\d+) layers", "library_cells": r"(\d+) library cells",
                "pins": r"Created (\d+) pins\.", "components": r"Created (\d+) components",
                "component_terminals": r"(\d+) component-terminals", "nets": r"Created (\d+) nets",
                "connections": r"(\d+) connections"}
    return {k: int(m.group(1)) for k, p in patterns.items() if (m := re.search(p, text))}


# --------------------------------------------------------------------------
# Session commands
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Step:
    name: str
    tool: str                    # "query" | "exec"
    command: str
    compare: bool = True         # also run directly and compare the result lines
    error: str | None = None     # the in-band error the server must report


TOOLS = {"query": "interactive_openroad_query", "exec": "interactive_openroad_exec"}


def session_steps(fixtures: Path, missing: Path) -> list[Step]:
    """The main session, in order. All of them run in the direct reference too."""
    return [
        Step("version", "exec", "ord::openroad_version"),
        Step("read_lef", "exec", f"read_lef {fixtures / 'tiny.lef'}"),
        Step("read_def", "exec", f"read_def {fixtures / 'tiny.def'}"),
        Step("set_cmd_units", "exec", "set_cmd_units -distance um"),
        Step("report_design_area", "query", "report_design_area"),
        Step("report_units", "query", "report_units"),
        Step("source_probe", "exec", f"source {fixtures / 'l1_probe.tcl'}"),
        Step("geometry", "exec", "l1_geometry"),
        Step("report_checks", "query", "report_checks"),
        Step("check_placement", "query", "check_placement -verbose"),
        Step("detailed_placement", "exec", "detailed_placement"),
        Step("geometry_after_dpl", "exec", "l1_geometry"),
        Step("mark", "query", f"puts {MARK}"),
        Step("unknown_command", "query", "report_no_such_thing", error="Invalid command: report_no_such_thing"),
        Step("missing_lef", "exec", f"read_lef {missing}", error=f"OpenROAD ORD-0001: {missing} does not exist."),
        Step("after_errors", "query", "report_design_area"),
    ]


BLOCKED_AFTER = 5     # blocked probes run after this many steps, so numbering is tested across them


def blocked_probes(fixtures: Path) -> list[tuple[str, str, str]]:
    """(tool, command, blocked verb): rejected by the allowlist before reaching openroad."""
    return [
        ("query", "exec ls", "exec"),
        ("query", f"read_def {fixtures / 'tiny.def'}", "read_def"),
        ("query", "puts [llength [[ord::get_db_block] getInsts]]", "[ord::get_db_block"),
        ("query", "gui::show", "gui::show"),
        ("exec", "socket localhost 1", "socket"),
        ("exec", "\\socket localhost 1", "socket"),
        ("exec", "glob *", "glob"),
        ("exec", "puts [subst {x}]", "subst"),
    ]


def bypass_command(marker: Path) -> str:
    """A query-tool command whose only verb is the read-only ``dict``; its body runs ``exec``."""
    return f"dict for {{k v}} {{a 1}} {{exec touch {marker}}}"


def _strip_prompt(line: str) -> str:
    while line.startswith("% "):
        line = line[2:]
    return line


def result_lines(output: str, command: str) -> tuple[list[str] | None, list[str]]:
    """(printed lines, junk before the echo) of an interactive_openroad_* ``output``.

    The output is the PTY echo of the command (sometimes behind a stale ``% `` prompt),
    the command's own lines, then a bare ``%`` prompt (not always). Lines are
    right-stripped; blank and bare-prompt lines are dropped. None if there is no echo."""
    lines = output.split("\n")
    for i, line in enumerate(lines):
        if _strip_prompt(line).rstrip() == command.strip():
            junk = [x for x in lines[:i] if x.strip() and x.strip() != "%"]
            return [x.rstrip() for x in lines[i + 1:] if x.strip() and x.strip() != "%"], junk
    return None, []


BEGIN, END = "@@OR_L1_BEGIN", "@@OR_L1_END"
SENTINEL_ECHO = 'puts "[join {ORMCP DONE '        # the server's completion sentinel, as the PTY echoes it
SENTINEL_FRAGMENT = re.compile(r'ORMCP|\[join \{|\} -\]"$')


def _echo_piece(text: str) -> bool:
    """A piece of the sentinel's echo: part of its fixed text (``put``, ``s "[jo``) or a marked fragment."""
    return bool(text) and (text in SENTINEL_ECHO or bool(SENTINEL_FRAGMENT.search(text)))


def echo_damage(got: list[str], want: list[str]) -> bool:
    """True if ``got`` is ``want`` damaged by the sentinel-echo race, and nothing else.

    The server writes its completion sentinel ``puts "[join {ORMCP DONE <nonce>} -]"`` right
    after the command; when the command is still printing, the PTY echo of that line
    interleaves with the output. The server then drops every line holding the marker (with
    the real output glued to it), and leaves the rest of the echo behind. So: every line of
    ``got`` is a line of ``want`` (in order), possibly with an echo piece glued to one end, or
    is an echo piece itself; lines of ``want`` may be missing."""
    if got == want:
        return False
    remaining = iter(want)
    for line in got:
        if _echo_piece(line):
            continue
        for w in remaining:
            if (line == w or (line.startswith(w) and _echo_piece(line[len(w):]))
                    or (line.endswith(w) and _echo_piece(line[:len(line) - len(w)]))):
                break
        else:
            return False
    return True


def direct_script(steps: list[Step]) -> str:
    """Tcl for ``openroad -exit``: each step between markers, its result or error message
    printed as the interactive shell prints it."""
    lines = ["proc l1_emit {text} { puts $text; flush stdout }"]
    for i, step in enumerate(steps):
        lines += [f"l1_emit {{{BEGIN} {i}}}",
                  f"set l1_rc [catch {{{step.command}}} l1_r]",
                  'if {$l1_r ne ""} { puts $l1_r }',
                  f'l1_emit "{END} {i} $l1_rc"']
    return "\n".join(lines) + "\n"


def parse_direct(stdout: str) -> dict[int, tuple[int, list[str]]]:
    """step index -> (Tcl return code, printed lines) from a direct run."""
    out: dict[int, tuple[int, list[str]]] = {}
    current: int | None = None
    buf: list[str] = []
    for raw in stdout.splitlines():
        line = raw.rstrip()
        if line.startswith(BEGIN + " "):
            current, buf = int(line.split()[1]), []
        elif line.startswith(END + " ") and current is not None:
            _, index, rc = line.split()
            if int(index) == current:
                out[current] = (int(rc), buf)
            current = None
        elif current is not None and line.strip() and line.strip() != "%":
            buf.append(line)
    return out


def run_direct(openroad: str, steps: list[Step], workdir: Path, env: dict) -> tuple[dict, str]:
    script = workdir / "direct.tcl"
    script.write_text(direct_script(steps), encoding="utf-8")
    proc = subprocess.run([openroad, "-no_init", "-no_splash", "-exit", str(script)], cwd=workdir, env=env,
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace", timeout=DIRECT_TIMEOUT)
    return parse_direct(proc.stdout), proc.stdout


# --------------------------------------------------------------------------
# Fake ORFS flow tree
# --------------------------------------------------------------------------

PLATFORM, DESIGN, RUN = "l1pdk", "l1design", "base"
SMALL_PNG = ("final_placement.png", 24, 16)       # under IMAGE_MIN_DIMENSION and the base64 budget: raw
LARGE_PNG = ("final_congestion.png", 2000, 100)   # long edge over IMAGE_MAX_DIMENSION: resized to webp
MAX_DIMENSION = 1568                               # OPENROAD_IMAGE_MAX_DIMENSION default
ESCAPE_LINK = "final_routing.png"                  # symlink to an image outside the flow tree
IMAGE_TYPES = {"final_placement.png": "cell_placement", "final_congestion.png": "congestion_heatmap"}

FLOORPLAN_JSON = """\
{
  "floorplan__design__instance__count": 3,
  "floorplan__design__core__area": 20.0,
  "floorplan__design__utilization": 0.25,
  "floorplan__design__utilization": 0.3
}
"""
FLOORPLAN_METRICS = {"floorplan__design__instance__count": 3, "floorplan__design__core__area": 20.0,
                     "floorplan__design__utilization": [0.25, 0.3]}
FLOORPLAN_LOG = ("[INFO FLW-0001] start\n[WARNING FLW-0002] w1\nerror: untagged text is not a diagnostic\n"
                 "[ERROR FLW-0003] e1\n  [WARNING FLW-0004] w2\n")
FLOORPLAN_LOG_ERRORS = ["[ERROR FLW-0003] e1"]
FLOORPLAN_LOG_WARNINGS = ["[WARNING FLW-0002] w1", "[WARNING FLW-0004] w2"]
IO_LOG = "[INFO FLW-0005] io only\n"
PLACE_GP_METRICS = {"globalplace__design__hpwl": 9.4}
RULES = {
    "floorplan__design__core__area": {"value": 25, "compare": "<="},
    "floorplan__design__instance__count": {"value": 2, "compare": "<=", "level": "warning"},
    "floorplan__design__utilization": {"value": 0.28, "compare": ">="},
    "globalplace__design__hpwl": {"value": 10, "compare": "<"},
    "route__drc_errors": {"value": 0, "compare": "=="},
}
MAKEFILE = (
    "floorplan:\n"
    "\t@echo OR_L1_FLOORPLAN $(DESIGN_CONFIG) $(FLOW_VARIANT) $(OR_L1_TAG)\n"
    f"\t@mkdir -p logs/{PLATFORM}/{DESIGN}/$(FLOW_VARIANT)\n"
    "\t@printf '{\"floorplan__design__instance__count\": 3, \"floorplan__design__core__area\": %s}\\n' "
    f"'$(OR_L1_AREA)' > logs/{PLATFORM}/{DESIGN}/$(FLOW_VARIANT)/2_1_floorplan.json\n"
    "place:\n"
    "\t@echo $$$$ > $(OR_L1_PIDFILE); exec sleep 120\n"
)
RUN_OVERRIDES = {"OR_L1_AREA": "20", "OR_L1_TAG": "x31415"}
RUN_VARIANT = "l1run"


def png_image(width: int, height: int) -> bytes:
    """A complete RGB PNG with a deterministic gradient."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    rows = b"".join(b"\x00" + bytes(v for x in range(width) for v in (x * 7 % 256, y * 13 % 256, 128))
                    for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def webp_size(data: bytes) -> tuple[int, int]:
    """Width and height of a WebP (VP8, VP8L or VP8X)."""
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ValueError("not a WebP")
    kind = data[12:16]
    if kind == b"VP8X":
        return 1 + int.from_bytes(data[24:27], "little"), 1 + int.from_bytes(data[27:30], "little")
    if kind == b"VP8L":
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if kind == b"VP8 ":
        return int.from_bytes(data[26:28], "little") & 0x3FFF, int.from_bytes(data[28:30], "little") & 0x3FFF
    raise ValueError(f"unknown WebP chunk {kind!r}")


def resized_box(width: int, height: int, limit: int = MAX_DIMENSION) -> tuple[float, float]:
    """The box a fit-inside resize to ``limit`` x ``limit`` gives (before integer rounding)."""
    scale = min(1.0, limit / max(width, height))
    return width * scale, height * scale


def build_flow_tree(home: Path, outside: Path) -> Path:
    """The fake ORFS tree at the server's default ORFS_FLOW_PATH; returns the flow dir."""
    flow = home / "OpenROAD-flow-scripts" / "flow"
    (flow / "platforms" / PLATFORM).mkdir(parents=True)
    design = flow / "designs" / PLATFORM / DESIGN
    design.mkdir(parents=True)
    (design / "config.mk").write_text(f"export DESIGN_NAME = {DESIGN}\nexport PLATFORM = {PLATFORM}\n")
    (design / "rules-base.json").write_text(json.dumps(RULES, indent=2) + "\n")
    reports = flow / "reports" / PLATFORM / DESIGN / RUN
    reports.mkdir(parents=True)
    for name, w, h in (SMALL_PNG, LARGE_PNG):
        (reports / name).write_bytes(png_image(w, h))
    (reports / "notes.txt").write_text("not an image\n")
    outside.write_bytes(png_image(8, 8))
    (reports / ESCAPE_LINK).symlink_to(outside)
    logs = flow / "logs" / PLATFORM / DESIGN / RUN
    logs.mkdir(parents=True)
    (logs / "2_1_floorplan.json").write_text(FLOORPLAN_JSON)
    (logs / "2_1_floorplan.log").write_text(FLOORPLAN_LOG)
    (logs / "2_2_floorplan_io.log").write_text(IO_LOG)
    (logs / "3_3_place_gp.json").write_text(json.dumps(PLACE_GP_METRICS) + "\n")
    (flow / "Makefile").write_text(MAKEFILE)
    return flow


def compare_gate(value, compare: str, threshold):
    """rules-base semantics: == / != on anything, ordering only on numbers (else unknown)."""
    if compare == "==":
        return value == threshold
    if compare == "!=":
        return value != threshold
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (value, threshold))
    ops = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b}
    return ops[compare](value, threshold) if numeric and compare in ops else None


def expected_gates(stages: list[tuple[str, dict]], rules: dict) -> tuple[list[dict], list[str], dict]:
    """(gates, unmatched metrics, summary) for (stage stem, metrics) pairs: a rule is checked in
    the first stage that has its metric, against the last value of a repeated metric."""
    gates, unmatched = [], []
    for metric, rule in rules.items():
        owner = next((s for s in stages if metric in s[1]), None)
        if owner is None:
            unmatched.append(metric)
            continue
        raw = owner[1][metric]
        value = raw[-1] if isinstance(raw, list) else raw
        verdict = compare_gate(value, rule["compare"], rule["value"])
        gates.append({"metric": metric, "stage": owner[0], "value": value, "threshold": rule["value"],
                      "compare": rule["compare"], "level": rule.get("level", "error"),
                      "status": "unknown" if verdict is None else "pass" if verdict else "fail",
                      "ambiguous": isinstance(raw, list)})
    failing = [g for g in gates if g["status"] == "fail"]
    summary = {"total": len(gates), "pass": sum(g["status"] == "pass" for g in gates), "fail": len(failing),
               "unknown": sum(g["status"] == "unknown" for g in gates),
               "failing_errors": sum(g["level"] == "error" for g in failing),
               "failing_warnings": sum(g["level"] != "error" for g in failing), "unmatched": len(unmatched)}
    return gates, unmatched, summary


def gate_digest(gate: dict) -> dict:
    """The comparable part of a returned gate."""
    keys = ("metric", "stage", "value", "threshold", "compare", "level", "status")
    return {**{k: gate.get(k) for k in keys}, "ambiguous": bool(gate.get("ambiguous"))}


# --------------------------------------------------------------------------
# Tool-call helpers
# --------------------------------------------------------------------------

def call_payload(session: Session, check: str, tool: str, arguments: dict, *, timeout: float | None = None,
                 count_in_band: bool = True):
    """(decoded JSON of the text block(s), raw result); (None, None) after a FAIL. Records
    in-band errors (``error`` set, ``isError`` false) for the summary check."""
    result = session.call(check, tool, arguments, allow_error=True, timeout=timeout)
    if result is None:
        return None, None
    try:
        data = json.loads(text_of(result))
    except json.JSONDecodeError:
        session.report.add("L1", check, "FAIL", f"result is not JSON: {text_of(result)[:200]!r}")
        return None, result
    if not isinstance(data, dict):
        session.report.add("L1", check, "FAIL", f"result is not a JSON object: {str(data)[:200]}")
        return None, result
    if count_in_band and data.get("error") is not None and not result.get("isError"):
        session.state.setdefault("in_band", Counter())[tool] += 1
    return data, result


def ok_payload(session: Session, check: str, tool: str, arguments: dict, **kw) -> dict | None:
    """A result that must succeed (``error`` null); FAIL otherwise."""
    data, result = call_payload(session, check, tool, arguments, **kw)
    if data is None:
        return None
    if data.get("error") is not None or (result or {}).get("isError"):
        session.report.add("L1", check, "FAIL", f"error {data.get('error')!r}: {str(data.get('message') or '')[:200]}")
        return None
    return data


def expect_error(session: Session, check: str, tool: str, arguments: dict, want: str, **kw) -> dict | None:
    """A result that must be refused with an error containing ``want``: PASS; a different
    error is WARN, success is FAIL."""
    data, result = call_payload(session, check, tool, arguments, **kw)
    if data is None:
        return None
    error = data.get("error")
    flag = "isError=true" if (result or {}).get("isError") else "in-band"
    if error is None:
        session.report.add("L1", check, "FAIL", f"accepted: {json.dumps(data)[:240]}")
    elif want in str(error) or want in str(data.get("message") or ""):
        session.report.add("L1", check, "PASS", f"{flag}: {error}")
    else:
        session.report.add("L1", check, "WARN", f"refused, but with {error!r} (expected {want!r})")
    return data


def problems_of(expected: dict, data: dict) -> list[str]:
    return [f"{k}={data.get(k)!r} (expected {v!r})" for k, v in expected.items() if data.get(k) != v]


def add(session: Session, check: str, problems: list[str], ok_detail: str = "", *, warn: bool = False, **data):
    status = ("WARN" if warn else "FAIL") if problems else "PASS"
    session.report.add("L1", check, status, "; ".join(problems)[:600] if problems else ok_detail, **data)


# --------------------------------------------------------------------------
# Session group
# --------------------------------------------------------------------------

def check_create(session: Session, sid: str, **options) -> bool:
    data = ok_payload(session, f"create_interactive_session[{sid}]", "create_interactive_session",
                      {"session_id": sid, **options})
    if data is None:
        return False
    add(session, f"create_interactive_session[{sid}]",
        problems_of({"session_id": sid, "is_alive": True, "command_count": 0, "state": "active"}, data),
        f"caller-chosen id, alive, 0 commands{' with ' + ', '.join(options) if options else ''}")
    return True


def run_step(session: Session, sid: str, step: Step, executed: list[dict]) -> dict | None:
    """One counted command; appends its record (command, number, output) to ``executed``."""
    data, _ = call_payload(session, f"{step.tool}[{step.name}]", TOOLS[step.tool],
                           {"session_id": sid, "command": step.command})
    if data is None:
        return None
    output = data.get("output") or ""
    record = {"step": step, "number": len(executed) + 1, "output": output, "error": data.get("error"),
              "data": data}
    executed.append(record)
    lines, junk = result_lines(output, step.command)
    record["lines"], record["junk"] = lines, junk
    problems = problems_of({"session_id": sid, "command_count": record["number"], "truncated": False,
                            "bytes_discarded": 0}, data)
    if lines is None:
        problems.append(f"no echo of the command in output {output[:200]!r}")
    if junk:
        problems.append(f"output of an earlier command before the echo: {junk[:3]}")
    if step.error is None and data.get("error") is not None and not step.compare:
        problems.append(f"error {data.get('error')!r}")
    if problems:
        session.report.add("L1", f"{step.tool}[{step.name}]", "FAIL", "; ".join(problems))
    return record


def compare_with_direct(session: Session, record: dict, direct: dict[int, tuple[int, list[str]]],
                        index: int) -> None:
    step: Step = record["step"]
    name = f"{step.tool}[{step.name}] vs direct openroad"
    if index not in direct:
        session.report.add("L1", name, "FAIL", "the direct run printed no result for this step")
        return
    rc, want = direct[index]
    got = record.get("lines")
    if got is None:
        return                                        # already a FAIL in run_step
    if got != want:
        if echo_damage(got, want):
            record["damaged"] = True
            session.report.add("L1", name, "WARN",
                               "output damaged by the PTY echo of the server's completion sentinel (upstream race: "
                               "the sentinel line is written while the command still prints, the echo interleaves "
                               f"with the output and the line holding the marker is dropped with it): MCP {got[:6]} "
                               f"vs direct {want[:6]}", mcp_output=record["output"], direct_lines=want)
        else:
            session.report.add("L1", name, "FAIL", f"MCP {got[:8]} vs direct {want[:8]}",
                               mcp_output=record["output"], direct_lines=want)
        return
    error = record["error"]
    if step.error is not None:
        status = "PASS" if error == step.error and rc != 0 else "WARN"
        detail = (f"{len(got)} identical line(s); Tcl error reported as {error!r}" if status == "PASS" else
                  f"identical lines, but error={error!r} (expected {step.error!r}), direct rc={rc}")
    elif rc != 0:
        status, detail = "WARN", f"identical lines, but the step fails on this openroad too (rc={rc}): {got[-2:]}"
    elif error is not None:
        status, detail = "WARN", f"identical lines, but the server reports error={error!r} for a successful command"
    else:
        status, detail = "PASS", f"{len(got)} identical line(s)" + (f": {got[0][:80]!r}" if len(got) == 1 else "")
    session.report.add("L1", name, status, detail)


def check_lines(session: Session, check: str, got: list[str] | None, want: list[str], ok_detail: str) -> None:
    """PASS if equal; WARN if the difference is the sentinel-echo race; FAIL otherwise."""
    if got == want:
        session.report.add("L1", check, "PASS", ok_detail)
    elif got is not None and echo_damage(got, want):
        session.report.add("L1", check, "WARN", f"output damaged by the sentinel-echo race: {got} vs {want}")
    else:
        session.report.add("L1", check, "FAIL", f"printed {got} (expected {want})")


def check_closed_form(session: Session, records: dict[str, dict]) -> None:
    cf = session.state["closed_form"]

    def lines(name: str) -> list[str]:
        """The MCP lines; the direct run's for an output damaged by the echo race (already WARN)."""
        record = records.get(name) or {}
        if record.get("damaged"):
            return session.state["direct_lines"].get(name, [])
        return record.get("lines") or []
    counts = {**read_counts(lines("read_lef")), **read_counts(lines("read_def"))}
    want = {"layers": cf["layers"], "library_cells": cf["library_cells"], "pins": cf["bterms"],
            "components": cf["instances"], "component_terminals": cf["component_terminals"],
            "nets": cf["nets"], "connections": cf["connections"]}
    add(session, "read_lef/read_def[counts vs closed form]",
        [f"{k}={counts.get(k)} (expected {v})" for k, v in want.items() if counts.get(k) != v],
        "1 layer, 1 cell; 2 pins, 3 components, 6 component terminals, 4 nets, 6 connections")
    area = design_area(lines("report_design_area"))
    problems = []
    if area is None:
        problems.append(f"no 'Design area' line in {lines('report_design_area')}")
    else:
        if abs(area[0] - cf["cell_area_um2"]) > 0.5:
            problems.append(f"area {area[0]} u^2 (expected {cf['cell_area_um2']:g})")
        if abs(area[1] - cf["utilization_percent"]) > 0.5:
            problems.append(f"utilization {area[1]}% (expected {cf['utilization_percent']:g}%)")
    add(session, "report_design_area[vs closed form]", problems,
        f"{cf['cell_area_um2']:g} u^2, {cf['utilization_percent']:g}% of the {cf['core_area_um2']:g} u^2 core "
        "(after set_cmd_units -distance um)")
    expected = geometry_lines(cf)
    got = lines("geometry")
    add(session, "l1_geometry[odb vs closed form]",
        [] if got == expected else [f"got {got}, expected {expected}"],
        f"counts, die {cf['die']}, core {cf['core']}, placements, HPWL total {cf['hpwl_total']} DBU")
    after = lines("geometry_after_dpl")
    if "geometry_after_dpl" in records:
        add(session, "detailed_placement[legal placement kept]",
            [] if after == expected else [f"after detailed_placement: {after}"],
            "an already legal placement is unchanged (same placements and HPWL)", warn=True)


def check_main_session(session: Session) -> list[dict]:
    """create -> steps (with blocked probes in the middle) -> bypass probe. Returns the records."""
    if not check_create(session, MAIN):
        return []
    fixtures, steps, direct = session.state["fixtures"], session.state["steps"], session.state["direct"]
    executed: list[dict] = []
    for index, step in enumerate(steps):
        if index == BLOCKED_AFTER:
            check_blocked(session, MAIN, len(executed))
        record = run_step(session, MAIN, step, executed)
        if record is not None and step.compare and direct is not None:
            compare_with_direct(session, record, direct, index)
    records = {r["step"].name: r for r in executed}
    version = (records.get("version") or {}).get("lines") or []
    session.state["openroad_version"] = version[0] if version else None
    check_closed_form(session, records)

    marker = session.tmp / "query-bypass-marker"
    bypass = Step("query_bypass", "query", bypass_command(marker), compare=False)
    record = run_step(session, MAIN, bypass, executed)
    if record is not None:
        if marker.exists():
            session.report.add("L1", "interactive_openroad_query[read-only bypass]", "WARN",
                               "the read-only query tool ran 'exec touch' inside a 'dict for' body: only the "
                               "statement verb (dict) and [bracketed] verbs are checked, so script-body "
                               "arguments of read-only builtins are not; query is not a sandbox")
        else:
            session.report.add("L1", "interactive_openroad_query[read-only bypass]", "PASS",
                               f"exec inside a dict-for body did not run (error={record['error']!r})")
    return executed


def check_blocked(session: Session, sid: str, count_before: int) -> None:
    fixtures = session.state["fixtures"]
    for tool, command, verb in blocked_probes(fixtures):
        name = f"{tool}[blocked {command.split()[0][:24]}]"
        data = expect_error(session, name, TOOLS[tool], {"session_id": sid, "command": command},
                            f"CommandBlocked: '{verb}'")
        if data is not None and data.get("error") is not None:
            problems = problems_of({"output": "", "command_count": 0}, data)
            if problems:
                session.report.add("L1", f"{name}[not run]", "FAIL", "; ".join(problems))
    # the next counted command must get number count_before + 1 (checked by run_step)


def check_grep(session: Session, executed: list[dict]) -> None:
    by_name = {r["step"].name: r for r in executed}
    mark, area = by_name.get("mark"), by_name.get("report_design_area")
    retained = [r for r in executed if r["output"]]
    if mark and MARK not in mark["output"].split("\n"):
        session.report.add("L1", "grep_session_output[anchored mark]", "WARN",
                           f"skipped: the mark line was lost from its own output ({mark['output']!r})")
    elif mark:
        data = ok_payload(session, "grep_session_output[anchored mark]", "grep_session_output",
                          {"session_id": MAIN, "pattern": f"^{MARK}$"})
        if data is not None:
            line_number = mark["output"].split("\n").index(MARK) + 1
            want = [{"command_number": mark["number"], "command": mark["step"].command, "line_number": line_number,
                     "line": MARK}]
            got = [{k: m.get(k) for k in ("command_number", "command", "line_number", "line")}
                   for m in data.get("matches") or []]
            problems = [] if got == want else [f"matches {got} (expected {want})"]
            problems += problems_of({"total_matches": 1, "truncated": False, "pattern_kind": "regex",
                                     "evicted_commands": 0,
                                     "retained_chars": sum(len(r["output"]) for r in retained)}, data)
            # +1 when a late prompt is still unread in the live buffer (searched as "(unread buffer)")
            if data.get("searched_commands") not in (len(retained), len(retained) + 1):
                problems.append(f"searched_commands={data.get('searched_commands')} for {len(retained)} outputs")
            add(session, "grep_session_output[anchored mark]", problems,
                f"one match: command #{mark['number']}, line {line_number}; searched {len(retained)} retained "
                "outputs, nothing evicted")
    if area and area.get("damaged"):
        session.report.add("L1", "grep_session_output[command_number + context]", "WARN",
                           "skipped: the report_design_area output was damaged by the echo race")
    elif area and area.get("lines"):
        data = ok_payload(session, "grep_session_output[command_number + context]", "grep_session_output",
                          {"session_id": MAIN, "pattern": "^Design area", "command_number": area["number"],
                           "context_lines": 1, "ignore_case": False})
        if data is not None:
            lines = area["output"].split("\n")
            index = next((i for i, x in enumerate(lines) if x.startswith("Design area")), None)
            want = None if index is None else {
                "command_number": area["number"], "line_number": index + 1, "line": lines[index],
                "before": lines[max(0, index - 1):index], "after": lines[index + 1:index + 2]}
            got = [{k: m.get(k) for k in ("command_number", "line_number", "line", "before", "after")}
                   for m in data.get("matches") or []]
            add(session, "grep_session_output[command_number + context]",
                [] if want and got == [want] else [f"matches {got} (expected [{want}])"],
                f"only command #{area['number']} searched; context lines are its neighbours")
        data = ok_payload(session, "grep_session_output[regex that matches nothing -> literal]",
                          "grep_session_output", {"session_id": MAIN, "pattern": "u^2"})
        if data is not None:
            hits = {(m.get("command_number"), m.get("line")) for m in data.get("matches") or []}
            want = {(r["number"], line) for r in executed for line in r["output"].split("\n") if "u^2" in line.lower()}
            add(session, "grep_session_output[regex that matches nothing -> literal]",
                problems_of({"pattern_kind": "substring-fallback"}, data)
                + ([] if hits == want else [f"hits {sorted(hits)} (expected {sorted(want)})"]),
                f"'u^2' is valid regex matching nothing; retried literally: {len(want)} line(s)")


def check_history(session: Session, executed: list[dict]) -> None:
    want = [{"command_number": r["number"], "command": r["step"].command.strip(),
             "output_length": len(r["output"])} for r in executed]
    data = ok_payload(session, "get_session_history[all]", "get_session_history", {"session_id": MAIN})
    if data is not None:
        history = data.get("history") or []
        got = sorted(({k: h.get(k) for k in ("command_number", "command", "output_length")} for h in history),
                     key=lambda h: h["command_number"] or 0)
        mismatch = [f"#{g['command_number']}: {g} vs {w}" for g, w in zip(got, want) if g != w]
        problems = mismatch[:4] + ([] if len(got) == len(want) else [f"{len(got)} entries (expected {len(want)})"])
        problems += problems_of({"total_commands": len(want)}, data)
        add(session, "get_session_history[all]", problems,
            f"{len(want)} commands numbered 1..{len(want)} (blocked ones absent), output_length = len(output)")
        # sorted by a millisecond timestamp: two commands in the same millisecond keep oldest-first
        numbers = [h.get("command_number") for h in history]
        if numbers != sorted(numbers, reverse=True):
            session.report.add("L1", "get_session_history[order]", "WARN", f"not most-recent-first: {numbers}")
    data = ok_payload(session, "get_session_history[limit=3]", "get_session_history",
                      {"session_id": MAIN, "limit": 3})
    if data is not None:
        numbers = sorted(h.get("command_number") for h in data.get("history") or [])
        expected = sorted([r["number"] for r in executed][::-1][:3])
        add(session, "get_session_history[limit=3]",
            ([] if numbers == expected else [f"numbers {numbers} (expected {expected})"])
            + problems_of({"total_commands": 3, "limit": 3}, data),
            "the 3 most recent; total_commands is the number returned (as docs/API.md shows)")
    data = ok_payload(session, "get_session_history[search]", "get_session_history",
                      {"session_id": MAIN, "search": "REPORT_"})
    if data is not None:
        got = sorted(h.get("command_number") for h in data.get("history") or [])
        expected = [r["number"] for r in executed if "report_" in r["step"].command.lower()]
        add(session, "get_session_history[search]",
            ([] if got == expected else [f"numbers {got} (expected {expected})"])
            + problems_of({"total_commands": len(expected), "search": "REPORT_"}, data),
            f"case-insensitive substring: {len(expected)} report_* commands")


def check_inspect(session: Session, sid: str, count: int) -> None:
    data = ok_payload(session, f"inspect_interactive_session[{sid}]", "inspect_interactive_session",
                      {"session_id": sid})
    if data is None:
        return
    metrics = data.get("metrics") or {}
    commands, buffer = metrics.get("commands") or {}, metrics.get("buffer") or {}
    problems = problems_of({"session_id": sid, "state": "active", "is_alive": True}, metrics)
    problems += problems_of({"total_executed": count, "current_count": count, "history_length": count}, commands)
    problems += problems_of({"max_size": BUFFER_SIZE}, buffer)
    problems += problems_of({"configured_seconds": None, "is_timed_out": False}, metrics.get("timeout") or {})
    add(session, f"inspect_interactive_session[{sid}]", problems,
        f"{count} commands executed, buffer {BUFFER_SIZE} B, no session timeout")


def check_side_session(session: Session) -> int:
    """cwd/env options, per-call timeout, isolation from the main session. Returns its command count."""
    fixtures = session.state["fixtures"]
    if not check_create(session, SIDE, cwd=str(fixtures), env=SIDE_ENV):
        return 0
    count = 0
    command = 'puts "[pwd] $env(OR_L1_ENV)"'
    data = ok_payload(session, "exec[cwd + env options]", TOOLS["exec"], {"session_id": SIDE, "command": command})
    if data is not None:
        count += 1
        lines, _ = result_lines(data.get("output") or "", command)
        want = [f"{os.path.realpath(fixtures)} {SIDE_ENV['OR_L1_ENV']}"]
        check_lines(session, "exec[cwd + env options]", lines, want,
                    "the session runs in the requested cwd with the extra environment variable")
    command = "read_lef tiny.lef"
    data = ok_payload(session, "exec[relative path in cwd]", TOOLS["exec"], {"session_id": SIDE, "command": command})
    if data is not None:
        count += 1
        lines, _ = result_lines(data.get("output") or "", command)
        absolute = str(fixtures / "tiny.lef")
        want = [line.replace(absolute, "tiny.lef") for line in session.state["direct_lines"].get("read_lef", [])]
        check_lines(session, "exec[relative path in cwd]", lines, want,
                    "tiny.lef read relative to the session cwd: the direct run's lines with the relative name")
    data = expect_error(session, "exec[timeout_ms]", TOOLS["exec"],
                        {"session_id": SIDE, "command": "exec sleep 3", "timeout_ms": 1000},
                        "CommandTimeout: command did not complete within 1000ms")
    if data is not None:
        count += 1
        time.sleep(3.5)
        data = ok_payload(session, "query[after a timed-out command]", TOOLS["query"],
                          {"session_id": SIDE, "command": f"puts {AFTER_TIMEOUT_MARK}"})
        if data is not None:
            count += 1
            lines, junk = result_lines(data.get("output") or "", f"puts {AFTER_TIMEOUT_MARK}")
            if junk:
                session.report.add("L1", "query[after a timed-out command]", "WARN",
                                   f"leftover output of the timed-out command before the echo: {junk[:3]}")
            else:
                check_lines(session, "query[after a timed-out command]", lines, [AFTER_TIMEOUT_MARK],
                            "the session recovers: the next command gets only its own output")
    data = ok_payload(session, f"get_session_history[{SIDE}]", "get_session_history", {"session_id": SIDE})
    if data is not None:
        got = sorted(h.get("command_number") for h in data.get("history") or [])
        add(session, f"get_session_history[{SIDE}]", [] if got == list(range(1, count + 1)) else [f"numbers {got}"],
            f"its own {count} commands only; the main session's history is separate")
    return count


def check_launch_options(session: Session) -> None:
    """Duplicate id and the launch-command allowlist of create_interactive_session."""
    expect_error(session, "create_interactive_session[duplicate id]", "create_interactive_session",
                 {"session_id": MAIN}, f"Session {MAIN} already exists")
    for sid, command, want in (
            ("l1c", ["sh"], "not in the allowed commands list"),
            ("l1d", ["openroad", "-no_init;id"], "shell metacharacters"),
            ("l1e", ["openroad", "../x"], "path traversal")):
        data = expect_error(session, f"create_interactive_session[command {command}]", "create_interactive_session",
                            {"session_id": sid, "command": command}, want)
        if data is not None and data.get("error") is None:
            call_payload(session, f"terminate_interactive_session[{sid}]", "terminate_interactive_session",
                         {"session_id": sid, "force": True})
    openroad = session.state.get("openroad")
    if not openroad:
        return
    alt = session.tmp / "alt"
    alt.mkdir(exist_ok=True)
    ran = alt / "ran"
    wrapper = alt / "openroad"
    wrapper.write_text(f'#!/bin/sh\n: > {ran}\nexec {openroad} "$@"\n')
    wrapper.chmod(0o755)
    data, _ = call_payload(session, "create_interactive_session[any executable named openroad]",
                           "create_interactive_session", {"session_id": "l1f", "command": [str(wrapper), "-no_init"]})
    if data is None:
        return
    if data.get("error") is None:
        session.report.add("L1", "create_interactive_session[any executable named openroad]",
                           "WARN" if ran.exists() else "FAIL",
                           "the launch allowlist compares only the basename: a script at an arbitrary absolute "
                           f"path named 'openroad' was started (it ran: {ran.exists()})")
        call_payload(session, "terminate_interactive_session[l1f]", "terminate_interactive_session",
                     {"session_id": "l1f", "force": True})
    else:
        session.report.add("L1", "create_interactive_session[any executable named openroad]", "PASS",
                           f"refused: {data.get('error')}")


def check_list_and_metrics(session: Session, counts: dict[str, int]) -> None:
    data = ok_payload(session, "list_interactive_sessions", "list_interactive_sessions", {})
    if data is not None:
        got = {s.get("session_id"): (s.get("command_count"), s.get("is_alive"), s.get("state"))
               for s in data.get("sessions") or []}
        want = {sid: (n, True, "active") for sid, n in counts.items()}
        add(session, "list_interactive_sessions",
            ([] if got == want else [f"sessions {got} (expected {want})"])
            + problems_of({"total_count": len(counts), "active_count": len(counts)}, data),
            f"{len(counts)} live sessions with their command counts; failed creates left nothing behind")
    data = ok_payload(session, "get_session_metrics", "get_session_metrics", {})
    if data is not None:
        metrics = data.get("metrics") or {}
        manager, aggregate = metrics.get("manager") or {}, metrics.get("aggregate") or {}
        problems = problems_of({"total_sessions": len(counts), "active_sessions": len(counts),
                                "terminated_sessions": 0, "max_sessions": MAX_SESSIONS}, manager)
        if abs((manager.get("utilization_percent") or 0) - 100 * len(counts) / MAX_SESSIONS) > 1e-9:
            problems.append(f"utilization_percent={manager.get('utilization_percent')}")
        problems += problems_of({"total_commands": sum(counts.values())}, aggregate)
        ids = sorted(s.get("session_id") for s in metrics.get("sessions") or [])
        if ids != sorted(counts):
            problems.append(f"sessions {ids}")
        add(session, "get_session_metrics", problems,
            f"{len(counts)}/{MAX_SESSIONS} sessions, {sum(counts.values())} commands in total")


def check_unknown_session(session: Session) -> None:
    for tool, arguments in (
            ("interactive_openroad_query", {"command": "puts x"}), ("interactive_openroad_exec", {"command": "puts x"}),
            ("get_session_history", {}), ("inspect_interactive_session", {}),
            ("grep_session_output", {"pattern": "x"}), ("terminate_interactive_session", {})):
        want = "SessionNotFound" if tool == "grep_session_output" else "not found"
        expect_error(session, f"{tool}[unknown session]", tool, {"session_id": "nosuch", **arguments}, want)


def check_auto_session(session: Session, live: set[str]) -> None:
    """A command without session_id: runs, but in a new session that stays alive."""
    data = ok_payload(session, "interactive_openroad_query[no session_id]", TOOLS["query"], {"command": "puts auto"})
    if data is None:
        return
    lines, _ = result_lines(data.get("output") or "", "puts auto")
    listing = ok_payload(session, "list_interactive_sessions[after no session_id]", "list_interactive_sessions", {})
    extra = sorted({s.get("session_id") for s in (listing or {}).get("sessions") or []} - live)
    if lines != ["auto"]:
        session.report.add("L1", "interactive_openroad_query[no session_id]", "FAIL", f"printed {lines}")
    elif extra:
        session.report.add("L1", "interactive_openroad_query[no session_id]", "WARN",
                           f"ran in a new session {extra} that is left running; every call without "
                           "session_id starts another openroad process")
    else:
        session.report.add("L1", "interactive_openroad_query[no session_id]", "PASS", "no session left behind")
    for sid in extra:
        call_payload(session, f"terminate_interactive_session[{sid}]", "terminate_interactive_session",
                     {"session_id": sid})


def check_terminate(session: Session, sid: str) -> None:
    data = ok_payload(session, f"terminate_interactive_session[{sid}]", "terminate_interactive_session",
                      {"session_id": sid})
    if data is not None:
        add(session, f"terminate_interactive_session[{sid}]",
            problems_of({"session_id": sid, "terminated": True, "was_alive": True, "force": False}, data))
    expect_error(session, f"interactive_openroad_query[{sid} after terminate]", TOOLS["query"],
                 {"session_id": sid, "command": "puts x"}, "not found")
    data = expect_error(session, f"terminate_interactive_session[{sid} twice]", "terminate_interactive_session",
                        {"session_id": sid}, "not found")
    if data is not None and data.get("terminated") is not False:
        session.report.add("L1", f"terminate_interactive_session[{sid} twice][terminated]", "FAIL",
                           f"terminated={data.get('terminated')}")


def run_session_group(session: Session) -> None:
    executed = check_main_session(session)
    if not executed:
        return
    check_grep(session, executed)
    check_history(session, executed)
    check_inspect(session, MAIN, len(executed))
    side = check_side_session(session)
    check_launch_options(session)
    counts = {MAIN: len(executed), **({SIDE: side} if side else {})}
    check_list_and_metrics(session, counts)
    check_unknown_session(session)
    check_auto_session(session, set(counts))
    for sid in counts:
        check_terminate(session, sid)
    data = ok_payload(session, "list_interactive_sessions[after terminate]", "list_interactive_sessions", {})
    if data is not None:
        add(session, "list_interactive_sessions[after terminate]",
            problems_of({"sessions": [], "total_count": 0, "active_count": 0}, data))


# --------------------------------------------------------------------------
# ORFS group
# --------------------------------------------------------------------------

def check_report_images(session: Session) -> None:
    flow = session.state["flow"]
    reports = flow / "reports" / PLATFORM / DESIGN / RUN
    where = {"platform": PLATFORM, "design": DESIGN, "run_slug": RUN}
    data = ok_payload(session, "list_report_images", "list_report_images", where)
    if data is not None:
        want = {"final": [{"filename": name, "path": str(reports / name), "size_bytes": (reports / name).stat().st_size,
                           "type": IMAGE_TYPES[name]} for name in sorted(IMAGE_TYPES)]}
        got = {stage: [{k: i.get(k) for k in ("filename", "path", "size_bytes", "type")} for i in items]
               for stage, items in (data.get("images_by_stage") or {}).items()}
        problems = problems_of({"run_path": str(reports), "total_images": 2}, data)
        problems += [] if got == want else [f"images {got} (expected {want})"]
        add(session, "list_report_images", problems,
            "2 images by stage with exact sizes and types; the escaping symlink and notes.txt are not listed")
    data = ok_payload(session, "list_report_images[stage filter]", "list_report_images", {**where, "stage": "cts"})
    if data is not None:
        add(session, "list_report_images[stage filter]",
            problems_of({"total_images": 0, "images_by_stage": {}}, data), "no cts images")

    name, w, h = SMALL_PNG
    result = session.call("read_report_image[raw]", "read_report_image", {**where, "image_name": name})
    if result is not None:
        blocks = [b for b in result.get("content", []) if b.get("type") == "image"]
        meta = (json.loads(text_of(result) or "{}").get("metadata") or {})
        raw = (reports / name).read_bytes()
        problems = [] if blocks and blocks[0].get("data") == base64.b64encode(raw).decode() else ["image bytes differ"]
        mime = blocks[0].get("mimeType") if blocks else None
        problems += [] if mime == "image/png" else [f"mime {mime}"]
        problems += problems_of({"filename": name, "format": "png", "size_bytes": len(raw), "width": w, "height": h,
                                 "stage": "final", "type": IMAGE_TYPES[name], "compression_applied": False,
                                 "original_size_bytes": None, "original_width": w, "original_height": h}, meta)
        add(session, "read_report_image[raw]", problems,
            f"{w}x{h} PNG returned byte-for-byte as an image block, metadata exact")

    name, w, h = LARGE_PNG
    result = session.call("read_report_image[resized]", "read_report_image", {**where, "image_name": name})
    if result is not None:
        blocks = [b for b in result.get("content", []) if b.get("type") == "image"]
        meta = (json.loads(text_of(result) or "{}").get("metadata") or {})
        problems = []
        try:
            data_bytes = base64.b64decode(blocks[0]["data"]) if blocks else b""
            size = webp_size(data_bytes)
        except (ValueError, KeyError) as exc:
            data_bytes, size = b"", None
            problems.append(f"not a WebP image block: {exc}")
        box = resized_box(w, h)
        if size is not None:
            if size != (meta.get("width"), meta.get("height")):
                problems.append(f"decoded {size} vs metadata {meta.get('width')}x{meta.get('height')}")
            if abs(size[0] - box[0]) > 1 or abs(size[1] - box[1]) > 1:
                problems.append(f"{size} (expected about {box[0]:g}x{box[1]:g}: long edge {MAX_DIMENSION})")
        problems += [] if blocks and blocks[0].get("mimeType") == "image/webp" else ["mime is not image/webp"]
        problems += problems_of({"format": "webp", "compression_applied": True, "size_bytes": len(data_bytes),
                                 "original_size_bytes": (session.state["flow"] / "reports" / PLATFORM / DESIGN / RUN
                                                         / name).stat().st_size,
                                 "original_width": w, "original_height": h}, meta)
        add(session, "read_report_image[resized]", problems,
            f"{w}x{h} PNG resized to {size and size[0]}x{size and size[1]} WebP (long edge {MAX_DIMENSION}), "
            "decoded size = metadata")

    for check, arguments, want in (
            ("read_report_image[symlink out of the tree]", {**where, "image_name": ESCAPE_LINK}, "not contained"),
            ("read_report_image[path in image_name]", {**where, "image_name": "../base/final_placement.png"},
             "ValidationError"),
            ("read_report_image[not an image name]", {**where, "image_name": "notes.txt"}, "InvalidImageName"),
            ("read_report_image[missing image]", {**where, "image_name": "final_clocks.png"}, "ImageNotFound"),
            ("read_report_image[unknown run]", {**where, "run_slug": "nosuch", "image_name": SMALL_PNG[0]},
             "RunNotFound")):
        data = expect_error(session, check, "read_report_image", arguments, want)
        if data is not None and data.get("error") is not None and check.endswith("[missing image]"):
            if not all(n in str(data.get("message")) for n in IMAGE_TYPES):
                session.report.add("L1", f"{check}[available list]", "FAIL", str(data.get("message"))[:200])
    expect_error(session, "list_report_images[unknown platform]", "list_report_images",
                 {**where, "platform": "nosuch"}, "ValidationError")
    expect_error(session, "list_report_images[unknown run]", "list_report_images", {**where, "run_slug": "nosuch"},
                 "RunNotFound")


def check_metrics(session: Session) -> None:
    stems = ["2_1_floorplan", "2_2_floorplan_io", "3_3_place_gp"]
    metrics = {"2_1_floorplan": FLOORPLAN_METRICS, "2_2_floorplan_io": {}, "3_3_place_gp": PLACE_GP_METRICS}
    logs = {"2_1_floorplan": (FLOORPLAN_LOG_ERRORS, FLOORPLAN_LOG_WARNINGS), "2_2_floorplan_io": ([], []),
            "3_3_place_gp": None}
    for stage, selected in (("all", stems), ("floorplan", stems[:2]), ("globalplace", stems[2:])):
        check = f"read_orfs_metrics[stage={stage}]"
        data = ok_payload(session, check, "read_orfs_metrics", {"design": DESIGN, "stage": stage})
        if data is None:
            continue
        problems = problems_of({"platform": PLATFORM, "design": DESIGN, "variant": RUN, "stage": stage,
                                "logs_path": f"logs/{PLATFORM}/{DESIGN}/{RUN}", "available_stages": stems,
                                "rules_path": f"designs/{PLATFORM}/{DESIGN}/rules-base.json", "message": None}, data)
        got_stages = data.get("stages") or []
        if [s.get("stage") for s in got_stages] != selected:
            problems.append(f"stages {[s.get('stage') for s in got_stages]} (expected {selected})")
        for s in got_stages:
            stem = s.get("stage")
            if stem not in metrics:
                continue
            json_path = f"logs/{PLATFORM}/{DESIGN}/{RUN}/{stem}.json" if metrics[stem] else None
            repeated = [k for k, v in metrics[stem].items() if isinstance(v, list)]
            problems += [f"{stem}: {p}" for p in problems_of(
                {"metrics": metrics[stem], "metrics_path": json_path, "repeated_metrics": repeated, "error": None}, s)]
            log = s.get("log")
            if logs[stem] is None:
                if log is not None:
                    problems.append(f"{stem}: log {log} (no log file)")
            else:
                errors, warnings = logs[stem]
                problems += [f"{stem}.log: {p}" for p in problems_of(
                    {"path": f"logs/{PLATFORM}/{DESIGN}/{RUN}/{stem}.log", "errors": errors, "warnings": warnings,
                     "error_count": len(errors), "warning_count": len(warnings), "truncated": False}, log or {})]
        gates, unmatched, summary = expected_gates([(s, metrics[s]) for s in selected], RULES)
        got_gates = [gate_digest(g) for g in data.get("gates") or []]
        if got_gates != gates:
            problems.append(f"gates {got_gates} (expected {gates})")
        if sorted(u.get("metric") for u in data.get("unmatched_gates") or []) != sorted(unmatched):
            problems.append(f"unmatched {data.get('unmatched_gates')} (expected {unmatched})")
        problems += problems_of({"gate_summary": summary}, data)
        add(session, check, problems,
            f"{len(selected)} stage(s) with exact metrics (repeated key kept as a list) and tagged log lines; "
            f"gates {summary['pass']} pass / {summary['fail']} fail / {summary['unmatched']} unmatched")
    for check, arguments, want in (
            ("read_orfs_metrics[unknown stage]", {"design": DESIGN, "stage": "nosuch"}, "StageNotFound"),
            ("read_orfs_metrics[unknown design]", {"design": "nosuch"}, "ValidationError"),
            ("read_orfs_metrics[unknown variant]", {"design": DESIGN, "variant": "nosuch"}, "RunNotFound"),
            ("read_orfs_metrics[path in design]", {"design": "../x"}, "ValidationError")):
        expect_error(session, check, "read_orfs_metrics", arguments, want)


def make_command(stage: str, variant: str, *, dry_run: bool = False, overrides: dict | None = None) -> str:
    parts = ["make", *(["-n"] if dry_run else []), stage, f"DESIGN_CONFIG=./designs/{PLATFORM}/{DESIGN}/config.mk",
             f"FLOW_VARIANT={variant}", *(f"{k}={v}" for k, v in (overrides or {}).items())]
    return " ".join(parts)


def wait_job(session: Session, check: str, job_id: str, limit: float = 30.0) -> dict | None:
    """Poll get_orfs_job until the job has finished. A cancelled or timed-out job carries its
    reason in ``error``: that describes the job, not a failed call, so it is not counted."""
    deadline = time.monotonic() + limit
    while True:
        data, _ = call_payload(session, check, "get_orfs_job", {"job_id": job_id}, count_in_band=False)
        if data is None or data.get("finished_at") is not None or time.monotonic() > deadline:
            return data
        time.sleep(0.5)


def process_gone(pid: int, limit: float = 5.0) -> bool:
    """True once ``pid`` no longer exists (or is a zombie awaiting its reaper)."""
    deadline = time.monotonic() + limit
    while True:
        try:
            os.kill(pid, 0)
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except ProcessLookupError:
            return True
        except (OSError, IndexError):
            state = None
        if state in ("Z", "X"):
            return True
        if time.monotonic() > deadline:
            return False
        time.sleep(0.2)


def read_pid(path: Path, limit: float = 15.0) -> int | None:
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        try:
            return int(path.read_text().strip())
        except (OSError, ValueError):
            time.sleep(0.2)
    return None


def check_flow_runs(session: Session) -> None:
    flow = session.state["flow"]
    base_metrics = flow / "logs" / PLATFORM / DESIGN / RUN / "2_1_floorplan.json"
    before = base_metrics.read_bytes()
    data = ok_payload(session, "run_orfs_stage[dry run]", "run_orfs_stage",
                      {"design": DESIGN, "stage": "floorplan", "dry_run": True, "wait_seconds": 30}, timeout=60)
    if data is not None:
        command = make_command("floorplan", RUN, dry_run=True)
        recent = data.get("recent_lines") or []
        problems = problems_of({"status": "succeeded", "exit_code": 0, "platform": PLATFORM, "design": DESIGN,
                                "variant": RUN, "stage": "floorplan", "dry_run": True, "command": command,
                                "stages": [], "gates": []}, data)
        if recent[:1] != [f"$ {command}"]:
            problems.append(f"log starts {recent[:1]}")
        if not any(line.startswith(f"echo OR_L1_FLOORPLAN ./designs/{PLATFORM}/{DESIGN}/config.mk {RUN}")
                   for line in recent):
            problems.append(f"make -n did not print the recipe: {recent}")
        if base_metrics.read_bytes() != before:
            problems.append("a dry run rewrote the stage metrics")
        log_path = Path(str(data.get("log_path") or ""))
        if not log_path.is_file() or not str(log_path).startswith(str(session.scratch)):
            problems.append(f"log_path {data.get('log_path')} is not a file under TMPDIR")
        add(session, "run_orfs_stage[dry run]", problems,
            "platform inferred; exact make -n command; recipe printed, nothing run; stale metrics not reported")

    data = ok_payload(session, "run_orfs_stage[overrides]", "run_orfs_stage",
                      {"design": DESIGN, "platform": PLATFORM, "stage": "floorplan", "variant": RUN_VARIANT,
                       "overrides": RUN_OVERRIDES, "wait_seconds": 60}, timeout=120)
    run_id = None
    if data is not None:
        run_id = data.get("job_id")
        run_metrics = {"floorplan__design__instance__count": 3,
                       "floorplan__design__core__area": float(RUN_OVERRIDES["OR_L1_AREA"])}
        gates, _, summary = expected_gates([("2_1_floorplan", run_metrics)], RULES)
        problems = problems_of({"status": "succeeded", "exit_code": 0, "variant": RUN_VARIANT, "dry_run": False,
                                "overrides": RUN_OVERRIDES,
                                "command": make_command("floorplan", RUN_VARIANT, overrides=RUN_OVERRIDES)}, data)
        echo = f"OR_L1_FLOORPLAN ./designs/{PLATFORM}/{DESIGN}/config.mk {RUN_VARIANT} {RUN_OVERRIDES['OR_L1_TAG']}"
        if echo not in [line.strip() for line in data.get("recent_lines") or []]:
            problems.append(f"recipe output missing: {data.get('recent_lines')}")
        stages = [(s.get("stage"), s.get("metrics")) for s in data.get("stages") or []]
        if stages != [("2_1_floorplan", run_metrics)]:
            problems.append(f"stages {stages}")
        if [gate_digest(g) for g in data.get("gates") or []] != gates:
            problems.append(f"gates {data.get('gates')} (expected {gates})")
        want_summary = {k: summary[k] for k in ("total", "pass", "fail", "failing_errors")}
        problems += problems_of({"gate_summary": want_summary}, data)
        add(session, "run_orfs_stage[overrides]", problems,
            "make variables passed through to the recipe; the run's own stage metrics parsed and gated "
            f"({summary['pass']} pass, {summary['fail']} fail)")
    if run_id:
        data = ok_payload(session, "get_orfs_job[finished run]", "get_orfs_job", {"job_id": run_id})
        if data is not None:
            add(session, "get_orfs_job[finished run]",
                problems_of({"job_id": run_id, "status": "succeeded", "exit_code": 0, "variant": RUN_VARIANT}, data))

    # two long runs fill the job limit; a third is refused; one is cancelled, one times out
    pidfiles = {v: session.scratch / f"{v}.pid" for v in ("slow1", "slow2")}
    jobs = {}
    for variant, extra in (("slow1", {"timeout_seconds": 2}), ("slow2", {})):
        data = ok_payload(session, f"run_orfs_stage[{variant}]", "run_orfs_stage",
                          {"design": DESIGN, "stage": "place", "variant": variant,
                           "overrides": {"OR_L1_PIDFILE": str(pidfiles[variant])}, **extra})
        if data is not None and data.get("status") == "running" and data.get("job_id"):
            jobs[variant] = data["job_id"]
        elif data is not None:
            session.report.add("L1", f"run_orfs_stage[{variant}]", "FAIL", f"not running: {json.dumps(data)[:300]}")
    expect_error(session, "run_orfs_stage[job limit]", "run_orfs_stage",
                 {"design": DESIGN, "stage": "place", "variant": "slow3", "dry_run": True},
                 f"already in progress (limit {MAX_FLOW_JOBS})")
    pids = {v: read_pid(pidfiles[v]) for v in jobs}
    if "slow2" in jobs:
        data = ok_payload(session, "cancel_orfs_job", "cancel_orfs_job", {"job_id": jobs["slow2"]})
        if data is not None:
            add(session, "cancel_orfs_job", problems_of({"job_id": jobs["slow2"], "status": "cancelled",
                                                          "cancelled": True}, data))
        done = wait_job(session, "get_orfs_job[cancelled]", jobs["slow2"])
        if done is not None:
            problems = problems_of({"status": "cancelled", "error": "Cancelled by request"}, done)
            if done.get("finished_at") is None or done.get("signal") is None:
                problems.append(f"finished_at={done.get('finished_at')} signal={done.get('signal')}")
            if pids.get("slow2") is None or not process_gone(pids["slow2"]):
                problems.append(f"the recipe's process {pids.get('slow2')} is still running")
            add(session, "get_orfs_job[cancelled]", problems,
                f"cancelled by signal {done.get('signal')}; the recipe's sleep (pid {pids.get('slow2')}) is gone")
    if "slow1" in jobs:
        done = wait_job(session, "get_orfs_job[timed out]", jobs["slow1"])
        if done is not None:
            problems = problems_of({"status": "timed_out", "error": "Flow run exceeded 2s and was terminated"}, done)
            if pids.get("slow1") is None or not process_gone(pids["slow1"]):
                problems.append(f"the recipe's process {pids.get('slow1')} is still running")
            add(session, "get_orfs_job[timed out]", problems,
                f"timeout_seconds=2 ended the run by signal {done.get('signal')}; its process group is gone")
    data = ok_payload(session, "get_orfs_job[list]", "get_orfs_job", {})
    if data is not None:
        got = {j.get("job_id"): j.get("status") for j in data.get("jobs") or []}
        want = {jid: status for jid, status in ((jobs.get("slow1"), "timed_out"), (jobs.get("slow2"), "cancelled"),
                                                (run_id, "succeeded")) if jid}
        problems = [f"{jid}: {got.get(jid)} (expected {s})" for jid, s in want.items() if got.get(jid) != s]
        problems += problems_of({"active_count": 0}, data)
        if data.get("total_count") != len(got):
            problems.append(f"total_count={data.get('total_count')} for {len(got)} jobs")
        add(session, "get_orfs_job[list]", problems, f"{len(got)} runs listed with their final status, none active")

    for check, arguments, want in (
            ("run_orfs_stage[target not allowed]", {"design": DESIGN, "stage": "-f/tmp/evil.mk"}, "ValidationError"),
            ("run_orfs_stage[SHELL override]", {"design": DESIGN, "stage": "floorplan",
                                                "overrides": {"SHELL": "/bin/sh"}}, "ValidationError"),
            ("run_orfs_stage[$( in override]", {"design": DESIGN, "stage": "floorplan",
                                                "overrides": {"X": "$(shell id)"}}, "ValidationError"),
            ("run_orfs_stage[path in variant]", {"design": DESIGN, "stage": "floorplan", "variant": "../x"},
             "ValidationError"),
            ("run_orfs_stage[unknown design]", {"design": "nosuch", "stage": "floorplan"}, "ValidationError")):
        expect_error(session, check, "run_orfs_stage", arguments, want)
    expect_error(session, "get_orfs_job[unknown job]", "get_orfs_job", {"job_id": "nope"}, "FlowJobNotFound")
    expect_error(session, "cancel_orfs_job[unknown job]", "cancel_orfs_job", {"job_id": "nope"}, "FlowJobNotFound")


def check_in_band_summary(session: Session) -> None:
    in_band = session.state.get("in_band") or Counter()
    if in_band:
        session.report.add("L1", "errors are in-band", "WARN",
                           f"{sum(in_band.values())} error results had isError=false (only the JSON 'error' field "
                           f"says so): {dict(sorted(in_band.items()))}; read_report_image alone sets isError",
                           in_band_errors=dict(in_band))
    else:
        session.report.add("L1", "errors are in-band", "PASS", "every error result set isError")


def run_l1(session: Session) -> None:
    if session.state.get("openroad"):
        run_session_group(session)
    check_report_images(session)
    check_metrics(session)
    check_flow_runs(session)
    check_in_band_summary(session)


# --------------------------------------------------------------------------
# Declaration
# --------------------------------------------------------------------------

def prepare(session: Session) -> None:
    """Fixtures, the direct reference run, and the fake ORFS tree (before the server starts)."""
    fixtures = session.tmp / "fixtures"
    fixtures.mkdir()
    (fixtures / "tiny.lef").write_text(TINY_LEF)
    (fixtures / "tiny.def").write_text(TINY_DEF)
    (fixtures / "l1_probe.tcl").write_text(PROBE_TCL)
    steps = session_steps(fixtures, session.tmp / "missing.lef")
    session.state.update(fixtures=fixtures, steps=steps, closed_form=closed_form(), direct=None, direct_lines={},
                         flow=build_flow_tree(session.home, session.tmp / "outside.png"))
    host_path = load_setup().HOST_PATH
    openroad = shutil.which("openroad", path=host_path)
    if openroad is None:
        session.report.add("L0", "openroad on the host", "FAIL",
                           f"not found in {host_path}; the session tools cannot work "
                           "(the ORFS tools are still checked)")
        return
    session.state["openroad"] = openroad
    workdir = session.tmp / "direct"
    workdir.mkdir()
    env = {"HOME": str(workdir), "TMPDIR": str(session.scratch), "PATH": host_path, "LANG": "C.UTF-8"}
    try:
        direct, stdout = run_direct(openroad, [s for s in steps if s.compare], workdir, env)
    except (OSError, subprocess.SubprocessError) as exc:
        session.report.add("L0", "direct openroad reference", "FAIL", str(exc))
        return
    compared = [s for s in steps if s.compare]
    missing = [s.name for i, s in enumerate(compared) if i not in direct]
    session.state.update(direct={steps.index(s): direct[i] for i, s in enumerate(compared) if i in direct},
                         direct_lines={s.name: direct[i][1] for i, s in enumerate(compared) if i in direct},
                         direct_stdout_tail=stdout[-4000:])
    session.report.add("L0", "direct openroad reference", "FAIL" if missing else "PASS",
                       f"no result for {missing}; stdout tail {stdout[-300:]!r}" if missing else
                       f"{openroad} -no_init -no_splash -exit ran {len(compared)} commands")


def _tool_versions(session: Session) -> dict:
    out = {}
    host_path = load_setup().HOST_PATH
    node = shutil.which("node", path=host_path)
    if node:
        try:
            out["node"] = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            out["node"] = None
    try:
        package = json.loads((session.checkout / "typescript" / "package.json").read_text(encoding="utf-8"))
        out["openroad-mcp"] = package.get("version")
    except (OSError, json.JSONDecodeError):
        out["openroad-mcp"] = None
    out["openroad"] = session.state.get("openroad_version")
    out["openroad_binary"] = session.state.get("openroad")
    return out


SMOKE = Smoke(
    server="openroad",
    run_l1=run_l1,
    prepare=prepare,
    report_fields=lambda session: {
        "backend": _tool_versions(session),
        "closed_form": {k: v for k, v in closed_form().items() if k != "placements"},
        "flow_tree": {"platform": PLATFORM, "design": DESIGN, "run": RUN},
        "direct_stdout_tail": session.state.get("direct_stdout_tail"),
    },
)
