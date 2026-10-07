"""Instance generator for the OpenROAD MCP E2E fake task.

A seeded RNG picks a tiny placed design: 4-7 inverters and buffers in one or
two standard-cell rows, a chain netlist from an input pin to an output pin with
at most one fan-out net, and a die around the rows. The generator writes it as
LEF + DEF of our own (one routing layer, one site, two macros; no third-party
PDK). Through the OpenROAD MCP server the agent must open an interactive
session, read both files, report the area / utilisation, print the die box and
the total half-perimeter wirelength from the database, and find the area
report again in the session's retained output with ``grep_session_output``.

Every scored quantity has a **closed form** (counts, the die box, the summed
master areas over the row box, HPWL from pin-shape centres), so the reference
does not depend on the openroad build and needs only the standard library. The
LEF/DEF text is the one the L1 smoke validated on the target host: with the L1
constants, :func:`render_lef` / :func:`render_def` reproduce its ``TINY_LEF`` /
``TINY_DEF`` byte for byte (tests/mcp_e2e).

A correct number does **not** prove that the agent used the server: every
quantity also follows from reading the two files. What does is the verifier
(e2e_check.json): each answer has to be a number some session command
*printed* (the prompts ask for in-session Tcl arithmetic), the session id the
server created must be the one later calls name, and ``no_bypass`` rejects a
direct openroad run, a database library imported into Python, and Tcl ``exec``
sent through the session tools.

Integer quantities only: master widths of 1 and 2 um on 2 um rows give whole
um^2 areas, and a utilisation within 0.05 of a .5 tie (``%.0f`` rounding) is
rejected; every pin-shape centre is an even DBU sum, so HPWL is exact.

Framework call: python generate_gt.py --output-dir <dir> --params '<json>'
Only the standard library: generate with ``asibench generate --sandbox none``
or ``--sandbox task``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

INPUT_SPEC = [
    {"name": "task.json", "description": "design name, the LEF/DEF file names, the session id and the grep pattern"},
    {"name": "design.lef", "description": "technology + library: one layer, one site, INV and BUF"},
    {"name": "design.def", "description": "the placed design: die, rows, components, I/O pins, nets"},
]
OUTPUT_SPEC = [
    {"name": "result.json", "description": "counts, die box, design area and utilisation, total HPWL "
                                           "and the area report line, as OpenROAD printed them"},
]
DEFAULT_PARAMS = {"seed": 0}

DBU = 1000
SITE = (200, 2000)                       # core site width, height (DBU)
LAYER = "metal1"
# name -> (width DBU, {pin: rect relative to the cell origin}); pin rects have even sums
MACROS = {
    "INV": (1000, {"A": (50, 950, 150, 1050), "Y": (850, 950, 950, 1050)}),
    "BUF": (2000, {"A": (50, 950, 150, 1050), "Y": (1850, 950, 1950, 1050)}),
}
BTERM_SHAPE = (-100, -50, 100, 50)
PIN_IN, PIN_OUT = "in", "out"
GREP_PATTERN = "^Design area"
DESIGNS = ("relay", "ripple", "skew_line", "tap_chain", "delay_line", "fanout_pair")
MCP_TOOLS = ("create_interactive_session", "interactive_openroad_exec", "interactive_openroad_query",
             "grep_session_output")


# --------------------------------------------------------------------------
# LEF / DEF text (the layout of servers/openroad.py TINY_LEF / TINY_DEF)
# --------------------------------------------------------------------------

def _um(dbu: int) -> str:
    """A DBU length in microns as LEF writes it: ``0.05``, ``1.0``, ``2.0``."""
    text = f"{dbu / DBU:.3f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def render_lef(library: list[str]) -> str:
    head = f"""\
VERSION 5.8 ;
BUSBITCHARS "[]" ;
DIVIDERCHAR "/" ;

UNITS
  DATABASE MICRONS {DBU} ;
END UNITS

MANUFACTURINGGRID 0.005 ;

LAYER {LAYER}
  TYPE ROUTING ;
  DIRECTION HORIZONTAL ;
  PITCH 0.2 ;
  WIDTH 0.1 ;
  SPACING 0.1 ;
END {LAYER}

SITE core
  CLASS CORE ;
  SYMMETRY Y ;
  SIZE {_um(SITE[0])} BY {_um(SITE[1])} ;
END core
"""
    blocks = []
    for name in library:
        width, pins = MACROS[name]
        text = (f"\nMACRO {name}\n  CLASS CORE ;\n  ORIGIN 0 0 ;\n  SIZE {_um(width)} BY {_um(SITE[1])} ;\n"
                f"  SYMMETRY X Y ;\n  SITE core ;\n")
        for pin, rect in pins.items():
            direction = "INPUT" if pin == "A" else "OUTPUT"
            text += (f"  PIN {pin}\n    DIRECTION {direction} ;\n    USE SIGNAL ;\n    PORT\n"
                     f"      LAYER {LAYER} ;\n        RECT {' '.join(_um(v) for v in rect)} ;\n"
                     f"    END\n  END {pin}\n")
        blocks.append(text + f"END {name}\n")
    return head + "".join(blocks) + "\nEND LIBRARY\n"


def render_def(case: dict) -> str:
    die = case["die"]
    rows = "".join(f"ROW ROW_{i} core {x} {y} N DO {case['row_sites']} BY 1 STEP {SITE[0]} 0 ;\n"
                   for i, (x, y) in enumerate(case["rows"]))
    comps = "".join(f"- {name} {master} + PLACED ( {x} {y} ) N ;\n"
                    for name, master, x, y in case["components"])
    s = BTERM_SHAPE
    pins = ""
    for name, direction in ((PIN_IN, "INPUT"), (PIN_OUT, "OUTPUT")):
        x, y = case["pins"][name]
        pins += (f"- {name} + NET {name} + DIRECTION {direction} + USE SIGNAL\n"
                 f"  + LAYER {LAYER} ( {s[0]} {s[1]} ) ( {s[2]} {s[3]} )\n  + PLACED ( {x} {y} ) N ;\n")
    nets = "".join(f"- {name} {' '.join(f'( {o} {p} )' for o, p in terms)} + USE SIGNAL ;\n"
                   for name, terms in case["nets"])
    return (f"VERSION 5.8 ;\nDIVIDERCHAR \"/\" ;\nBUSBITCHARS \"[]\" ;\nDESIGN {case['design']} ;\n"
            f"UNITS DISTANCE MICRONS {DBU} ;\n\nDIEAREA ( {die[0]} {die[1]} ) ( {die[2]} {die[3]} ) ;\n\n"
            f"{rows}\nCOMPONENTS {len(case['components'])} ;\n{comps}END COMPONENTS\n\n"
            f"PINS 2 ;\n{pins}END PINS\n\nNETS {len(case['nets'])} ;\n{nets}END NETS\n\nEND DESIGN\n")


# --------------------------------------------------------------------------
# The instance
# --------------------------------------------------------------------------

def _netlist(n: int, rng: random.Random) -> list[tuple[str, list[tuple[str, str]]]]:
    """``in`` -> u1 -> ... -> un -> ``out``; with probability 1/2 one cell instead hangs off
    its grandparent, so that driver fans out to two loads and its parent's output dangles."""
    parent = {i: i - 1 for i in range(2, n + 1)}
    if rng.random() < 0.5:
        k = rng.randint(3, n)
        parent[k] = k - 2
    nets = [(PIN_IN, [("PIN", PIN_IN), ("u1", "A")])]
    for driver in range(1, n):
        loads = [i for i in range(2, n + 1) if parent[i] == driver]
        if loads:
            nets.append((f"n{driver}", [(f"u{driver}", "Y")] + [(f"u{i}", "A") for i in loads]))
    nets.append((PIN_OUT, [(f"u{n}", "Y"), ("PIN", PIN_OUT)]))
    return nets


def _utilisation(cell_area: int, core_area: int) -> float:
    return 100.0 * cell_area / core_area


def build_case(seed: int) -> dict:
    rng = random.Random(seed)
    while True:
        n = rng.randint(4, 7)
        masters = [rng.choice(("INV", "INV", "BUF")) for _ in range(n)]
        n_rows = rng.choice((1, 2))
        sites = rng.randint(30, 70)
        margin_x = rng.choice((1000, 1400, 2000))
        margin_y = rng.choice((2000, 3000))
        per_row = [masters[i::n_rows] for i in range(n_rows)]              # u1 row 0, u2 row 1, ...
        widths = [sum(MACROS[m][0] // SITE[0] for m in row) for row in per_row]
        if max(widths) > sites:
            continue
        cell_area = sum(MACROS[m][0] * SITE[1] for m in masters)
        core_area = sites * SITE[0] * n_rows * SITE[1]
        util = _utilisation(cell_area, core_area)
        if abs(util - math.floor(util) - 0.5) < 0.05:
            continue
        break
    rows = [(margin_x, margin_y + r * SITE[1]) for r in range(n_rows)]
    die = (0, 0, 2 * margin_x + sites * SITE[0], 2 * margin_y + n_rows * SITE[1])
    slots: dict[int, tuple[int, int]] = {}
    for r, row in enumerate(per_row):
        free = sites - widths[r]
        gaps = sorted(rng.randint(0, free) for _ in row)                    # cumulative gaps, in sites
        x_site = 0
        for j, master in enumerate(row):
            index = r + j * n_rows                                         # back to u-numbering (0-based)
            slots[index] = (rows[r][0] + (x_site + gaps[j]) * SITE[0], rows[r][1])
            x_site += MACROS[master][0] // SITE[0]
    pin_y = [rng.randrange(400, die[3] - 400 + 1, 200) for _ in range(2)]
    design = f"{rng.choice(DESIGNS)}_{seed % 1000}"
    return {
        "seed": seed,
        "design": design,
        "lef_file": "data/design.lef",
        "def_file": "data/design.def",
        "session_id": f"or{seed}",
        "grep_pattern": GREP_PATTERN,
        "library": sorted(MACROS, reverse=True),                           # INV, BUF
        "die": list(die),
        "rows": [list(r) for r in rows],
        "row_sites": sites,
        "components": [[f"u{i + 1}", masters[i], *slots[i]] for i in range(n)],
        "pins": {PIN_IN: [100, pin_y[0]], PIN_OUT: [die[2] - 100, pin_y[1]]},
        "nets": [[name, [list(t) for t in terms]] for name, terms in _netlist(n, rng)],
    }


# --------------------------------------------------------------------------
# Closed-form reference
# --------------------------------------------------------------------------

def _centre(rect) -> tuple[int, int]:
    return (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2


def terminal_centre(case: dict, owner: str, pin: str) -> tuple[int, int]:
    if owner == "PIN":
        x, y = case["pins"][pin]
        s = BTERM_SHAPE
        return _centre((x + s[0], y + s[1], x + s[2], y + s[3]))
    _, master, x, y = next(c for c in case["components"] if c[0] == owner)
    r = MACROS[master][1][pin]
    return _centre((x + r[0], y + r[1], x + r[2], y + r[3]))


def reference(case: dict) -> dict:
    hpwl = {}
    for name, terms in case["nets"]:
        points = [terminal_centre(case, owner, pin) for owner, pin in terms]
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        hpwl[name] = max(xs) - min(xs) + max(ys) - min(ys)
    n = len(case["components"])
    rows = case["rows"]
    core = (rows[0][0], rows[0][1], rows[0][0] + case["row_sites"] * SITE[0], rows[-1][1] + SITE[1])
    cell_area = sum(MACROS[m][0] * SITE[1] for _, m, _, _ in case["components"])
    core_area = (core[2] - core[0]) * (core[3] - core[1])
    design_area = cell_area // DBU ** 2
    util = math.floor(_utilisation(cell_area, core_area) + 0.5)
    connections = sum(1 for _, terms in case["nets"] for owner, _ in terms if owner != "PIN")
    die = case["die"]
    return {
        "created_command_count": 0,
        "lef_facts": ["layers=1", f"library_cells={len(case['library'])}"],
        "def_facts": ["pins=2", f"components={n}", f"component_terminals={2 * n}",
                      f"nets={len(case['nets'])}", f"connections={connections}"],
        "area_facts": [f"design_area_um2={design_area}", f"utilization_percent={util}"],
        "die_numbers": [die[2], die[3]],
        "hpwl_numbers": [sum(hpwl.values())],
        "instance_count": n,
        "net_count": len(case["nets"]),
        "io_pin_count": 2,
        "die_width_dbu": die[2] - die[0],
        "die_height_dbu": die[3] - die[1],
        "design_area_um2": design_area,
        "utilization_percent": util,
        "hpwl_total_dbu": sum(hpwl.values()),
        "area_report_line": f"Design area {design_area} u^2 {util}% utilization.",
        "core": list(core),
        "core_area_um2": core_area / DBU ** 2,
        "hpwl_per_net_dbu": hpwl,
        "mcp_tools": list(MCP_TOOLS),
    }


# --------------------------------------------------------------------------
# Framework entry point
# --------------------------------------------------------------------------

def render_prompts(task_dir: Path, output_dir: Path, values: dict) -> None:
    for level in ("b1", "b2", "b3", "b4"):
        text = (task_dir / f"prompt_{level}.md").read_text(encoding="utf-8")
        for key, value in values.items():
            text = text.replace("{{" + key + "}}", str(value))
        if "{{" in text:
            raise RuntimeError(f"unresolved placeholder in prompt_{level}.md")
        (output_dir / f"prompt_{level}.md").write_text(text, encoding="utf-8")


def generate(output_dir: Path, params: dict) -> dict:
    p = {**DEFAULT_PARAMS, **params}
    t0 = time.time()
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    ref_dir = output_dir / "reference"
    data_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    case = build_case(int(p["seed"]))
    ref = reference(case)
    task = {key: case[key] for key in ("design", "lef_file", "def_file", "session_id", "grep_pattern")}
    lef, deff = Path(case["lef_file"]).name, Path(case["def_file"]).name
    (data_dir / lef).write_text(render_lef(case["library"]), encoding="utf-8")
    (data_dir / deff).write_text(render_def(case), encoding="utf-8")
    (data_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
    (ref_dir / "reference.json").write_text(json.dumps({**case, **ref}, indent=2) + "\n", encoding="utf-8")
    render_prompts(Path(__file__).resolve().parent, output_dir, task)
    meta = {
        "params_used": p,
        "input_files": ["task.json", lef, deff],
        "reference_files": ["reference.json"],
        "generation_time_seconds": round(time.time() - t0, 2),
    }
    (output_dir / "instance_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params", type=str, default="{}")
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, json.loads(args.params)), indent=2))
