#!/usr/bin/env python3
"""Measure the ground-truth table of the quantum_espresso MCP E2E fake task.

The ``mcp_e2e.qe_si_bandstructure`` task cannot recompute DFT in its
``generate_gt.py``: ``pw.x`` and ``bands.x`` only exist inside the pinned conda
environment of the MCP server, and a task runtime may not build one. The task
therefore carries a **measured** table of the server's own numbers, and this
script is what produces it — run once per (server revision, conda lock) pair::

    ~/mcp/quantum_espresso/.venv/bin/python scripts/mcp/e2e/measure_qe_table.py \\
        --config ~/mcp/quantum_espresso.mcp.json --output ~/mcp/qe-table.json

It drives the same five-tool chain the agent is asked to drive over the task's
grid and writes a JSON record of every reply plus a ready-to-paste Python
literal for ``generate_gt.py``. ``--check`` instead compares a fresh
measurement with the table the generator already carries and exits non-zero on
any difference: that is how a second host (another platform, a re-installed
server) proves the committed table is its own answer, value for value.

What is measured and what is derived
------------------------------------
The DFT depends on the cutoff, the realized SCF grid and the number of
band-path points, so only ``(ecutwfc, grid, npoints_band)`` is measured. The
task's instance parameter for the grid is a k-spacing; the grid it realizes is
derived in the generator by the server's documented rule (the L1 smoke checks
that rule), and every ``qe_suggest_kpoints`` answer for the task's spacings is
recorded here so the derivation is reconciled with the server, not assumed.
``qe_list_pseudopotentials`` is called once: which Si file the server indexes
depends on the host's directory order (D1), so the table records the pick it
was measured with and the generator puts it into every reference.

Cost: 27 chains of seconds each (2-atom Si, at most 9x9x9 and 40 band points);
``--ecut`` measures one cutoff at a time for hosts with a short command limit,
and ``--merge`` joins the partial JSON records into one table.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

E2E_DIR = Path(__file__).resolve().parent
REPO = E2E_DIR.parents[2]
sys.path.insert(0, str(E2E_DIR))

from e2e_smoke.client import MCPError, StdioMCP, text_of   # noqa: E402
from e2e_smoke.runner import load_setup, server_env        # noqa: E402

SERVER = "quantum_espresso"
TASK_GENERATOR = REPO / "examples/mcp-e2e-tasks/mcp_e2e/qe_si_bandstructure/generate_gt.py"

# Keep in step with generate_gt.py (the offline tests compare the two).
STRUCTURE = "Si"
ECUTWFC = (20, 25, 30)
KSPACING = (0.06, 0.04, 0.03)
NPOINTS_BAND = (20, 30, 40)

CALL_TIMEOUT = 1800.0        # a correct run must never fail on the client's clock


class MeasureError(RuntimeError):
    pass


def payload(result: dict) -> dict:
    data = result.get("structuredContent")
    if isinstance(data, dict) and isinstance(data.get("result"), dict) and len(data) == 1:
        data = data["result"]
    if not isinstance(data, dict):
        data = json.loads(text_of(result))
    if not isinstance(data, dict):
        raise MeasureError("result is not a JSON object")
    return data


def call(client: StdioMCP, label: str, tool: str, arguments: dict, seconds: dict) -> dict:
    print(f"[ .. ] {label}: {tool}", flush=True)
    started = time.monotonic()
    try:
        response = client.call_tool(tool, arguments, timeout=CALL_TIMEOUT)
    except MCPError as exc:
        raise MeasureError(f"{label}: {tool} failed: {exc}") from exc
    finally:
        seconds[f"{label} {tool}"] = round(time.monotonic() - started, 2)
    if "error" in response:
        raise MeasureError(f"{label}: {tool} JSON-RPC error: {response['error']}")
    result = response["result"]
    if result.get("isError"):
        raise MeasureError(f"{label}: {tool} returned isError: {text_of(result)[:300]!r}")
    data = payload(result)
    if data.get("success") is False:
        raise MeasureError(f"{label}: {tool} returned success=false: {data.get('error')!r}")
    return data


def measure_chain(client: StdioMCP, ecut: int, grid: list[int], npoints: int, seconds: dict) -> dict:
    """bandstructure -> list_files -> read_bands on the gnu file, as the task asks."""
    label = f"ecut={ecut} grid={grid[0]} npoints={npoints}"
    bands = call(client, label, "qe_workflow_bandstructure",
                 {"structure": STRUCTURE, "kpoints": ",".join(map(str, grid)),
                  "ecutwfc": ecut, "npoints_band": npoints}, seconds)
    folder = bands.get("output_dir")
    if not isinstance(folder, str):
        raise MeasureError(f"{label}: no output_dir in {sorted(bands)}")
    files = call(client, label, "qe_list_files", {"output_dir": folder}, seconds)
    gnu = files.get("band_files") or []
    if len(gnu) != 1:
        raise MeasureError(f"{label}: expected one band file, got {gnu}")
    read = call(client, label, "qe_read_bands", {"output_dir": gnu[0]}, seconds)
    pseudo = sorted(p.name for p in (Path(folder) / "pseudo").glob("*.upf"))
    names = sorted(Path(p).name for key, value in files.items()
                   if key.endswith("_files") and isinstance(value, list) for p in value)
    return {
        "ecutwfc": ecut, "grid": list(grid), "npoints_band": npoints,
        "bands": {k: bands.get(k) for k in (
            "total_energy_eV", "fermi_energy_eV", "band_gap_eV", "is_metal", "is_direct_gap",
            "vbm_eV", "cbm_eV", "n_bands", "n_kpoints", "high_symmetry_points")},
        "workflow_id": bands.get("workflow_id"),
        "file_names": names,
        "read_bands": {"n_bands": read.get("n_bands"), "n_kpoints": read.get("n_kpoints"),
                       "path_length": max(read.get("k_distances") or [float("nan")])},
        "pseudo_copied": pseudo,
    }


# --------------------------------------------------------------------------
# The literal the generator carries
# --------------------------------------------------------------------------

def table_from(records: list[dict], kgrids: dict, pick: dict) -> dict:
    """The ``MEASURED`` mapping: the server's own numbers, full precision."""
    points = {}
    for r in sorted(records, key=lambda r: (r["ecutwfc"], r["grid"][0], r["npoints_band"])):
        b = r["bands"]
        points[(r["ecutwfc"], r["grid"][0], r["npoints_band"])] = {
            "total_energy_eV": b["total_energy_eV"],
            "fermi_energy_eV": b["fermi_energy_eV"],
            "band_gap_eV": b["band_gap_eV"],
            "is_direct_gap": b["is_direct_gap"],
            "is_metal": b["is_metal"],
            "vbm_eV": b["vbm_eV"],
            "cbm_eV": b["cbm_eV"],
            "n_bands": b["n_bands"],
            "n_kpoints": b["n_kpoints"],
            "path_length": r["read_bands"]["path_length"],
            "read_n_kpoints": r["read_bands"]["n_kpoints"],
            "file_names": r["file_names"],
            "pseudo_copied": r["pseudo_copied"],
        }
    return {"si_pseudopotential": pick.get("filename"),
            "si_cutoff_hints_ry": [pick.get("ecutwfc_Ry"), pick.get("ecutrho_Ry")],
            "suggested_grid_by_kspacing": {repr(k): v for k, v in sorted(kgrids.items())},
            "points": points}


def render_literal(table: dict, header: list[str]) -> str:
    lines = [*(f"# {h}" for h in header), "MEASURED = {"]
    for key in ("si_pseudopotential", "si_cutoff_hints_ry", "suggested_grid_by_kspacing"):
        lines.append(f"    {key!r}: {table[key]!r},")
    lines.append("    'points': {")
    for point, row in table["points"].items():
        lines.append(f"        {point!r}: {{")
        for key, value in row.items():
            lines.append(f"            {key!r}: {value!r},")
        lines.append("        },")
    lines.append("    },")
    lines.append("}")
    return "\n".join(lines) + "\n"


def spread(table: dict) -> dict:
    """Per scored field: how many distinct values the grid gives, and the smallest
    non-zero distance between two of them (what a scorer's zero-credit tolerance
    has to stay below for a copied wrong instance to score nothing)."""
    out = {}
    rows = list(table["points"].values())
    for field in ("total_energy_eV", "fermi_energy_eV", "band_gap_eV", "vbm_eV", "cbm_eV", "path_length"):
        values = sorted({row[field] for row in rows})
        gaps = [b - a for a, b in zip(values, values[1:])]
        out[field] = {"distinct": len(values), "min_gap": min(gaps) if gaps else None,
                      "min_relative_gap": min((g / max(abs(a), abs(b)) for g, a, b in
                                               zip(gaps, values, values[1:])), default=None)}
    return out


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

def git_revision(checkout: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict:
    import importlib.metadata as md
    out = {"python": platform.python_version(), "platform": platform.platform(),
           "machine": platform.machine(), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")}
    for package in ("mcp", "ase", "numpy"):
        try:
            out[package] = md.version(package)
        except md.PackageNotFoundError:
            out[package] = None
    return out


def committed_table() -> dict:
    spec = importlib.util.spec_from_file_location("qe_generate_gt", TASK_GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MEASURED


def compare(fresh: dict, committed: dict) -> list[str]:
    problems = []
    for key in ("si_pseudopotential", "si_cutoff_hints_ry", "suggested_grid_by_kspacing"):
        if fresh[key] != committed.get(key):
            problems.append(f"{key}: measured {fresh[key]!r}, committed {committed.get(key)!r}")
    for point, row in fresh["points"].items():
        old = committed.get("points", {}).get(point)
        if old is None:
            problems.append(f"{point}: not in the committed table")
            continue
        problems += [f"{point} {k}: measured {v!r}, committed {old.get(k)!r}"
                     for k, v in row.items() if old.get(k) != v]
    return problems


def measure(args, entry: dict, server: dict) -> dict:
    ecuts = [e for e in ECUTWFC if not args.ecut or e in args.ecut]
    seconds: dict[str, float] = {}
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="mcp-e2e-qe-measure-") as tmp:
        tmp_path = Path(tmp)
        home, cwd, scratch = tmp_path / "home", tmp_path / "cwd", tmp_path / "tmpdir"
        for directory in (home, cwd, scratch):
            directory.mkdir()
        client = StdioMCP(server["command"], server["args"], cwd=cwd,
                          env=server_env(server, home, scratch),
                          stderr_path=tmp_path / "server.stderr.log")
        try:
            client.initialize()
            pseudos = call(client, "setup", "qe_list_pseudopotentials", {}, seconds)
            pick = (pseudos.get("details") or {}).get("Si") or {}
            kgrids = {}
            for spacing in KSPACING:
                data = call(client, f"kspacing={spacing}", "qe_suggest_kpoints",
                            {"structure": STRUCTURE, "kspacing": spacing}, seconds)
                kgrids[spacing] = data.get("kpoints")
            records = []
            for ecut in ecuts:
                for spacing in KSPACING:
                    for npoints in NPOINTS_BAND:
                        records.append(measure_chain(client, ecut, kgrids[spacing], npoints, seconds))
        finally:
            client.close()
    return {"server": SERVER, "revision": entry["revision"],
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "environment": package_versions(), "si_pick": pick, "suggested_grid_by_kspacing":
                {repr(k): v for k, v in kgrids.items()},
            "records": records, "seconds": seconds,
            "seconds_total": round(time.monotonic() - started, 1)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the quantum_espresso E2E ground-truth table")
    parser.add_argument("--config", help="generated <root>/quantum_espresso.mcp.json")
    parser.add_argument("--output", default="qe-table.json", help="where the measurement goes")
    parser.add_argument("--ecut", type=int, action="append",
                        help="measure only this cutoff (repeatable); join parts with --merge")
    parser.add_argument("--merge", nargs="+", type=Path,
                        help="no server: join partial JSON records into one table")
    parser.add_argument("--check", action="store_true",
                        help="compare with the MEASURED table generate_gt.py carries; exit 1 on a difference")
    args = parser.parse_args(argv)

    if args.merge:
        parts = [json.loads(p.read_text(encoding="utf-8")) for p in args.merge]
        if len({(p["revision"], json.dumps(p["si_pick"], sort_keys=True)) for p in parts}) != 1:
            raise MeasureError("the parts disagree on the revision or the Si pick")
        report = {**parts[0], "records": [r for p in parts for r in p["records"]],
                  "seconds": {k: v for p in parts for k, v in p["seconds"].items()},
                  "seconds_total": round(sum(p["seconds_total"] for p in parts), 1),
                  "merged_from": [str(p) for p in args.merge]}
    else:
        if not args.config:
            parser.error("--config is required unless --merge is given")
        entry = load_setup().load_manifest()[SERVER]
        config = json.loads(Path(args.config).expanduser().read_text(encoding="utf-8"))
        server = config["mcpServers"][SERVER]
        revision = git_revision(Path(server["cwd"]))
        if revision != entry["revision"]:
            raise MeasureError(f"checkout at {revision}, manifest pins {entry['revision']}")
        report = measure(args, entry, server)

    kgrids = {float(k): v for k, v in report["suggested_grid_by_kspacing"].items()}
    table = table_from(report["records"], kgrids, report["si_pick"])
    report["table"] = {**table, "points": {repr(k): v for k, v in table["points"].items()}}
    report["spread"] = spread(table)
    output = Path(args.output).expanduser()
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    env = report["environment"]
    header = [f"Generated by scripts/mcp/e2e/measure_qe_table.py against qe-mcp {report['revision']}",
              f"and the committed conda lock, on {env['machine']} ({env['platform']}),",
              f"{len(report['records'])} chains in {report['seconds_total']:.0f} s, {report['measured_at'][:10]}.",
              "Replace wholesale when either is bumped; never hand-edit a value."]
    literal = output.with_suffix(".py")
    literal.write_text(render_literal(table, header), encoding="utf-8")
    print(f"\n[ ok ] {len(report['records'])} chains, {report['seconds_total']:.0f} s -> {output}"
          f"\n       literal -> {literal}\n       Si pick: {report['si_pick']}")
    print(json.dumps(report["spread"], indent=2))

    if args.check:
        complete = len(table["points"]) == len(ECUTWFC) * len(KSPACING) * len(NPOINTS_BAND)
        problems = compare(table, committed_table())
        if not complete:
            problems.append(f"only {len(table['points'])} points measured; --check needs the whole grid")
        for problem in problems:
            print(f"[DIFF] {problem}")
        print("[ ok ] identical to the committed table" if not problems
              else f"[FAIL] {len(problems)} difference(s) from the committed table")
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MeasureError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1) from None
