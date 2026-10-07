#!/usr/bin/env python3
"""Measure the ground-truth table of the gpaw MCP E2E fake task.

The ``mcp_e2e.gpaw_mos2_bandgap`` task cannot recompute plane-wave DFT in its
``generate_gt.py``: GPAW only exists inside the pinned conda environment of the
MCP server, and a task runtime may not build one. The task therefore carries a
**measured** table of the server's own numbers, and this script is what
produces it — run once per (server revision, conda lock) pair, on the platform
the benchmark will run on::

    ~/mcp/gpaw/.venv/bin/python scripts/mcp/e2e/measure_gpaw_table.py \\
        --config ~/mcp/gpaw.mcp.json --output ~/mcp/gpaw-table.json

It drives the same six-tool chain the agent is asked to drive, over the task's
``(ecut, kpts_density)`` grid, and writes both a JSON record of every reply and
a ready-to-paste Python literal for ``generate_gt.py``.

What is measured and what is derived
------------------------------------
Only the plane-wave SCF depends on ``(ecut, kpts_density)``, so only that grid
is measured. The task's other two instance parameters are free:

* ``tol_mev_per_atom`` changes nothing that is computed — ``check_convergence``
  always sweeps the same hard-coded range (ecut 300-800 eV on the auto_kpts(15)
  grid, densities 10-45 at ecut 400) and the tolerance only picks a row. The
  generator therefore recomputes the recommendation from the measured sweep
  rows with the same pure policy function the L1 smoke validates.
* ``gap_tol_ev`` only selects a branch of ``verify_run``'s state machine, which
  is likewise recomputed from the measured artefacts.

Both derivations are checked against the server here, not just asserted: the
``CROSS_CHECK`` chain re-runs ``check_convergence`` and ``calc_band_dos`` at the
second tolerance, and every chain calls ``verify_run`` at both gap tolerances.

Cost: one chain of fetch -> relax -> convergence -> bands -> verify -> artifacts
per grid point, plus two extra calls in the cross-check chain. Measured on a
single-threaded amd64 host (2026-10-07): eight chains in 3799 s, the convergence
sweep being about 60% of each. Each step prints before it starts and the report
records its wall-clock time.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

E2E_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(E2E_DIR))

from e2e_smoke.client import MCPError, StdioMCP, text_of   # noqa: E402
from e2e_smoke.runner import load_setup, server_env        # noqa: E402

SERVER = "gpaw"

# The task's instance grid. Only these two parameters move the DFT.
ECUTS = (350, 400, 450, 500)
KPTS_DENSITIES = (15.0, 25.0)
# The task's free parameters; the first of each is the one the chains use.
TOL_MEV_PER_ATOM = (5.0, 0.3)
# Keep in step with GAP_TOL_EV in the task's generate_gt.py: measuring verify_run
# at tolerances no instance uses would confirm nothing about any instance.
GAP_TOL_EV = (0.005, 0.3)
# The chain that additionally re-runs the gate and the bands at TOL_MEV_PER_ATOM[1],
# so that the generator's derivation of the recommendation and of params_verified
# is confirmed against the server instead of assumed.
CROSS_CHECK = (400, 25.0)

# Fixed arguments of the chain: the prompt pins them, so they are not instance
# parameters and the agent cannot trade accuracy for time.
FMAX = 0.05
MAX_STEPS = 100
NPOINTS = 60
WINDOW_EV = 8.0
ENGINE = "gpaw"
QUERY = "MoS2"

CALL_TIMEOUT = 7200.0        # a correct run must never fail on the client's clock
ARTIFACT_NAMES = ("bands.png", "convergence.json", "dos.png", "fetch.json", "gs.gpw",
                  "relax.json", "relaxed.cif", "structure.cif", "summary.json", "verify.json")


class MeasureError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Calling the server
# --------------------------------------------------------------------------

def payload(result: dict) -> dict:
    """The business dict of a tools/call result (bare ``object`` outputSchema)."""
    data = result.get("structuredContent")
    if not isinstance(data, dict):
        data = json.loads(text_of(result))
    if not isinstance(data, dict):
        raise MeasureError("result is not a JSON object")
    return data


class Chain:
    """One run of the agent's tool chain, with per-step wall-clock times."""

    def __init__(self, client: StdioMCP, label: str) -> None:
        self.client = client
        self.label = label
        self.seconds: dict[str, float] = {}

    def call(self, step: str, tool: str, arguments: dict) -> dict:
        print(f"[ .. ] {self.label} {step}: {tool}", flush=True)
        started = time.monotonic()
        try:
            response = self.client.call_tool(tool, arguments, timeout=CALL_TIMEOUT)
        except MCPError as exc:
            raise MeasureError(f"{self.label} {step}: {tool} failed: {exc}") from exc
        finally:
            self.seconds[step] = round(time.monotonic() - started, 1)
        if "error" in response:
            raise MeasureError(f"{self.label} {step}: {tool} JSON-RPC error: {response['error']}")
        result = response["result"]
        if result.get("isError"):
            raise MeasureError(f"{self.label} {step}: {tool} returned isError: "
                               f"{text_of(result)[:300]!r}")
        data = payload(result)
        if data.get("ok") is not True:
            raise MeasureError(f"{self.label} {step}: {tool} returned ok={data.get('ok')!r}: "
                               f"{data.get('error')} {data.get('traceback_tail')}")
        print(f"[ ok ] {self.label} {step}: {self.seconds[step]:.1f} s", flush=True)
        return data


def strip_ok(data: dict) -> dict:
    return {k: v for k, v in data.items() if k != "ok"}


# --------------------------------------------------------------------------
# One chain
# --------------------------------------------------------------------------

def measure_chain(client: StdioMCP, runs_dir: Path, ecut: int, kd: float, *,
                  tol: float, cross_check: bool) -> dict:
    """fetch -> relax -> convergence -> bands -> verify -> artifacts, as the task asks."""
    label = f"ecut={ecut} kd={kd:g} tol={tol:g}"
    chain = Chain(client, label)

    fetch = chain.call("fetch", "fetch_structure", {"query": QUERY, "use_builtin": True})
    run_id = fetch.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise MeasureError(f"{label}: fetch_structure returned run_id={run_id!r}")
    run_path = runs_dir / run_id

    relax = chain.call("relax", "relax_structure",
                       {"run_id": run_id, "ecut": ecut, "kpts_density": kd,
                        "fmax": FMAX, "max_steps": MAX_STEPS, "engine": ENGINE})
    convergence = chain.call("convergence", "check_convergence",
                             {"run_id": run_id, "tol_mev_per_atom": tol, "engine": ENGINE})
    bands = chain.call("bands", "calc_band_dos",
                       {"run_id": run_id, "ecut": ecut, "kpts_density": kd,
                        "npoints": NPOINTS, "window_ev": WINDOW_EV, "engine": ENGINE})
    verify = {}
    for gap_tol in GAP_TOL_EV:
        # verify_run reads the artefacts and runs no SCF, so every branch is cheap;
        # the last call is the one that stays in verify.json.
        verify[repr(gap_tol)] = strip_ok(chain.call(f"verify[gap_tol={gap_tol:g}]", "verify_run",
                                                    {"run_id": run_id, "gap_tol_ev": gap_tol}))
    artifacts = chain.call("artifacts", "get_run_artifacts", {"run_id": run_id})

    names = sorted(Path(a["path"]).name for a in artifacts.get("artifacts") or [])
    missing = sorted(set(ARTIFACT_NAMES) - set(names))
    if missing:
        raise MeasureError(f"{label}: the chain left no {missing} in {run_path}")

    record = {
        "ecut": ecut, "kpts_density": kd, "tol_mev_per_atom": tol, "run_id": run_id,
        "fetch": strip_ok(fetch), "relax": strip_ok(relax),
        "convergence": strip_ok(convergence), "bands": strip_ok(bands),
        "verify_by_gap_tol": verify, "artifact_names": names,
        "artifact_count": len(names), "seconds": dict(chain.seconds),
    }

    if cross_check:
        # The same run, re-gated at the second tolerance: this is what tells us
        # whether the generator may derive the recommendation (and with it
        # params_verified) from the sweep rows instead of measuring it.
        other = TOL_MEV_PER_ATOM[1] if tol == TOL_MEV_PER_ATOM[0] else TOL_MEV_PER_ATOM[0]
        again = chain.call(f"convergence[tol={other:g}]", "check_convergence",
                           {"run_id": run_id, "tol_mev_per_atom": other, "engine": ENGINE})
        regated = chain.call(f"bands[tol={other:g}]", "calc_band_dos",
                             {"run_id": run_id, "ecut": ecut, "kpts_density": kd,
                              "npoints": NPOINTS, "window_ev": WINDOW_EV, "engine": ENGINE})
        record["cross_check"] = {"tol_mev_per_atom": other,
                                 "convergence": strip_ok(again), "bands": strip_ok(regated)}
        record["seconds"] = dict(chain.seconds)
    return record


# --------------------------------------------------------------------------
# Consistency of the measurement itself
# --------------------------------------------------------------------------

def sweep_spread(records: list[dict]) -> dict:
    """How far the convergence sweeps of the different chains drift apart.

    Every chain sweeps the same hard-coded range, but on *its own* relaxed
    geometry. If the spread is tiny the generator may keep one sweep table; if
    it is not, it needs one per grid point. The answer goes into the report so
    the choice is on the record rather than assumed.
    """
    out = {}
    for sweep, key in (("ecut_sweep", "ecut_ev"), ("kpts_sweep", "kpts_density")):
        by_point: dict = {}
        for record in records:
            for row in record["convergence"].get(sweep) or []:
                point = by_point.setdefault(row[key], {"energy_ev": [], "gap_ev": []})
                point["energy_ev"].append(row.get("energy_ev"))
                point["gap_ev"].append(row.get("gap_ev"))
        out[sweep] = {
            str(point): {
                field: (max(values) - min(values)) if values and None not in values else None
                for field, values in fields.items()
            }
            for point, fields in sorted(by_point.items())
        }
    return out


def grid_spread(records: list[dict]) -> dict:
    """The span of each scored quantity over the grid: the task is only worth
    running if the instance parameters actually move the answers."""
    fields = {
        "relax_total_energy_ev": lambda r: r["relax"].get("total_energy_ev"),
        "relax_max_force_ev_per_a": lambda r: r["relax"].get("max_force_ev_per_a"),
        "scf_total_energy_ev": lambda r: r["bands"].get("total_energy_ev"),
        "fermi_ev": lambda r: r["bands"].get("fermi_ev"),
        "band_gap_ev": lambda r: r["bands"].get("band_gap_ev"),
    }
    out = {}
    for name, read in fields.items():
        values = [read(r) for r in records]
        if any(v is None for v in values):
            out[name] = None
            continue
        out[name] = {"min": min(values), "max": max(values), "span": max(values) - min(values)}
    return out


# --------------------------------------------------------------------------
# The literal the generator carries
# --------------------------------------------------------------------------

def number(value) -> str:
    """A Python literal, not JSON: ``repr`` round-trips floats exactly and keeps
    ``True``/``None`` spelled the way the generator can import them."""
    return repr(value)


def render_literal(records: list[dict]) -> str:
    """``MEASURED`` for generate_gt.py: the server's own numbers, full precision."""
    lines = [
        "# Measured with scripts/mcp/e2e/measure_gpaw_table.py against the pinned matmcp",
        "# revision and conda lock; see that script for what is measured and what the",
        "# generator derives. Keyed by (ecut_ev, kpts_density).",
        "MEASURED = {",
    ]
    for record in sorted(records, key=lambda r: (r["ecut"], r["kpts_density"])):
        relax, bands, conv = record["relax"], record["bands"], record["convergence"]
        lines.append(f"    ({record['ecut']}, {record['kpts_density']!r}): {{")
        for key, value in (
            ("relax_total_energy_ev", relax.get("total_energy_ev")),
            ("relax_max_force_ev_per_a", relax.get("max_force_ev_per_a")),
            ("relax_n_steps", relax.get("n_steps")),
            ("relax_kpts", relax.get("kpts")),
            ("scf_total_energy_ev", bands.get("total_energy_ev")),
            ("fermi_ev", bands.get("fermi_ev")),
            ("band_gap_ev", bands.get("band_gap_ev")),
            ("gap_type", bands.get("gap_type")),
            ("vbm_label", (bands.get("vbm") or {}).get("label")),
            ("cbm_label", (bands.get("cbm") or {}).get("label")),
            ("kpts_scf", bands.get("kpts_scf")),
            ("band_path", bands.get("band_path")),
            ("artifact_count", record["artifact_count"]),
            # Measured at the chain's own tolerance; the generator derives these
            # from the sweep rows for any tolerance and is unit-tested against
            # the values recorded here.
            ("measured_at_tol_mev_per_atom", record["tol_mev_per_atom"]),
            ("measured_recommended_ecut_ev", conv.get("recommended_ecut_ev")),
            ("measured_recommended_kpts_density", conv.get("recommended_kpts_density")),
            ("measured_converged", conv.get("converged")),
            ("measured_params_verified", bands.get("params_verified")),
            ("measured_band_gap_ref", (record["fetch"] or {}).get("band_gap_ref")),
            # verify_run's own answer at each gap tolerance: the generator derives
            # these from the state machine and the offline tests compare.
            ("measured_verdict_by_gap_tol", {
                tol: data.get("verdict") for tol, data in record["verify_by_gap_tol"].items()}),
            ("measured_verify_labels_by_gap_tol", {
                tol: [f"{c.get('check')}:{c.get('status')}" for c in data.get("checks") or []]
                for tol, data in record["verify_by_gap_tol"].items()}),
        ):
            lines.append(f"        {json.dumps(key)}: {number(value)},")
        for sweep in ("ecut_sweep", "kpts_sweep"):
            rows = record["convergence"].get(sweep) or []
            key = "ecut_ev" if sweep == "ecut_sweep" else "kpts_density"
            lines.append(f"        {json.dumps(sweep)}: [")
            for row in rows:
                parts = [f"{json.dumps(key)}: {number(row.get(key))}",
                         f'"energy_ev": {number(row.get("energy_ev"))}',
                         f'"gap_ev": {number(row.get("gap_ev"))}']
                if row.get("kpts") is not None:
                    parts.append(f'"kpts": {json.dumps(row["kpts"])}')
                lines.append("            {" + ", ".join(parts) + "},")
            lines.append("        ],")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines) + "\n"


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
           "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")}
    for package in ("gpaw", "ase", "fastmcp", "numpy", "scipy"):
        try:
            out[package] = md.version(package)
        except md.PackageNotFoundError:
            out[package] = None
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the gpaw E2E ground-truth table")
    parser.add_argument("--config", required=True, help="generated <root>/gpaw.mcp.json")
    parser.add_argument("--output", default="gpaw-table.json", help="where the measurement goes")
    parser.add_argument("--literal", default=None,
                        help="also write the MEASURED literal here (default: <output>.py)")
    parser.add_argument("--only", default=None,
                        help="comma-separated ecut:density pairs, e.g. 400:25 — for a dry run")
    args = parser.parse_args(argv)

    setup = load_setup()
    entry = setup.load_manifest()[SERVER]
    config = json.loads(Path(args.config).expanduser().read_text(encoding="utf-8"))
    server = config["mcpServers"][SERVER]
    checkout = Path(server["cwd"])
    revision = git_revision(checkout)
    if revision != entry["revision"]:
        raise MeasureError(f"checkout at {revision}, manifest pins {entry['revision']}")

    grid = [(e, k) for e in ECUTS for k in KPTS_DENSITIES]
    if args.only:
        wanted = {(int(p.split(":")[0]), float(p.split(":")[1]))
                  for p in args.only.split(",") if p.strip()}
        grid = [point for point in grid if point in wanted]
        if not grid:
            raise MeasureError(f"--only {args.only!r} selects none of {grid}")

    env = server.get("env", {})
    runs_dir = (Path(env["MATMCP_RUNS"]) if env.get("MATMCP_RUNS")
                else Path(env.get("MATMCP_REPO") or checkout) / "runs")
    print(f"[ .. ] {len(grid)} chains over {grid}; runs under {runs_dir}", flush=True)

    started = time.monotonic()
    records = []
    with tempfile.TemporaryDirectory(prefix="mcp-e2e-gpaw-measure-") as tmp:
        tmp_path = Path(tmp)
        home, cwd, scratch = tmp_path / "home", tmp_path / "cwd", tmp_path / "tmpdir"
        for directory in (home, cwd, scratch):
            directory.mkdir()
        client = StdioMCP(server["command"], server["args"], cwd=cwd,
                          env=server_env(server, home, scratch),
                          stderr_path=tmp_path / "server.stderr.log")
        try:
            client.initialize()
            for index, (ecut, kd) in enumerate(grid, start=1):
                print(f"[ .. ] chain {index}/{len(grid)}", flush=True)
                records.append(measure_chain(client, runs_dir, ecut, kd,
                                             tol=TOL_MEV_PER_ATOM[0],
                                             cross_check=(ecut, kd) == CROSS_CHECK))
        finally:
            client.close()

    elapsed = round(time.monotonic() - started, 1)
    report = {
        "server": SERVER, "revision": revision, "config": str(Path(args.config).expanduser()),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": package_versions(),
        "fixed_arguments": {"query": QUERY, "use_builtin": True, "fmax": FMAX,
                            "max_steps": MAX_STEPS, "npoints": NPOINTS,
                            "window_ev": WINDOW_EV, "engine": ENGINE},
        "grid": [list(point) for point in grid],
        "tol_mev_per_atom": list(TOL_MEV_PER_ATOM), "gap_tol_ev": list(GAP_TOL_EV),
        "records": records,
        "sweep_spread_across_chains": sweep_spread(records),
        "grid_spread": grid_spread(records),
        "seconds_total": elapsed,
    }
    output = Path(args.output).expanduser()
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    literal = Path(args.literal).expanduser() if args.literal else output.with_suffix(".py")
    literal.write_text(render_literal(records), encoding="utf-8")

    print(f"\n[ ok ] {len(records)} chains in {elapsed:.0f} s -> {output}\n       literal -> {literal}")
    print("\ngrid spread (does the instance parameter move the answer?):")
    print(json.dumps(report["grid_spread"], indent=2))
    print("\nconvergence sweep spread across chains (one shared table, or one per point?):")
    print(json.dumps(report["sweep_spread_across_chains"], indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MeasureError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1) from None
