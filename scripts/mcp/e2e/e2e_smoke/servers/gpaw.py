"""Direct (agent-free) E2E smoke test for the pinned matmcp MCP server (GPAW engine).

Run with the server's own conda prefix (it provides GPAW and ASE for the
references)::

    ~/mcp/gpaw/.venv/bin/python scripts/mcp/e2e/smoke.py gpaw --config ~/mcp/gpaw.mcp.json

No network and no Materials Project key: every physics check uses the built-in
2H-MoS2 monolayer builder, which is the server's only credential-free structure
path (``use_builtin=true``, or a query in ``BUILTIN_BUILDERS``).

References are computed here, in this process, never in the server:

* the k-grid policy, the convergence recommendation, the gap analysis and the
  verifier state machine are reimplemented in stdlib (pure functions below,
  unit-tested offline in ``tests/mcp_e2e/test_smoke_gpaw.py``);
* the DFT numbers come from GPAW called directly in this process with the
  settings the engine documents (PW(ecut), PBE, FermiDirac(0.05),
  Gamma-centered grid, ASE BFGS) — one relaxation and three SCF points, which
  together cover the headline numbers of every tool:

      relax_structure     energy / max force / step count / relaxed geometry
      calc_band_dos       total energy, Fermi level, gap, gap character and
                          VBM/CBM from our own analysis of our own SCF, plus
                          the server's ``gs.gpw`` reopened and compared
      check_convergence   the full delta/recommendation arithmetic recomputed
                          from the returned sweep, two sweep points re-run
      verify_run          all five checks and the gap_tol_ev state machine
      run_verified_workflow  the converged chain (bit-identical gap) and a
                          deliberately unconverged one (the gate must produce
                          no gap and no figures)
      get_run_artifacts   reconciled against the run directory on disk

L1 classification: a wrong number for a correctly used tool is FAIL. Upstream
defects that do not make correct use wrong are WARN, and each WARN names the
defect precisely, because fake-task authors have to work around exactly these:
non-JSON stdout from ``calc_band_dos``, the single shared ``gs.gpw`` restart
file, ``query`` ignored for built-ins, the ``verification_note`` that is
returned but never written to ``summary.json``, in-band errors with
``isError=false``, and the whole chain hiding behind one
``run_verified_workflow`` call.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from ..client import text_of
from ..helpers import max_abs_diff, png_size, quiet_fds, sorted_distances
from ..runner import Caller, Session, Smoke, check_rejected

# The built-in structure: workflows._build_mos2_monolayer().
MOS2_BUILDER = {"formula": "MoS2", "kind": "2H", "a": 3.18, "thickness": 3.17, "vacuum": 10.0}
MOS2_FORMULA = "MoS2 (monolayer 2H)"
MP_GAP_REF = 1.66             # fetch_structure's hard-coded MP mp-1023924 reference
RUN_ID_RE = re.compile(r"\d{8}-\d{6}_[0-9a-z-]{1,36}_[0-9a-f]{4}")

# Server defaults we rely on (and pass explicitly, so the smoke does not depend
# on them staying the defaults).
ECUT = 400                    # eV
KD_RELAX = 25.0               # auto_kpts -> 9x9x1 on this cell
KD_BANDS = 30.0               # calc_band_dos default; the same realized 9x9x1 grid
KD_UNVERIFIED = 10.0          # below the gate's recommendation -> params_verified=false
FMAX = 0.05                   # eV/A
MAX_STEPS = 100
NPOINTS = 60                  # band path points (calc_band_dos default)
TOL_MEV_PER_ATOM = 5.0        # check_convergence default
TOL_GAP_EV = 0.02             # check_convergence default, NOT exposed over MCP
ECUT_ENERGY_TOL = 100.0       # check_convergence default pathology backstop
JUNK = {"ecut": 200, "kpts_density": 3.0}   # cannot pass the gate (survey: rejected_unconverged)

# Tolerances. Same program, same settings, our own call path: GPAW is
# deterministic for a fixed grid (repeated calls agree bit for bit; only the
# thread count moves the last digits, and the launch env pins
# OMP_NUM_THREADS=1), so these are generous by two orders of magnitude.
ENERGY_TOL = 1e-4             # eV, total energies
FORCE_TOL = 1e-4              # eV/A
GAP_TOL = 1e-4                # eV, band gap / Fermi level
DIST_TOL = 1e-3               # A, interatomic distances through a CIF round-trip
CELL_TOL = 1e-4               # A / degrees
EIG_TOL = 1e-6                # eV, eigenvalues of the same SCF read back
SAME_GRID_TOL = 1e-6          # eV, two calculations of one realized grid
ARITHMETIC_TOL = 1e-9         # the server's own delta arithmetic, relative
GAP_DRIFT_NOTE = 2e-3         # eV, gap difference between gpaw releases (26.7 vs 25.7)

# verify_run thresholds: |gap - MP ref| is about 0.015 eV, so these three
# tolerances select the three verdicts of the state machine.
GAP_TOL_FAIL, GAP_TOL_WARN, GAP_TOL_PASS = 0.005, 0.012, 0.3

# Plane-wave DFT is minutes per call, and a loaded or throttled host multiplies
# that: the tools/call timeout must never be what fails a correct result. The
# longest call is run_verified_workflow (relax + a ten-point sweep + bands).
CALL_TIMEOUT = 7200.0         # seconds

# Wall-clock seconds per step, filled as the run proceeds and written to the
# report: this is the only record of how slow a host was, and what an L2 task
# has to budget for.
TIMINGS: dict[str, float] = {}


def progress(what: str) -> None:
    """A single step here can take minutes; say what is running before it starts."""
    print(f"[ .. ] L1 {what}", flush=True)


def timed(label: str, call, *args, **kwargs):
    """Run ``call``, record its wall-clock time under ``label`` in :data:`TIMINGS`."""
    started = time.monotonic()
    try:
        return call(*args, **kwargs)
    finally:
        TIMINGS[label] = round(time.monotonic() - started, 1)


# --------------------------------------------------------------------------
# Pure helpers (stdlib only; unit-tested offline)
# --------------------------------------------------------------------------

def payload(result: dict) -> dict:
    """The business dict of a tools/call result.

    Every tool declares a bare ``object`` outputSchema, so ``structuredContent``
    is the dict itself (no ``{"result": ...}`` wrapper); the text block carries
    the same JSON."""
    data = result.get("structuredContent")
    if not isinstance(data, dict):
        data = json.loads(text_of(result))
    if not isinstance(data, dict):
        raise ValueError("result is not a JSON object")
    return data


def in_band_error(result: dict) -> str | None:
    """``{"ok": false, "error": ...}`` returned with isError=false, for check_rejected."""
    try:
        data = payload(result)
    except (ValueError, json.JSONDecodeError):
        return None
    if data.get("ok") is False and data.get("error"):
        return f"ok=false, {data['error']}"
    return None


def vacuum_axes_reference(lengths, scaled, min_gap_a: float = 8.0) -> tuple[bool, bool, bool]:
    """``workflows._vacuum_axes``: an axis is vacuum when the largest gap between
    atoms along it (wrapped, in Angstrom) exceeds ``min_gap_a`` — the atom span,
    not the cell length, so layered bulk is not mistaken for a slab."""
    out = []
    for i in range(3):
        frac = sorted(s[i] % 1.0 for s in scaled)
        gaps = [b - a for a, b in zip(frac, frac[1:])] + [frac[0] + 1.0 - frac[-1]]
        out.append(bool(max(gaps) * lengths[i] > min_gap_a))
    return tuple(out)


def auto_kpts_reference(lengths, angles, scaled, density: float) -> tuple[int, int, int]:
    """``workflows.auto_kpts``: n_i = round(density / length_i), 1 on vacuum axes,
    and on the in-plane axes of a hexagonal slab rounded up to a multiple of 3
    (>= 3) so that the Gamma-centered grid contains K = (1/3, 1/3, 0)."""
    vacuum = vacuum_axes_reference(lengths, scaled)
    hex2d = (vacuum[2] and abs(lengths[0] - lengths[1]) / lengths[0] < 0.05
             and abs(angles[2] - 120.0) < 2.0)
    kpts = []
    for i, length in enumerate(lengths):
        if vacuum[i]:
            kpts.append(1)
            continue
        n = max(1, int(round(density / length)))
        if hex2d and i < 2:
            n = max(3, ((n + 2) // 3) * 3)
        kpts.append(n)
    return tuple(kpts)


def klabel_reference(k) -> str:
    """``engines.klabel``: first named high-symmetry point within 0.06 of k, in the
    order of ``engines._NAMED_KPTS``; otherwise the rounded coordinates."""
    named = [((0, 0, 0), "Γ"), ((1 / 3, 1 / 3, 0), "K"), ((-1 / 3, -1 / 3, 0), "K"),
             ((1 / 3, -1 / 3, 0), "K'"), ((0.5, 0, 0), "M"), ((0, 0.5, 0), "M"),
             ((0.5, 0.5, 0), "M"), ((0, 0, 0.5), "A"), ((1 / 3, 1 / 3, 0.5), "H")]
    for coords, name in named:
        if all(abs(a - b) <= 0.06 for a, b in zip(k, coords)):
            return name
    return f"[{k[0]:.2f},{k[1]:.2f},{k[2]:.2f}]"


def gap_reference(eigvals, fermi_ev: float, kpts) -> dict:
    """``engines.gap_analysis``: VBM = highest eigenvalue strictly below the Fermi
    level, CBM = lowest strictly above, first occurrence in (k, band) order."""
    best_v = best_c = None
    for ik, row in enumerate(eigvals):
        for nb, value in enumerate(row):
            if value < fermi_ev and (best_v is None or value > best_v[0]):
                best_v = (value, ik, nb)
            if value > fermi_ev and (best_c is None or value < best_c[0]):
                best_c = (value, ik, nb)
    if best_v is None or best_c is None:
        return {"gap": None, "gap_type": "metallic", "vbm": None, "cbm": None}

    def position(entry):
        value, ik, nb = entry
        return {"k": list(kpts[ik]), "label": klabel_reference(kpts[ik]),
                "band": nb, "energy_ev": round(value, 4)}

    return {"gap": best_c[0] - best_v[0],
            "gap_type": "direct" if best_v[1] == best_c[1] else "indirect",
            "vbm": position(best_v), "cbm": position(best_c)}


def sweep_deltas(rows, n_atoms: int) -> list[dict]:
    """``verify.check_convergence``'s per-row deltas, recomputed from the energies
    and gaps the server reported: |Delta E| in meV/atom and |Delta gap| in eV to
    the next point, None for the last one."""
    out = []
    for i, row in enumerate(rows):
        if i == len(rows) - 1:
            out.append({"delta_mev_per_atom": None, "delta_gap_ev": None})
            continue
        nxt = rows[i + 1]
        g0, g1 = row.get("gap_ev"), nxt.get("gap_ev")
        out.append({
            "delta_mev_per_atom": abs(nxt["energy_ev"] - row["energy_ev"]) / n_atoms * 1000,
            "delta_gap_ev": abs(g1 - g0) if g0 is not None and g1 is not None else None,
        })
    return out


def recommend_reference(rows, deltas, key: str, energy_tol: float, tol_gap_ev: float):
    """``verify.check_convergence.recommend``: the first point whose gap has
    settled (and whose energy delta is not pathological) wins; metallic systems
    fall back to the energy criterion; hitting the sweep ceiling is not
    converged."""
    for row, delta in zip(rows, deltas):
        d_e, d_gap = delta["delta_mev_per_atom"], delta["delta_gap_ev"]
        if d_gap is not None:
            ok = d_gap < tol_gap_ev and (d_e is None or d_e < energy_tol)
        else:
            ok = d_e is not None and d_e < energy_tol
        if ok:
            return row[key], True
    return rows[-1][key], False


def convergence_reference(ecut_rows, kpts_rows, n_atoms: int, tol_mev_per_atom: float,
                          tol_gap_ev: float = TOL_GAP_EV,
                          ecut_energy_tol: float = ECUT_ENERGY_TOL) -> dict:
    """The whole convergence policy as a pure function of the sweep rows."""
    ecut_deltas, kpts_deltas = sweep_deltas(ecut_rows, n_atoms), sweep_deltas(kpts_rows, n_atoms)
    rec_ecut, ok_e = recommend_reference(ecut_rows, ecut_deltas, "ecut_ev",
                                         ecut_energy_tol, tol_gap_ev)
    rec_kd, ok_k = recommend_reference(kpts_rows, kpts_deltas, "kpts_density",
                                       tol_mev_per_atom, tol_gap_ev)
    return {"ecut_deltas": ecut_deltas, "kpts_deltas": kpts_deltas,
            "recommended_ecut_ev": int(rec_ecut), "ecut_converged": ok_e,
            "recommended_kpts_density": float(rec_kd), "kpts_converged": ok_k,
            "converged": bool(ok_e and ok_k)}


def verify_reference(fetch, relax, summary, conv, drift, gap_tol_ev: float,
                     fmax_tol: float = 0.05, drift_tol_a: float = 0.5) -> dict:
    """``verify.verify_run``'s state machine over a run's artifacts: the ordered
    (check, status) list, and fail > warn > pass for the verdict."""
    checks: list[tuple[str, str]] = []
    if conv:
        checks.append(("convergence_gate", "pass" if conv.get("converged") else "warn"))
    if relax:
        ok = bool(relax.get("converged")) and relax.get("max_force_ev_per_a", 9) <= fmax_tol
        checks.append(("relax_convergence", "pass" if ok else "fail"))
        if drift is not None:
            checks.append(("structure_drift", "pass" if drift <= drift_tol_a else "warn"))
    else:
        checks.append(("relax_convergence", "fail"))
    if summary and fetch:
        gap, ref = summary.get("band_gap_ev"), fetch.get("band_gap_ref")
        if gap is not None and ref is not None:
            diff = abs(gap - ref)
            checks.append(("band_gap_vs_mp", "pass" if diff <= gap_tol_ev
                           else ("warn" if diff <= 1.5 * gap_tol_ev else "fail")))
        elif ref is None:
            checks.append(("band_gap_vs_mp", "skip"))
        if summary.get("gap_type"):
            checks.append(("gap_character", "info"))
    elif not summary:
        checks.append(("band_gap_vs_mp", "fail"))
    if fetch and fetch.get("is_stable") is not None:
        checks.append(("mp_stability", "pass" if fetch["is_stable"] else "warn"))
    fails = [name for name, status in checks if status == "fail"]
    warns = [name for name, status in checks if status == "warn"]
    return {"verdict": "fail" if fails else ("pass_with_warnings" if warns else "pass"),
            "checks": checks, "blocking_failures": fails}


def relative_diff(got, want) -> float:
    if got is None or want is None:
        return 0.0 if got == want else math.inf
    return abs(got - want) / max(1.0, abs(want))


def deltas_match(rows, reference, tol: float = ARITHMETIC_TOL) -> list[str]:
    """Rows whose own delta fields disagree with the recomputed ones."""
    problems = []
    for i, (row, want) in enumerate(zip(rows, reference)):
        for field in ("delta_mev_per_atom", "delta_gap_ev"):
            got = row.get(field)
            if (got is None) != (want[field] is None) or relative_diff(got, want[field]) > tol:
                problems.append(f"row {i} {field}: {got} != {want[field]}")
    return problems


# --------------------------------------------------------------------------
# Independent references: GPAW and ASE in THIS process
# --------------------------------------------------------------------------

@dataclass
class Run:
    """One server run directory."""
    run_id: str
    path: Path


class GpawRef:
    """GPAW/ASE references, computed in this process with the engine's documented
    settings but through our own code path."""

    def __init__(self, scratch: Path) -> None:
        # Must precede the first BLAS/GPAW import: the server runs pinned to one
        # thread (launch env), and the thread count moves the last digits.
        os.environ["OMP_NUM_THREADS"] = "1"
        with quiet_fds():
            import ase
            import gpaw
        self.ase_version = ase.__version__
        self.gpaw_version = gpaw.__version__
        self.scratch = scratch

    # ---- structures

    def build_mos2(self):
        from ase.build import mx2
        return mx2(**MOS2_BUILDER)

    def read(self, path: Path):
        from ase.io import read as ase_read
        return ase_read(str(path))

    @staticmethod
    def geometry(atoms) -> dict:
        """Cell, scaled positions and symbols as plain Python."""
        return {"lengths": [float(x) for x in atoms.cell.lengths()],
                "angles": [float(x) for x in atoms.cell.angles()],
                "scaled": [[float(c) for c in row] for row in atoms.get_scaled_positions()],
                "symbols": list(atoms.get_chemical_symbols()),
                "atoms": [(sym, tuple(float(c) for c in pos))
                          for sym, pos in zip(atoms.get_chemical_symbols(), atoms.positions)]}

    def kpts_for(self, atoms, density: float) -> tuple[int, int, int]:
        geom = self.geometry(atoms)
        return auto_kpts_reference(geom["lengths"], geom["angles"], geom["scaled"], density)

    def band_path_labels(self, atoms, path_str: str = "GMKG", npoints: int = NPOINTS) -> int:
        return len(atoms.cell.bandpath(path_str, npoints=npoints).kpts)

    # ---- DFT

    def _calc(self, ecut: float, kpts, txt: Path):
        from gpaw import GPAW, PW, FermiDirac
        return GPAW(mode=PW(ecut), xc="PBE", kpts={"size": tuple(kpts), "gamma": True},
                    occupations=FermiDirac(0.05), txt=str(txt))

    def relax(self, source: Path, ecut: float, kpts, tag: str = "ref_relax") -> dict:
        from ase.optimize import BFGS
        progress(f"reference: GPAW relaxation at {ecut} eV on {list(kpts)}")
        started = time.monotonic()
        atoms = self.read(source)
        atoms.calc = self._calc(ecut, kpts, self.scratch / f"{tag}.txt")
        with quiet_fds():
            opt = BFGS(atoms, trajectory=str(self.scratch / f"{tag}.traj"),
                       logfile=str(self.scratch / f"{tag}.log"))
            opt.run(fmax=FMAX, steps=MAX_STEPS)
            forces = atoms.get_forces()
            energy = float(atoms.get_potential_energy())
        TIMINGS[f"reference {tag}"] = round(time.monotonic() - started, 1)
        return {"converged": bool(opt.converged()), "n_steps": int(opt.get_number_of_steps()),
                "energy_ev": energy,
                "max_force_ev_per_a": max(math.dist(f, (0, 0, 0)) for f in forces),
                "geometry": self.geometry(atoms)}

    def scf(self, source: Path, ecut: float, kpts, tag: str) -> dict:
        progress(f"reference: GPAW SCF at {ecut} eV on {list(kpts)}")
        started = time.monotonic()
        atoms = self.read(source)
        atoms.calc = self._calc(ecut, kpts, self.scratch / f"{tag}.txt")
        with quiet_fds():
            energy = float(atoms.get_potential_energy())
            calc = atoms.calc
            fermi = float(calc.get_fermi_level())
            kpts_ibz = [[float(c) for c in k] for k in calc.get_ibz_k_points()]
            eigvals = [[float(e) for e in calc.get_eigenvalues(kpt=k)] for k in range(len(kpts_ibz))]
        TIMINGS[f"reference {tag}"] = round(time.monotonic() - started, 1)
        return {"energy_ev": energy, "fermi_ev": fermi, "kpts_ibz": kpts_ibz, "eigvals": eigvals,
                **gap_reference(eigvals, fermi, kpts_ibz)}

    def from_gpw(self, gpw: Path) -> dict:
        """The server's own restart file, re-analysed here (no new SCF)."""
        from gpaw import GPAW
        with quiet_fds():
            calc = GPAW(str(gpw))
            fermi = float(calc.get_fermi_level())
            kpts_ibz = [[float(c) for c in k] for k in calc.get_ibz_k_points()]
            eigvals = [[float(e) for e in calc.get_eigenvalues(kpt=k)] for k in range(len(kpts_ibz))]
        return {"fermi_ev": fermi, "kpts_ibz": kpts_ibz, "eigvals": eigvals,
                **gap_reference(eigvals, fermi, kpts_ibz)}

    def drift(self, before: Path, after: Path) -> float:
        a0, a1 = self.read(before), self.read(after)
        return max(math.dist(p, q) for p, q in zip(a1.positions, a0.positions))


# --------------------------------------------------------------------------
# Call helpers
# --------------------------------------------------------------------------

def tool_ok(call: Caller, name: str, tool: str, arguments: dict) -> dict | None:
    """A successful call returning ``ok=true``; anything else FAILs a valid request."""
    progress(f"{name}: calling {tool}")
    result = timed(f"call {name}", call, name, tool, arguments)
    if result is None:
        return None
    try:
        data = payload(result)
    except (ValueError, json.JSONDecodeError) as exc:
        call.report.add("L1", name, "FAIL", f"unparseable result ({exc}): {text_of(result)[:200]!r}")
        return None
    if data.get("ok") is not True:
        call.report.add("L1", name, "FAIL", f"ok={data.get('ok')!r} for a valid request: "
                        f"{data.get('error')} {data.get('traceback_tail')}")
        return None
    try:
        text = json.loads(text_of(result))
    except json.JSONDecodeError:
        text = None
    if text != data:
        call.report.add("L1", f"{name}[text == structuredContent]", "FAIL",
                        "the text block and structuredContent differ")
    return data


def new_run(call: Caller, name: str, query: str, runs_dir: Path) -> Run | None:
    data = tool_ok(call, name, "fetch_structure", {"query": query, "use_builtin": True})
    if data is None:
        return None
    run_id = data.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        call.report.add("L1", name, "FAIL", f"run_id {run_id!r} is not <timestamp>_<label>_<uuid4>")
        return None
    return Run(run_id, runs_dir / run_id)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def require(report, path: Path, check: str) -> bool:
    """A later step cannot run without an earlier step's artefact."""
    if path.is_file():
        return True
    report.add("L1", check, "FAIL", f"{path.name} is missing; the previous step failed — skipped")
    return False


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_fetch(session: Session) -> Run | None:
    """The built-in builder: run directory, CIF and reference gap."""
    call, report, ref = session.call, session.report, session.state["ref"]
    name = "fetch_structure[builtin MoS2]"
    data = tool_ok(call, name, "fetch_structure", {"query": "MoS2", "use_builtin": True})
    if data is None:
        return None
    run_id = data.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        report.add("L1", name, "FAIL", f"run_id {run_id!r} is not <timestamp>_<label>_<uuid4hex>")
        return None
    run = Run(run_id, session.state["runs_dir"] / run_id)

    fields = {"source": "ase-mx2-builtin", "formula": MOS2_FORMULA, "n_atoms": 3,
              "mp_id": None, "band_gap_ref": MP_GAP_REF, "query": "MoS2",
              "cif": f"runs/{run_id}/structure.cif"}
    wrong = {k: data.get(k) for k, v in fields.items() if data.get(k) != v}
    report.add("L1", name, "FAIL" if wrong else "PASS",
               f"wrong fields {wrong} (expected {fields})" if wrong
               else f"run_id={run_id}, band_gap_ref={MP_GAP_REF} eV, 3 atoms",
               server_value=data)

    # The CIF must be the ase.build.mx2 cell this builder documents.
    cif = run.path / "structure.cif"
    name = "fetch_structure[structure.cif == ase.build.mx2]"
    if not cif.is_file():
        report.add("L1", name, "FAIL", f"{cif} was not written")
        return run
    want, got = ref.geometry(ref.build_mos2()), ref.geometry(ref.read(cif))
    problems = []
    if got["symbols"] != want["symbols"]:
        problems.append(f"symbols {got['symbols']} != {want['symbols']}")
    for field, tol in (("lengths", CELL_TOL), ("angles", CELL_TOL)):
        worst = max_abs_diff(got[field], want[field])
        if worst > tol:
            problems.append(f"{field} {got[field]} != {want[field]} (max|d|={worst:.1e})")
    bonds = max_abs_diff(sorted_distances(got["atoms"]), sorted_distances(want["atoms"]))
    if bonds > DIST_TOL:
        problems.append(f"interatomic distances differ by {bonds:.1e} A")
    report.add("L1", name, "FAIL" if problems else "PASS",
               "; ".join(problems) if problems else
               f"a={got['lengths'][0]:.4f} c={got['lengths'][2]:.4f} A, "
               f"gamma={got['angles'][2]:.2f} deg, Mo-S={sorted_distances(got['atoms'])[0]:.5f} A, "
               f"max|d(distances)|={bonds:.1e} A")

    on_disk = read_json(run.path / "fetch.json")
    expected = {k: v for k, v in data.items() if k not in ("ok", "cif")}
    report.add("L1", "fetch_structure[fetch.json]",
               "PASS" if on_disk == expected else "FAIL",
               "" if on_disk == expected else f"{on_disk} != {expected}")
    session.state["fetch"] = data
    return run


def check_fetch_variants(session: Session) -> Run | None:
    """``query`` is ignored for built-ins, and every other query needs credentials."""
    call, report = session.call, session.report
    name = "fetch_structure[use_builtin ignores query]"
    data = tool_ok(call, name, "fetch_structure", {"query": "NaCl", "use_builtin": True})
    run = None
    if data is not None:
        run_id = data.get("run_id", "")
        run = Run(run_id, session.state["runs_dir"] / run_id)
        if data.get("formula") == MOS2_FORMULA and data.get("query") == "NaCl":
            report.add("L1", name, "WARN",
                       "use_builtin=true silently ignores query='NaCl' and returns the MoS2 "
                       f"monolayer; the query only names the run directory ({run_id}) — a task "
                       "cannot select a structure this way", server_value=data)
        else:
            report.add("L1", name, "FAIL", f"expected the built-in MoS2, got {data}")

    check_rejected(call, "fetch_structure[MP query without credentials]", "fetch_structure",
                   {"query": "Si"}, in_band=in_band_error,
                   on_accept=lambda result: ("FAIL", "a Materials Project query succeeded without "
                                             f"MP_API_KEY: {text_of(result)[:200]!r}"))
    return run


def check_relax(session: Session, run: Run) -> None:
    """One GPAW relaxation, re-run here from the same CIF with the same settings."""
    call, report, ref = session.call, session.report, session.state["ref"]
    if not require(report, run.path / "structure.cif", "relax_structure[input]"):
        return
    atoms = ref.read(run.path / "structure.cif")
    kpts = ref.kpts_for(atoms, KD_RELAX)
    name = f"relax_structure[MoS2 ecut={ECUT} kd={KD_RELAX:g}]"
    data = tool_ok(call, name, "relax_structure",
                   {"run_id": run.run_id, "ecut": ECUT, "kpts_density": KD_RELAX,
                    "fmax": FMAX, "max_steps": MAX_STEPS, "engine": "gpaw"})
    if data is None:
        return
    reference = ref.relax(run.path / "structure.cif", ECUT, kpts)
    problems = []
    if tuple(data.get("kpts") or ()) != kpts:
        problems.append(f"kpts {data.get('kpts')} != auto_kpts reference {list(kpts)}")
    if data.get("engine") != "gpaw-pw" or data.get("ecut_ev") != ECUT:
        problems.append(f"engine/ecut echoed as {data.get('engine')}/{data.get('ecut_ev')}")
    if data.get("converged") is not True or data.get("n_steps") != reference["n_steps"]:
        problems.append(f"converged={data.get('converged')} n_steps={data.get('n_steps')} "
                        f"(reference {reference['converged']}/{reference['n_steps']})")
    d_e = abs((data.get("total_energy_ev") or math.inf) - reference["energy_ev"])
    d_f = abs((data.get("max_force_ev_per_a") or math.inf) - reference["max_force_ev_per_a"])
    if d_e > ENERGY_TOL:
        problems.append(f"energy {data.get('total_energy_ev')} vs {reference['energy_ev']:.10f} eV")
    if d_f > FORCE_TOL:
        problems.append(f"max force {data.get('max_force_ev_per_a')} vs "
                        f"{reference['max_force_ev_per_a']:.6f} eV/A")
    relaxed = run.path / "relaxed.cif"
    if not relaxed.is_file():
        problems.append("relaxed.cif was not written")
        bonds = math.inf
    else:
        bonds = max_abs_diff(sorted_distances(ref.geometry(ref.read(relaxed))["atoms"]),
                             sorted_distances(reference["geometry"]["atoms"]))
        if bonds > DIST_TOL:
            problems.append(f"relaxed geometry differs: max|d(distances)|={bonds:.1e} A")
    report.add("L1", name, "FAIL" if problems else "PASS",
               "; ".join(problems) if problems else
               f"E={data['total_energy_ev']:.8f} eV (|dE|={d_e:.1e}), fmax="
               f"{data['max_force_ev_per_a']:.6f} eV/A (|df|={d_f:.1e}), {data['n_steps']} steps, "
               f"kpts={list(kpts)}, max|d(distances)|={bonds:.1e} A",
               server_value=data, reference_value={k: v for k, v in reference.items()
                                                   if k != "geometry"})
    on_disk = read_json(run.path / "relax.json")
    expected = {k: v for k, v in data.items() if k != "ok"}
    report.add("L1", "relax_structure[relax.json]", "PASS" if on_disk == expected else "FAIL",
               "" if on_disk == expected else f"{on_disk} != {expected}")
    session.state["relax"] = data


def check_convergence_tool(session: Session, run: Run) -> None:
    """The sweep: realized grids, de-duplication, the delta/recommendation
    arithmetic and two independently recomputed sweep points."""
    call, report, ref = session.call, session.report, session.state["ref"]
    if not require(report, run.path / "relaxed.cif", "check_convergence[input]"):
        return
    atoms = ref.read(run.path / "relaxed.cif")
    n_atoms = len(atoms.get_chemical_symbols())
    name = f"check_convergence[tol={TOL_MEV_PER_ATOM:g} meV/atom]"
    data = tool_ok(call, name, "check_convergence",
                   {"run_id": run.run_id, "tol_mev_per_atom": TOL_MEV_PER_ATOM, "engine": "gpaw"})
    if data is None:
        return
    ecut_rows, kpts_rows = data.get("ecut_sweep") or [], data.get("kpts_sweep") or []
    reference = convergence_reference(ecut_rows, kpts_rows, n_atoms, TOL_MEV_PER_ATOM)

    problems = deltas_match(ecut_rows, reference["ecut_deltas"]) + \
        deltas_match(kpts_rows, reference["kpts_deltas"])
    for field in ("recommended_ecut_ev", "recommended_kpts_density", "ecut_converged",
                  "kpts_converged", "converged"):
        if data.get(field) != reference[field]:
            problems.append(f"{field}: {data.get(field)!r} != {reference[field]!r}")
    if data.get("tol_gap_ev") != TOL_GAP_EV:
        problems.append(f"tol_gap_ev echoed as {data.get('tol_gap_ev')}")
    report.add("L1", name, "FAIL" if problems else "PASS",
               "; ".join(problems) if problems else
               f"{len(ecut_rows)} ecut + {len(kpts_rows)} k-grid points, all deltas and the "
               f"recommendation (ecut={data['recommended_ecut_ev']} eV, "
               f"kd={data['recommended_kpts_density']:g}, converged={data['converged']}) "
               "reproduced from the reported energies and gaps",
               server_value={k: v for k, v in data.items() if not k.endswith("_sweep")},
               ecut_sweep=ecut_rows, kpts_sweep=kpts_rows)

    # Realized grids and the de-duplication of densities that collapse onto one grid.
    name = "check_convergence[realized grids and dedupe]"
    want_ecuts = [300, 400, 500, 600, 800]
    want_grids = {}
    for density in (10, 15, 20, 25, 35, 45):
        want_grids.setdefault(ref.kpts_for(atoms, float(density)), float(density))
    expected_k = [{"kpts_density": d, "kpts": list(g)} for g, d in want_grids.items()]
    got_k = [{"kpts_density": r.get("kpts_density"), "kpts": r.get("kpts")} for r in kpts_rows]
    got_e = [r.get("ecut_ev") for r in ecut_rows]
    ecut_grid_rows = [r for r in ecut_rows if "kpts" in r]
    problems = []
    if got_e != want_ecuts:
        problems.append(f"ecut points {got_e} != {want_ecuts}")
    if got_k != expected_k:
        problems.append(f"k-grid points {got_k} != {expected_k}")
    if ecut_grid_rows:
        problems.append("ecut rows unexpectedly report a grid")
    report.add("L1", name, "FAIL" if problems else "PASS",
               "; ".join(problems) if problems else
               f"densities {[r['kpts_density'] for r in expected_k]} -> grids "
               f"{[r['kpts'] for r in expected_k]}; density 20 collapses onto the 15 grid and is "
               "evaluated once; the ecut sweep runs on the auto_kpts(15) grid and reports no grid")

    # The recommendation depends only on the tolerance, not on the caller's
    # parameters: the sweep range is fixed, so a stricter tolerance can be
    # evaluated on the same rows without a second 60 s sweep.
    name = "check_convergence[recommendation follows the tolerance]"
    strict = convergence_reference(ecut_rows, kpts_rows, n_atoms, 0.3)
    if strict["recommended_kpts_density"] > reference["recommended_kpts_density"]:
        report.add("L1", name, "PASS",
                   f"tol=5 meV/atom -> kd={reference['recommended_kpts_density']:g}, "
                   f"tol=0.3 -> kd={strict['recommended_kpts_density']:g} on the same rows")
    else:
        report.add("L1", name, "FAIL",
                   f"a 0.3 meV/atom tolerance recommends the same grid "
                   f"({strict['recommended_kpts_density']:g}) as 5 meV/atom")
    report.add("L1", "check_convergence[sweep range is fixed]", "WARN",
               "the sweep is hard-coded (ecut 300-800 eV at the auto_kpts(15) grid, densities "
               "10-45 at ecut 400) and ignores the run's own parameters; tol_gap_ev is not on "
               "the MCP surface, so only tol_mev_per_atom can move the k-grid recommendation")

    # Two sweep points recomputed with our own SCF.
    spots = (("ecut sweep 300 eV", ref.kpts_for(atoms, 15.0), 300,
              next((r for r in ecut_rows if r.get("ecut_ev") == 300), None)),
             ("k-grid sweep density 10", ref.kpts_for(atoms, 10.0), ECUT,
              next((r for r in kpts_rows if r.get("kpts_density") == 10.0), None)))
    for label, kpts, ecut, row in spots:
        name = f"check_convergence[{label}]"
        if row is None:
            report.add("L1", name, "FAIL", "the sweep has no such point")
            continue
        got = ref.scf(run.path / "relaxed.cif", ecut, kpts, f"ref_conv_{ecut}_{kpts[0]}")
        d_e = abs(row["energy_ev"] - got["energy_ev"])
        d_g = abs((row.get("gap_ev") or math.inf) - (got["gap"] or math.inf))
        ok = d_e <= ENERGY_TOL and d_g <= GAP_TOL
        report.add("L1", name, "PASS" if ok else "FAIL",
                   f"grid {list(kpts)} at {ecut} eV: E={row['energy_ev']:.8f} vs "
                   f"{got['energy_ev']:.8f} eV (|dE|={d_e:.1e}), gap={row.get('gap_ev')} vs "
                   f"{got['gap']} eV (|dgap|={d_g:.1e})",
                   server_value=row, reference_value={"energy_ev": got["energy_ev"],
                                                      "gap_ev": got["gap"]})

    on_disk = read_json(run.path / "convergence.json")
    expected = {k: v for k, v in data.items() if k != "ok"}
    report.add("L1", "check_convergence[convergence.json]",
               "PASS" if on_disk == expected else "FAIL",
               "" if on_disk == expected else "the file differs from the returned result")
    session.state["convergence"] = data


def check_band_dos(session: Session, run: Run) -> None:
    """SCF + bands + DOS, with our own SCF and our own gap analysis as reference."""
    call, report, ref = session.call, session.report, session.state["ref"]
    if not require(report, run.path / "relaxed.cif", "calc_band_dos[input]"):
        return
    atoms = ref.read(run.path / "relaxed.cif")
    kpts = ref.kpts_for(atoms, KD_BANDS)
    name = f"calc_band_dos[MoS2 ecut={ECUT} kd={KD_BANDS:g}]"
    data = tool_ok(call, name, "calc_band_dos",
                   {"run_id": run.run_id, "ecut": ECUT, "kpts_density": KD_BANDS,
                    "npoints": NPOINTS, "window_ev": 8.0, "engine": "gpaw"})
    if data is None:
        return
    reference = ref.scf(run.path / "relaxed.cif", ECUT, kpts, "ref_bands_scf")
    problems = []
    if tuple(data.get("kpts_scf") or ()) != kpts:
        problems.append(f"kpts_scf {data.get('kpts_scf')} != auto_kpts reference {list(kpts)}")
    d_e = abs((data.get("total_energy_ev") or math.inf) - reference["energy_ev"])
    d_f = abs((data.get("fermi_ev") or math.inf) - reference["fermi_ev"])
    d_g = abs((data.get("band_gap_ev") or math.inf) - (reference["gap"] or math.inf))
    if d_e > ENERGY_TOL:
        problems.append(f"total energy {data.get('total_energy_ev')} vs {reference['energy_ev']:.10f}")
    if d_f > GAP_TOL:
        problems.append(f"fermi level {data.get('fermi_ev')} vs {reference['fermi_ev']:.10f}")
    if d_g > GAP_TOL:
        problems.append(f"band gap {data.get('band_gap_ev')} vs {reference['gap']}")
    if data.get("gap_type") != reference["gap_type"]:
        problems.append(f"gap_type {data.get('gap_type')!r} != {reference['gap_type']!r}")
    for edge in ("vbm", "cbm"):
        got, want = data.get(edge) or {}, reference[edge] or {}
        if (got.get("label"), got.get("band")) != (want.get("label"), want.get("band")) or \
                abs((got.get("energy_ev") or math.inf) - want.get("energy_ev", math.inf)) > GAP_TOL:
            problems.append(f"{edge} {got} != {want}")
    if data.get("band_path") != "GMKG":
        problems.append(f"band_path {data.get('band_path')!r} != 'GMKG' for a 2D slab")
    report.add("L1", name, "FAIL" if problems else "PASS",
               "; ".join(problems) if problems else
               f"gap={data['band_gap_ev']:.6f} eV {data['gap_type']} "
               f"{data['vbm']['label']}->{data['cbm']['label']} (|dgap|={d_g:.1e}), "
               f"E={data['total_energy_ev']:.8f} eV (|dE|={d_e:.1e}), "
               f"E_F={data['fermi_ev']:.6f} eV (|dE_F|={d_f:.1e}), grid {list(kpts)}",
               server_value=data,
               reference_value={k: reference[k] for k in ("energy_ev", "fermi_ev", "gap",
                                                          "gap_type", "vbm", "cbm")})

    # params_verified is the run's own convergence gate, and it is honest.
    conv = session.state.get("convergence") or {}
    name = "calc_band_dos[params_verified]"
    gate = (conv.get("converged") and ECUT >= conv.get("recommended_ecut_ev", 10 ** 9)
            and KD_BANDS >= conv.get("recommended_kpts_density", 10 ** 9))
    if data.get("params_verified") is bool(gate) and "verification_note" not in data:
        report.add("L1", name, "PASS",
                   f"params_verified={data['params_verified']} after a passed gate "
                   f"(ecut {ECUT} >= {conv.get('recommended_ecut_ev')}, kd {KD_BANDS:g} >= "
                   f"{conv.get('recommended_kpts_density')}), no verification_note")
    else:
        report.add("L1", name, "FAIL",
                   f"params_verified={data.get('params_verified')} (gate says {bool(gate)}), "
                   f"note={data.get('verification_note')!r}")

    # The restart file must be the one this call wrote.
    name = "calc_band_dos[gs.gpw is this call's SCF]"
    gpw = run.path / "gs.gpw"
    if not gpw.is_file():
        report.add("L1", name, "FAIL", "gs.gpw was not written")
    else:
        stored = ref.from_gpw(gpw)
        worst = max((max_abs_diff(a, b) for a, b in zip(stored["eigvals"], reference["eigvals"])),
                    default=math.inf)
        same = len(stored["kpts_ibz"]) == len(reference["kpts_ibz"]) and worst <= EIG_TOL
        report.add("L1", name, "PASS" if same else "FAIL",
                   f"{len(stored['kpts_ibz'])} irreducible k-points, max|d(eigenvalue)|="
                   f"{worst:.1e} eV, gap from the restart file {stored['gap']}"
                   if same else f"restart file has {len(stored['kpts_ibz'])} k-points "
                   f"(reference {len(reference['kpts_ibz'])}), max|d(eigenvalue)|={worst:.1e} eV")

    # Same realized grid as the k-grid sweep point at density 25 -> same number.
    name = "calc_band_dos[consistent with the sweep]"
    row = next((r for r in (conv.get("kpts_sweep") or [])
                if tuple(r.get("kpts") or ()) == kpts), None)
    if row is None:
        report.add("L1", name, "FAIL", f"the sweep has no point on the realized grid {list(kpts)}")
    else:
        d_e = abs(row["energy_ev"] - data["total_energy_ev"])
        d_g = abs((row.get("gap_ev") or math.inf) - data["band_gap_ev"])
        ok = d_e <= SAME_GRID_TOL and d_g <= SAME_GRID_TOL
        report.add("L1", name, "PASS" if ok else "FAIL",
                   f"density {row['kpts_density']:g} and {KD_BANDS:g} realize the same grid "
                   f"{list(kpts)}: |dE|={d_e:.1e} eV, |dgap|={d_g:.1e} eV")

    # Figures: real PNGs of a plausible size (the content is not scored).
    for key, png in (("bands_png", run.path / "bands.png"), ("dos_png", run.path / "dos.png")):
        name = f"calc_band_dos[{key}]"
        if data.get(key) != f"runs/{run.run_id}/{png.name}" or not png.is_file():
            report.add("L1", name, "FAIL", f"{data.get(key)!r} / exists={png.is_file()}")
            continue
        blob = png.read_bytes()
        try:
            width, height = png_size(blob)
        except ValueError as exc:
            report.add("L1", name, "FAIL", f"{png.name} is not a PNG ({exc})")
            continue
        ok = width >= 400 and height >= 300 and len(blob) > 10_000
        report.add("L1", name, "PASS" if ok else "FAIL",
                   f"{png.name}: {width}x{height} px, {len(blob)} bytes")

    name = "calc_band_dos[band path]"
    points = ref.band_path_labels(atoms, "GMKG", NPOINTS)
    report.add("L1", name, "PASS" if points == NPOINTS else "WARN",
               f"ase bandpath('GMKG', npoints={NPOINTS}) has {points} k-points"
               if points == NPOINTS else
               f"ase gives {points} k-points for npoints={NPOINTS} (ASE rounds the request)")

    on_disk = read_json(run.path / "summary.json")
    expected = {k: v for k, v in data.items() if k not in ("ok", "verification_note")}
    report.add("L1", "calc_band_dos[summary.json]", "PASS" if on_disk == expected else "FAIL",
               "" if on_disk == expected else "the file differs from the returned result")
    session.state["summary"] = data


def check_unverified_params(session: Session) -> None:
    """A run without a convergence gate, and the single shared restart file."""
    call, report, ref = session.call, session.report, session.state["ref"]
    run = new_run(call, "calc_band_dos[unverified run: fetch]", "MoS2", session.state["runs_dir"])
    if run is None:
        return
    name = f"calc_band_dos[no gate, kd={KD_UNVERIFIED:g}]"
    data = tool_ok(call, name, "calc_band_dos",
                   {"run_id": run.run_id, "ecut": ECUT, "kpts_density": KD_UNVERIFIED,
                    "engine": "gpaw"})
    if data is None:
        return
    atoms = ref.read(run.path / "structure.cif")
    small = ref.kpts_for(atoms, KD_UNVERIFIED)
    problems = []
    if data.get("params_verified") is not False:
        problems.append(f"params_verified={data.get('params_verified')} without a convergence gate")
    if "no convergence gate passed" not in (data.get("verification_note") or ""):
        problems.append(f"verification_note={data.get('verification_note')!r}")
    if tuple(data.get("kpts_scf") or ()) != small:
        problems.append(f"kpts_scf {data.get('kpts_scf')} != {list(small)}")
    report.add("L1", name, "FAIL" if problems else "PASS",
               "; ".join(problems) if problems else
               f"params_verified=false with the documented note, grid {list(small)} "
               "(the structure was never relaxed in this run)", server_value=data)

    # The note exists only in the reply, never on disk.
    on_disk = read_json(run.path / "summary.json") or {}
    name = "calc_band_dos[verification_note is not persisted]"
    if on_disk.get("params_verified") is False and "verification_note" not in on_disk:
        report.add("L1", name, "WARN",
                   "summary.json records params_verified=false but not the verification_note: "
                   "the warning is added to the reply after the file is written, so an agent "
                   "that copies summary.json forwards unverified numbers without the warning")
    else:
        report.add("L1", name, "FAIL", f"unexpected summary.json: {on_disk}")

    # gs.gpw is one shared mutable file per run: last writer wins.
    name = "calc_band_dos[gs.gpw is shared and overwritten]"
    if not (run.path / "gs.gpw").is_file():
        report.add("L1", name, "FAIL", "the call left no gs.gpw")
        return
    before = len(ref.from_gpw(run.path / "gs.gpw")["kpts_ibz"])
    again = tool_ok(call, f"{name}: second call", "calc_band_dos",
                    {"run_id": run.run_id, "ecut": ECUT, "kpts_density": KD_RELAX,
                     "engine": "gpaw"})
    if again is None:
        return
    after = len(ref.from_gpw(run.path / "gs.gpw")["kpts_ibz"])
    if before != after and tuple(again.get("kpts_scf") or ()) != small:
        report.add("L1", name, "WARN",
                   f"the run's single gs.gpw went from {before} to {after} irreducible k-points "
                   f"when the grid changed from {list(small)} to {again.get('kpts_scf')}: any tool "
                   "that runs an SCF in the run directory (check_convergence included) replaces "
                   "the restart file that bands are computed from, so a task must fix the call "
                   "order and must not reuse an older summary.json with a newer gs.gpw",
                   ibz_before=before, ibz_after=after)
    else:
        report.add("L1", name, "FAIL",
                   f"expected the restart file to change with the grid: {before} -> {after} "
                   f"k-points for grids {list(small)} -> {again.get('kpts_scf')}")


def check_verify(session: Session, run: Run, empty: Run | None) -> None:
    """The physics verifier: all five checks, and the gap tolerance state machine."""
    call, report, ref = session.call, session.report, session.state["ref"]
    fetch = read_json(run.path / "fetch.json")
    relax = read_json(run.path / "relax.json")
    summary = read_json(run.path / "summary.json")
    conv = read_json(run.path / "convergence.json")
    if not (fetch and relax and summary and conv):
        report.add("L1", "verify_run[artefacts]", "FAIL",
                   "the chain left no complete set of artefacts to verify; skipped")
        return
    drift = ref.drift(run.path / "structure.cif", run.path / "relaxed.cif")

    for gap_tol, label in ((GAP_TOL_PASS, "pass"), (GAP_TOL_WARN, "warn"), (GAP_TOL_FAIL, "fail")):
        name = f"verify_run[gap_tol_ev={gap_tol}]"
        data = tool_ok(call, name, "verify_run", {"run_id": run.run_id, "gap_tol_ev": gap_tol})
        if data is None:
            continue
        reference = verify_reference(fetch, relax, summary, conv, drift, gap_tol)
        got = [(c.get("check"), c.get("status")) for c in data.get("checks") or []]
        problems = []
        if got != reference["checks"]:
            problems.append(f"checks {got} != {reference['checks']}")
        if data.get("verdict") != reference["verdict"]:
            problems.append(f"verdict {data.get('verdict')!r} != {reference['verdict']!r}")
        if (data.get("blocking_failures") or []) != reference["blocking_failures"]:
            problems.append(f"blocking_failures {data.get('blocking_failures')} != "
                            f"{reference['blocking_failures']}")
        report.add("L1", name, "FAIL" if problems else "PASS",
                   "; ".join(problems) if problems else
                   f"verdict={data['verdict']} ({label} branch), checks {got}, drift "
                   f"{drift:.3f} A, |gap - MP ref|="
                   f"{abs(summary['band_gap_ev'] - fetch['band_gap_ref']):.4f} eV",
                   server_value=data, reference_value=reference)
        if label == "pass":
            on_disk = read_json(run.path / "verify.json")
            expected = {k: v for k, v in data.items() if k != "ok"}
            report.add("L1", "verify_run[verify.json]", "PASS" if on_disk == expected else "FAIL",
                       "" if on_disk == expected else "the file differs from the returned result")

    if empty is None:
        return
    name = "verify_run[run with no results]"
    data = tool_ok(call, name, "verify_run", {"run_id": empty.run_id})
    if data is None:
        return
    reference = verify_reference(read_json(empty.path / "fetch.json"), None, None, None, None, 0.3)
    got = [(c.get("check"), c.get("status")) for c in data.get("checks") or []]
    ok = got == reference["checks"] and data.get("verdict") == reference["verdict"] == "fail"
    report.add("L1", name, "PASS" if ok else "FAIL",
               f"verdict={data.get('verdict')}, checks {got}"
               + ("" if ok else f" != reference {reference['checks']}"),
               server_value=data, reference_value=reference)


def check_artifacts(session: Session, run: Run) -> None:
    """The artifact listing against the directory on disk, and run_id handling."""
    call, report = session.call, session.report
    name = "get_run_artifacts[listing matches the run directory]"
    data = tool_ok(call, name, "get_run_artifacts", {"run_id": run.run_id})
    if data is not None:
        want = sorted(({"path": f"runs/{run.run_id}/{p.name}", "bytes": p.stat().st_size}
                       for p in run.path.iterdir() if p.is_file()), key=lambda a: a["path"])
        got = data.get("artifacts") or []
        ok = got == want
        report.add("L1", name, "PASS" if ok else "FAIL",
                   f"{len(got)} artefacts, paths relative to the checkout, sizes match"
                   if ok else f"{got} != {want}",
                   server_value=got, reference_value=want)
        expected_names = {"fetch.json", "structure.cif", "relax.json", "relaxed.cif",
                          "convergence.json", "summary.json", "verify.json", "gs.gpw",
                          "bands.png", "dos.png"}
        missing = sorted(expected_names - {Path(a["path"]).name for a in got})
        report.add("L1", "get_run_artifacts[chain artefacts]",
                   "PASS" if not missing else "FAIL",
                   "the full chain left fetch/relax/convergence/summary/verify JSON, both CIFs, "
                   "gs.gpw and both figures" if not missing else f"missing {missing}")

    check_rejected(call, "get_run_artifacts[path traversal]", "get_run_artifacts",
                   {"run_id": "../etc"}, in_band=in_band_error,
                   on_accept=lambda result: ("FAIL", f"traversal accepted: {text_of(result)[:200]!r}"))
    check_rejected(call, "get_run_artifacts[unknown run]", "get_run_artifacts",
                   {"run_id": "20260101-000000_nope_0000"}, in_band=in_band_error,
                   on_accept=lambda result: ("FAIL", f"unknown run accepted: {text_of(result)[:200]!r}"))


def check_engines(session: Session, run: Run | None) -> None:
    """Engine selection fails loudly instead of falling back to GPAW."""
    call = session.call
    if run is None:
        return
    check_rejected(call, "relax_structure[engine=qe without pseudopotentials]", "relax_structure",
                   {"run_id": run.run_id, "engine": "qe", "ecut": ECUT}, in_band=in_band_error,
                   on_accept=lambda result: ("FAIL", "engine='qe' silently fell back to GPAW: "
                                             f"{text_of(result)[:200]!r}"))
    check_rejected(call, "relax_structure[unknown engine]", "relax_structure",
                   {"run_id": run.run_id, "engine": "vasp"}, in_band=in_band_error,
                   on_accept=lambda result: ("FAIL", "an unknown engine was accepted: "
                                             f"{text_of(result)[:200]!r}"))
    check_rejected(call, "relax_structure[unknown run]", "relax_structure",
                   {"run_id": "20260101-000000_nope_0000"}, in_band=in_band_error,
                   on_accept=lambda result: ("FAIL", "an unknown run was relaxed: "
                                             f"{text_of(result)[:200]!r}"))


def check_workflow_pass(session: Session) -> None:
    """The one-call pipeline on parameters that pass the gate."""
    call, report, ref = session.call, session.report, session.state["ref"]
    runs_dir = session.state["runs_dir"]
    summary = session.state.get("summary") or {}

    name = f"run_verified_workflow[converged ecut={ECUT} kd={KD_RELAX:g}]"
    data = tool_ok(call, name, "run_verified_workflow",
                   {"query": "MoS2", "use_builtin": True, "ecut": ECUT,
                    "kpts_density": KD_RELAX, "max_retries": 2, "engine": "gpaw"})
    if data is None:
        return
    run = Run(data.get("run_id", ""), runs_dir / data.get("run_id", ""))
    attempts = data.get("attempts") or []
    conv = session.state.get("convergence") or {}
    problems = []
    if data.get("status") != "pass" or data.get("verdict") != "pass":
        problems.append(f"status={data.get('status')!r} verdict={data.get('verdict')!r}")
    if len(attempts) != 1 or attempts[0].get("params_converged") is not True or \
            attempts[0].get("relax_converged") is not True:
        problems.append(f"attempts={attempts}")
    if attempts and attempts[0].get("recommended") != {
            "ecut_ev": conv.get("recommended_ecut_ev"),
            "kpts_density": conv.get("recommended_kpts_density")}:
        problems.append(f"recommended {attempts[0].get('recommended')} != the manual chain's "
                        f"{conv.get('recommended_ecut_ev')}/{conv.get('recommended_kpts_density')}")
    d_g = abs((data.get("band_gap_ev") or math.inf) - (summary.get("band_gap_ev") or math.inf))
    if d_g > SAME_GRID_TOL or data.get("gap_type") != summary.get("gap_type"):
        problems.append(f"gap {data.get('band_gap_ev')} {data.get('gap_type')} vs the manual "
                        f"chain's {summary.get('band_gap_ev')} {summary.get('gap_type')} "
                        f"(|dgap|={d_g:.1e})")
    if data.get("blocking_failures"):
        problems.append(f"blocking_failures={data.get('blocking_failures')}")
    own = read_json(run.path / "summary.json") or {}
    if own.get("params_verified") is not True:
        problems.append(f"the workflow's own summary.json has "
                        f"params_verified={own.get('params_verified')}")
    report.add("L1", name, "FAIL" if problems else "PASS",
               "; ".join(problems) if problems else
               f"status=pass in {len(attempts)} attempt, gap={data['band_gap_ev']:.6f} eV "
               f"{data['gap_type']} — identical to the manual chain (|dgap|={d_g:.1e}), "
               f"params_verified=true, verdict={data['verdict']}",
               server_value=data)

    # One MCP call, the whole chain's artefacts: it bypasses every per-step tool.
    produced = sorted(p.name for p in run.path.iterdir() if p.is_file())
    chain = {"fetch.json", "relax.json", "convergence.json", "summary.json", "verify.json",
             "attempts.json", "workflow_result.json", "bands.png", "dos.png"}
    report.add("L1", "run_verified_workflow[one call, whole chain]",
               "WARN" if chain <= set(produced) else "FAIL",
               f"a single tools/call produced {len(produced)} artefacts including "
               f"{sorted(chain)}: it calls fetch/relax/convergence/band/verify as Python "
               "functions, so an agent's trajectory shows one call and no evidence of the "
               "individual steps — a task that wants the chain must forbid this tool"
               if chain <= set(produced) else f"only {produced}")

    # The reported gap must be reproducible from the run's own restart file.
    name = "run_verified_workflow[gap from its own gs.gpw]"
    gpw = run.path / "gs.gpw"
    if not gpw.is_file():
        report.add("L1", name, "FAIL", "the workflow left no gs.gpw")
        return
    stored = ref.from_gpw(gpw)
    d_g = abs((stored["gap"] or math.inf) - (data.get("band_gap_ev") or math.inf))
    ok = d_g <= GAP_TOL and stored["gap_type"] == data.get("gap_type")
    report.add("L1", name, "PASS" if ok else "FAIL",
               f"{len(stored['kpts_ibz'])} irreducible k-points in the restart file give "
               f"{stored['gap']} eV {stored['gap_type']} (|dgap|={d_g:.1e}) — the band "
               "SCF is the last writer, after the gate"
               if ok else f"the restart file gives {stored['gap']} eV "
               f"{stored['gap_type']} but the result says {data.get('band_gap_ev')} "
               f"{data.get('gap_type')}")


def check_workflow_gate(session: Session) -> None:
    """The one-call pipeline on parameters that cannot pass the gate."""
    call, report = session.call, session.report
    name = f"run_verified_workflow[gate rejects ecut={JUNK['ecut']} kd={JUNK['kpts_density']:g}]"
    data = tool_ok(call, name, "run_verified_workflow",
                   {"query": "MoS2", "use_builtin": True, "max_retries": 0, "engine": "gpaw",
                    **JUNK})
    if data is None:
        return
    run = Run(data.get("run_id", ""), session.state["runs_dir"] / data.get("run_id", ""))
    forbidden = [f for f in ("summary.json", "bands.png", "dos.png", "verify.json")
                 if (run.path / f).exists()]
    problems = []
    if data.get("status") != "rejected_unconverged":
        problems.append(f"status={data.get('status')!r}")
    if forbidden:
        problems.append(f"the rejected run still produced {forbidden}")
    for key in ("band_gap_ev", "gap_type", "bands_png", "dos_png"):
        if key in data:
            problems.append(f"the rejected result still reports {key}={data[key]!r}")
    if not data.get("root_cause") or "收敛门未开启" not in (data.get("message") or ""):
        problems.append(f"root_cause={data.get('root_cause')!r} message={data.get('message')!r}")
    if data.get("final_params") != {"ecut_ev": JUNK["ecut"], "kpts_density": JUNK["kpts_density"]}:
        problems.append(f"final_params={data.get('final_params')}")
    report.add("L1", name, "FAIL" if problems else "PASS",
               "; ".join(problems) if problems else
               f"status=rejected_unconverged after {data.get('n_attempts')} attempt: no "
               "summary.json, no figures, no gap in the result — the convergence gate is real "
               f"(root cause {data['root_cause']!r})", server_value=data)


def check_stdout(session: Session) -> None:
    """Which tool corrupts the stdio transport (the shared check only counts lines)."""
    call, report = session.call, session.report
    noisy = {tool: n for tool, n in call.stdout_by_tool.items() if n}
    band = noisy.get("calc_band_dos", 0)
    others = {tool: n for tool, n in noisy.items()
              if tool not in ("calc_band_dos", "run_verified_workflow")}
    if band and not others:
        report.add("L1", "calc_band_dos[stdout pollution]", "WARN",
                   f"calc_band_dos wrote {band} non-JSON lines to stdout (and "
                   f"run_verified_workflow {noisy.get('run_verified_workflow', 0)}, which calls "
                   "it): engines.GpawEngine.band_eigs opens the restart file with GPAW(gpw) and "
                   "no txt=, so GPAW's default txt='-' prints the SCF table into the stdio "
                   "transport; every other tool passes txt=. Claude Code and Codex tolerate it, "
                   "stricter clients may not", non_json_by_tool=noisy)
    elif not noisy:
        report.add("L1", "calc_band_dos[stdout pollution]", "PASS",
                   "no tool wrote non-JSON lines to stdout (upstream defect fixed?)")
    else:
        report.add("L1", "calc_band_dos[stdout pollution]", "WARN",
                   f"non-JSON stdout from more tools than the known defect: {noisy}",
                   non_json_by_tool=noisy)


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------

def run_l1(session: Session) -> None:
    run = check_fetch(session)
    empty = check_fetch_variants(session)
    if run is not None:
        check_relax(session, run)
        # Order matters: the gate must pass before the band SCF (params_verified),
        # and the band SCF must be the last writer of the run's gs.gpw.
        check_convergence_tool(session, run)
        check_band_dos(session, run)
        check_verify(session, run, empty)
        check_artifacts(session, run)
    check_engines(session, empty)
    check_unverified_params(session)
    check_workflow_pass(session)
    check_workflow_gate(session)
    check_stdout(session)


def prepare(session: Session) -> None:
    env = session.server.get("env", {})
    repo = env.get("MATMCP_REPO")
    runs_dir = Path(env["MATMCP_RUNS"]) if env.get("MATMCP_RUNS") else Path(repo or session.checkout) / "runs"
    session.state["runs_dir"] = runs_dir
    scratch = session.tmp / "reference"
    scratch.mkdir()
    session.state["ref"] = GpawRef(scratch)
    session.report.add("L0", "reference environment", "PASS",
                       f"gpaw {session.state['ref'].gpaw_version}, ase "
                       f"{session.state['ref'].ase_version}, OMP_NUM_THREADS="
                       f"{os.environ.get('OMP_NUM_THREADS')}; runs under {runs_dir}")


def report_fields(session: Session) -> dict:
    summary = session.state.get("summary") or {}
    conv = session.state.get("convergence") or {}
    return {"mos2_monolayer": {
        "band_gap_ev": summary.get("band_gap_ev"),
        "gap_type": summary.get("gap_type"),
        "total_energy_ev": summary.get("total_energy_ev"),
        "fermi_ev": summary.get("fermi_ev"),
        "recommended_ecut_ev": conv.get("recommended_ecut_ev"),
        "recommended_kpts_density": conv.get("recommended_kpts_density"),
        "note": "PBE/PAW plane-wave values of this gpaw build; they drift by a few meV "
                f"between gpaw releases (>{GAP_DRIFT_NOTE} eV against gpaw 25.7), so a task's "
                "ground truth must pin the conda lock",
    }, "seconds_by_step": dict(TIMINGS),
        "seconds_total_measured": round(sum(TIMINGS.values()), 1)}


SMOKE = Smoke(
    server="gpaw",
    run_l1=run_l1,
    packages=("gpaw", "ase", "fastmcp", "mcp", "numpy", "scipy", "matplotlib", "mp-api"),
    call_timeout=CALL_TIMEOUT,
    prepare=prepare,
    report_fields=report_fields,
)
