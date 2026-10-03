"""Instance generator for the gpaw MoS2 band-structure MCP E2E fake task.

A seeded RNG picks a plane-wave cutoff, a k-point density, a convergence
tolerance and a band-gap tolerance. Through the matmcp MCP server (GPAW
engine) the agent must drive the whole six-tool chain on one run:

    fetch_structure -> relax_structure -> check_convergence -> calc_band_dos
                    -> verify_run -> get_run_artifacts

and report the numbers plus the artefacts the server wrote.

Where the reference comes from
------------------------------
Plane-wave DFT cannot be recomputed here: GPAW exists only inside the server's
pinned conda environment, and a task runtime may not build one. The DFT
numbers are therefore **measured** against that pinned environment by
``scripts/mcp/e2e/measure_gpaw_table.py`` and carried in ``MEASURED`` below,
keyed by ``(ecut, kpts_density)`` — the only two instance parameters the SCF
depends on. Everything else is *derived* here, in the standard library, by the
same pure policy functions the L1 smoke validates against the live server:

* ``auto_kpts`` — the realized Gamma-centred grid, from the ASE structure the
  server's built-in builder produces (hexagonal in-plane axes rounded up to a
  multiple of three so the grid contains K);
* ``check_convergence``'s deltas and recommendation — recomputed from the
  measured sweep rows, so ``tol_mev_per_atom`` is a free instance parameter
  even though the sweep itself is hard-coded upstream;
* ``calc_band_dos``'s ``params_verified`` gate;
* ``verify_run``'s state machine, so ``gap_tol_ev`` is free as well.

Each derivation is checked against the server's own answer in the measurement
record (the ``measured_*`` fields), and the offline tests assert that.

Two notes on ``e2e_check.json``
-------------------------------
* ``run_verified_workflow`` is a **bypass** tool, not merely suspicious: it runs
  the entire chain in one call, which is the one thing this task exists to
  prevent, and every prompt level forbids it by name.
* The bypass and suspicious patterns deliberately do **not** include a bare
  ``gpaw``. The tools report artefact paths relative to the server's own
  working directory, so locating and copying the two figures necessarily puts
  the server's directory — which contains ``gpaw`` — into a shell command. A
  blanket pattern would therefore warn on every correct run and drown the real
  signal. What is matched instead is importing, installing or executing GPAW,
  ASE or another solver.

Framework call: python generate_gt.py --output-dir <dir> --params '<json>'
Needs ASE for the structure: generate with ``asibench generate --sandbox task``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

INPUT_SPEC = [
    {"name": "calculation.json", "description": "the plane-wave cutoff, k-point density, "
                                                "convergence tolerance and band-gap tolerance "
                                                "the chain must use, and the fixed arguments"},
]
OUTPUT_SPEC = [
    {"name": "result.json", "description": "relaxation, ground state, convergence gate and "
                                           "verification numbers as the tools returned them"},
    {"name": "bands.png", "description": "the band-structure figure the server wrote"},
    {"name": "dos.png", "description": "the density-of-states figure the server wrote"},
]
DEFAULT_PARAMS = {"seed": 0}

# ---------------------------------------------------------------------------
# The instance grid
# ---------------------------------------------------------------------------
# Only (ecut, kpts_density) move the SCF, so only those are measured; the other
# two are derived from the measured rows and cost nothing to vary.
ECUTS = (350, 400, 450, 500)
KPTS_DENSITIES = (15.0, 25.0)
# 5.0 meV/atom is the server default and recommends the coarse grid; 0.3 is
# strict enough to recommend a finer one, which flips params_verified for some
# cutoff/density pairs.
TOL_MEV_PER_ATOM = (5.0, 0.3)
# |gap - the server's hard-coded MP reference| is about 0.015 eV, so these two
# select the outer branches of verify_run's state machine: 0.005 eV fails the
# gap check (verdict "fail"), 0.3 eV passes it. The middle "warn" branch is
# deliberately unused: it spans (tol, 1.5*tol], i.e. only 0.5*tol ~ 6 meV at the
# tolerance that would centre it on this gap, which leaves no room for the few
# meV GPAW releases move the gap by (see GAP_BRANCH_MARGIN_EV). A
# pass_with_warnings verdict is still reachable, through a convergence gate that
# did not converge at the strict tolerance.
GAP_TOL_EV = (0.005, 0.3)

# Fixed arguments: the prompt pins them at every level, so they are not a way to
# trade accuracy for time and the reference is unambiguous.
QUERY = "MoS2"
FMAX = 0.05
MAX_STEPS = 100
NPOINTS = 60
WINDOW_EV = 8.0
ENGINE = "gpaw"

# The server's built-in structure: workflows._build_mos2_monolayer().
MOS2 = {"formula": "MoS2", "kind": "2H", "a": 3.18, "thickness": 3.17, "vacuum": 10.0}
N_ATOMS = 3
# fetch_structure returns this hard-coded Materials Project reference (mp-1023924)
# for the built-in monolayer; verify_run compares the computed gap against it.
BAND_GAP_REF = 1.66

# check_convergence's own defaults, which are not on the MCP surface.
TOL_GAP_EV = 0.02
ECUT_ENERGY_TOL = 100.0
# verify_run's thresholds (verify.verify_run).
RELAX_FMAX_TOL = 0.05
DRIFT_TOL_A = 0.5
# verify_run's gap check is a three-way branch on |gap - BAND_GAP_REF| against
# gap_tol_ev and 1.5*gap_tol_ev. The measured gap is reproducible to ~1e-10 eV
# on one gpaw build but drifts 2-3 meV between releases, so an instance whose
# gap sits on a branch boundary would have a ground truth that a version bump
# silently flips. Generation fails instead; the fix is to move GAP_TOL_EV.
GAP_BRANCH_MARGIN_EV = 0.003

MCP_TOOLS = ("fetch_structure", "relax_structure", "check_convergence", "calc_band_dos",
             "verify_run", "get_run_artifacts")
# Everything the full chain leaves in the run directory. The two figures exist
# only there — the tools return their paths, never their bytes — so they are
# what the agent has to locate with get_run_artifacts and copy out.
RUN_ARTIFACTS = ("bands.png", "convergence.json", "dos.png", "fetch.json", "gs.gpw",
                 "relax.json", "relaxed.cif", "structure.cif", "summary.json", "verify.json")
COPIED_PNG = ("bands.png", "dos.png")


# >>> BEGIN MEASURED TABLE >>>
# Generated by scripts/mcp/e2e/measure_gpaw_table.py against matmcp
# 378f50512a987dedd5533f98979775e0f4efe474 and the committed conda lock.
# Replace wholesale when either is bumped; never hand-edit a value.
MEASURED: dict[tuple[int, float], dict] = {}
# <<< END MEASURED TABLE <<<


class GenerationError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# The structure and the k-grid policy (workflows.auto_kpts)
# ---------------------------------------------------------------------------

def mos2_structure() -> dict:
    """Cell lengths, angles and scaled positions of the server's built-in monolayer."""
    from ase.build import mx2                            # noqa: PLC0415 (task runtime only)

    atoms = mx2(formula=MOS2["formula"], kind=MOS2["kind"], a=MOS2["a"],
                thickness=MOS2["thickness"], vacuum=MOS2["vacuum"])
    cell = atoms.cell
    return {
        "symbols": list(atoms.get_chemical_symbols()),
        "lengths": [float(v) for v in cell.lengths()],
        "angles": [float(v) for v in cell.angles()],
        "scaled": [[float(c) for c in row] for row in atoms.get_scaled_positions()],
    }


def vacuum_axes(lengths, scaled, min_gap_a: float = 8.0) -> tuple[bool, bool, bool]:
    """``workflows._vacuum_axes``: an axis is vacuum when the largest wrapped gap
    between atoms along it exceeds ``min_gap_a`` — the atom span, not the cell
    length, so layered bulk is not mistaken for a slab."""
    out = []
    for axis in range(3):
        frac = sorted(position[axis] % 1.0 for position in scaled)
        gaps = [b - a for a, b in zip(frac, frac[1:])] + [frac[0] + 1.0 - frac[-1]]
        out.append(bool(max(gaps) * lengths[axis] > min_gap_a))
    return tuple(out)


def auto_kpts(lengths, angles, scaled, density: float) -> tuple[int, int, int]:
    """``workflows.auto_kpts``: n_i = round(density / length_i), 1 on vacuum axes,
    and on the in-plane axes of a hexagonal slab rounded up to a multiple of 3
    (at least 3) so the Gamma-centred grid contains K = (1/3, 1/3, 0)."""
    vacuum = vacuum_axes(lengths, scaled)
    hex2d = (vacuum[2] and abs(lengths[0] - lengths[1]) / lengths[0] < 0.05
             and abs(angles[2] - 120.0) < 2.0)
    kpts = []
    for axis, length in enumerate(lengths):
        if vacuum[axis]:
            kpts.append(1)
            continue
        n = max(1, int(round(density / length)))
        if hex2d and axis < 2:
            n = max(3, ((n + 2) // 3) * 3)
        kpts.append(n)
    return tuple(kpts)


# ---------------------------------------------------------------------------
# The convergence policy (verify.check_convergence)
# ---------------------------------------------------------------------------

def sweep_deltas(rows, n_atoms: int = N_ATOMS) -> list[dict]:
    """Per-row distance to the next sweep point: |dE| in meV/atom and |dgap| in eV."""
    out = []
    for index, row in enumerate(rows):
        if index == len(rows) - 1:
            out.append({"delta_mev_per_atom": None, "delta_gap_ev": None})
            continue
        nxt = rows[index + 1]
        here, there = row.get("gap_ev"), nxt.get("gap_ev")
        out.append({
            "delta_mev_per_atom": abs(nxt["energy_ev"] - row["energy_ev"]) / n_atoms * 1000,
            "delta_gap_ev": abs(there - here) if here is not None and there is not None else None,
        })
    return out


def recommend(rows, deltas, key: str, energy_tol: float, tol_gap_ev: float = TOL_GAP_EV):
    """``check_convergence.recommend``: the first point whose gap has settled (and
    whose energy delta is not pathological) wins; a metallic system falls back to
    the energy criterion; reaching the ceiling of the sweep is not converged."""
    for row, delta in zip(rows, deltas):
        d_e, d_gap = delta["delta_mev_per_atom"], delta["delta_gap_ev"]
        if d_gap is not None:
            settled = d_gap < tol_gap_ev and (d_e is None or d_e < energy_tol)
        else:
            settled = d_e is not None and d_e < energy_tol
        if settled:
            return row[key], True
    return rows[-1][key], False


def convergence_policy(ecut_rows, kpts_rows, tol_mev_per_atom: float) -> dict:
    """The whole gate as a pure function of the measured sweep rows."""
    rec_ecut, ecut_ok = recommend(ecut_rows, sweep_deltas(ecut_rows), "ecut_ev",
                                  ECUT_ENERGY_TOL)
    rec_kd, kpts_ok = recommend(kpts_rows, sweep_deltas(kpts_rows), "kpts_density",
                                tol_mev_per_atom)
    return {"recommended_ecut_ev": int(rec_ecut), "ecut_converged": bool(ecut_ok),
            "recommended_kpts_density": float(rec_kd), "kpts_converged": bool(kpts_ok),
            "converged": bool(ecut_ok and kpts_ok)}


def params_verified(gate: dict, ecut: int, kpts_density: float) -> bool:
    """``calc_band_dos``'s own check that the run was computed at or above the
    parameters the convergence gate recommended."""
    return bool(gate["converged"]
                and ecut >= gate["recommended_ecut_ev"]
                and kpts_density >= gate["recommended_kpts_density"])


# ---------------------------------------------------------------------------
# The verification state machine (verify.verify_run)
# ---------------------------------------------------------------------------

def gap_branch_margin(band_gap_ev: float, band_gap_ref: float, gap_tol_ev: float) -> float:
    """Distance of ``|gap - ref|`` from the nearest ``verify_run`` branch boundary."""
    difference = abs(band_gap_ev - band_gap_ref)
    return min(abs(difference - gap_tol_ev), abs(difference - 1.5 * gap_tol_ev))


def verify_checks(*, gate_converged: bool, relax_converged: bool, max_force: float,
                  drift_a: float | None, band_gap_ev: float, band_gap_ref: float | None,
                  gap_type: str | None, gap_tol_ev: float) -> dict:
    """The ordered (check, status) list and the verdict, with fail > warn > pass.

    The built-in structure carries no ``is_stable``, so the ``mp_stability``
    check never appears; ``structure_drift`` only appears when the run relaxed.
    """
    checks: list[tuple[str, str]] = [("convergence_gate", "pass" if gate_converged else "warn")]
    ok = relax_converged and max_force <= RELAX_FMAX_TOL
    checks.append(("relax_convergence", "pass" if ok else "fail"))
    if drift_a is not None:
        checks.append(("structure_drift", "pass" if drift_a <= DRIFT_TOL_A else "warn"))
    if band_gap_ref is None:
        checks.append(("band_gap_vs_mp", "skip"))
    else:
        difference = abs(band_gap_ev - band_gap_ref)
        checks.append(("band_gap_vs_mp",
                       "pass" if difference <= gap_tol_ev
                       else ("warn" if difference <= 1.5 * gap_tol_ev else "fail")))
    if gap_type:
        checks.append(("gap_character", "info"))
    fails = [name for name, status in checks if status == "fail"]
    warns = [name for name, status in checks if status == "warn"]
    return {"verdict": "fail" if fails else ("pass_with_warnings" if warns else "pass"),
            "checks": [{"check": name, "status": status} for name, status in checks],
            "blocking_failures": fails}


# ---------------------------------------------------------------------------
# The instance
# ---------------------------------------------------------------------------

def build_case(seed: int) -> dict:
    rng = random.Random(seed)
    ecut = ECUTS[rng.randrange(len(ECUTS))]
    kpts_density = KPTS_DENSITIES[rng.randrange(len(KPTS_DENSITIES))]
    return {
        "structure": "MoS2 monolayer (2H), the server's built-in builder",
        "query": QUERY,
        "use_builtin": True,
        "ecut": ecut,
        "kpts_density": kpts_density,
        "tol_mev_per_atom": TOL_MEV_PER_ATOM[rng.randrange(len(TOL_MEV_PER_ATOM))],
        "gap_tol_ev": GAP_TOL_EV[rng.randrange(len(GAP_TOL_EV))],
        "fmax": FMAX,
        "max_steps": MAX_STEPS,
        "npoints": NPOINTS,
        "window_ev": WINDOW_EV,
        "engine": ENGINE,
        "result_file": "result.json",
        "copied_files": list(COPIED_PNG),
    }


def reference(case: dict) -> dict:
    point = (case["ecut"], case["kpts_density"])
    measured = MEASURED.get(point)
    if measured is None:
        raise GenerationError(
            f"no measured DFT for ecut={point[0]} kpts_density={point[1]}; regenerate the "
            "MEASURED table with scripts/mcp/e2e/measure_gpaw_table.py against the pinned "
            f"server (have {sorted(MEASURED)})")

    structure = mos2_structure()
    kpts = auto_kpts(structure["lengths"], structure["angles"], structure["scaled"],
                     case["kpts_density"])
    if list(kpts) != list(measured["relax_kpts"]):
        raise GenerationError(f"auto_kpts gives {list(kpts)} but the server realized "
                              f"{measured['relax_kpts']} at density {case['kpts_density']}")
    if len(RUN_ARTIFACTS) != measured["artifact_count"]:
        raise GenerationError(f"the chain left {measured['artifact_count']} artefacts but "
                              f"RUN_ARTIFACTS names {len(RUN_ARTIFACTS)}")

    gate = convergence_policy(measured["ecut_sweep"], measured["kpts_sweep"],
                              case["tol_mev_per_atom"])
    verified = params_verified(gate, case["ecut"], case["kpts_density"])
    margin = gap_branch_margin(measured["band_gap_ev"], BAND_GAP_REF, case["gap_tol_ev"])
    if margin < GAP_BRANCH_MARGIN_EV:
        raise GenerationError(
            f"|gap - MP reference| = {abs(measured['band_gap_ev'] - BAND_GAP_REF):.6f} eV is "
            f"{margin:.6f} eV from a verify_run branch boundary at gap_tol_ev="
            f"{case['gap_tol_ev']}; the verdict would be decided by noise, so choose other "
            "GAP_TOL_EV values")
    verdict = verify_checks(
        gate_converged=gate["converged"],
        relax_converged=True,
        max_force=measured["relax_max_force_ev_per_a"],
        # The monolayer relaxes by well under 0.01 A, two orders below the
        # 0.5 A threshold, so structure_drift is a constant pass. verify_run
        # does not report the drift itself, but it does report that check, and
        # the offline tests compare the whole measured check list with the one
        # derived here for every grid point.
        drift_a=0.0,
        band_gap_ev=measured["band_gap_ev"],
        band_gap_ref=BAND_GAP_REF,
        gap_type=measured["gap_type"],
        gap_tol_ev=case["gap_tol_ev"])

    return {
        "relax_total_energy_ev": measured["relax_total_energy_ev"],
        "relax_max_force_ev_per_a": measured["relax_max_force_ev_per_a"],
        "relax_n_steps": measured["relax_n_steps"],
        "relax_kpts": list(measured["relax_kpts"]),
        "scf_total_energy_ev": measured["scf_total_energy_ev"],
        "fermi_ev": measured["fermi_ev"],
        "band_gap_ev": measured["band_gap_ev"],
        "gap_type": measured["gap_type"],
        "vbm_label": measured["vbm_label"],
        "cbm_label": measured["cbm_label"],
        "band_path": measured["band_path"],
        "params_verified": verified,
        "recommended_ecut_ev": gate["recommended_ecut_ev"],
        "recommended_kpts_density": gate["recommended_kpts_density"],
        "converged": gate["converged"],
        # Not in the answer contract; kept so a scoring dispute shows which half
        # of the gate decided `converged`.
        "ecut_converged": gate["ecut_converged"],
        "kpts_converged": gate["kpts_converged"],
        "verdict": verdict["verdict"],
        "verify_checks": verdict["checks"],
        # What the verifier compares get_run_artifacts and verify_run against.
        "verify_check_labels": [f"{c['check']}:{c['status']}" for c in verdict["checks"]],
        "run_artifact_names": list(RUN_ARTIFACTS),
        "blocking_failures": verdict["blocking_failures"],
        "artifact_count": measured["artifact_count"],
        "band_gap_ref": BAND_GAP_REF,
        "n_atoms": N_ATOMS,
        "mcp_tools": list(MCP_TOOLS),
    }


# ---------------------------------------------------------------------------
# Framework entry point
# ---------------------------------------------------------------------------

def render_prompts(task_dir: Path, output_dir: Path, case: dict) -> None:
    for level in ("b1", "b2", "b3", "b4"):
        text = (task_dir / f"prompt_{level}.md").read_text(encoding="utf-8")
        for key, value in case.items():
            rendered = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
            text = text.replace("{{" + key + "}}", rendered)
        if "{{" in text:
            raise GenerationError(f"unresolved placeholder in prompt_{level}.md")
        (output_dir / f"prompt_{level}.md").write_text(text, encoding="utf-8")


def generate(output_dir: Path, params: dict) -> dict:
    merged = {**DEFAULT_PARAMS, **params}
    started = time.time()
    output_dir = Path(output_dir)
    data_dir, ref_dir = output_dir / "data", output_dir / "reference"
    data_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    case = build_case(int(merged["seed"]))
    ref = reference(case)
    numbers = [value for value in ref.values()
               if isinstance(value, float) and not isinstance(value, bool)]
    if not all(math.isfinite(value) for value in numbers) or ref["band_gap_ev"] <= 0:
        raise GenerationError("non-finite or empty reference")

    (data_dir / "calculation.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    (ref_dir / "reference.json").write_text(json.dumps({**case, **ref}, indent=2) + "\n",
                                            encoding="utf-8")
    render_prompts(Path(__file__).resolve().parent, output_dir, case)
    meta = {
        "params_used": merged,
        "input_files": [spec["name"] for spec in INPUT_SPEC],
        "reference_files": ["reference.json"],
        "generation_time_seconds": round(time.time() - started, 2),
    }
    (output_dir / "instance_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params", type=str, default="{}")
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, json.loads(args.params)), indent=2))
