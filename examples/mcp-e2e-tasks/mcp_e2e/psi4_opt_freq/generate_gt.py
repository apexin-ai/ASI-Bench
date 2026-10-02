"""Instance generator for the psi4 optimize → frequency MCP E2E fake task.

A seeded RNG picks a small closed-shell molecule, a basis set (STO-3G or
cc-pVDZ) and a random distortion of a near-equilibrium starting geometry. The
agent must optimise the geometry with RHF through the psi4 MCP server and then
compute harmonic frequencies at the optimised geometry.

The reference is computed with a different program, PySCF, set up to match
what the server's psi4 does:

* density-fitted RHF with psi4's default JK fitting basis for the orbital basis
  (``def2-universal-jkfit`` for STO-3G, ``cc-pvdz-jkfit`` for cc-pVDZ; psi4
  treats Pople-type 6-31G auxiliary functions as Cartesian, so 6-31G is not
  used);
* a tight geomeTRIC minimisation from the same starting geometry;
* the analytic DF-RHF Hessian at that minimum, harmonic analysis with
  translations/rotations projected out and most-abundant-isotope masses (psi4's
  convention; PySCF's default average masses shift frequencies by ~0.3 cm⁻¹);
* ZPE = ½ Σ ν with 1 cm⁻¹ = 4.556335252912e-6 Eh.

Measured agreement with the pinned server (scripts/mcp/e2e/README.md):
energies ≤ 4e-9 Eh, frequencies ≤ 0.12 cm⁻¹, ZPE ≤ 5e-7 Eh.

Framework call: python generate_gt.py --output-dir <dir> --params '<json>'
Needs PySCF and geomeTRIC: generate with ``asibench generate --sandbox task``.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

INPUT_SPEC = [
    {"name": "molecule.json", "description": "molecule, starting geometry (Angstrom), charge, multiplicity, method, basis"},
]
OUTPUT_SPEC = [
    {"name": "result.json", "description": "final_energy_hartree, frequencies_cm_inv, zpe_hartree"},
]
DEFAULT_PARAMS = {"seed": 0}

# Near-equilibrium geometries (Angstrom); every molecule is a rigid closed-shell minimum.
MOLECULES = {
    "water": "O 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692",
    "ammonia": "N 0.0 0.0 0.1124\nH 0.0 0.9372 -0.2623\nH 0.8116 -0.4686 -0.2623\nH -0.8116 -0.4686 -0.2623",
    "formaldehyde": "C 0.0 0.0 -0.5296\nO 0.0 0.0 0.6766\nH 0.0 0.9377 -1.1175\nH 0.0 -0.9377 -1.1175",
    "hydrogen cyanide": "C 0.0 0.0 -0.4990\nN 0.0 0.0 0.6560\nH 0.0 0.0 -1.5650",
    "hydrogen fluoride": "F 0.0 0.0 0.0\nH 0.0 0.0 0.917",
    "methane": ("C 0.0 0.0 0.0\nH 0.6291 0.6291 0.6291\nH -0.6291 -0.6291 0.6291\n"
                "H -0.6291 0.6291 -0.6291\nH 0.6291 -0.6291 -0.6291"),
}
# Orbital basis -> psi4's default DF_BASIS_SCF (checked in psi4 1.11 output).
JK_FIT = {"sto-3g": "def2-universal-jkfit", "cc-pvdz": "cc-pvdz-jkfit"}
DISTORTION = 0.04                    # Angstrom, uniform per Cartesian component
CM_TO_HARTREE = 4.556335252912e-6    # 1 cm^-1 in Eh (CODATA 2018)
# geomeTRIC criteria ~10x tighter than Gaussian "tight" (psi4 gau_tight).
GEOMETRIC_TIGHT = {"convergence_energy": 1e-8, "convergence_grms": 1e-6, "convergence_gmax": 1.5e-6,
                   "convergence_drms": 4e-6, "convergence_dmax": 6e-6}


def build_case(seed: int) -> dict:
    rng = random.Random(seed)
    name = sorted(MOLECULES)[rng.randrange(len(MOLECULES))]
    basis = sorted(JK_FIT)[rng.randrange(len(JK_FIT))]
    lines = []
    for line in MOLECULES[name].splitlines():
        symbol, *xyz = line.split()
        lines.append(symbol + " " + " ".join(f"{float(v) + rng.uniform(-DISTORTION, DISTORTION):.4f}" for v in xyz))
    return {"molecule": name, "geometry_xyz": "\n".join(lines), "charge": 0, "multiplicity": 1,
            "method": "HF", "basis": basis,
            "units": {"geometry": "Angstrom", "energy": "Hartree", "frequency": "cm^-1"}}


@contextlib.contextmanager
def _quiet():
    """geomeTRIC prints a banner and per-step logs to stdout; the framework reads our stdout."""
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


def reference(case: dict) -> dict:
    import geometric
    import numpy as np
    import pyscf
    from pyscf import gto, scf
    from pyscf.data.elements import COMMON_ISOTOPE_MASSES
    from pyscf.geomopt.geometric_solver import optimize
    from pyscf.hessian import thermo

    aux = JK_FIT[case["basis"]]
    atoms = "; ".join(case["geometry_xyz"].splitlines())
    mol = gto.M(atom=atoms, basis=case["basis"], charge=case["charge"], spin=case["multiplicity"] - 1,
                symmetry=False, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=aux)
    mf.conv_tol = 1e-11
    with _quiet():
        mol_eq = optimize(mf, maxsteps=200, **GEOMETRIC_TIGHT)
    mf_eq = scf.RHF(mol_eq).density_fit(auxbasis=aux)
    mf_eq.conv_tol = 1e-12
    energy = float(mf_eq.kernel())
    if not mf_eq.converged:
        raise RuntimeError("reference SCF at the minimum did not converge")
    hessian = mf_eq.Hessian().kernel()
    masses = np.array([COMMON_ISOTOPE_MASSES[int(z)] for z in mol_eq.atom_charges()])
    modes = thermo.harmonic_analysis(mol_eq, hessian, mass=masses)
    raw = np.atleast_1d(modes["freq_wavenumber"])
    freqs = sorted(float(w.real) if abs(np.imag(w)) < 1e-9 else -float(abs(w)) for w in raw)
    natm = mol_eq.natm
    linear = natm == 2 or case["molecule"] == "hydrogen cyanide"
    if len(freqs) != 3 * natm - (5 if linear else 6) or min(freqs) <= 0:
        raise RuntimeError(f"reference is not a minimum: {freqs}")
    coords = mol_eq.atom_coords(unit="Angstrom")
    return {
        "final_energy_hartree": energy,
        "frequencies_cm_inv": freqs,
        "zpe_hartree": 0.5 * sum(freqs) * CM_TO_HARTREE,
        "n_imaginary": 0,
        "optimized_geometry_xyz": "\n".join(f"{mol_eq.atom_symbol(k)} {x:.10f} {y:.10f} {z:.10f}"
                                            for k, (x, y, z) in enumerate(coords.tolist())),
        "jk_fit_basis": aux,
        "pyscf_version": pyscf.__version__,
        "geometric_version": geometric.__version__,
    }


def render_prompts(task_dir: Path, output_dir: Path, case: dict) -> None:
    for level in ("b1", "b2", "b3", "b4"):
        text = (task_dir / f"prompt_{level}.md").read_text(encoding="utf-8")
        for key in ("molecule", "method", "basis"):
            text = text.replace("{{" + key + "}}", str(case[key]))
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
    if not all(math.isfinite(v) for v in (ref["final_energy_hartree"], ref["zpe_hartree"], *ref["frequencies_cm_inv"])):
        raise RuntimeError("non-finite reference")
    (data_dir / "molecule.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    reference_doc = {**case, **ref, "mcp_tools": ["optimize", "frequency"]}
    (ref_dir / "reference.json").write_text(json.dumps(reference_doc, indent=2) + "\n", encoding="utf-8")

    render_prompts(Path(__file__).resolve().parent, output_dir, case)
    meta = {
        "params_used": p,
        "input_files": [s["name"] for s in INPUT_SPEC],
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
