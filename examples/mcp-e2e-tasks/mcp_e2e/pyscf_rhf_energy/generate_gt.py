"""Instance generator for the pyscf MCP E2E fake task.

A seeded RNG picks a small closed-shell molecule and basis set and perturbs the
equilibrium-like geometry by up to ±0.02 Å per coordinate, so the exact energy
cannot be recalled from memory. The reference RHF energy is computed here with
PySCF using the same settings as the MCP tool (``symmetry=True``).

Framework call: python generate_gt.py --output-dir <dir> --params '<json>'
Needs PySCF: generate with ``asibench generate --sandbox task``.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

INPUT_SPEC = [
    {"name": "molecule.json", "description": "molecule name, PySCF atom string (Angstrom), basis"},
]
OUTPUT_SPEC = [
    {"name": "result.json", "description": "energy_hartree, basis, molecule"},
]
DEFAULT_PARAMS = {"seed": 0}

# Approximate equilibrium geometries in Angstrom (closed shell, neutral).
MOLECULES = {
    "water": [("O", 0.0, 0.0, 0.117790), ("H", 0.0, 0.755453, -0.471161), ("H", 0.0, -0.755453, -0.471161)],
    "ammonia": [("N", 0.0, 0.0, 0.112700), ("H", 0.0, 0.937700, -0.263000),
                ("H", 0.812100, -0.468900, -0.263000), ("H", -0.812100, -0.468900, -0.263000)],
    "methane": [("C", 0.0, 0.0, 0.0), ("H", 0.629100, 0.629100, 0.629100), ("H", -0.629100, -0.629100, 0.629100),
                ("H", -0.629100, 0.629100, -0.629100), ("H", 0.629100, -0.629100, -0.629100)],
    "hydrogen fluoride": [("F", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.917000)],
    "carbon monoxide": [("C", 0.0, 0.0, 0.0), ("O", 0.0, 0.0, 1.128000)],
}
BASES = ["sto-3g", "6-31g"]
PERTURBATION_ANGSTROM = 0.02


def build_case(seed: int) -> dict:
    rng = random.Random(seed)
    name = rng.choice(sorted(MOLECULES))
    basis = rng.choice(BASES)
    atoms = []
    for symbol, *xyz in MOLECULES[name]:
        coords = [round(c + rng.uniform(-PERTURBATION_ANGSTROM, PERTURBATION_ANGSTROM), 6) for c in xyz]
        atoms.append(f"{symbol} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}")
    return {"molecule": name, "atom": "; ".join(atoms), "basis": basis, "charge": 0, "spin": 0,
            "units": "Angstrom"}


def reference_energy(atom: str, basis: str) -> tuple[float, str]:
    import pyscf
    from pyscf import gto, scf

    mol = gto.M(atom=atom, basis=basis, symmetry=True, verbose=0)
    mf = scf.HF(mol)
    mf.verbose = 0
    energy = float(mf.kernel())
    if not mf.converged:
        raise RuntimeError(f"reference RHF did not converge for {atom!r}/{basis}")
    return energy, pyscf.__version__


def render_prompts(task_dir: Path, output_dir: Path, case: dict) -> None:
    for level in ("b1", "b2", "b3", "b4"):
        text = (task_dir / f"prompt_{level}.md").read_text(encoding="utf-8")
        for key in ("molecule", "basis"):
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
    (data_dir / "molecule.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")

    energy, pyscf_version = reference_energy(case["atom"], case["basis"])
    reference = {**case, "energy_hartree": energy, "method": "RHF", "symmetry": True,
                 "pyscf_version": pyscf_version, "mcp_tool": "pyscf_rhf_energy"}
    (ref_dir / "reference.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")

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
