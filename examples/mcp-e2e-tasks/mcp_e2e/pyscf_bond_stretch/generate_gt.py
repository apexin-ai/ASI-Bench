"""Instance generator for the pyscf bond-stretch MCP E2E fake task.

A seeded RNG picks a small rigid molecule, one bond and a scan range. The
reference reproduces what the MCP tool `run_bond_stretch_calculation_mcp` is
meant to compute, independently of the server code: an RDKit ETKDG conformer
(seeded here, unseeded in the server) relaxed with UFF, then the second atom is
moved along the atom1→atom2 direction to each distance while all other atoms
stay fixed, and the RHF/STO-3G energy is computed at every point.

Only rigid molecules are used: their UFF minimum is unique up to rotation and
H permutation, so the server's unseeded geometry gives the same energies
(measured agreement ~1e-8 Ha, see scripts/mcp/e2e/e2e_smoke/servers/pyscf.py).

Framework call: python generate_gt.py --output-dir <dir> --params '<json>'
Needs PySCF and RDKit: generate with ``asibench generate --sandbox task``.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

INPUT_SPEC = [
    {"name": "scan.json", "description": "molecule, SMILES, bond atom indices, scan range and point count"},
]
OUTPUT_SPEC = [
    {"name": "result.json", "description": "bond_lengths, energies_hartree, min_bond_length, min_energy_hartree"},
]
DEFAULT_PARAMS = {"seed": 0}

# (SMILES, name, atom1_idx, atom2_idx, bond label, approximate bond length in Å).
# RDKit atom order: heavy atoms in SMILES order, then the added hydrogens.
BONDS = [
    ("O", "water", 0, 1, "O-H", 0.96),
    ("N", "ammonia", 0, 1, "N-H", 1.01),
    ("F", "hydrogen fluoride", 0, 1, "F-H", 0.92),
    ("C#N", "hydrogen cyanide", 0, 1, "C#N", 1.15),
    ("C#N", "hydrogen cyanide", 0, 2, "C-H", 1.07),
    ("C=O", "formaldehyde", 0, 1, "C=O", 1.21),
    ("CF", "fluoromethane", 0, 1, "C-F", 1.38),
]
BASIS = "sto-3g"  # the MCP tool always uses STO-3G, whatever basis it is given
RDKIT_SEED = 20260929


def build_case(seed: int) -> dict:
    rng = random.Random(seed)
    smiles, name, i, j, label, r_eq = BONDS[rng.randrange(len(BONDS))]
    start = round(r_eq * rng.uniform(0.85, 0.95), 3)
    end = round(r_eq * rng.uniform(1.25, 1.45), 3)
    return {"molecule": name, "smiles": smiles, "atom1_idx": i, "atom2_idx": j, "bond": label,
            "start_dist": start, "end_dist": end, "num_points": rng.randint(5, 9),
            "method": "RHF", "basis": BASIS, "units": {"distance": "Angstrom", "energy": "Hartree"}}


def uff_geometry(smiles: str) -> list[tuple[str, list[float]]]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDG()
    params.randomSeed = RDKIT_SEED
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"RDKit embedding failed for {smiles!r}")
    AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    return [(a.GetSymbol(), [float(v) for v in conf.GetAtomPosition(k)]) for k, a in enumerate(mol.GetAtoms())]


def reference_scan(case: dict) -> dict:
    import numpy as np
    import pyscf
    import rdkit
    from pyscf import gto, scf

    atoms = uff_geometry(case["smiles"])
    i, j = case["atom1_idx"], case["atom2_idx"]
    lengths = np.linspace(case["start_dist"], case["end_dist"], case["num_points"]).tolist()
    energies = []
    for dist in lengths:
        coords = np.array([xyz for _, xyz in atoms])
        vec = coords[j] - coords[i]
        coords[j] = coords[i] + vec * (dist / np.linalg.norm(vec))
        mol = gto.M(atom=[[sym, *xyz] for (sym, _), xyz in zip(atoms, coords.tolist())], basis=BASIS, verbose=0)
        mf = scf.RHF(mol)
        mf.verbose = 0
        energy = float(mf.kernel())
        if not mf.converged:
            raise RuntimeError(f"reference RHF did not converge at {dist} Å for {case['smiles']}")
        energies.append(energy)
    k = int(np.argmin(energies))
    return {"bond_lengths": lengths, "energies_hartree": energies,
            "min_bond_length": lengths[k], "min_energy_hartree": energies[k],
            "atoms_in_order": [sym for sym, _ in atoms],
            "pyscf_version": pyscf.__version__, "rdkit_version": rdkit.__version__}


def render_prompts(task_dir: Path, output_dir: Path, case: dict) -> None:
    for level in ("b1", "b2", "b3", "b4"):
        text = (task_dir / f"prompt_{level}.md").read_text(encoding="utf-8")
        for key in ("molecule", "bond"):
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
    scan = reference_scan(case)
    case["atoms_in_order"] = scan["atoms_in_order"]
    (data_dir / "scan.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    reference = {**case, **scan, "mcp_tools": ["run_bond_stretch_calculation_mcp", "plot_energy_scan_image_mcp"]}
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
