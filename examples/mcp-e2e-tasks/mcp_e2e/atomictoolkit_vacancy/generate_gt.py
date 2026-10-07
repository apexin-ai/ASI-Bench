"""Instance generator for the atomictoolkit vacancy MCP E2E fake task.

A seeded RNG picks an fcc metal that ASE's EMT potential covers, a lattice
constant near its experimental one, a cubic supercell size, the atom to remove
and the strain range of the bulk-modulus fit. Through the atomictoolkit MCP
server (XirtamEsrevni/mcp-atomictoolkit) the agent must build the conventional
cubic cell (``build_structure_workflow``), repeat it (``manipulate_structure_workflow``
``supercell``), remove one atom (``vacancy``), compute both supercells' EMT energies
(``single_point_workflow`` twice), analyse the defective one
(``analyze_structure_workflow``) and fit the unit cell's bulk modulus
(``estimate_elastic_workflow``), then report the unrelaxed vacancy formation energy
``E_vac = E_vacancy - E_perfect * (N - 1) / N``.

The reference is computed here with ASE directly (never through the server),
following what the pinned server does:

* structures go through an extxyz write / read at every step, as the server's
  files do (the 8-decimal positions then match the server's bit for bit);
* energies with ``EMT()``; ``estimate_elastic`` as the server fits it: five
  energies at strains ``(-s, -s/2, 0, s/2, s)`` of the cell, ``np.polyfit`` degree
  2, ``B = 2 a / (9 V0)`` — the curvature of E(strain) at the given cell, which
  includes a pressure term when the cell is not at equilibrium;
* coordination numbers with ``NeighborList`` at covalent radius x 1.2 and ASE's
  default skin (0.3 Angstrom), which the server leaves in place (smoke defect D4):
  the server reports second neighbours too whenever they fall inside
  ``2 * 1.2 * r_cov + 0.6``. The physical (skin 0) average is recorded alongside.

Lattice constants within 0.02 Angstrom of a neighbour-shell boundary are
skipped, so the coordination count is not decided by rounding. Al is not used:
at +2 % its first shell leaves the 1.2 x covalent cutoff.

Framework call: python generate_gt.py --output-dir <dir> --params '<json>'
Needs ASE 3.29.0: generate with ``asibench generate --sandbox task``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
import time
from pathlib import Path

INPUT_SPEC = [
    {"name": "calculation.json",
     "description": "element, lattice constant, supercell size, vacancy index, strain range, file names"},
]
OUTPUT_SPEC = [
    {"name": "result.json",
     "description": "atom count, both energies, vacancy formation energy, coordination, bulk modulus"},
]
DEFAULT_PARAMS = {"seed": 0}

# Experimental fcc lattice constants (ASE reference_states) and Cordero covalent
# radii (ase.data.covalent_radii), in Angstrom. Pure data, so a case can be built
# (and tested) without ASE.
ELEMENTS = {
    "Ag": {"a0": 4.09, "r_cov": 1.45},
    "Au": {"a0": 4.08, "r_cov": 1.36},
    "Cu": {"a0": 3.61, "r_cov": 1.32},
    "Ni": {"a0": 3.52, "r_cov": 1.24},
    "Pd": {"a0": 3.89, "r_cov": 1.39},
    "Pt": {"a0": 3.92, "r_cov": 1.36},
}
STRAIN_RANGE = 0.02                    # lattice constant within +-2 % of a0
SUPERCELLS = (2, 3)
STRAIN_MAX = (0.01, 0.015, 0.02)
COORDINATION_FACTOR = 1.2              # analyze_structure_workflow default
UPSTREAM_SKIN = 0.3                    # ASE NeighborList default, kept by the server
SHELL_MARGIN = 0.02                    # Angstrom between a neighbour shell and a cutoff
FILES = {"unit_cell_file": "unit_cell.extxyz", "supercell_file": "supercell.extxyz",
         "vacancy_file": "vacancy.extxyz", "analysis_dir": "analysis"}
CALCULATOR = "emt"
MCP_TOOLS = ["build_structure_workflow", "manipulate_structure_workflow", "single_point_workflow",
             "analyze_structure_workflow", "estimate_elastic_workflow"]


def shell_distances(a: float) -> list[float]:
    """The first six fcc neighbour distances, a * sqrt(k / 2)."""
    return [a * math.sqrt(k / 2) for k in range(1, 7)]


def cutoffs(r_cov: float) -> tuple[float, float]:
    """Pair cutoff of the server's coordination count (with skin) and the physical one."""
    physical = 2 * COORDINATION_FACTOR * r_cov
    return physical + 2 * UPSTREAM_SKIN, physical


def clear_of_shells(a: float, r_cov: float) -> bool:
    return all(abs(d - c) >= SHELL_MARGIN for d in shell_distances(a) for c in cutoffs(r_cov))


def build_case(seed: int) -> dict:
    rng = random.Random(seed)
    element = sorted(ELEMENTS)[rng.randrange(len(ELEMENTS))]
    a0, r_cov = ELEMENTS[element]["a0"], ELEMENTS[element]["r_cov"]
    while True:
        a = round(a0 * (1 + rng.uniform(-STRAIN_RANGE, STRAIN_RANGE)), 3)
        if clear_of_shells(a, r_cov):
            break
    n = SUPERCELLS[rng.randrange(len(SUPERCELLS))]
    return {"element": element, "crystal_system": "fcc", "lattice_constant": a, "supercell": n,
            "vacancy_index": rng.randrange(4 * n ** 3),
            "strain_max": STRAIN_MAX[rng.randrange(len(STRAIN_MAX))],
            "calculator": CALCULATOR, **FILES}


def _round_trip(atoms, path: Path):
    from ase.io import read, write
    write(path, atoms)
    return read(path)


def _energy(atoms) -> float:
    from ase.calculators.emt import EMT
    atoms = atoms.copy()
    atoms.calc = EMT()
    return float(atoms.get_potential_energy())


def _coordination(atoms, skin: float) -> float:
    import numpy as np
    from ase.data import covalent_radii
    from ase.neighborlist import NeighborList
    radii = [covalent_radii[atom.number] * COORDINATION_FACTOR for atom in atoms]
    neighbours = NeighborList(radii, self_interaction=False, bothways=True, skin=skin)
    neighbours.update(atoms)
    return float(np.mean([len(neighbours.get_neighbors(i)[0]) for i in range(len(atoms))]))


def _bulk_modulus(atoms, strain_max: float) -> tuple[float, float]:
    """(B in GPa, V0 in Angstrom^3) the way estimate_elastic_workflow fits them."""
    import numpy as np
    from ase import units
    strains = [-strain_max, -0.5 * strain_max, 0.0, 0.5 * strain_max, strain_max]
    energies = []
    for strain in strains:
        strained = atoms.copy()
        strained.set_cell(atoms.cell * (1.0 + strain), scale_atoms=True)
        energies.append(_energy(strained))
    volume0 = float(atoms.get_volume())
    curvature = float(np.polyfit(np.array(strains), np.array(energies), 2)[0])
    return float(2.0 * curvature / (9.0 * volume0) / units.GPa), volume0


def reference(case: dict) -> dict:
    import ase
    from ase.build import bulk

    n = case["supercell"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        unit = _round_trip(bulk(case["element"], "fcc", a=case["lattice_constant"], cubic=True),
                           tmp / case["unit_cell_file"])
        perfect = _round_trip(unit * (n, n, n), tmp / case["supercell_file"])
        defect = perfect.copy()
        del defect[case["vacancy_index"]]
        defect = _round_trip(defect, tmp / case["vacancy_file"])
    n_atoms = len(perfect)
    e_perfect, e_vacancy = _energy(perfect), _energy(defect)
    bulk_modulus, volume = _bulk_modulus(unit, case["strain_max"])
    return {
        "n_atoms_unit_cell": len(unit),
        "n_atoms_perfect": n_atoms,
        "n_atoms_vacancy": len(defect),
        "energy_perfect_eV": e_perfect,
        "energy_vacancy_eV": e_vacancy,
        "vacancy_formation_energy_eV": e_vacancy - e_perfect * (n_atoms - 1) / n_atoms,
        "coordination_average": _coordination(defect, UPSTREAM_SKIN),
        "coordination_average_skin0": _coordination(defect, 0.0),
        "bulk_modulus_GPa": bulk_modulus,
        "unit_cell_volume_A3": volume,
        "operation_supercell": "supercell",
        "operation_vacancy": "vacancy",
        "ase_version": ase.__version__,
    }


def render_prompts(task_dir: Path, output_dir: Path, case: dict) -> None:
    for level in ("b1", "b2", "b3", "b4"):
        text = (task_dir / f"prompt_{level}.md").read_text(encoding="utf-8")
        for key, value in case.items():
            text = text.replace("{{" + key + "}}", str(value))
        if "{{" in text:
            raise RuntimeError(f"prompt_{level}.md has an unknown placeholder")
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
    if not all(math.isfinite(v) for v in ref.values() if isinstance(v, float)):
        raise RuntimeError("non-finite reference")
    (data_dir / "calculation.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    reference_doc = {**case, **ref, "mcp_tools": MCP_TOOLS}
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
