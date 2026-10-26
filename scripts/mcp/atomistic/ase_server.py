"""Bounded ASE tools using ordinary MCP tools/call; no arbitrary calculator/code."""
from __future__ import annotations
import math
from typing import Any
from mcp.server.fastmcp import FastMCP
from pydantic import StrictInt, StrictFloat
from common import (MAX_ATOMS, bounded_int, positive, read_atoms, output_path,
                    save_atoms, structure, workspace)

mcp = FastMCP('ASI-Bench ASE')


@mcp.tool()
def get_info() -> dict[str, Any]:
    """Return installed versions, supported calculators and units."""
    import ase
    return {'ase_version': ase.__version__, 'workspace': str(workspace()),
            'calculators': ['emt', 'lennard_jones'],
            'units': {'length': 'angstrom', 'energy': 'eV', 'forces': 'eV/angstrom'}}


@mcp.tool()
def build_bulk(symbol: str, crystalstructure: str, lattice_constant: StrictFloat,
               repeat: list[StrictInt], output_name: str = 'bulk.extxyz') -> dict[str, Any]:
    """Build conventional cubic fcc/bcc/diamond cells and repeat them (<=4096 atoms)."""
    from ase.build import bulk
    from ase.data import atomic_numbers
    if symbol not in atomic_numbers or symbol == 'X':
        raise ValueError('symbol must be a chemical element')
    if crystalstructure not in {'fcc', 'bcc', 'diamond'}:
        raise ValueError('crystalstructure must be fcc, bcc or diamond')
    a = positive(lattice_constant, .5, 50, 'lattice_constant')
    counts = _repeat(repeat)
    cell = bulk(symbol, crystalstructure, a=a, cubic=True)
    if len(cell)*math.prod(counts) > MAX_ATOMS:
        raise ValueError('requested structure exceeds 4096 atoms')
    atoms = cell.repeat(counts)
    path = output_path(output_name)
    return {**structure(atoms), 'structure_path': save_atoms(atoms, path)}


def _repeat(repeat):
    if len(repeat) != 3:
        raise ValueError('repeat must contain three integers')
    return [bounded_int(n, 1, 32, 'repeat') for n in repeat]


@mcp.tool()
def read_structure(input_name: str) -> dict[str, Any]:
    """Read the first frame of a workspace XYZ/extxyz/CIF structure."""
    return structure(read_atoms(input_name))


@mcp.tool()
def repeat_structure(input_name: str, repeat: list[StrictInt],
                     output_name: str = 'supercell.extxyz') -> dict[str, Any]:
    """Repeat a structure along each cell vector, within the atom budget."""
    atoms = read_atoms(input_name)
    counts = _repeat(repeat)
    if len(atoms)*math.prod(counts) > MAX_ATOMS:
        raise ValueError('requested structure exceeds 4096 atoms')
    atoms = atoms.repeat(counts)
    return {**structure(atoms), 'structure_path': save_atoms(atoms, output_path(output_name))}


def _calculator(atoms, name):
    # Bound interactions before the backend constructs potentially huge neighbor lists.
    import numpy as np
    if len(atoms) > 512:
        raise ValueError('energy/optimization tools allow at most 512 atoms')
    if atoms.pbc.any() and np.min(atoms.cell.lengths()) < 2:
        raise ValueError('calculator periodic cell lengths must be at least 2 angstrom')
    if len(atoms) > 1:
        distances = atoms.get_all_distances(mic=bool(atoms.pbc.any()))
        np.fill_diagonal(distances, np.inf)
        if np.min(distances) < .25:
            raise ValueError('interatomic separation must be at least 0.25 angstrom')
    if name == 'emt':
        from ase.calculators.emt import EMT
        if set(atoms.get_chemical_symbols()) - {'Al', 'Cu', 'Ag', 'Au', 'Ni', 'Pd', 'Pt', 'H', 'C', 'N', 'O'}:
            raise ValueError('EMT does not support these elements; see ASE EMT documentation')
        return EMT()
    if name == 'lennard_jones':
        from ase.calculators.lj import LennardJones
        return LennardJones(epsilon=1.0, sigma=1.0, rc=3.0)
    raise ValueError('calculator must be emt or lennard_jones; external engines are not installed')


def _prediction(atoms):
    import numpy as np
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    if not math.isfinite(energy) or not np.isfinite(forces).all():
        raise RuntimeError('calculator returned nonfinite values')
    return {'energy_eV': energy, 'forces_eV_per_A': forces.tolist(),
            'max_force_eV_per_A': float(np.linalg.norm(forces, axis=1).max())}


@mcp.tool()
def single_point(input_name: str, calculator: str = 'emt') -> dict[str, Any]:
    """Compute energy and forces with EMT or LJ (epsilon=1 eV, sigma=1 A, rc=3 A)."""
    atoms = read_atoms(input_name)
    atoms.calc = _calculator(atoms, calculator)
    return {'calculator': calculator, 'atom_count': len(atoms), **_prediction(atoms)}


@mcp.tool()
def optimize_structure(input_name: str, calculator: str = 'emt', fmax: StrictFloat = .05,
                       steps: StrictInt = 100, output_name: str = 'optimized.extxyz') -> dict[str, Any]:
    """BFGS position-only relaxation; fixed cell. Reports convergence even on step exhaustion."""
    from ase.optimize import BFGS
    target = positive(fmax, .001, 1, 'fmax')
    count = bounded_int(steps, 1, 200, 'steps')
    path = output_path(output_name)
    atoms = read_atoms(input_name)
    atoms.calc = _calculator(atoms, calculator)
    optimizer = BFGS(atoms, logfile=None)
    converged = bool(optimizer.run(fmax=target, steps=count))
    prediction = _prediction(atoms)
    return {'calculator': calculator, 'converged': converged, 'steps': optimizer.nsteps,
            **prediction, **structure(atoms), 'structure_path': save_atoms(atoms, path)}


if __name__ == '__main__':
    mcp.run(transport='stdio')
