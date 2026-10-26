"""Workspace and size contracts shared by the local atomistic MCP adapters."""
from __future__ import annotations
import math
import os
from pathlib import Path

MAX_ATOMS = 4096
MAX_FILE_BYTES = 16 * 1024 * 1024
FORMATS = {'.xyz': 'extxyz', '.extxyz': 'extxyz', '.cif': 'cif'}


def workspace() -> Path:
    return Path(os.environ.get('ATOMISTIC_MCP_WORKSPACE', '/workspace')).resolve()


def input_path(name: str) -> Path:
    if not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('input must be a relative workspace path')
    path = (workspace()/name).resolve()
    if not path.is_relative_to(workspace()) or not path.is_file():
        raise ValueError('input must be an existing workspace file')
    if path.suffix.lower() not in FORMATS or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError('input must be XYZ/extxyz/CIF and at most 16 MiB')
    return path


def output_path(name: str, suffixes=('.extxyz', '.xyz', '.cif')) -> Path:
    if not name or Path(name).name != name or Path(name).suffix.lower() not in suffixes:
        raise ValueError(f'output must be a simple filename ending in {suffixes}')
    path = workspace()/name
    # Do not follow output symlinks, including links back inside the workspace.
    if path.is_symlink() or not path.resolve().is_relative_to(workspace()):
        raise ValueError('output symlinks or workspace escapes are forbidden')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def bounded_int(value: int, lower: int, upper: int, label: str) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f'{label} must be an integer in [{lower}, {upper}]')
    return value


def positive(value: float, lower: float, upper: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f'{label} must be finite in [{lower}, {upper}]')
    return float(value)


def read_atoms(name: str):
    from ase.io import read
    path = input_path(name)
    if path.suffix.lower() in {'.xyz', '.extxyz'}:
        with path.open() as handle:
            bounded_int(int(handle.readline().strip()), 1, MAX_ATOMS, 'atom count')
    # Explicit formats avoid calculator execution, URLs and ASE pickle formats.
    atoms = read(str(path), index=0, format=FORMATS[path.suffix.lower()])
    validate_atoms(atoms)
    return atoms


def validate_atoms(atoms):
    import numpy as np
    if not 1 <= len(atoms) <= MAX_ATOMS:
        raise ValueError(f'structure must contain 1..{MAX_ATOMS} atoms')
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
        raise ValueError('structure coordinates/cell must be finite')
    if atoms.pbc.any() and (atoms.cell.rank != 3 or abs(atoms.get_volume()) < 1e-8):
        raise ValueError('periodic structures require a nondegenerate 3D cell')


def structure(atoms) -> dict:
    return {'atom_count': len(atoms), 'symbols': atoms.get_chemical_symbols(),
            'positions_A': atoms.positions.tolist(), 'cell_A': atoms.cell.array.tolist(),
            'pbc': atoms.pbc.tolist(), 'formula': atoms.get_chemical_formula()}


def save_atoms(atoms, path: Path) -> str:
    from ase.io import write
    write(str(path), atoms, format=FORMATS[path.suffix.lower()])
    return str(path)
