"""Small stdio MCP adapter for the official MatterSim ASE calculator.

The adapter intentionally keeps the scientific contract narrow: callers pass a
JSON ASE-like structure and receive energies in eV, forces in eV/A, and stress
in eV/A^3.  The model is selected by ``MATTERSIM_MODEL`` (default 1M) and an
optional local checkpoint can be pinned with ``MATTERSIM_LOAD_PATH``.  The
adapter does not silently substitute another calculator.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("asibench-mattersim")
logging.basicConfig(level=logging.INFO, stream=__import__("sys").stderr)
mcp = FastMCP("MatterSim")

_calculator = None
_calculator_key: tuple[str, str] | None = None


def _atoms(data: dict[str, Any]):
    from ase import Atoms

    if not isinstance(data, dict):
        raise ValueError("structure_data must be an object")
    symbols = data.get("symbols")
    positions = data.get("positions")
    if isinstance(symbols, str):
        symbols = symbols.split()
    if not isinstance(symbols, list) or not symbols or not all(isinstance(x, str) for x in symbols):
        raise ValueError("structure_data.symbols must be a non-empty list of element symbols")
    if not isinstance(positions, list) or len(positions) != len(symbols):
        raise ValueError("structure_data.positions must have one XYZ row per symbol")
    kwargs: dict[str, Any] = {"symbols": symbols, "positions": positions}
    if "cell" in data:
        kwargs["cell"] = data["cell"]
    if "pbc" in data:
        kwargs["pbc"] = data["pbc"]
    return Atoms(**kwargs)


def _structure(atoms) -> dict[str, Any]:
    return {
        "symbols": atoms.get_chemical_symbols(),
        "positions": atoms.get_positions().tolist(),
        "cell": atoms.cell.array.tolist(),
        "pbc": atoms.get_pbc().tolist(),
    }


def _calculator_for(model_name: str, device: str):
    global _calculator, _calculator_key
    key = (model_name, device)
    if _calculator is not None and _calculator_key == key:
        return _calculator
    from mattersim.forcefield import MatterSimCalculator

    load_path = os.environ.get("MATTERSIM_LOAD_PATH") or model_name
    _calculator = MatterSimCalculator(load_path=load_path, device=device)
    _calculator_key = key
    return _calculator


def _settings(model_name: str | None, device: str | None) -> tuple[str, str]:
    return (model_name or os.environ.get("MATTERSIM_MODEL", "MatterSim-v1.0.0-1M"),
            device or os.environ.get("MATTERSIM_DEVICE", "auto"))


@mcp.tool()
def get_info() -> dict[str, Any]:
    """Return the pinned model selection without loading the checkpoint."""
    model, device = _settings(None, None)
    return {
        "model": model,
        "device": device,
        "checkpoint": os.environ.get("MATTERSIM_LOAD_PATH") or "MatterSim package cache",
        "calculator": "microsoft/mattersim",
    }


@mcp.tool()
def predict_structure(
    structure_data: dict[str, Any],
    model_name: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Compute energy, forces, and (for periodic cells) stress for one structure."""
    try:
        model, target = _settings(model_name, device)
        atoms = _atoms(structure_data)
        atoms.calc = _calculator_for(model, target)
        result: dict[str, Any] = {
            "energy_eV": float(atoms.get_potential_energy()),
            "forces_eV_per_A": atoms.get_forces().tolist(),
        }
        try:
            result["stress_eV_per_A3"] = atoms.get_stress().tolist()
        except Exception as exc:  # Molecules do not define a virial stress.
            result["stress_unavailable"] = str(exc)
        return result
    except Exception as exc:
        logger.exception("MatterSim prediction failed")
        return {"error": f"MatterSim prediction failed: {exc}"}


@mcp.tool()
def relax_structure(
    structure_data: dict[str, Any],
    fmax: float = 0.02,
    steps: int = 200,
    model_name: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Relax atomic positions with ASE BFGS and return the final structure."""
    if fmax <= 0 or steps <= 0:
        return {"error": "fmax and steps must be positive"}
    try:
        from ase.optimize import BFGS

        model, target = _settings(model_name, device)
        atoms = _atoms(structure_data)
        atoms.calc = _calculator_for(model, target)
        optimizer = BFGS(atoms, logfile=None)
        converged = bool(optimizer.run(fmax=fmax, steps=steps))
        result: dict[str, Any] = {
            "converged": converged,
            "steps": int(optimizer.nsteps),
            "energy_eV": float(atoms.get_potential_energy()),
            "forces_eV_per_A": atoms.get_forces().tolist(),
            "structure": _structure(atoms),
        }
        return result
    except Exception as exc:
        logger.exception("MatterSim relaxation failed")
        return {"error": f"MatterSim relaxation failed: {exc}"}


if __name__ == "__main__":
    mcp.run(transport="stdio")
