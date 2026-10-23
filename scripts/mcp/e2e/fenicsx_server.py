"""Minimal FEniCSx MCP entry point for a pinned DOLFINx image.

FEniCSx is imported on the first tool call: protocol handshake can complete
without loading PETSc, while runtime info and scientific tools fail with
an explicit deployment error when the image is not a DOLFINx image.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ASI-Bench FEniCSx")
WORKSPACE = Path(os.environ.get("FENICSX_MCP_WORKSPACE", "/workspace")).resolve()


def _dolfinx():
    try:
        import dolfinx
        import ufl
        from mpi4py import MPI
        from petsc4py import PETSc
    except ImportError as exc:  # pragma: no cover - requires DOLFINx image
        raise RuntimeError("FEniCSx MCP requires a pinned DOLFINx/PETSc/MPI image") from exc
    return dolfinx, ufl, MPI, PETSc


def _output(name: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name or not name:
        raise ValueError("output_name must be a simple workspace file name")
    path = (WORKSPACE / name).resolve()
    if not path.is_relative_to(WORKSPACE):
        raise ValueError("output must stay inside the MCP workspace")
    return path


@mcp.tool()
def get_info() -> dict[str, Any]:
    """Load and report the actual FEniCSx runtime."""
    dolfinx, _, _, PETSc = _dolfinx()
    return {"dolfinx_version": dolfinx.__version__, "petsc_version": PETSc.Sys.getVersion(), "solver": "FEniCSx/DOLFINx", "workspace": str(WORKSPACE), "requires": ["MPI", "PETSc", "UFL"]}


@mcp.tool()
def solve_poisson_manufactured(nx: int, ny: int, degree: int = 1, output_name: str = "poisson.xdmf") -> dict[str, Any]:
    """Solve a bounded unit-square manufactured Poisson problem and export XDMF."""
    if isinstance(nx, bool) or not isinstance(nx, int) or not 1 <= nx <= 256:
        raise ValueError("nx must be an integer in [1, 256]")
    if isinstance(ny, bool) or not isinstance(ny, int) or not 1 <= ny <= 256:
        raise ValueError("ny must be an integer in [1, 256]")
    if isinstance(degree, bool) or not isinstance(degree, int) or not 1 <= degree <= 2:
        raise ValueError("degree must be 1 or 2")
    path = _output(output_name)
    if path.suffix.lower() != ".xdmf":
        raise ValueError("output_name must end in .xdmf")
    # XDMF writes a sibling HDF5 file as well. Validate it before opening either.
    _output(path.with_suffix(".h5").name)
    dolfinx, ufl, MPI, PETSc = _dolfinx()
    from dolfinx import fem, mesh
    from dolfinx.fem.petsc import LinearProblem
    from dolfinx.io import XDMFFile
    from petsc4py.PETSc import ScalarType
    import numpy as np

    domain = mesh.create_unit_square(MPI.COMM_SELF, nx, ny, mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    exact = ufl.sin(math.pi * x[0]) * ufl.sin(math.pi * x[1])
    source = 2 * math.pi**2 * exact
    boundaries = mesh.locate_entities_boundary(domain, 1, lambda x: np.isclose(x[0], 0) | np.isclose(x[0], 1) | np.isclose(x[1], 0) | np.isclose(x[1], 1))
    dofs = fem.locate_dofs_topological(V, 1, boundaries)
    bc = fem.dirichletbc(ScalarType(0), dofs, V)
    problem = LinearProblem(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx, source * v * ufl.dx, bcs=[bc],
        petsc_options_prefix="asibench_poisson_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True})
    uh = problem.solve()
    reason = int(problem.solver.getConvergedReason())
    if reason <= 0:
        raise RuntimeError(f"PETSc solve failed: convergence reason {reason}")
    uh.x.scatter_forward()
    l2_error = math.sqrt(float(domain.comm.allreduce(
        fem.assemble_scalar(fem.form((uh - exact) ** 2 * ufl.dx)), op=MPI.SUM)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with XDMFFile(MPI.COMM_SELF, str(path), "w") as file:
        file.write_mesh(domain)
        # XDMF requires function degree to match the linear mesh geometry.
        exported = uh
        if degree == 2:
            exported = fem.Function(fem.functionspace(domain, ("Lagrange", 1)))
            exported.interpolate(uh)
        file.write_function(exported)
    return {"convergence_reason": reason, "export_degree": 1, "l2_error": l2_error, "mesh_cells": int(domain.topology.index_map(domain.topology.dim).size_local), "dofs": int(V.dofmap.index_map.size_local), "solution_path": str(path), "finite": bool(np.isfinite(uh.x.array).all())}


if __name__ == "__main__":
    mcp.run(transport=os.environ.get("FENICSX_MCP_TRANSPORT", "stdio"))
