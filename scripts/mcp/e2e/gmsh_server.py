"""Small, deterministic Gmsh MCP server for ASI-Bench tasks.

The server intentionally exposes a narrow JSON contract instead of arbitrary
Python/Gmsh execution.  All files are confined to the MCP workspace.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ASI-Bench Gmsh")
WORKSPACE = Path(os.environ.get("GMSH_MCP_WORKSPACE", "/workspace")).resolve()


def _safe_name(name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("output_name must be a simple file name")
    path = (WORKSPACE / name).resolve()
    if not path.is_relative_to(WORKSPACE):
        raise ValueError("output must stay inside the MCP workspace")
    return path


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return float(value)


def _gmsh():
    try:
        import gmsh
    except ImportError as exc:  # pragma: no cover - exercised in image smoke
        raise RuntimeError("gmsh Python package is not installed in this server image") from exc
    return gmsh


def _inventory(gmsh) -> dict[str, Any]:
    nodes, coords, _ = gmsh.model.mesh.getNodes()
    element_types, element_tags, _ = gmsh.model.mesh.getElements()
    return {
        "nodes": int(len(nodes)),
        "elements": int(sum(len(tags) for tags in element_tags)),
        "element_types": [int(x) for x in element_types],
        "physical_groups": [
            {"dimension": int(dim), "tag": int(tag), "name": gmsh.model.getPhysicalName(dim, tag)}
            for dim, tag in gmsh.model.getPhysicalGroups()
        ],
    }


@mcp.tool()
def get_info() -> dict[str, Any]:
    """Return the Gmsh runtime and the constrained workspace."""
    gmsh = _gmsh()
    return {"gmsh_version": gmsh.option.getString("General.Version"), "workspace": str(WORKSPACE)}


@mcp.tool()
def mesh_box(length: float, width: float, height: float, element_size: float, output_name: str = "box.msh") -> dict[str, Any]:
    """Generate a tetrahedral box mesh and write an MSH 2.2 file."""
    dimensions = [_positive(length, "length"), _positive(width, "width"), _positive(height, "height")]
    size = _positive(element_size, "element_size")
    if any(x > 1e4 for x in dimensions) or size < 1e-6:
        raise ValueError("dimensions or element_size exceed safe limits")
    output = _safe_name(output_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    gmsh = _gmsh()
    gmsh.initialize()
    try:
        gmsh.model.add("box")
        volume = gmsh.model.occ.addBox(*([0.0, 0.0, 0.0] + dimensions))
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [volume], 1)
        gmsh.model.setPhysicalName(3, 1, "volume")
        gmsh.option.setNumber("Mesh.MeshSizeMin", size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(output))
        result = _inventory(gmsh)
    finally:
        gmsh.finalize()
    result.update({"mesh_path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
    return result


@mcp.tool()
def inspect_mesh(mesh_name: str) -> dict[str, Any]:
    """Open a workspace MSH file and return bounded inventory metadata."""
    path = _safe_name(mesh_name)
    if path.suffix.lower() != ".msh" or not path.is_file():
        raise ValueError("mesh_name must identify an existing .msh workspace file")
    gmsh = _gmsh()
    gmsh.initialize()
    try:
        gmsh.open(str(path))
        result = _inventory(gmsh)
    finally:
        gmsh.finalize()
    result.update({"mesh_path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return result


if __name__ == "__main__":
    mcp.run(transport=os.environ.get("GMSH_MCP_TRANSPORT", "stdio"))
