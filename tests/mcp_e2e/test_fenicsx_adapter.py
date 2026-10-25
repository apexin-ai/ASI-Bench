"""Offline checks for the FEniCSx image and adapter contract."""
from pathlib import Path


ROOT = Path(__file__).parents[2] / "scripts" / "mcp" / "e2e"


def test_fenicsx_adapter_is_lazy_and_image_pinned():
    source = (ROOT / "fenicsx_server.py").read_text()
    dockerfile = (ROOT / "fenicsx.Dockerfile").read_text()
    assert "def solve_poisson_manufactured" in source
    assert "def _dolfinx" in source
    assert "dolfinx/dolfinx:v0.11.0@sha256:" in dockerfile
    assert "mcp==1.30.0" in dockerfile


def test_fenicsx_tool_limits_are_explicit():
    source = (ROOT / "fenicsx_server.py").read_text()
    assert "1 <= nx <= 256" in source
    assert "1 <= ny <= 256" in source
    assert "1 <= degree <= 2" in source
    assert "is_relative_to(WORKSPACE)" in source


def test_gmsh_image_declares_native_glu_runtime():
    dockerfile = (ROOT / "gmsh.Dockerfile").read_text()
    assert "libglu1-mesa" in dockerfile
    assert "gmsh==4.15.2" in dockerfile
