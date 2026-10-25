"""L0/L1 checks for the constrained ASI-Bench Gmsh adapter."""
from __future__ import annotations

from pathlib import Path

from ..materials import in_band_error, payload
from ..runner import Smoke, check_rejected


def run_l1(session) -> None:
    info = payload(session.call("get_info succeeds", "get_info", {}))
    session.report.add("L1", "get_info succeeds", "PASS" if isinstance(info, dict) and not info.get("error") else "FAIL", str(info)[:300])
    check_rejected(session.call, "mesh_box rejects negative dimensions", "mesh_box",
                   {"length": -1, "width": 1, "height": 1, "element_size": 0.2}, in_band=in_band_error)
    result = payload(session.call("mesh_box creates MSH", "mesh_box", {
        "length": 1.0, "width": 1.0, "height": 1.0, "element_size": 0.5, "output_name": "smoke.msh"}))
    path = session.cwd / "smoke.msh"
    ok = isinstance(result, dict) and result.get("elements", 0) > 0 and path.is_file() and path.stat().st_size > 0
    session.report.add("L1", "mesh_box creates MSH", "PASS" if ok else "FAIL", str(result)[:1000])
    if ok:
        inspected = payload(session.call("inspect_mesh agrees", "inspect_mesh", {"mesh_name": "smoke.msh"}))
        same = isinstance(inspected, dict) and inspected.get("nodes") == result.get("nodes") and inspected.get("elements") == result.get("elements")
        session.report.add("L1", "inspect_mesh agrees", "PASS" if same else "FAIL", str(inspected)[:1000])


SMOKE = Smoke(server="gmsh", run_l1=run_l1, packages=("gmsh", "mcp", "numpy"), expected_cwd_files=("smoke.msh",))
