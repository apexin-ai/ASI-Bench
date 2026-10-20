"""L0/L1 checks for the pinned ASI-Bench MatterSim adapter."""
from __future__ import annotations

from ..runner import Session, Smoke

_INVALID = {"symbols": [], "positions": []}


def run_l1(session: Session) -> None:
    assert session.call is not None
    info = session.call("get_info responds", "get_info", {})
    session.report.add("L1", "get_info responds", "PASS" if info is not None else "FAIL")
    for tool in ("predict_structure", "relax_structure"):
        result = session.call(f"{tool} rejects invalid structure", tool,
                              {"structure_data": _INVALID}, allow_error=True)
        session.report.add("L1", f"{tool} rejects invalid structure",
                           "PASS" if result is not None else "FAIL")


SMOKE = Smoke(server="mattersim", run_l1=run_l1, call_timeout=300.0,
              packages=("mattersim", "mcp"))
