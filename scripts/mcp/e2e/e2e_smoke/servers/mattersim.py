"""L0/L1 checks for the pinned ASI-Bench MatterSim adapter."""
from __future__ import annotations

from ..runner import Session, Smoke, check_rejected
from ..materials import in_band_error, prediction, payload

_INVALID = {"symbols": [], "positions": []}


def run_l1(session: Session) -> None:
    assert session.call is not None
    info = payload(session.call("get_info succeeds", "get_info", {}))
    session.report.add("L1", "get_info succeeds", "PASS" if isinstance(info, dict) and not info.get("error") else "FAIL")
    for tool in ("predict_structure", "relax_structure"):
        check_rejected(session.call, f"{tool} rejects invalid structure", tool,
                       {"structure_data": _INVALID}, in_band=in_band_error)
    if session.args.load_model:
        prediction(session, info["model"])
    else:
        session.report.add("L1", "prediction precondition", "WARN", "rerun with --load-model for real inference")


def arguments(parser):
    parser.add_argument("--load-model", action="store_true", help="run real inference and an independent ASE reference")


SMOKE = Smoke(server="mattersim", run_l1=run_l1, call_timeout=600.0,
              packages=("mattersim", "mcp", "torch"), add_arguments=arguments)
