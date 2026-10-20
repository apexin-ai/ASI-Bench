"""L0/L1 checks for the AtomisticSkills MACE server."""
from __future__ import annotations

from ..runner import Session, Smoke

TOOLS = (
    "get_info", "load_model", "predict_atomic_features", "predict_structure",
    "relax_structure", "run_md",
)


def run_l1(session: Session) -> None:
    assert session.call is not None
    for tool in ("get_info", "predict_atomic_features", "predict_structure", "relax_structure", "run_md"):
        result = session.call(f"{tool} responds", tool, {}, allow_error=True)
        session.report.add("L1", f"{tool} responds", "PASS" if result is not None else "FAIL")
    if session.args.load_model:
        result = session.call("load_model responds", "load_model",
                              {"model_name": session.args.model_name, "device": "cpu"}, allow_error=True,
                              timeout=session.smoke.call_timeout)
        session.report.add("L1", "load_model responds", "PASS" if result is not None else "FAIL")
    else:
        session.report.add("L1", "load_model precondition", "WARN",
                           "skipped checkpoint loading; rerun with --load-model")


def arguments(parser) -> None:
    parser.add_argument("--load-model", action="store_true",
                        help="load the configured MACE checkpoint (may download weights)")
    parser.add_argument("--model-name", default="MACE-OMAT-0-small")


SMOKE = Smoke(server="mace", run_l1=run_l1, call_timeout=120.0, packages=("mace-torch", "mcp"),
              add_arguments=arguments)
