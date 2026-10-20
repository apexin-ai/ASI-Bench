"""L0/L1 checks for the AtomisticSkills MACE server."""
from __future__ import annotations

from ..runner import Session, Smoke, check_rejected
from ..materials import in_band_error, load_model, prediction, payload

TOOLS = (
    "get_info", "load_model", "predict_atomic_features", "predict_structure",
    "relax_structure", "run_md",
)


def run_l1(session: Session) -> None:
    assert session.call is not None
    for tool in ("get_info", "predict_atomic_features", "predict_structure", "relax_structure", "run_md"):
        if tool == "get_info":
            value = payload(session.call("get_info succeeds", tool, {}))
            session.report.add("L1", "get_info succeeds", "PASS" if isinstance(value, dict) and not value.get("error") else "FAIL")
        else:
            check_rejected(session.call, f"{tool} rejects missing structure", tool, {}, in_band=in_band_error)
    if session.args.load_model:
        if load_model(session):
            prediction(session, session.args.model_name)
    else:
        session.report.add("L1", "load_model precondition", "WARN",
                           "skipped checkpoint loading; rerun with --load-model")


def arguments(parser) -> None:
    parser.add_argument("--load-model", action="store_true",
                        help="load the configured MACE checkpoint (may download weights)")
    parser.add_argument("--model-name", default="MACE-OMAT-0-small")


SMOKE = Smoke(server="mace", run_l1=run_l1, call_timeout=600.0, packages=("mace-torch", "mcp", "torch"),
              add_arguments=arguments)
