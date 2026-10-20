"""L0/L1 checks for the AtomisticSkills MatterGen server."""
from __future__ import annotations

from ..runner import Session, Smoke


def run_l1(session: Session) -> None:
    assert session.call is not None
    if session.args.generate:
        result = session.call("generate_structures responds", "generate_structures", {
            "num_structures": session.args.num_structures,
            "chemical_system": session.args.chemical_system,
            "device": "cpu",
        }, allow_error=True, timeout=session.smoke.call_timeout)
        session.report.add("L1", "generate_structures responds", "PASS" if result is not None else "FAIL")
    else:
        session.report.add("L1", "generate_structures precondition", "WARN",
                           "skipped generative checkpoint; rerun with --generate")


def arguments(parser) -> None:
    parser.add_argument("--generate", action="store_true",
                        help="run MatterGen and write generated structures")
    parser.add_argument("--num-structures", type=int, default=1)
    parser.add_argument("--chemical-system", default="Li-Zr-Cl")


SMOKE = Smoke(server="mattergen", run_l1=run_l1, call_timeout=900.0,
              packages=("mattergen", "mcp"), add_arguments=arguments)
