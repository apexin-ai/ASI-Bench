"""L0/L1 checks for the AtomisticSkills MatterGen server."""
from __future__ import annotations

from ..runner import Session, Smoke
from ..materials import payload


def run_l1(session: Session) -> None:
    assert session.call is not None
    if session.args.generate:
        result = payload(session.call("generate_structures produces CIFs", "generate_structures", {
            "num_structures": session.args.num_structures,
            "batch_size": session.args.num_structures,
            "chemical_system": session.args.chemical_system,
            "device": "cpu",
            "output_dir": str(session.cwd / "generated"),
        }, timeout=session.smoke.call_timeout))
        ok = isinstance(result, dict) and not result.get("error") and result.get("num_generated") == session.args.num_structures
        if ok:
            try:
                from pymatgen.core import Structure
                from pathlib import Path

                files = result["structures"]
                ok = len(files) == session.args.num_structures and all(
                    Path(p).is_file() and len(Structure.from_file(p)) > 0 for p in files)
            except Exception as exc:
                ok = False
                result = {**result, "validation_error": str(exc)}
        session.report.add("L1", "generate_structures produces CIFs", "PASS" if ok else "FAIL", str(result)[:1500])
    else:
        session.report.add("L1", "generate_structures precondition", "WARN",
                           "skipped generative checkpoint; rerun with --generate")


def arguments(parser) -> None:
    parser.add_argument("--generate", action="store_true",
                        help="run MatterGen and write generated structures")
    parser.add_argument("--num-structures", type=int, default=1)
    parser.add_argument("--chemical-system", default="Li-Zr-Cl")


SMOKE = Smoke(server="mattergen", run_l1=run_l1, call_timeout=3600.0,
              packages=("mattergen", "mcp", "torch", "torch-scatter"), add_arguments=arguments,
              expected_cwd_files=("generated",))
