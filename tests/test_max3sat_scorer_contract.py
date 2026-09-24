"""Error classification for the public Max-3-SAT scorer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_scorer_module():
    path = Path("tasks/computer_science/max3sat_assignment_optimization/custom_scorer.py")
    name = "max3sat_scorer_contract_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return name, module


def test_missing_solver_is_submission_failure_not_evaluator_error(tmp_path):
    name, module = _load_scorer_module()
    try:
        module._evaluation = lambda *_args, **_kwargs: module._EvaluationRecord(
            False,
            "solver.py is missing",
            False,
            3,
            1,
            0,
            1,
            0,
            "",
            None,
        )
        detail = module.Max3SATSolverValidityScorer().score(
            tmp_path, tmp_path, {}
        )
    finally:
        sys.modules.pop(name, None)

    assert detail.details["scorer_internal_error"] is False
    assert detail.details["failure_kind"] == "submission_error"
