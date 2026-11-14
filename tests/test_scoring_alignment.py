"""Scientific termination and finite-trajectory scoring contracts."""

import json

import numpy as np
import pytest

from tasks.materials.ising_domain_growth.custom_scorer import _growth_exponent_score
from tasks.math.homotopy_poly_roots.custom_scorer import HomotopyPathDiagnosticsScorer


@pytest.mark.parametrize(
    "exponent, expected",
    [
        (0.09, 0),
        (0.14, 0.5),
        (0.19, 1),
        (0.3, 1),
        (0.5, 1),
        (0.55, 0.5),
        (0.61, 0),
        (float("nan"), 0),
    ],
)
def test_finite_trajectory_acceptance(exponent, expected):
    assert _growth_exponent_score(exponent) == pytest.approx(expected)


@pytest.mark.parametrize(
    "case, valid",
    [
        ("valid", 2),
        ("false_divergence", 1),
        ("fake_finite_residual", 1),
        ("invalid_sentinel", 1),
    ],
)
def test_diagnostics_validate_each_termination(tmp_path, case, valid):
    pred = tmp_path / "prediction"
    ref = tmp_path / "reference"
    (pred / "data").mkdir(parents=True)
    ref.mkdir()
    starts = np.array([[1, 1], [-1, 1]], dtype=complex)
    roots = np.array([[0, 0], [1e8 + 1e8j, 1e8 + 1e8j]])
    status = np.array([0, 1])
    trusted = np.array([0, 1])
    if case == "false_divergence":
        trusted[1] = 0
    if case == "fake_finite_residual":
        roots[0] = [1, 1]
    if case == "invalid_sentinel":
        roots[1] = [100, 100]
    system = {
        "d1": 2,
        "d2": 1,
        "polynomials": [
            {"terms": [{"coeff_re": 1, "coeff_im": 0, "exponents": [1, 0]}]},
            {"terms": [{"coeff_re": 1, "coeff_im": 0, "exponents": [0, 1]}]},
        ],
    }
    (pred / "data" / "system.json").write_text(json.dumps(system))
    for name, array in [
        ("path_log", starts),
        ("roots", roots),
        ("residuals", np.array([0.0, 1e9])),
        ("path_status", status),
    ]:
        np.save(pred / (name + ".npy"), array)
    np.save(pred / "data" / "start_roots.npy", starts)
    np.save(ref / "path_status_ref.npy", trusted)
    result = HomotopyPathDiagnosticsScorer().score(pred, ref, {"weight": 5})
    assert result.details["converged_paths"] == valid
    assert result.score == pytest.approx(5 if valid == 2 else 4.375)
