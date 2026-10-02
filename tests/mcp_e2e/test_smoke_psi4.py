"""e2e_smoke/servers/psi4.py without psi4: geometry parsing, frequency/state classification, tool payload checks."""
import json
import math

import pytest

from . import support
from .support import StubClient, report_statuses, runner

p4 = support.smoke_module("psi4")


def test_psi4_parse_geometry_accepts_psi4_xyz_and_bare_lines():
    psi4_out = "0 1\n O    0.0  0.0  0.07\n H    0.0  0.75 -0.56\n H    0.0 -0.75 -0.56\n"
    xyz = "3\nwater\nO 0 0 0.07\nH 0 0.75 -0.56\nH 0 -0.75 -0.56"
    assert p4.parse_geometry(psi4_out) == p4.parse_geometry(xyz)
    assert [s for s, _ in p4.parse_geometry(p4.H2O_START)] == ["O", "H", "H"]
    with pytest.raises(ValueError):
        p4.parse_geometry("0 1\n")


def test_psi4_classify_frequencies():
    ref = [-1081.6, 1866.4, 1866.4, 4023.5, 4363.4, 4363.4]
    assert p4.classify_frequencies(ref, ref, 1)[0] == "PASS"
    dropped_i = [abs(f) for f in ref]
    status, detail = p4.classify_frequencies(dropped_i, ref, 0)
    assert status == "WARN" and "-1081.6" in detail
    assert p4.classify_frequencies(ref, ref, 0)[0] == "FAIL"                 # right values, wrong count
    assert p4.classify_frequencies([f + 5 for f in dropped_i], ref, 0)[0] == "FAIL"
    minimum = [2170.28, 4139.70, 4390.74]
    assert p4.classify_frequencies(minimum, minimum, 0)[0] == "PASS"
    assert p4.classify_frequencies(minimum[:2], minimum, 0)[0] == "FAIL"


def test_psi4_classify_alternative_and_value_of():
    assert p4.classify_alternative(1.0, 1.0, 2.0, 1e-6) == "PASS"
    assert p4.classify_alternative(2.0, 1.0, 2.0, 1e-6) == "WARN"
    assert p4.classify_alternative(3.0, 1.0, 2.0, 1e-6) == "FAIL"
    assert p4.classify_alternative(None, 1.0, 2.0, 1e-6) == "FAIL"
    assert p4.value_of({"value": -1.5, "unit": "Hartree"}) == -1.5
    assert p4.value_of(0.3) == 0.3 and p4.value_of(None) is None and p4.value_of(True) is None


def test_psi4_compare_states():
    server = [{"excitation_energy": {"value": 7.0001, "unit": "eV"}, "oscillator_strength": 0.0157},
              {"excitation_energy": {"value": 9.2, "unit": "eV"}, "oscillator_strength": 0.0}]
    de, df = p4.compare_states(server, [(7.0, 0.0157), (9.2, 0.0)])
    assert de == pytest.approx(1e-4) and df == 0.0
    assert p4.compare_states([{"excitation_energy": None, "oscillator_strength": 0.1}], [(1.0, 0.1)])[0] == math.inf


def _p4json(obj):
    return {"result": {"content": [{"type": "text", "text": json.dumps(obj)}], "structuredContent": obj,
                       "isError": False}}


class _FakeRef:
    """Psi4Ref stand-in with fixed reference numbers."""

    def __init__(self, ground_minimum=-75.3231):
        self.ground_minimum = ground_minimum

    def excited_total(self, atoms, *args):
        return -75.3231, -74.9473

    def excited_gradient_norm(self, *args):
        return 1e-4

    def optimize(self, *args, **kwargs):
        return self.ground_minimum, p4.parse_geometry(_WATER)


_WATER = "O 0 0 0.076\nH 0 0.771 -0.603\nH 0 -0.771 -0.603"


@pytest.mark.parametrize("energy,exc,ground_min,expected", [
    (-74.9473, 10.2264, -75.3231, ("PASS", None)),                        # a real S1 minimum
    (-75.3231, 11.0565, -75.3231, ("WARN", "WARN")),                      # upstream: ground-state minimum
    (-75.3231, 10.2264, -75.0, ("FAIL", None)),                           # E(S0) but not the S0 minimum
    (-70.0, 10.2264, -75.3231, ("FAIL", None)),
])
def test_psi4_excited_state_opt_classification(energy, exc, ground_min, expected):
    result = {"ok": True, "result": {"final_total_energy": {"value": energy, "unit": "Hartree"},
                                     "excitation_energy_at_opt": {"value": exc, "unit": "eV"},
                                     "optimized_geometry_xyz": "0 1\n" + _WATER, "converged": True}}
    report = p4.Report()
    call = p4.Caller(StubClient({"optimize_excited_state": _p4json(result)}), report)
    p4.check_excited_state_opt(call, report, _FakeRef(ground_min), p4.parse_geometry(_WATER))
    statuses = report_statuses(report)
    main = next(v for k, v in statuses.items() if not k.endswith("[excitation_energy_at_opt]"))
    side = next((v for k, v in statuses.items() if k.endswith("[excitation_energy_at_opt]")), None)
    assert (main, side) == expected
    assert call.client.calls[0][1]["memory_gb"] == 1


def test_psi4_tool_payload_fails_on_in_band_error_for_valid_requests():
    report = p4.Report()
    bad = {"ok": False, "error_code": "PSI4_INTERNAL_ERROR", "details": "No module named 'psi4'"}
    call = p4.Caller(StubClient({"single_point": _p4json(bad)}), report)
    assert p4.tool_payload(call, report, "sp", "single_point", {}) is None
    assert report_statuses(report) == {"sp": "FAIL"} and "psi4" in report.checks[0]["detail"]


def test_psi4_server_env_is_minimal_and_applies_config_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    env = runner.server_env({"command": "/m/psi4/.venv/bin/python", "env": {"PYTHONPATH": "/m/psi4"}},
                            tmp_path / "home", tmp_path / "tmp")
    assert env["PYTHONPATH"] == "/m/psi4" and env["TMPDIR"] == str(tmp_path / "tmp")
    assert "ANTHROPIC_API_KEY" not in env
