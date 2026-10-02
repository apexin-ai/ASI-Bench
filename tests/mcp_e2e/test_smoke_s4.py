"""e2e_smoke/servers/s4.py without S4: TMM/1D RCWA references, grids, spectrum helpers, checks and defect probes."""
import math

import pytest

from . import support
from .support import StubClient, report_statuses, rpc_json, rpc_text, runner

s4 = support.smoke_module("s4")


def test_tmm_matches_closed_forms():
    n = 3.47
    r, t = s4.tmm("TE", [1.0, n], [], 0.5, 0.0)
    assert r == pytest.approx(((n - 1) / (n + 1)) ** 2, abs=1e-14) and r + t == pytest.approx(1.0, abs=1e-14)
    brewster = math.degrees(math.atan(1.5))
    assert s4.tmm("TM", [1.0, 1.5], [], 1.0, brewster)[0] == pytest.approx(0.0, abs=1e-14)
    assert s4.tmm("TE", [1.0, 1.5], [], 1.0, brewster)[0] > 0.1
    # quarter-wave antireflection coating n1 = sqrt(n0 ns): zero reflectance at the design wavelength
    n1 = math.sqrt(1.5)
    assert s4.tmm("TE", [1.0, n1, 1.5], [1.0 / (4 * n1)], 1.0, 0.0)[0] == pytest.approx(0.0, abs=1e-14)
    # quarter-wave mirror air | (L H)^N | ns: each layer maps the admittance Y -> n^2 / Y,
    # so Y = (nL / nH)^(2N) ns and R = ((1 - Y) / (1 + Y))^2 at the design wavelength
    nh, nl, ns, pairs = 2.3, 1.45, 1.5, 4
    indices = [1.0] + [nl, nh] * pairs + [ns]
    thick = [1.55 / (4 * x) for x in indices[1:-1]]
    y = (nl / nh) ** (2 * pairs) * ns
    assert s4.tmm("TE", indices, thick, 1.55, 0.0)[0] == pytest.approx(((1 - y) / (1 + y)) ** 2, abs=1e-12)


def test_tmm_conserves_energy_and_absorbs_with_positive_k():
    for pol in ("TE", "TM"):
        r, t = s4.tmm(pol, [1.0, 2.3, 1.45, 1.5], [0.17, 0.27], 1.3, 50.0)
        assert r + t == pytest.approx(1.0, abs=1e-13)
        r, t = s4.tmm(pol, [1.0, complex(2.0, 0.2), 1.5], [0.05], 0.8, 30.0)
        assert 0.0 < 1.0 - r - t < 1.0
    with pytest.raises(ValueError):
        s4.tmm("TE", [1.0, 1.5], [0.1], 1.0, 0.0)


def test_refractive_index_accepts_both_notations():
    assert s4.refractive_index({"n": 2.0, "k": 0.2}) == pytest.approx(
        s4.refractive_index({"eps_real": 3.96, "eps_imag": 0.8}), abs=1e-14)
    assert s4.refractive_index({"eps_real": 2.25}) == 1.5


def test_rcwa_uniform_limit_equals_tmm():
    for pol in ("TE", "TM"):
        for theta in (0.0, 40.0):
            layers = [(0.25, 2.3 ** 2, 2.3 ** 2, 0.1, 0.0), (0.3, 1.45 ** 2 + 0.1j, 1.45 ** 2 + 0.1j, 0.2, 0.0)]
            got = s4.rcwa_1d(pol, 1.3, theta, 0.6, 1.0, 1.5, layers, orders=5)
            ref = s4.tmm(pol, [1.0, 2.3, (1.45 ** 2 + 0.1j) ** 0.5, 1.5], [0.25, 0.3], 1.3, theta)
            assert got == pytest.approx(ref, abs=1e-12)


def test_rcwa_grating_conserves_energy_and_rules_converge():
    layer = [(0.3, 1.0, 4.0, 0.2, 0.0)]
    for pol in ("TE", "TM"):
        r, t = s4.rcwa_1d(pol, 0.7, 10.0, 0.8, 1.0, 1.5, layer, orders=15)    # diffracting
        assert r + t == pytest.approx(1.0, abs=1e-10)
    assert s4.rcwa_1d("TE", 1.0, 0.0, 0.8, 1.0, 1.5, layer, 8, "li") == \
        s4.rcwa_1d("TE", 1.0, 0.0, 0.8, 1.0, 1.5, layer, 8, "laurent")
    li = s4.rcwa_1d("TM", 1.0, 0.0, 0.8, 1.0, 1.5, layer, 40, "li")
    few = s4.rcwa_1d("TM", 1.0, 0.0, 0.8, 1.0, 1.5, layer, 4, "laurent")
    many = s4.rcwa_1d("TM", 1.0, 0.0, 0.8, 1.0, 1.5, layer, 80, "laurent")
    assert abs(many[0] - li[0]) < abs(few[0] - li[0]) and abs(many[0] - li[0]) < 2e-3
    with pytest.raises(ValueError):
        s4.rcwa_1d("TM", 1.0, 0.0, 0.8, 1.0, 1.5, layer, 4, "lalanne")


@pytest.mark.parametrize("harmonics,orders", [(1, 0), (5, 1), (9, 1), (13, 2), (21, 2), (49, 4), (81, 5),
                                              (11, None), (51, None), (2, None)])
def test_square_shell_orders(harmonics, orders):
    assert s4.square_shell_orders(harmonics) == orders


def test_smoke_grids_avoid_rayleigh_anomalies():
    indices = (1.0, 1.5)
    grids = [(theta, lams) for _, theta, _, lams in s4.GRATING_CASES]
    grids += [(s4.METAL_CASE[1], s4.METAL_CASE[3]), (0.0, s4.OBLIQUE_RANGE), (s4.GEOMETRY_THETA, s4.OBLIQUE_RANGE)]
    for theta, lams in grids:
        anomalies = s4.rayleigh_wavelengths(s4.GRATING_PERIOD, theta, indices)
        for lam in s4.wavelength_grid({"wavelength_start": lams[0], "wavelength_stop": lams[1],
                                       "wavelength_points": 4}):
            assert min(abs(lam - a) for a in anomalies) >= s4.ANOMALY_CLEARANCE, (theta, lam)
    assert 0.8 in [round(x, 12) for x in s4.rayleigh_wavelengths(0.8, 0.0, indices)]
    for _, _, harmonics, _ in s4.GRATING_CASES:
        assert s4.square_shell_orders(harmonics) is not None


def test_spectrum_helpers():
    arguments = {"wavelength_start": 1.0, "wavelength_stop": 2.0, "wavelength_points": 3}
    good = {"wavelength": [1.0, 1.5, 2.0], "R": [0.1] * 3, "T": [0.9] * 3, "A": [0.0] * 3}
    assert s4.spectrum_problems(good, arguments) == []
    assert s4.spectrum_error(good, [(0.1, 0.9)] * 3) == pytest.approx(0.0, abs=1e-15)
    assert s4.spectrum_error(good, [(0.1, 0.8)] * 3) == pytest.approx(0.1)
    assert s4.spectrum_problems({**good, "wavelength": [1.0, 1.4, 2.0]}, arguments)
    assert s4.spectrum_problems({**good, "R": [0.1, float("nan"), 0.1]}, arguments)
    assert s4.spectrum_problems([1, 2], arguments)
    assert s4.unphysical({"R": [0.0], "T": [1.1], "A": [-0.1]}) == ["T in [1.1, 1.1]", "A in [-0.1, -0.1]"]


def _s4_simulator(*, swap_polarisation=False, rule="laurent"):
    """Fake simulate_stack_spectrum that answers from the smoke's own references."""
    def answer(args):
        args = dict(args)
        if swap_polarisation:
            args["polarization"] = {"TE": "TM", "TM": "TE"}[args.get("polarization", "TE")]
        if any(layer.get("pattern") for layer in args["layers"]):
            ref = s4.grating_reference(args, s4.square_shell_orders(args["n_harmonics"]), rule)
        else:
            ref = s4.stack_reference(args)
        spectrum = {"wavelength": s4.wavelength_grid(args), "R": [r for r, _ in ref], "T": [t for _, t in ref],
                    "A": [1 - r - t for r, t in ref]}
        return rpc_json(spectrum)
    return answer


def test_s4_checks_pass_on_correct_spectra_and_catch_polarisation_or_formulation_errors():
    def run(check, **kwargs):
        report = s4.Report()
        check(s4.Caller(StubClient({"simulate_stack_spectrum": _s4_simulator(**kwargs)}), report), report)
        return report

    assert not run(s4.check_mirror).failed
    assert not run(s4.check_absorber).failed
    assert not run(s4.check_gratings).failed
    swapped = report_statuses(run(s4.check_mirror, swap_polarisation=True))
    assert swapped["simulate_stack_spectrum[quarter-wave mirror TM 30 deg vs TMM]"] == "FAIL"
    assert swapped["simulate_stack_spectrum[quarter-wave mirror TM 0 deg vs TMM]"] == "PASS"   # s = p at normal
    li = report_statuses(run(s4.check_gratings, rule="li"))
    assert li["simulate_stack_spectrum[grating TM 0 deg, 49 harmonics = orders +-4 vs 1D RCWA]"] == "FAIL"
    assert li["simulate_stack_spectrum[grating TE 0 deg, 49 harmonics = orders +-4 vs 1D RCWA]"] == "PASS"


def test_s4_defect_probe_classification():
    report = s4.Report()
    call = s4.Caller(StubClient({"simulate_stack_spectrum": rpc_text("rejected", is_error=True)}), report)
    s4._probe(call, report, "x", {}, lambda s: "accepted")
    accepted = {"wavelength": [1.0], "R": [0.0], "T": [1.1], "A": [-0.1]}
    call = s4.Caller(StubClient({"simulate_stack_spectrum": rpc_json(accepted)}), report)
    s4._probe(call, report, "y", {}, lambda s: f"accepted {s4.unphysical(s)}")
    assert report_statuses(report) == {"simulate_stack_spectrum[x]": "PASS", "simulate_stack_spectrum[y]": "WARN"}


def test_s4_sanity_check_uses_exact_fresnel():
    exact = ((3.47 - 1) / 4.47) ** 2
    def answer(r):
        return rpc_json({"R": r, "T": 1 - r, "A": 0.0, "expected_R": 0.3055, "ok": True})
    for value, status in ((exact, "PASS"), (0.3055, "FAIL")):
        report = s4.Report()
        s4.check_sanity(s4.Caller(StubClient({"check_engine_sanity": answer(value)}), report), report)
        assert set(report_statuses(report).values()) == {status}


def test_s4_server_env_is_minimal_and_applies_config_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    env = runner.server_env({"command": "/m/s4/.venv/bin/python", "env": {"PYTHONPATH": "/m/s4/src"}},
                            tmp_path / "home", tmp_path / "tmp")
    assert env["PYTHONPATH"] == "/m/s4/src" and env["PATH"].startswith("/m/s4/.venv/bin:")
    assert "OPENAI_API_KEY" not in env
