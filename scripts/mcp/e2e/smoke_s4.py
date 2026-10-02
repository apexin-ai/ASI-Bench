"""Direct (agent-free) E2E smoke test for the pinned mcp-s4-rcwa (S4 RCWA) tools.

Run with the server's own virtualenv (it provides numpy for the references)::

    ~/mcp/s4/.venv/bin/python scripts/mcp/e2e/smoke_s4.py --config ~/mcp/s4.mcp.json

No network is needed. The references are computed here, in this process, with
code that shares nothing with S4 or the server:

* a coherent transfer-matrix method (TMM) for unpatterned stacks — exact, so
  any harmonic count must reproduce it to rounding;
* a 1D rigorous coupled-wave analysis (RCWA, enhanced transmittance matrix)
  for lamellar gratings. S4's default formulation uses Laurent's rule for both
  polarisations and keeps the reciprocal-lattice vectors inside a circle; with
  a complete shell of the square lattice (21, 49, 81 harmonics ...) a grating
  that is uniform along y couples only the |m| <= M orders on the x axis, so
  the 1D RCWA with Laurent's rule and the same M must agree to rounding.
  Li's (inverse) rule with many orders gives the converged answer, used to
  report how far S4's default formulation is from convergence.

Levels reported:
  L0  initialize + tools/list (stable, matches manifest), launch environment,
      the vendored libS4.so is the upstream binary
  L1  real tools/call of both tools: Fresnel self-test; unpatterned stacks
      (README example, quarter-wave mirror at 0/30/60 deg in TE and TM, an
      absorbing film, both material notations) against the TMM; gratings
      against the 1D RCWA (TE/TM, oblique, diffracting, absorbing ridge);
      2D-pattern symmetry and translation invariance; the PNG plot;
      repeatability; validation errors; defect probes

Upstream defects that do not make a correctly used tool wrong (silently
accepted inputs that give unphysical or misleading spectra, in-band errors,
stdout chatter) are WARN; wrong numbers for correct inputs are FAIL.

The server runs from a temporary cwd, HOME and TMPDIR with a minimal
environment and no operator credentials.
"""
from __future__ import annotations

import argparse
import base64
import cmath
import datetime as dt
import hashlib
import json
import math
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from smoke_common import Caller, MCPError, Report, StdioMCP, package_versions, text_of  # noqa: E402

# sha256 of src/mcp_s4_rcwa/s4lib/libS4.so at the pinned revision (x86-64, GCC 13.3).
UPSTREAM_LIBS4_SHA256 = "f4097479f835fd429b1865264c29d8338ddce5fb3a6612c78ecba782c25e4d98"
LIBS4 = Path("src/mcp_s4_rcwa/s4lib/libS4.so")

EXACT_TOL = 1e-9          # TMM / matched-truncation RCWA agree to ~1e-14; leave room for other BLAS builds
SANITY_N = 3.47           # check_engine_sanity: air / n=3.47 at normal incidence
CONVERGED_NH = 201        # S4 harmonics for the convergence report
CONVERGED_ORDERS = 60     # 1D RCWA orders (Li's rule) for the converged reference
CONVERGENCE_WARN = 5e-3   # |S4 - converged| above this at CONVERGED_NH is worth a WARN

# Upstream README example (wavelength_points reduced from 200).
README_EXAMPLE = {
    "period": 1.0,
    "materials": {"Si": {"n": 3.47}, "SiO2": {"n": 1.44}, "air": {"n": 1.0}},
    "layers": [{"name": "substrate", "thickness": 0.5, "material": "Si"},
               {"name": "spacer", "thickness": 0.3, "material": "SiO2"},
               {"name": "superstrate", "thickness": 0.5, "material": "air"}],
    "incidence_layer": "superstrate", "substrate_layer": "substrate",
    "polarization": "TE", "wavelength_start": 1.4, "wavelength_stop": 1.8, "wavelength_points": 41,
}

# Quarter-wave mirror centred at 1.55 um: glass | (H L) x 4 | air.
DBR_CENTER = 1.55
DBR_MATERIALS = {"glass": {"n": 1.5}, "H": {"n": 2.3}, "L": {"n": 1.45}, "air": {"n": 1.0}}
DBR_LAYERS = ([{"name": "substrate", "thickness": 1.0, "material": "glass"}]
              + [layer for i in range(4) for layer in (
                  {"name": f"H{i}", "thickness": DBR_CENTER / 4 / 2.3, "material": "H"},
                  {"name": f"L{i}", "thickness": DBR_CENTER / 4 / 1.45, "material": "L"})]
              + [{"name": "air", "thickness": 1.0, "material": "air"}])
DBR_ANGLES = (0.0, 30.0, 60.0)

# Absorbing film (n + ik, Im(eps) > 0 is loss in S4) on glass, and the same in eps notation.
ABSORBER_NK = {"n": 2.0, "k": 0.2}
ABSORBER_EPS = {"eps_real": 2.0 ** 2 - 0.2 ** 2, "eps_imag": 2 * 2.0 * 0.2}

# Lamellar grating: glass | ridges (n=2, 0.4 um wide, 0.3 um high) in air | air; period 0.8 um.
GRATING_PERIOD = 0.8
GRATING_HALFWIDTH = 0.2
GRATING_HEIGHT = 0.3
GRATING_MATERIALS = {"glass": {"n": 1.5}, "air": {"n": 1.0}, "ridge": {"n": 2.0}}
METAL_RIDGE = {"n": 0.5, "k": 3.0}
# (polarisation, theta, harmonics, wavelength range): wavelengths below the period
# diffract; every grid point stays clear of the Rayleigh anomalies (an order at
# grazing emergence), where RCWA results are ill-conditioned.
NORMAL_RANGE, OBLIQUE_RANGE = (0.65, 1.0), (0.95, 1.25)
GRATING_CASES = (("TE", 0.0, 49, NORMAL_RANGE), ("TM", 0.0, 49, NORMAL_RANGE),
                 ("TE", 20.0, 21, OBLIQUE_RANGE), ("TM", 20.0, 81, OBLIQUE_RANGE))
METAL_CASE = ("TM", 10.0, 49, OBLIQUE_RANGE)
GEOMETRY_THETA = 15.0
ANOMALY_CLEARANCE = 0.01   # um


# --------------------------------------------------------------------------
# Pure helpers and independent references (numpy only; unit-tested offline)
# --------------------------------------------------------------------------

def refractive_index(material: dict) -> complex:
    """Complex index n + ik of a server material spec ({"n","k"} or {"eps_real","eps_imag"})."""
    if "n" in material:
        return complex(material["n"], material.get("k", 0.0))
    root = cmath.sqrt(complex(material.get("eps_real", 1.0), material.get("eps_imag", 0.0)))
    return root if root.imag >= 0 else -root


def _forward_cos(n: complex, s0: complex) -> complex:
    c = cmath.sqrt(1 - (s0 / n) ** 2)
    kz = n * c
    if kz.imag < -1e-15 or (abs(kz.imag) <= 1e-15 and kz.real < 0):
        c = -c
    return c


def tmm(pol: str, indices: list[complex], thicknesses: list[float], wavelength: float,
        theta_deg: float) -> tuple[float, float]:
    """Coherent transfer-matrix R, T of a planar stack.

    ``indices`` run from the incidence medium (first) to the substrate (last),
    both semi-infinite and non-absorbing; ``thicknesses`` are the inner layers
    (um). exp(-iwt) convention, Im(n) > 0 absorbs. TE = s, TM = p.
    """
    n = [complex(x) for x in indices]
    if len(thicknesses) != len(n) - 2:
        raise ValueError("need one thickness per inner layer")
    s0 = n[0] * math.sin(math.radians(theta_deg))
    cos = [_forward_cos(nj, s0) for nj in n]

    def interface(i: int, j: int) -> tuple[complex, complex]:
        ni, nj, ci, cj = n[i], n[j], cos[i], cos[j]
        if pol == "TE":
            return (ni * ci - nj * cj) / (ni * ci + nj * cj), 2 * ni * ci / (ni * ci + nj * cj)
        return (nj * ci - ni * cj) / (nj * ci + ni * cj), 2 * ni * ci / (nj * ci + ni * cj)

    m = np.eye(2, dtype=complex)
    for k in range(len(n) - 1):
        r, t = interface(k, k + 1)
        if k:
            delta = 2 * math.pi * n[k] * cos[k] * thicknesses[k - 1] / wavelength
            m = m @ np.array([[cmath.exp(-1j * delta), 0], [0, cmath.exp(1j * delta)]])
        m = m @ (np.array([[1, r], [r, 1]], dtype=complex) / t)
    r, t = m[1, 0] / m[0, 0], 1 / m[0, 0]
    if pol == "TE":
        ratio = (n[-1] * cos[-1]).real / (n[0] * cos[0]).real
    else:
        ratio = (n[-1].conjugate() * cos[-1]).real / (n[0].conjugate() * cos[0]).real
    return float(abs(r) ** 2), float(abs(t) ** 2 * ratio)


def stack_reference(arguments: dict) -> list[tuple[float, float]]:
    """TMM (R, T) per wavelength for unpatterned simulate_stack_spectrum arguments.

    Layers are bottom (substrate) -> top (incidence medium) as the tool expects;
    the first and last layers are the semi-infinite media.
    """
    layers = arguments["layers"]
    if any(layer.get("pattern") for layer in layers):
        raise ValueError("stack_reference handles unpatterned stacks only")
    mats = arguments["materials"]
    top_down = list(reversed(layers))
    indices = [refractive_index(mats[layer["material"]]) for layer in top_down]
    thicknesses = [float(layer["thickness"]) for layer in top_down[1:-1]]
    return [tmm(arguments.get("polarization", "TE"), indices, thicknesses, lam, arguments.get("theta_deg", 0.0))
            for lam in wavelength_grid(arguments)]


def wavelength_grid(arguments: dict) -> list[float]:
    return np.linspace(arguments["wavelength_start"], arguments["wavelength_stop"],
                       arguments.get("wavelength_points", 200)).tolist()


def _eps_fourier(eps_bg: complex, eps_ridge: complex, halfwidth: float, center: float,
                 period: float, orders: int) -> np.ndarray:
    """Fourier coefficients eps_n, n = -2M..2M, of a lamellar profile (ridge |x - center| < halfwidth)."""
    n = np.arange(-2 * orders, 2 * orders + 1)
    fill = 2 * halfwidth / period
    coeff = (eps_ridge - eps_bg) * fill * np.sinc(n * fill) * np.exp(-2j * np.pi * n * center / period)
    coeff[2 * orders] += eps_bg
    return coeff


def _toeplitz(coeff: np.ndarray, orders: int) -> np.ndarray:
    idx = np.arange(2 * orders + 1)
    return coeff[(idx[:, None] - idx[None, :]) + 2 * orders]


def _decaying_root(values: np.ndarray) -> np.ndarray:
    q = np.sqrt(values.astype(complex))
    return np.where(q.real < 0, -q, q)


def rcwa_1d(pol: str, wavelength: float, theta_deg: float, period: float, n_incidence: complex,
            n_substrate: complex, layers: list[tuple], orders: int, rule: str = "li") -> tuple[float, float]:
    """Total R, T of lamellar gratings (grooves along y, plane of incidence xz).

    ``layers`` run top -> bottom as (thickness, eps_background, eps_ridge,
    halfwidth, center), permittivities in the exp(-iwt) convention.
    ``orders`` = M keeps diffraction orders -M..M. ``rule`` picks the TM
    factorisation: "li" (inverse rule, fast convergence) or "laurent" (S4's
    default). TE is the same for both. Enhanced transmittance matrix
    (Moharam et al. 1995), internally exp(+jwt).
    """
    if pol not in ("TE", "TM") or rule not in ("li", "laurent"):
        raise ValueError("pol must be TE/TM and rule li/laurent")
    k0 = 2 * math.pi / wavelength
    m = np.arange(-orders, orders + 1)
    kx = (complex(n_incidence).real * math.sin(math.radians(theta_deg)) - m * wavelength / period).astype(complex)
    kxm = np.diag(kx)
    eye = np.eye(len(m), dtype=complex)
    e_inc, e_sub = np.conj(complex(n_incidence) ** 2), np.conj(complex(n_substrate) ** 2)
    q_inc, q_sub = _decaying_root(kx ** 2 - e_inc), _decaying_root(kx ** 2 - e_sub)
    tm = pol == "TM"
    f, g = eye.copy(), -np.diag(q_sub) / (e_sub if tm else 1.0)
    transfers = []
    for thickness, eps_bg, eps_ridge, halfwidth, center in reversed(layers):
        eps_bg, eps_ridge = np.conj(complex(eps_bg)), np.conj(complex(eps_ridge))
        e_toe = _toeplitz(_eps_fourier(eps_bg, eps_ridge, halfwidth, center, period, orders), orders)
        if not tm:
            a_mat, p_mat = kxm @ kxm - e_toe, eye
        else:
            e_inv = np.linalg.inv(e_toe)
            p_mat = (_toeplitz(_eps_fourier(1 / eps_bg, 1 / eps_ridge, halfwidth, center, period, orders), orders)
                     if rule == "li" else e_inv)
            a_mat = np.linalg.solve(p_mat, kxm @ e_inv @ kxm - eye)
        w2, w = np.linalg.eig(a_mat)
        q = _decaying_root(w2)
        v = p_mat @ w @ np.diag(q)
        x = np.diag(np.exp(-q * k0 * thickness))
        ab = np.linalg.solve(np.block([[w, w], [-v, v]]), np.vstack([f, g]))
        a_inv = np.linalg.inv(ab[:len(m)])
        xbax = x @ ab[len(m):] @ a_inv @ x
        f, g = w @ (eye + xbax), -v @ (eye - xbax)
        transfers.append(a_inv @ x)
    delta = (m == 0).astype(complex)
    y_inc = np.diag(q_inc) / (e_inc if tm else 1.0)
    t = np.linalg.solve(y_inc @ f - g, 2 * y_inc @ delta)
    r = f @ t - delta
    for transfer in reversed(transfers):
        t = transfer @ t
    kz_inc, kz_sub = -1j * q_inc, -1j * q_sub
    norm_inc, norm_sub = (e_inc, e_sub) if tm else (1.0, 1.0)
    base = np.real(kz_inc[orders] / norm_inc)
    big_r = np.abs(r) ** 2 * np.real(kz_inc / norm_inc) / base
    big_t = np.abs(t) ** 2 * np.real(kz_sub / norm_sub) / base
    return float(big_r.sum()), float(big_t.sum())


def square_shell_orders(harmonics: int) -> int | None:
    """M such that S4's circular truncation with ``harmonics`` keeps x-axis orders -M..M.

    Only defined (non-None) when ``harmonics`` is exactly a complete shell of
    the square lattice, i.e. the number of (i, j) with i^2 + j^2 <= r^2.
    """
    radius_sq = 0
    while True:
        r = math.isqrt(radius_sq)
        count = sum(1 for i in range(-r, r + 1) for j in range(-r, r + 1) if i * i + j * j <= radius_sq)
        if count == harmonics:
            return r
        if count > harmonics:
            return None
        radius_sq += 1


def rayleigh_wavelengths(period: float, theta_deg: float, indices: tuple[float, ...],
                         max_order: int = 4) -> list[float]:
    """Wavelengths at which diffraction order m emerges at grazing angle in a medium of index n.

    In-plane incidence from a medium of index n0 = indices[0]:
    |n0 sin(theta) - m lambda / period| = n.
    """
    s = indices[0] * math.sin(math.radians(theta_deg))
    out = []
    for n in indices:
        for m in range(1, max_order + 1):
            for sign in (1, -1):
                lam = period * (n + sign * s) / m
                if lam > 0:
                    out.append(lam)
    return sorted(out)


def grating_arguments(pol: str, theta: float, harmonics: int, wavelengths: tuple[float, float],
                      points: int = 4, ridge: dict | None = None, halfwidths: list | None = None,
                      center: list | None = None) -> dict:
    pattern = {"material": "ridge", "halfwidths": halfwidths or [GRATING_HALFWIDTH, GRATING_PERIOD / 2]}
    if center is not None:
        pattern["center"] = center
    return {
        "period": GRATING_PERIOD,
        "materials": {**GRATING_MATERIALS, **({"ridge": ridge} if ridge else {})},
        "layers": [{"name": "substrate", "thickness": 1.0, "material": "glass"},
                   {"name": "grating", "thickness": GRATING_HEIGHT, "material": "air", "pattern": pattern},
                   {"name": "superstrate", "thickness": 1.0, "material": "air"}],
        "incidence_layer": "superstrate", "substrate_layer": "substrate",
        "polarization": pol, "theta_deg": theta, "n_harmonics": harmonics,
        "wavelength_start": wavelengths[0], "wavelength_stop": wavelengths[1], "wavelength_points": points,
        "include_plot": False,
    }


def grating_reference(arguments: dict, orders: int, rule: str) -> list[tuple[float, float]]:
    mats = arguments["materials"]
    grating = arguments["layers"][1]
    eps_bg = refractive_index(mats[grating["material"]]) ** 2
    eps_ridge = refractive_index(mats[grating["pattern"]["material"]]) ** 2
    center = grating["pattern"].get("center", [0.0, 0.0])[0]
    layer = (grating["thickness"], eps_bg, eps_ridge, grating["pattern"]["halfwidths"][0], center)
    n_inc = refractive_index(mats[arguments["layers"][-1]["material"]])
    n_sub = refractive_index(mats[arguments["layers"][0]["material"]])
    return [rcwa_1d(arguments["polarization"], lam, arguments["theta_deg"], arguments["period"], n_inc, n_sub,
                    [layer], orders, rule) for lam in wavelength_grid(arguments)]


def spectrum_error(spectrum: dict, reference: list[tuple[float, float]]) -> float:
    """Largest |R - R_ref|, |T - T_ref| and |A - (1 - R_ref - T_ref)| over the sweep."""
    worst = 0.0
    for i, (r_ref, t_ref) in enumerate(reference):
        worst = max(worst, abs(spectrum["R"][i] - r_ref), abs(spectrum["T"][i] - t_ref),
                    abs(spectrum["A"][i] - (1.0 - r_ref - t_ref)))
    return worst


def spectrum_problems(spectrum, arguments: dict) -> list[str]:
    """Shape problems of a simulate_stack_spectrum payload (keys, lengths, grid, finiteness)."""
    if not isinstance(spectrum, dict):
        return [f"not a JSON object: {str(spectrum)[:120]}"]
    problems = []
    keys = ("wavelength", "R", "T", "A")
    if sorted(spectrum) != sorted(keys):
        problems.append(f"keys {sorted(spectrum)}")
    points = arguments.get("wavelength_points", 200)
    for key in keys:
        values = spectrum.get(key)
        if not isinstance(values, list) or len(values) != points \
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            problems.append(f"{key}: expected {points} finite numbers")
    if not problems:
        grid = wavelength_grid(arguments)
        if max(abs(a - b) for a, b in zip(spectrum["wavelength"], grid)) > 1e-12:
            problems.append("wavelength grid is not linspace(start, stop, points)")
    return problems


def png_size(data: bytes) -> tuple[int, int]:
    """Width and height from a PNG IHDR chunk."""
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("not a PNG")
    return struct.unpack(">II", data[16:24])


def unphysical(spectrum: dict, slack: float = 1e-9) -> list[str]:
    """Values outside 0 <= R, T, A <= 1 (for passive structures)."""
    bad = []
    for key in ("R", "T", "A"):
        values = spectrum.get(key) or []
        if any(v < -slack or v > 1 + slack for v in values):
            bad.append(f"{key} in [{min(values):.4g}, {max(values):.4g}]")
    return bad


# --------------------------------------------------------------------------
# Tool-call helpers
# --------------------------------------------------------------------------

def simulate(call: Caller, report: Report, check: str, arguments: dict, *, allow_error: bool = False):
    """simulate_stack_spectrum -> (spectrum | {"_isError": True, "_text": ...} | None, image blocks)."""
    result = call(check, "simulate_stack_spectrum", arguments, allow_error=allow_error)
    if result is None:
        return None, []
    images = [b for b in result.get("content", []) if b.get("type") == "image"]
    if result.get("isError"):
        return {"_isError": True, "_text": text_of(result)}, images
    try:
        return json.loads(text_of(result)), images
    except json.JSONDecodeError:
        report.add("L1", check, "FAIL", f"result is not JSON: {text_of(result)[:200]!r}")
        return None, images


def compare(report: Report, check: str, spectrum, arguments: dict, reference, tol: float, what: str) -> None:
    if spectrum is None:
        return
    problems = spectrum_problems(spectrum, arguments)
    if problems:
        report.add("L1", check, "FAIL", "; ".join(problems))
        return
    err = spectrum_error(spectrum, reference)
    report.add("L1", check, "PASS" if err <= tol else "FAIL",
               f"max |dR|,|dT|,|dA| = {err:.1e} vs {what} over {len(reference)} wavelengths (tol {tol:g})",
               max_error=err)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_sanity(call: Caller, report: Report) -> None:
    exact = ((SANITY_N - 1) / (SANITY_N + 1)) ** 2
    for arguments in ({}, {"n_harmonics": 1}, {"n_harmonics": 51}):
        name = f"check_engine_sanity{json.dumps(arguments) if arguments else ''}"
        result = call(name, "check_engine_sanity", arguments)
        if result is None:
            continue
        try:
            payload = json.loads(text_of(result))
        except json.JSONDecodeError:
            report.add("L1", name, "FAIL", f"not JSON: {text_of(result)[:120]!r}")
            continue
        r, t, a = payload.get("R"), payload.get("T"), payload.get("A")
        ok = payload.get("ok") is True and all(isinstance(v, float) for v in (r, t, a)) \
            and abs(r - exact) <= EXACT_TOL and abs(t - (1 - exact)) <= EXACT_TOL and abs(a) <= EXACT_TOL
        report.add("L1", name, "PASS" if ok else "FAIL",
                   f"R={r} vs analytic Fresnel {exact:.12f}, T={t}, A={a}, ok={payload.get('ok')} "
                   f"(the tool's own expected_R {payload.get('expected_R')} is a rounded value)")


def check_readme_example(call: Caller, report: Report) -> None:
    arguments = dict(README_EXAMPLE)
    spectrum, images = simulate(call, report, "simulate_stack_spectrum[README example, plot]", arguments)
    compare(report, "simulate_stack_spectrum[README example vs TMM]", spectrum, arguments,
            stack_reference(arguments), EXACT_TOL, "TMM")
    name = "simulate_stack_spectrum[PNG plot]"
    if len(images) != 1:
        report.add("L1", name, "FAIL", f"{len(images)} image blocks (include_plot defaults to true)")
    else:
        try:
            width, height = png_size(base64.b64decode(images[0].get("data", "")))
            ok = images[0].get("mimeType") == "image/png" and width > 100 and height > 100
            report.add("L1", name, "PASS" if ok else "FAIL", f"{images[0].get('mimeType')} {width}x{height}")
        except (ValueError, struct.error) as exc:
            report.add("L1", name, "FAIL", f"image is not a PNG: {exc}")
    arguments = {**README_EXAMPLE, "include_plot": False}
    result = call("simulate_stack_spectrum[include_plot=false]", "simulate_stack_spectrum", arguments)
    if result is not None:
        kinds = [b.get("type") for b in result.get("content", [])]
        same = False
        try:
            same = json.loads(text_of(result)) == spectrum
        except json.JSONDecodeError:
            pass
        report.add("L1", "simulate_stack_spectrum[include_plot=false]", "PASS" if kinds == ["text"] and same
                   else "FAIL", f"content {kinds}; spectrum identical to the plotted call: {same}")


def check_mirror(call: Caller, report: Report) -> None:
    for pol in ("TE", "TM"):
        for theta in DBR_ANGLES:
            arguments = {"period": 0.5, "materials": DBR_MATERIALS, "layers": DBR_LAYERS,
                         "incidence_layer": "air", "substrate_layer": "substrate", "polarization": pol,
                         "theta_deg": theta, "wavelength_start": 1.2, "wavelength_stop": 1.9,
                         "wavelength_points": 15, "include_plot": False}
            name = f"simulate_stack_spectrum[quarter-wave mirror {pol} {theta:g} deg vs TMM]"
            spectrum, _ = simulate(call, report, name, arguments)
            compare(report, name, spectrum, arguments, stack_reference(arguments), EXACT_TOL, "TMM")


def absorber_arguments(material: dict, pol: str, theta: float) -> dict:
    return {"period": 0.5, "materials": {"glass": {"n": 1.5}, "film": material, "air": {"n": 1.0}},
            "layers": [{"name": "substrate", "thickness": 1.0, "material": "glass"},
                       {"name": "film", "thickness": 0.05, "material": "film"},
                       {"name": "superstrate", "thickness": 1.0, "material": "air"}],
            "incidence_layer": "superstrate", "substrate_layer": "substrate", "polarization": pol,
            "theta_deg": theta, "wavelength_start": 0.5, "wavelength_stop": 1.5, "wavelength_points": 11,
            "include_plot": False}


def check_absorber(call: Caller, report: Report) -> None:
    for pol in ("TE", "TM"):
        for theta in (0.0, 45.0):
            arguments = absorber_arguments(ABSORBER_NK, pol, theta)
            name = f"simulate_stack_spectrum[absorbing film n+ik {pol} {theta:g} deg vs TMM]"
            spectrum, _ = simulate(call, report, name, arguments)
            compare(report, name, spectrum, arguments, stack_reference(arguments), EXACT_TOL, "TMM")
            if pol == "TM" and theta == 45.0 and isinstance(spectrum, dict) and "A" in spectrum:
                eps_args = absorber_arguments(ABSORBER_EPS, pol, theta)
                other, _ = simulate(call, report, "simulate_stack_spectrum[eps notation]", eps_args)
                if isinstance(other, dict) and "A" in other:
                    err = max(abs(a - b) for key in ("R", "T", "A") for a, b in zip(spectrum[key], other[key]))
                    report.add("L1", "simulate_stack_spectrum[eps_real/eps_imag == n/k]",
                               "PASS" if err <= EXACT_TOL else "FAIL",
                               f"max difference {err:.1e}; A up to {max(spectrum['A']):.3f}")


def check_gratings(call: Caller, report: Report) -> None:
    cases = [(pol, theta, nh, lams, None) for pol, theta, nh, lams in GRATING_CASES]
    cases.append((*METAL_CASE, METAL_RIDGE))
    for pol, theta, nh, lams, ridge in cases:
        orders = square_shell_orders(nh)
        arguments = grating_arguments(pol, theta, nh, lams, ridge=ridge)
        label = f"{pol} {theta:g} deg, {nh} harmonics = orders +-{orders}" + (", absorbing ridge" if ridge else "")
        name = f"simulate_stack_spectrum[grating {label} vs 1D RCWA]"
        spectrum, _ = simulate(call, report, name, arguments)
        compare(report, name, spectrum, arguments, grating_reference(arguments, orders, "laurent"), EXACT_TOL,
                "own 1D RCWA (Laurent's rule, same truncation)")


def check_convergence(call: Caller, report: Report) -> None:
    """Informational: S4's default (Laurent) formulation vs the converged Li-rule answer."""
    for pol in ("TE", "TM"):
        errors = {}
        reference = None
        for nh in (51, CONVERGED_NH):
            arguments = grating_arguments(pol, 0.0, nh, (1.0, 1.05), points=2)
            if reference is None:
                reference = grating_reference(arguments, CONVERGED_ORDERS, "li")
            spectrum, _ = simulate(call, report, f"simulate_stack_spectrum[convergence {pol} {nh}]", arguments)
            if isinstance(spectrum, dict) and not spectrum_problems(spectrum, arguments):
                errors[nh] = spectrum_error(spectrum, reference)
        if CONVERGED_NH not in errors:
            continue
        status = "PASS" if errors[CONVERGED_NH] <= CONVERGENCE_WARN else "WARN"
        report.add("L1", f"grating convergence {pol} (vs Li-rule RCWA, {CONVERGED_ORDERS} orders)", status,
                   "; ".join(f"n_harmonics={nh}: {err:.1e}" for nh, err in errors.items())
                   + ("" if pol == "TE" else " — S4's default TM formulation (Laurent's rule) converges slowly; "
                      "results should be checked against n_harmonics as the tool description says"),
                   errors={str(k): v for k, v in errors.items()})


def check_pattern_geometry(call: Caller, report: Report) -> None:
    rect = [0.2, 0.3]
    a, _ = simulate(call, report, "simulate_stack_spectrum[2D rect TE]",
                    grating_arguments("TE", 0.0, 49, OBLIQUE_RANGE, halfwidths=rect))
    b, _ = simulate(call, report, "simulate_stack_spectrum[2D rect rotated TM]",
                    grating_arguments("TM", 0.0, 49, OBLIQUE_RANGE, halfwidths=rect[::-1]))
    if isinstance(a, dict) and isinstance(b, dict) and "R" in a and "R" in b:
        err = max(abs(x - y) for key in ("R", "T", "A") for x, y in zip(a[key], b[key]))
        report.add("L1", "pattern halfwidths [hx, hy] (2D symmetry: rotate 90 deg + swap TE/TM)",
                   "PASS" if err <= EXACT_TOL else "FAIL", f"max difference {err:.1e}")
    c, _ = simulate(call, report, "simulate_stack_spectrum[centred]",
                    grating_arguments("TM", GEOMETRY_THETA, 49, OBLIQUE_RANGE))
    d, _ = simulate(call, report, "simulate_stack_spectrum[shifted centre]",
                    grating_arguments("TM", GEOMETRY_THETA, 49, OBLIQUE_RANGE, center=[0.13, 0.05]))
    if isinstance(c, dict) and isinstance(d, dict) and "R" in c and "R" in d:
        err = max(abs(x - y) for key in ("R", "T", "A") for x, y in zip(c[key], d[key]))
        report.add("L1", "pattern center (translation invariance)", "PASS" if err <= EXACT_TOL else "FAIL",
                   f"max difference {err:.1e}")


def check_repeatability(call: Caller, report: Report) -> None:
    arguments = grating_arguments("TM", 20.0, 81, OBLIQUE_RANGE)
    first, _ = simulate(call, report, "simulate_stack_spectrum[repeat 1]", arguments)
    second, _ = simulate(call, report, "simulate_stack_spectrum[repeat 2]", arguments)
    if first is not None and second is not None:
        report.add("L1", "repeatability", "PASS" if first == second else "FAIL",
                   "bit-identical spectra" if first == second else "spectra differ between identical calls")


def check_errors(call: Caller, report: Report, client: StdioMCP) -> None:
    base = dict(README_EXAMPLE, include_plot=False, wavelength_points=3)
    cases = {
        "wavelength_points=1": {**base, "wavelength_points": 1},
        "wavelength_stop <= start": {**base, "wavelength_stop": 1.4},
        "unknown incidence_layer": {**base, "incidence_layer": "nope"},
        "polarization 'XY'": {**base, "polarization": "XY"},
        "unknown pattern material": {**base, "layers": [
            base["layers"][0], {**base["layers"][1], "pattern": {"material": "nope", "halfwidths": [0.1, 0.5]}},
            base["layers"][2]]},
    }
    for label, arguments in cases.items():
        name = f"simulate_stack_spectrum[{label}]"
        payload, _ = simulate(call, report, name, arguments, allow_error=True)
        if payload is None:
            continue
        if payload.get("_isError"):
            report.add("L1", name, "PASS", f"isError=true: {payload['_text'][:120]}")
        else:
            report.add("L1", name, "FAIL", f"accepted: {str(payload)[:160]}")
    try:
        response = client.call_tool("__nonexistent__", {})
        ok = "error" in response or response.get("result", {}).get("isError") is True
        report.add("L1", "unknown tool is an error", "PASS" if ok else "FAIL", "" if ok else f"got {response}")
    except MCPError as exc:
        report.add("L1", "unknown tool is an error", "FAIL", str(exc))


def _probe(call: Caller, report: Report, label: str, arguments: dict, explain) -> None:
    """A misuse the server should reject; WARN with ``explain(spectrum)`` if it is silently accepted."""
    name = f"simulate_stack_spectrum[{label}]"
    payload, _ = simulate(call, report, name, arguments, allow_error=True)
    if payload is None:
        return
    if payload.get("_isError"):
        report.add("L1", name, "PASS", f"rejected: {payload['_text'][:120]}")
    else:
        report.add("L1", name, "WARN", explain(payload), spectrum=payload)


def check_defects(call: Caller, report: Report) -> None:
    base = dict(README_EXAMPLE, include_plot=False, wavelength_points=3)

    def first(spectrum: dict) -> str:
        return ", ".join(f"{k}={spectrum[k][0]:.4g}" for k in ("R", "T", "A") if spectrum.get(k))

    _probe(call, report, "incidence/substrate swapped",
           {**base, "incidence_layer": "substrate", "substrate_layer": "superstrate"},
           lambda s: f"accepted with an unphysical spectrum ({first(s)}; {'; '.join(unphysical(s))}): "
                     "S4 always illuminates from the last layer and the tool only reads fluxes from the named "
                     "layers, so it never checks that incidence_layer is the top one")
    _probe(call, report, "incidence_layer is an inner layer", {**base, "incidence_layer": "spacer"},
           lambda s: f"accepted ({first(s)}, wavelength-independent) although the light enters from the "
                     "superstrate; the result is not the stack's reflectance")
    reference = stack_reference(base)
    _probe(call, report, "layer material not in materials",
           {**base, "layers": [base["layers"][0], {**base["layers"][1], "material": "nope"}, base["layers"][2]]},
           lambda s: f"accepted ({first(s)}; with the real spacer R={reference[0][0]:.4g}): an unknown layer "
                     "material is passed to S4 as material -1 and the layer silently disappears")
    for theta in (90.0, 120.0):
        _probe(call, report, f"theta_deg={theta:g}", {**base, "theta_deg": theta},
               lambda s: f"accepted, returns {first(s)}: no incident flux reaches the stack, the 1e-15 guard "
                         "sets R=T=0 and A=1-R-T reports total absorption")
    _probe(call, report, "negative layer thickness",
           {**base, "layers": [base["layers"][0], {**base["layers"][1], "thickness": -0.3}, base["layers"][2]]},
           lambda s: f"accepted without validation ({first(s)})")
    _probe(call, report, "duplicate layer names",
           {**base, "layers": [base["layers"][0], base["layers"][1], {**base["layers"][1], "material": "Si"},
                               base["layers"][2]]},
           lambda s: f"accepted ({first(s)}); patterns and flux reads address layers by name, so the "
                     "first 'spacer' becomes unreachable")


def run_l1(client: StdioMCP, report: Report) -> Caller:
    call = Caller(client, report)
    check_sanity(call, report)
    check_readme_example(call, report)
    check_mirror(call, report)
    check_absorber(call, report)
    check_gratings(call, report)
    check_pattern_geometry(call, report)
    check_convergence(call, report)
    check_repeatability(call, report)
    check_errors(call, report, client)
    check_defects(call, report)
    return call


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def server_env(server: dict, home: Path, tmpdir: Path) -> dict:
    venv_bin = str(Path(server["command"]).parent)
    env = {"HOME": str(home), "TMPDIR": str(tmpdir), "PATH": f"{venv_bin}:/usr/bin:/bin",
           "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1", "LANG": "C.UTF-8"}
    env.update(server.get("env", {}))
    return env


def sha256_of(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="generated <root>/s4.mcp.json")
    parser.add_argument("--server", default="s4")
    parser.add_argument("--report", default="s4-smoke-report.json")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).expanduser().read_text(encoding="utf-8"))
    server = config["mcpServers"][args.server]
    manifest = {e["id"]: e for e in json.loads((HERE / "manifest.json").read_text())["servers"]}
    entry = manifest[args.server]
    checkout = Path(server["cwd"])
    try:
        revision = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None

    report = Report()
    if revision != entry["revision"]:
        report.add("L0", "pinned revision", "FAIL", f"checkout at {revision}, manifest pins {entry['revision']}")
    want_path = str(checkout / "src")
    got_path = server.get("env", {}).get("PYTHONPATH")
    report.add("L0", "launch env in config", "PASS" if got_path == want_path else "FAIL",
               f"PYTHONPATH={got_path}" + ("" if got_path == want_path else f" (expected {want_path})"))
    digest = sha256_of(checkout / LIBS4)
    if digest == UPSTREAM_LIBS4_SHA256:
        report.add("L0", "vendored libS4.so", "PASS", f"upstream binary, sha256 {digest[:16]}…")
    else:
        report.add("L0", "vendored libS4.so", "WARN",
                   f"sha256 {digest} is not the upstream x86-64 binary ({UPSTREAM_LIBS4_SHA256[:16]}…): "
                   "a locally built S4 — fine for development, not evidence for the pinned server")

    call = None
    stderr_tail = ""
    with tempfile.TemporaryDirectory(prefix="mcp-e2e-s4-") as tmp:
        tmp_path = Path(tmp)
        home, cwd, scratch = tmp_path / "home", tmp_path / "cwd", tmp_path / "tmpdir"
        for d in (home, cwd, scratch):
            d.mkdir()
        stderr_path = tmp_path / "server.stderr.log"
        client = StdioMCP(server["command"], server["args"], cwd=cwd, env=server_env(server, home, scratch),
                          stderr_path=stderr_path)
        try:
            try:
                info = client.initialize()
                report.add("L0", "initialize", "PASS",
                           f"server={info.get('serverInfo')} protocol={info.get('protocolVersion')}")
                names = sorted(t["name"] for t in client.list_tools())
                expected = sorted(entry["expected_tools"])
                report.add("L0", "tools/list", "PASS" if names == expected else "FAIL",
                           f"{len(names)} tools" + ("" if names == expected else
                                                    f"; expected {expected}, got {names}"))
                again = sorted(t["name"] for t in client.list_tools())
                report.add("L0", "tools/list stable", "PASS" if again == names else "FAIL")
            except MCPError as exc:
                report.add("L0", "handshake", "FAIL", str(exc))
            else:
                call = run_l1(client, report)
            alive = client.proc.poll() is None
            report.add("L1", "server alive after calls", "PASS" if alive else "FAIL",
                       "" if alive else f"exit code {client.proc.returncode}")
        finally:
            client.close()
            stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]

        polluted = client.non_json_stdout
        per_tool = {k: v for k, v in sorted(call.stdout_by_tool.items()) if v} if call else {}
        report.add("L1", "stdout is pure JSON-RPC", "WARN" if polluted else "PASS",
                   f"{len(polluted)} non-JSON line(s), per tool {per_tool}, e.g. {polluted[:3]}" if polluted else "",
                   non_json_stdout=polluted[:50], non_json_stdout_by_tool=per_tool)
        leftovers = sorted(p.name for p in cwd.iterdir())
        report.add("L1", "server leaves cwd untouched", "WARN" if leftovers else "PASS",
                   f"created {leftovers}" if leftovers else "")

    document = {
        "server": args.server,
        "repository": entry["repository"],
        "revision": revision,
        "libS4_sha256": digest,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "environment": package_versions(("mcp", "numpy", "matplotlib", "pydantic")),
        "result": "FAIL" if report.failed else "PASS",
        "summary": dict(Counter(c["status"] for c in report.checks)),
        "checks": report.checks,
        "server_stderr_tail": stderr_tail,
    }
    Path(args.report).write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{document['result']} {document['summary']}  report -> {Path(args.report).resolve()}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
