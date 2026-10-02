"""Instance generator for the s4 grating-spectrum MCP E2E fake task.

A seeded RNG picks a lamellar dielectric grating on glass (period, ridge
material, width and height), an oblique TM plane wave, a harmonic count and a
wavelength sweep. The agent must simulate it with the ``simulate_stack_spectrum``
tool of the s4 (S4 RCWA) MCP server and report R and T at one sweep point plus
the maximum reflectance and where it occurs.

The reference is computed here with numpy only, independently of S4 and the
server: a 1D rigorous coupled-wave analysis (enhanced transmittance matrix).
S4's default formulation uses Laurent's rule for both polarisations and keeps
the reciprocal-lattice vectors inside a circle, so for a complete shell of the
square lattice (21 / 37 / 81 harmonics) and a grating that is uniform along y
it couples exactly the x-axis orders -M..M (M = 2 / 3 / 5). The 1D RCWA with
Laurent's rule and the same M agrees with the server to ~1e-14
(scripts/mcp/e2e/e2e_smoke/servers/s4.py checks this on every L1 run).

The cases are chosen so that the tool is the only practical source of the
numbers: at the reported point, the converged answer (Li's inverse rule, many
orders) and a neighbouring truncation (M + 1) both differ from S4's value by
far more than the full-credit tolerance, so a home-made RCWA does not
reproduce it by accident. Grid points stay clear of Rayleigh anomalies, the
reported point is not the reflectance maximum, and the maximum is unique and
inside the sweep.

Framework call: python generate_gt.py --output-dir <dir> --params '<json>'
Needs numpy: generate with ``asibench generate --sandbox task``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

INPUT_SPEC = [
    {"name": "grating.json", "description": "grating geometry, materials, illumination, harmonics and sweep"},
]
OUTPUT_SPEC = [
    {"name": "result.json", "description": "R_at_report, T_at_report, R_max, wavelength_at_R_max_um"},
]
DEFAULT_PARAMS = {"seed": 0}

# Complete square-lattice shells -> x-axis orders kept by S4 for a y-uniform grating.
# The server default (51 harmonics) keeps +-4, which none of these does.
HARMONICS_ORDERS = {21: 2, 37: 3, 81: 5}
RIDGE_MATERIALS = [("Si3N4", 2.0), ("Ta2O5", 2.1), ("TiO2", 2.5), ("Si", 3.48)]
SUBSTRATE = ("glass", 1.5)
AMBIENT = ("air", 1.0)
SEMI_INFINITE_THICKNESS = 1.0       # first/last S4 layers are semi-infinite; any positive value
CONVERGED_ORDERS = 40               # Li's rule, the "converged" answer a generic RCWA would give
MIN_BYPASS_MARGIN = 1e-3            # |S4 value - converged| and |S4 value - (M+1) value| at the report point
MIN_MAX_GAP = 1e-4                  # largest R must beat the second largest by this
ANOMALY_CLEARANCE = 0.01            # um between every grid point and every Rayleigh anomaly
SANITY_N = 3.47                     # check_engine_sanity: air / n = 3.47, normal incidence


# --------------------------------------------------------------------------
# Independent 1D RCWA (numpy only; same code as scripts/mcp/e2e/e2e_smoke/servers/s4.py)
# --------------------------------------------------------------------------

def _eps_fourier(eps_bg: complex, eps_ridge: complex, halfwidth: float, center: float,
                 period: float, orders: int) -> np.ndarray:
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
    """Total R, T of lamellar gratings; layers top -> bottom as (thickness, eps_bg, eps_ridge, halfwidth, center)."""
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


def rayleigh_wavelengths(period: float, theta_deg: float, indices: tuple[float, ...], max_order: int = 4) -> list[float]:
    s = indices[0] * math.sin(math.radians(theta_deg))
    return sorted(period * (n + sign * s) / m for n in indices for m in range(1, max_order + 1)
                  for sign in (1, -1) if period * (n + sign * s) / m > 0)


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

def tool_arguments(case: dict) -> dict:
    """The simulate_stack_spectrum arguments that describe the case (layers bottom -> top)."""
    g = case["grating"]
    ridge, groove = g["ridge_material"], g["groove_material"]
    sub, sup = case["substrate"], case["superstrate"]
    return {
        "period": case["period_um"],
        "materials": {sub["name"]: {"n": sub["n"]}, groove["name"]: {"n": groove["n"]},
                      ridge["name"]: {"n": ridge["n"]}},
        "layers": [
            {"name": "substrate", "thickness": SEMI_INFINITE_THICKNESS, "material": sub["name"]},
            {"name": "grating", "thickness": g["thickness_um"], "material": groove["name"],
             "pattern": {"material": ridge["name"], "halfwidths": [g["ridge_width_um"] / 2, case["period_um"] / 2],
                         "center": [0.0, 0.0]}},
            {"name": "superstrate", "thickness": SEMI_INFINITE_THICKNESS, "material": sup["name"]},
        ],
        "incidence_layer": "superstrate",
        "substrate_layer": "substrate",
        "polarization": case["illumination"]["polarization"],
        "theta_deg": case["illumination"]["theta_deg"],
        "n_harmonics": case["n_harmonics"],
        "wavelength_start": case["sweep"]["wavelength_start_um"],
        "wavelength_stop": case["sweep"]["wavelength_stop_um"],
        "wavelength_points": case["sweep"]["wavelength_points"],
        "include_plot": False,
    }


def wavelength_grid(case: dict) -> list[float]:
    sw = case["sweep"]
    return np.linspace(sw["wavelength_start_um"], sw["wavelength_stop_um"], sw["wavelength_points"]).tolist()


def spectrum(case: dict, orders: int, rule: str, wavelengths: list[float] | None = None) -> list[tuple[float, float]]:
    g = case["grating"]
    layer = (g["thickness_um"], g["groove_material"]["n"] ** 2, g["ridge_material"]["n"] ** 2,
             g["ridge_width_um"] / 2, 0.0)
    ill = case["illumination"]
    return [rcwa_1d(ill["polarization"], lam, ill["theta_deg"], case["period_um"], case["superstrate"]["n"],
                    case["substrate"]["n"], [layer], orders, rule)
            for lam in (wavelengths if wavelengths is not None else wavelength_grid(case))]


def _candidate(rng: random.Random) -> dict:
    period = rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])
    name, n_ridge = RIDGE_MATERIALS[rng.randrange(len(RIDGE_MATERIALS))]
    fill = rng.choice([0.3, 0.4, 0.5, 0.6, 0.7])
    harmonics = rng.choice(sorted(HARMONICS_ORDERS))
    points = rng.choice([11, 13, 15, 17, 21])
    step = rng.choice([0.01, 0.015, 0.02])                    # um; grid points are multiples of 5 nm
    start = round(round(period * rng.uniform(1.04, 1.2) / 0.005) * 0.005, 3)
    stop = round(start + step * (points - 1), 3)
    return {
        "period_um": period,
        "superstrate": {"name": AMBIENT[0], "n": AMBIENT[1]},
        "grating": {"thickness_um": rng.choice([0.1, 0.15, 0.2, 0.25, 0.3, 0.4]),
                    "groove_material": {"name": AMBIENT[0], "n": AMBIENT[1]},
                    "ridge_material": {"name": name, "n": n_ridge},
                    "ridge_width_um": round(fill * period, 4),
                    "geometry": "ridges parallel to y (uniform along y), periodic along x, centred at x = 0"},
        "substrate": {"name": SUBSTRATE[0], "n": SUBSTRATE[1]},
        "illumination": {"polarization": "TM", "theta_deg": float(rng.choice([5, 10, 15, 20])),
                         "plane_of_incidence": "xz", "from": "superstrate"},
        "n_harmonics": harmonics,
        "sweep": {"wavelength_start_um": start, "wavelength_stop_um": stop, "wavelength_points": points},
        "units": {"length": "um", "wavelength": "um, vacuum", "angle": "deg from the surface normal"},
    }


def evaluate(case: dict) -> dict | None:
    """Reference values for a candidate, or None if it fails a selection rule."""
    grid = wavelength_grid(case)
    anomalies = rayleigh_wavelengths(case["period_um"], case["illumination"]["theta_deg"],
                                     (case["superstrate"]["n"], case["substrate"]["n"]))
    if any(min(abs(lam - a) for a in anomalies) < ANOMALY_CLEARANCE for lam in grid):
        return None
    orders = HARMONICS_ORDERS[case["n_harmonics"]]
    spec = spectrum(case, orders, "laurent")
    r = [v[0] for v in spec]
    t = [v[1] for v in spec]
    ranked = sorted(range(len(r)), key=lambda i: r[i], reverse=True)
    if r[ranked[0]] - r[ranked[1]] < MIN_MAX_GAP or ranked[0] in (0, len(r) - 1):
        return None                      # a unique maximum inside the sweep, not at an end
    if not all(-1e-9 <= x <= 1 + 1e-9 for x in r + t) or max(abs(a + b - 1.0) for a, b in spec) > 1e-9:
        return None
    return {"wavelengths": grid, "R": r, "T": t, "argmax": ranked[0], "orders": orders}


def build_case(seed: int) -> dict:
    """A case that passes every selection rule, plus its reference values."""
    rng = random.Random(seed)
    for _ in range(500):
        case = _candidate(rng)
        ref = evaluate(case)
        if ref is None:
            continue
        others = [i for i in range(len(ref["R"])) if i != ref["argmax"]]
        rng.shuffle(others)
        for index in others:
            lam = ref["wavelengths"][index]
            converged = spectrum(case, CONVERGED_ORDERS, "li", [lam])[0][0]
            next_trunc = spectrum(case, ref["orders"] + 1, "laurent", [lam])[0][0]
            margin = min(abs(ref["R"][index] - converged), abs(ref["R"][index] - next_trunc))
            if margin >= MIN_BYPASS_MARGIN:
                case["report_point_index"] = index
                case["report_wavelength_um"] = lam
                return {"case": case, "ref": ref, "converged_R_at_report": converged,
                        "next_truncation_R_at_report": next_trunc, "bypass_margin": margin}
    raise RuntimeError(f"no admissible case for seed {seed}")


def render_prompts(task_dir: Path, output_dir: Path, case: dict) -> None:
    g = case["grating"]
    values = {"ridge": g["ridge_material"]["name"], "period_um": f"{case['period_um']:g}",
              "polarization": case["illumination"]["polarization"]}
    for level in ("b1", "b2", "b3", "b4"):
        text = (task_dir / f"prompt_{level}.md").read_text(encoding="utf-8")
        for key, value in values.items():
            text = text.replace("{{" + key + "}}", value)
        (output_dir / f"prompt_{level}.md").write_text(text, encoding="utf-8")


def generate(output_dir: Path, params: dict) -> dict:
    p = {**DEFAULT_PARAMS, **params}
    t0 = time.time()
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    ref_dir = output_dir / "reference"
    data_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    built = build_case(int(p["seed"]))
    case, ref = built["case"], built["ref"]
    k, j = case["report_point_index"], ref["argmax"]
    (data_dir / "grating.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    args = tool_arguments(case)
    reference = {
        "case": case,
        "tool_arguments": args,
        "period_um": case["period_um"],
        "polarization": case["illumination"]["polarization"],
        "theta_deg": case["illumination"]["theta_deg"],
        "n_harmonics": case["n_harmonics"],
        "wavelength_start_um": case["sweep"]["wavelength_start_um"],
        "wavelength_stop_um": case["sweep"]["wavelength_stop_um"],
        "wavelength_points": case["sweep"]["wavelength_points"],
        "orders": ref["orders"],
        "wavelengths_um": ref["wavelengths"],
        "R_spectrum": ref["R"],
        "T_spectrum": ref["T"],
        "report_point_index": k,
        "report_wavelength_um": case["report_wavelength_um"],
        "R_at_report": ref["R"][k],
        "T_at_report": ref["T"][k],
        "R_max": ref["R"][j],
        "wavelength_at_R_max_um": ref["wavelengths"][j],
        "R_max_index": j,
        "converged_R_at_report": built["converged_R_at_report"],
        "next_truncation_R_at_report": built["next_truncation_R_at_report"],
        "bypass_margin": built["bypass_margin"],
        "sanity_R": ((SANITY_N - 1) / (SANITY_N + 1)) ** 2,
        "method": "1D RCWA, Laurent's rule, orders -M..M (S4 default formulation, circular truncation)",
        "numpy_version": np.__version__,
        "mcp_tools": ["simulate_stack_spectrum", "check_engine_sanity"],
    }
    (ref_dir / "reference.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")

    render_prompts(Path(__file__).resolve().parent, output_dir, case)
    meta = {
        "params_used": p,
        "input_files": [s["name"] for s in INPUT_SPEC],
        "reference_files": ["reference.json"],
        "generation_time_seconds": round(time.time() - t0, 2),
    }
    (output_dir / "instance_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params", type=str, default="{}")
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, json.loads(args.params)), indent=2))
