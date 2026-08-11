"""Parametric ground-truth generator for materials.phonon_dispersion.

Computes phonon dispersion, density of states (DOS), and harmonic heat
capacity for a 1D diatomic chain with equal masses and alternating spring
constants (Simon, "The Oxford Solid State Basics", Ch.10).

Model:
  - N atoms, all mass m
  - Alternating springs κ1, κ2, κ1, κ2, ...
  - Lattice constant a (unit cell = 2 atoms)
  - Reduced units: a = k_B = ħ = 1

Output directory layout:
  <output_dir>/
  ├── data/
  │   └── params.json           physical and numerical parameters for the agent
  ├── reference/
  │   ├── dispersion_ref.npy    shape (2, n_kpoints) float64
  │   ├── dos_ref.npy           shape (2, N_DOS_BINS) float64
  │   └── heat_capacity_ref.npy shape (2, N_TEMPS) float64
  ├── prompt_b1.md
  ├── prompt_b2.md
  ├── prompt_b3.md
  ├── prompt_b4.md
  └── instance_meta.json
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np


# ── Fixed numerical constants ─────────────────────────────────────────────────

LATTICE_CONSTANT = 1.0   # a (reduced units)
N_DOS_BINS = 200         # frequency bins in DOS output
N_TEMPS = 100            # temperature points for Cv output
T_MIN = 0.1              # minimum temperature (reduced: k_B = ħ = 1)
T_MAX = 10.0             # maximum temperature
N_DENSE = 5000           # k-points for dense DOS/Cv mesh

TEMPLATE_PATTERN = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")

DEFAULT_PARAMS = {
    "mass": 1.0,
    "kappa1": 1.0,
    "kappa2": 0.5,
    "n_kpoints": 200,
    "seed": 0,
}

# ── I/O specification (must match task.yaml) ──────────────────────────────────

INPUT_SPEC = [
    {"name": "params.json",
     "description": "Physical and numerical parameters for the phonon calculation"},
]

OUTPUT_SPEC = [
    {"name": "dispersion.npy",
     "description": "Phonon dispersion, shape (2, n_kpoints) float64; row 0 = acoustic, row 1 = optical"},
    {"name": "dos.npy",
     "description": "DOS, shape (2, N_DOS_BINS) float64; row 0 = freq axis, row 1 = DOS"},
    {"name": "heat_capacity.npy",
     "description": "Heat capacity, shape (2, N_TEMPS) float64; row 0 = T axis, row 1 = Cv"},
]


# ── Physics ───────────────────────────────────────────────────────────────────

def dispersion_1d(mass: float, kappa1: float, kappa2: float,
                  k: np.ndarray) -> np.ndarray:
    """1D diatomic chain dispersion (Simon Ch.10: equal masses, alternating springs).

    ω²±(k) = (κ1+κ2)/m ± (1/m)√(κ1²+κ2²+2κ1κ2·cos(k·a))

    Args:
        k: wavevectors in [0, π/a], shape (n_k,)

    Returns:
        shape (2, n_k): row 0 = acoustic branch, row 1 = optical branch
    """
    a = LATTICE_CONSTANT
    term1 = (kappa1 + kappa2) / mass
    under_root = kappa1**2 + kappa2**2 + 2.0 * kappa1 * kappa2 * np.cos(k * a)
    term2 = np.sqrt(np.maximum(under_root, 0.0)) / mass
    omega_ac = np.sqrt(np.maximum(term1 - term2, 0.0))
    omega_op = np.sqrt(np.maximum(term1 + term2, 0.0))
    return np.stack([omega_ac, omega_op], axis=0)  # (2, n_k)


def compute_dos(omega_flat: np.ndarray, omega_max: float) -> np.ndarray:
    """Build normalized DOS histogram.

    Returns shape (2, N_DOS_BINS): row 0 = bin-center frequencies, row 1 = DOS.
    DOS is normalized per branch: ∫ DOS(ω) dω = n_branches = 2.
    (omega_flat has shape (2, N_DENSE), so omega_flat.size = 2*N_DENSE;
     we normalize by N_DENSE * d_omega so that ∫ DOS dω = 2.)
    """
    edges = np.linspace(0.0, omega_max, N_DOS_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(omega_flat.ravel(), bins=edges)
    d_omega = omega_max / N_DOS_BINS
    n_kpoints_per_branch = omega_flat.size // 2  # = N_DENSE
    total = n_kpoints_per_branch * d_omega
    dos = counts.astype(np.float64) / total if total > 0 else counts.astype(np.float64)
    return np.stack([centers, dos], axis=0)  # (2, N_DOS_BINS)


def compute_heat_capacity(omega_flat: np.ndarray, T_array: np.ndarray) -> np.ndarray:
    """Harmonic heat capacity per k-point (quantum Einstein formula, summed over branches).

    Cv(T) = (1/N_DENSE) × Σ_{all modes} (ω/T)² · exp(ω/T) / (exp(ω/T) − 1)²

    Sum over 2*N_DENSE modes, divide by N_DENSE (k-points per branch), so Cv → 2
    (= n_branches) at high temperature (Dulong-Petit limit per k-point).
    Uses reduced units k_B = ħ = 1. Zero modes (ω < 1e-10) are excluded.
    """
    n_kpoints_per_branch = omega_flat.size // 2  # = N_DENSE
    omega = omega_flat.ravel()
    omega = omega[omega > 1e-10]
    Cv = np.empty(len(T_array))
    for i, T in enumerate(T_array):
        x = np.clip(omega / T, 0.0, 500.0)
        ex = np.exp(x)
        Cv[i] = float(np.sum(x**2 * ex / (ex - 1.0)**2)) / n_kpoints_per_branch
    return Cv


def omega_max(mass: float, kappa1: float, kappa2: float) -> float:
    """Analytical ω_max: optical branch at k=0."""
    return float(np.sqrt(2.0 * (kappa1 + kappa2) / mass))


# ── Reference runner ──────────────────────────────────────────────────────────

def _run_reference(p: dict):
    a = LATTICE_CONSTANT
    mass, kappa1, kappa2 = p["mass"], p["kappa1"], p["kappa2"]
    n_kpoints = p["n_kpoints"]
    T_array = np.linspace(T_MIN, T_MAX, N_TEMPS)

    omax = omega_max(mass, kappa1, kappa2)

    # Dispersion on requested k-path [0, π/a]
    k_path = np.linspace(0.0, np.pi / a, n_kpoints)
    dispersion = dispersion_1d(mass, kappa1, kappa2, k_path)  # (2, n_kpoints)

    # Dense mesh for DOS / Cv (both branches → 2*N_DENSE frequencies)
    k_dense = np.linspace(0.0, np.pi / a, N_DENSE)
    omega_flat = dispersion_1d(mass, kappa1, kappa2, k_dense)  # (2, N_DENSE)

    dos = compute_dos(omega_flat, omax * 1.005)
    Cv = compute_heat_capacity(omega_flat, T_array)
    heat_capacity = np.stack([T_array, Cv], axis=0)  # (2, N_TEMPS)

    return dispersion, dos, heat_capacity, omax


# ── Prompt rendering ──────────────────────────────────────────────────────────

def _build_prompt_context(p: dict, omax: float) -> dict:
    return {
        "mass":      f"{p['mass']:.4g}",
        "kappa1":    f"{p['kappa1']:.4g}",
        "kappa2":    f"{p['kappa2']:.4g}",
        "n_kpoints": str(p["n_kpoints"]),
        "omega_max": f"{omax * 1.005:.6f}",
        "T_min":     str(T_MIN),
        "T_max":     str(T_MAX),
        "n_temps":   str(N_TEMPS),
        "n_dos_bins": str(N_DOS_BINS),
        "lattice_constant": str(LATTICE_CONSTANT),
    }


def _render(template: str, ctx: dict) -> str:
    return TEMPLATE_PATTERN.sub(lambda m: ctx.get(m.group(1), m.group(0)), template)


# ── Main generator ────────────────────────────────────────────────────────────

def generate(output_dir: Path, params: dict) -> dict:
    p = {**DEFAULT_PARAMS, **params}
    t0 = time.time()

    data_dir = output_dir / "data"
    ref_dir = output_dir / "reference"
    data_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    dispersion, dos, heat_capacity, omax = _run_reference(p)

    np.save(ref_dir / "dispersion_ref.npy", dispersion)
    np.save(ref_dir / "dos_ref.npy", dos)
    np.save(ref_dir / "heat_capacity_ref.npy", heat_capacity)

    params_json = {
        "mass":             p["mass"],
        "kappa1":           p["kappa1"],
        "kappa2":           p["kappa2"],
        "n_kpoints":        p["n_kpoints"],
        "lattice_constant": LATTICE_CONSTANT,
        "omega_max":        round(omax * 1.005, 6),
        "n_dos_bins":       N_DOS_BINS,
        "T_min":            T_MIN,
        "T_max":            T_MAX,
        "n_temps":          N_TEMPS,
    }
    (data_dir / "params.json").write_text(json.dumps(params_json, indent=2), encoding="utf-8")

    task_dir = Path(__file__).parent
    ctx = _build_prompt_context(p, omax)
    for level in ["b1", "b2", "b3", "b4"]:
        tpl = task_dir / f"prompt_{level}.md"
        text = _render(tpl.read_text(encoding="utf-8"), ctx) if tpl.exists() \
            else f"# Phonon Dispersion\n\nPrompt level {level}."
        (output_dir / f"prompt_{level}.md").write_text(text, encoding="utf-8")

    meta = {
        "params_used": p,
        "omega_max": round(omax, 6),
        "dispersion_shape": list(dispersion.shape),
        "dos_shape": list(dos.shape),
        "heat_capacity_shape": list(heat_capacity.shape),
        "input_files": [s["name"] for s in INPUT_SPEC],
        "reference_files": ["dispersion_ref.npy", "dos_ref.npy", "heat_capacity_ref.npy"],
        "generation_time_seconds": round(time.time() - t0, 2),
    }
    (output_dir / "instance_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params", type=str, default="{}")
    args = parser.parse_args()
    result = generate(args.output_dir, json.loads(args.params))
    print(json.dumps(result, indent=2))
