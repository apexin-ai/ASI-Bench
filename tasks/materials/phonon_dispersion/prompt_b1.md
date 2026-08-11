# 1D Diatomic Chain Phonon Dispersion, DOS, and Heat Capacity

> **Level B1**: Full algorithm description — provides complete implementation details.

## Problem

Compute the phonon dispersion relation, density of states (DOS), and harmonic heat capacity for a **1D diatomic chain** using the model from Simon, *The Oxford Solid State Basics*, Chapter 10.

Read all parameters from `data/params.json`.

## Parameters

| Key | Value | Description |
|-----|-------|-------------|
| `mass` | `{{ mass }}` | Atomic mass m (same for all atoms, reduced units) |
| `kappa1` | `{{ kappa1 }}` | Spring constant κ1 |
| `kappa2` | `{{ kappa2 }}` | Spring constant κ2 |
| `n_kpoints` | `{{ n_kpoints }}` | Number of k-points along dispersion path |
| `lattice_constant` | `{{ lattice_constant }}` | Lattice constant a |
| `omega_max` | `{{ omega_max }}` | Upper bound for DOS frequency axis |
| `n_dos_bins` | `{{ n_dos_bins }}` | Number of frequency bins for DOS |
| `T_min` / `T_max` / `n_temps` | `{{ T_min }}` / `{{ T_max }}` / `{{ n_temps }}` | Temperature grid for Cv |

All calculations use **reduced units**: `a = k_B = ħ = 1`.

---

## Step 1 — Phonon Dispersion

**Model**: N atoms all with mass m, connected by alternating springs κ1 and κ2. Unit cell contains 2 atoms.

**Dispersion relation:**

```
ω²±(k) = (κ1+κ2)/m ± (1/m) × sqrt(κ1² + κ2² + 2κ1κ2·cos(k·a))
```

- `−` → **acoustic branch** (ω → 0 as k → 0)
- `+` → **optical branch**
- k = `n_kpoints` evenly spaced values in `[0, π/a]` (inclusive)

**Output** `dispersion.npy`: shape `(2, n_kpoints)` float64
- Row 0: acoustic branch
- Row 1: optical branch

---

## Step 2 — Density of States

Use a **dense k-mesh** of `5000` evenly spaced k-points in `[0, π/a]`. Compute both branches → 10000 total frequencies.

Build a histogram with `n_dos_bins` bins over `[0, omega_max]`. Normalize so that `∫ DOS(ω) dω = 2` (one per branch): divide counts by `(n_samples/2) × d_omega` where `n_samples = 10000` (total frequencies from 2 branches × 5000 k-points) and `d_omega = omega_max / n_dos_bins`.

**Output** `dos.npy`: shape `(2, n_dos_bins)` float64
- Row 0: bin-center frequencies
- Row 1: normalized DOS values

---

## Step 3 — Harmonic Heat Capacity

For each temperature T in `n_temps` linearly spaced values from `T_min` to `T_max`:

```
Cv(T) = (1 / N_k) × Σ_{all modes} (ω/T)² × exp(ω/T) / (exp(ω/T) − 1)²
```

where `N_k = 5000` (k-points per branch, **not** the total number of modes). The sum runs over all `2 × N_k` modes (both branches). At high temperature this approaches `2` (Dulong-Petit limit, 2 modes per k-point).

Use the same dense k-mesh frequencies as Step 2. Exclude modes with `ω < 1e-10`. Clip `ω/T` at 500 to prevent overflow.

**Output** `heat_capacity.npy`: shape `(2, n_temps)` float64
- Row 0: temperature values (linspace from T_min to T_max)
- Row 1: Cv values

---

## Output Summary

| File | Shape | dtype |
|------|-------|-------|
| `dispersion.npy` | `(2, {{ n_kpoints }})` | float64 |
| `dos.npy` | `(2, {{ n_dos_bins }})` | float64 |
| `heat_capacity.npy` | `(2, {{ n_temps }})` | float64 |

Save all output files to the **current working directory**. Use only `numpy`. Write a single runnable script with no command-line arguments.
