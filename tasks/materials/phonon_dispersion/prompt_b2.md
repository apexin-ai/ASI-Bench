# 1D Diatomic Chain Phonon Dispersion, DOS, and Heat Capacity

> **Level B2**: Scientific background + method name — no algorithm details.

## Background

In a crystal with more than one atom per unit cell, the phonon dispersion relation splits into acoustic and optical branches. The simplest model is the **1D diatomic chain**: atoms of equal mass connected by alternating spring constants κ1 and κ2. When κ1 ≠ κ2, a band gap opens at the Brillouin zone boundary.

Key quantities:
- **Dispersion relation** ω(k): phonon frequency as a function of wavevector
- **Density of states (DOS)**: distribution of phonon frequencies
- **Heat capacity** Cv: obtained by averaging the quantum harmonic oscillator contribution over all phonon modes

## Task

Read parameters from `data/params.json`:
- `mass`: `{{ mass }}` — atomic mass m (same for all atoms)
- `kappa1`: `{{ kappa1 }}`, `kappa2`: `{{ kappa2 }}` — alternating spring constants
- `n_kpoints`: `{{ n_kpoints }}` — resolution of the dispersion path
- `omega_max`, `n_dos_bins`, `T_min`, `T_max`, `n_temps` — grid parameters for DOS and Cv

Use reduced units throughout (`a = k_B = ħ = 1`, where `a` = `lattice_constant` in params).

## Method

Apply the **harmonic lattice dynamics** model for a 1D diatomic chain:

1. **Dispersion**: Diagonalize the dynamical matrix to get acoustic and optical branches for k in [0, π/a]
2. **DOS**: Sample frequencies on a dense k-mesh and build a normalized histogram
3. **Heat capacity**: Average the quantum Einstein heat capacity formula over all sampled modes for each temperature

## Output Files

| File | Shape | dtype | Description |
|------|-------|-------|-------------|
| `dispersion.npy` | `(2, {{ n_kpoints }})` | float64 | Row 0 = acoustic, row 1 = optical |
| `dos.npy` | `(2, {{ n_dos_bins }})` | float64 | Row 0 = freq axis `[0, omega_max]`; row 1 = normalized DOS |
| `heat_capacity.npy` | `(2, {{ n_temps }})` | float64 | Row 0 = T values; row 1 = Cv |

Save to current working directory. Use only `numpy`.
