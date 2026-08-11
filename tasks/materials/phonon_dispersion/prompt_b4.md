# 1D Diatomic Chain

> **Level B4**: B3 + factually correct but redundant information.

## Task

`data/params.json` contains the physical parameters of a 1D diatomic chain. Compute and save:

1. **`dispersion.npy`** — shape `(2, {{ n_kpoints }})`, dtype `float64`
2. **`dos.npy`** — shape `(2, {{ n_dos_bins }})`, dtype `float64`; row 0 = frequency axis `[0, omega_max]`, row 1 = values
3. **`heat_capacity.npy`** — shape `(2, {{ n_temps }})`, dtype `float64`; row 0 = temperature axis `[T_min, T_max]`, row 1 = values

Save all files to the current working directory. Use only `numpy`.

---

## Additional Context

> The following information is factually correct but not all of it is necessary for this task.

### Phonon Density of States

The phonon DOS can be computed by several methods:

- **Histogram (linear sampling)**: Uniformly sample k-points in the Brillouin zone, compute frequencies, build a histogram. Simple and sufficient for smooth dispersions.
- **Gaussian smearing**: Replace each delta function with a Gaussian of finite width σ. Smooths Van Hove singularities but introduces a free parameter σ.
- **Linear tetrahedron method**: Partition the Brillouin zone into tetrahedra and interpolate frequencies linearly within each. Most accurate near Van Hove singularities; standard in ab initio codes.
- **Adaptive smearing**: Automatically adjusts the smearing width based on the local group velocity; balances accuracy and smoothness.

### Heat Capacity Models

Several models exist for lattice heat capacity:

- **Einstein model** (1907): All atoms oscillate at a single frequency ω_E. Cv = (ω_E/T)²·exp(ω_E/T)/(exp(ω_E/T)−1)². Captures quantum freeze-out at low T but misses the acoustic branch contribution.
- **Debye model** (1912): Assumes linear acoustic dispersion ω = v_s·k up to a cutoff ω_D. Correctly gives Cv ∝ T³ at low temperature. The Debye temperature Θ_D = ħω_D/k_B is a widely tabulated material property.
- **Full phonon calculation**: Sums the quantum harmonic oscillator contribution over all modes using the actual phonon DOS. Reduces to Einstein or Debye limits in the appropriate regimes.

### Anharmonic Effects

The harmonic approximation treats interatomic forces as linear (spring-like). In reality, at high temperatures anharmonic effects become important:

- Thermal expansion: lattice constant increases with temperature, shifting phonon frequencies
- Phonon-phonon scattering: limits thermal conductivity
- Quasi-harmonic approximation (QHA): accounts for volume-dependent force constants without full anharmonic treatment

For the current task parameters, the harmonic approximation is sufficient.
