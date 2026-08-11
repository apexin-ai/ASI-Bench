# 1D Diatomic Chain

> **Level B3**: Research goal + output format only — minimal information.

## Task

`data/params.json` contains the physical parameters of a 1D diatomic chain. Compute and save:

1. **`dispersion.npy`** — shape `(2, {{ n_kpoints }})`, dtype `float64`
2. **`dos.npy`** — shape `(2, {{ n_dos_bins }})`, dtype `float64`; row 0 = frequency axis `[0, omega_max]`, row 1 = values
3. **`heat_capacity.npy`** — shape `(2, {{ n_temps }})`, dtype `float64`; row 0 = temperature axis `[T_min, T_max]`, row 1 = values

Save all files to the current working directory. Use only `numpy`.
